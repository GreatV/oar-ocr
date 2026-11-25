//! Script router for text lines.
//!
//! The model scores each line crop for four recognizer groups and for being upside down. Page-level routing follows LiteParse's ocria-nano runtime (Apache-2.0). See `processors/NOTICE`.
use crate::core::OCRError;
use crate::core::config::{ModelInferenceConfig, OrtSessionConfig};
use crate::core::inference::{ModelSource, OrtInfer, TensorInput};
use crate::models::recognition::crnn::{GRAY_LINE_HEIGHT, gray_line};
use image::RgbImage;

const WIDTH: usize = 512;
const BATCH: usize = 64;

/// Page-level routing of line crops.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LineRouting {
    /// Recognizer group of each line: 0 Latin, 1 CJK, 2 right-to-left, 3 other scripts.
    pub groups: Vec<usize>,
    /// Whether the lines read upside down.
    pub upside_down: bool,
    /// Each group's share of the total line width.
    pub shares: [f64; 4],
}

/// Four-group script router with an upside-down vote.
#[derive(Debug)]
pub struct LineRouterModel {
    inference: OrtInfer,
}

/// The first maximum, as `np.argmax` returns.
fn argmax(values: &[f64; 4]) -> usize {
    let mut best = 0;
    for i in 1..4 {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

fn softmax<const N: usize>(logits: [f32; N]) -> [f32; N] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits.map(|v| (v - max).exp());
    let sum: f32 = exp.iter().sum();
    exp.map(|v| v / sum)
}

impl LineRouterModel {
    /// Load the router, which takes `[N, 1, 32, 512]` lines and returns `group` and `flip` logits.
    pub fn new(
        model: impl Into<ModelSource>,
        ort_config: Option<OrtSessionConfig>,
    ) -> Result<Self, OCRError> {
        let inference = if ort_config.is_some() {
            let common = ModelInferenceConfig {
                ort_session: ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common, model, None)?
        } else {
            OrtInfer::new(model, None)?
        };
        Ok(Self { inference })
    }

    /// Group probabilities and the upside-down probability of each crop.
    fn probabilities(&self, crops: &[&RgbImage]) -> Result<Vec<([f32; 4], f32)>, OCRError> {
        let height = GRAY_LINE_HEIGHT as usize;
        let mut out = Vec::with_capacity(crops.len());
        for part in crops.chunks(BATCH) {
            let mut input = ndarray::Array4::zeros((part.len(), 1, height, WIDTH));
            for (n, crop) in part.iter().enumerate() {
                let line = gray_line(crop, 8, u32::MAX);
                // Short lines are tiled across the router width.
                for y in 0..height {
                    for x in 0..WIDTH {
                        let v = line[((x % line.width() as usize) as u32, y as u32)][0];
                        input[[n, 0, y, x]] = v as f32 / 127.5 - 1.0;
                    }
                }
            }
            let model = &self.inference;
            let (mut groups, mut flips) = (None, None);
            for (name, output) in
                model.infer(&[(model.input_name(), TensorInput::Array4(&input))])?
            {
                match name.as_str() {
                    "group" => groups = Some(output.try_into_array2_f32()?),
                    "flip" => flips = Some(output.try_into_array2_f32()?),
                    _ => {}
                }
            }
            let (Some(groups), Some(flips)) = (groups, flips) else {
                return Err(OCRError::invalid_input(
                    "router has no `group` and `flip` outputs",
                ));
            };
            if groups.dim() != (part.len(), 4) || flips.dim() != (part.len(), 2) {
                return Err(OCRError::invalid_input(format!(
                    "unexpected router outputs {:?} and {:?} for {} lines",
                    groups.dim(),
                    flips.dim(),
                    part.len()
                )));
            }
            for n in 0..part.len() {
                let group: [f32; 4] = softmax(std::array::from_fn(|k| groups[[n, k]]));
                let flip = softmax([flips[[n, 0]], flips[[n, 1]]]);
                out.push((group, flip[1]));
            }
        }
        Ok(out)
    }

    /// Route the line crops of one page.
    ///
    /// Every line takes the page's dominant group unless it is long enough to judge and confidently another. The page is Latin unless another group holds a clear majority of the line width.
    pub fn route(
        &self,
        crops: &[impl std::borrow::Borrow<RgbImage>],
    ) -> Result<LineRouting, OCRError> {
        const OVERRIDE: f32 = 0.9;
        const MIN_ASPECT: f64 = 4.0;
        const FLIP_MIN_LINES: usize = 3;
        const FLIP_VOTE: f64 = 0.7;
        const OTHER_SHARE: f64 = 0.65;
        if crops.is_empty() {
            return Ok(LineRouting::default());
        }
        let crops: Vec<&RgbImage> = crops.iter().map(std::borrow::Borrow::borrow).collect();
        let probabilities = self.probabilities(&crops)?;
        let widths: Vec<f64> = crops.iter().map(|c| f64::from(c.width())).collect();
        let total: f64 = widths.iter().sum();
        let mut shares = [0f64; 4];
        for ((group, _), &w) in probabilities.iter().zip(&widths) {
            for k in 0..4 {
                shares[k] += f64::from(group[k]) * w;
            }
        }
        for share in &mut shares {
            *share /= total;
        }
        let mut dominant = argmax(&shares);
        if dominant != 0 && shares[dominant] <= OTHER_SHARE {
            dominant = 0;
        }
        let groups = probabilities
            .iter()
            .zip(&crops)
            .map(|((group, _), crop)| {
                let own = argmax(&group.map(f64::from));
                let confident = group.iter().copied().fold(f32::NEG_INFINITY, f32::max) > OVERRIDE;
                let long = f64::from(crop.width()) >= MIN_ASPECT * f64::from(crop.height());
                if own != dominant && confident && long {
                    own
                } else {
                    dominant
                }
            })
            .collect();
        let vote: f64 = probabilities
            .iter()
            .zip(&widths)
            .map(|(&(_, flip), &w)| f64::from(flip) * w)
            .sum();
        Ok(LineRouting {
            groups,
            upside_down: crops.len() >= FLIP_MIN_LINES && vote > FLIP_VOTE * total,
            shares,
        })
    }
}
