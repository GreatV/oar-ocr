//! Dynamic-width recognition batches with optional background flattening and CTC run endpoints.
use super::CRNNModel;
use crate::{
    core::{OCRError, inference::TensorInput},
    processors::{
        CtcReading,
        raster::{self, Interp},
    },
};
use image::{GrayImage, RgbImage};
use ndarray::Array4;

/// Pixel normalization and width policy for line recognition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecognitionPreprocessing {
    /// Use the recognizer's configured RGB resize.
    #[default]
    Standard,
    /// Remove local column backgrounds after resizing to a 48-pixel height.
    Flatten,
    /// Feed grayscale lines at a 32-pixel height, each batch padded with white to a multiple of 128 pixels.
    ///
    /// For single-channel recognizers that reduce their own output; it implies [`CtcDecoding::GraphArgmax`].
    GrayBuckets,
}
/// Output reduction strategy for a CTC recognizer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CtcDecoding {
    /// Apply the default CTC decoder.
    #[default]
    Standard,
    /// Keep the first score and both horizontal endpoints of each nonblank run.
    FrameSpans,
    /// Read the class indices and scores a model reduces in its own graph as outputs `ids` and `conf`.
    ///
    /// It implies [`RecognitionPreprocessing::GrayBuckets`].
    GraphArgmax,
}
/// Height of a [`RecognitionPreprocessing::GrayBuckets`] line.
pub(crate) const GRAY_LINE_HEIGHT: u32 = 32;

/// Grayscale line at a 32-pixel height with its width scaled by the aspect ratio and held within `min_w..=max_w`.
pub(crate) fn gray_line(img: &RgbImage, min_w: u32, max_w: u32) -> GrayImage {
    let gray = raster::scalar::grayscale(img);
    let (w, h) = gray.dimensions();
    let scaled = (w as f64 * GRAY_LINE_HEIGHT as f64 / h.max(1) as f64).round_ties_even();
    let nw = (scaled as u32).clamp(min_w, max_w);
    let interp = if h > GRAY_LINE_HEIGHT {
        Interp::Area
    } else {
        Interp::Linear
    };
    raster::scalar::resize(&gray, nw, GRAY_LINE_HEIGHT, interp)
}

fn prepare(img: &RgbImage) -> GrayImage {
    let w = (48 * img.width() as u64).div_ceil(img.height() as u64) as u32;
    let resized = raster::resize_rgb(img, w.max(1), 48);
    raster::flatten(&raster::grayscale(&resized))
}

impl CRNNModel {
    pub(crate) fn read_lines(
        &self,
        crops: &[impl std::borrow::Borrow<RgbImage>],
        policy: RecognitionPreprocessing,
        decoding: CtcDecoding,
        batch_size: usize,
        return_word_boxes: bool,
    ) -> Result<Vec<CtcReading>, OCRError> {
        if policy == RecognitionPreprocessing::GrayBuckets || decoding == CtcDecoding::GraphArgmax {
            return self.read_gray_lines(crops, batch_size);
        }
        let prepared: Vec<_> = if policy == RecognitionPreprocessing::Flatten {
            crops.iter().map(|img| prepare(img.borrow())).collect()
        } else {
            Vec::new()
        };
        let width = |i: usize| {
            if prepared.is_empty() {
                (self.resizer.rec_image_shape[1] as f32
                    * (crops[i].borrow().width() as f32 / crops[i].borrow().height().max(1) as f32))
                    .ceil() as u32
            } else {
                prepared[i].width()
            }
        };
        let mut order: Vec<_> = (0..crops.len()).collect();
        order.sort_by_key(|&i| width(i));
        let mut result = vec![CtcReading::default(); crops.len()];
        for ids in order.chunks(batch_size.max(1)) {
            let flattened = policy == RecognitionPreprocessing::Flatten;
            let w = width(*ids.last().unwrap()).clamp(16, 3200);
            let input = if flattened {
                let mut input = Array4::zeros((ids.len(), 3, 48, w as usize));
                for (n, &i) in ids.iter().enumerate() {
                    let rw = width(i).min(w);
                    let resized = raster::resize(&prepared[i], rw, 48, Interp::Linear);
                    for (x, y, p) in resized.enumerate_pixels() {
                        let val = (p[0] as f32 / 255. - 0.5) / 0.5;
                        for c in 0..3 {
                            input[[n, c, y as usize, x as usize]] = val;
                        }
                    }
                }
                input
            } else {
                self.preprocess_refs(&ids.iter().map(|&i| crops[i].borrow()).collect::<Vec<_>>())?
            };
            let w = input.dim().3 as u32;
            let model = &self.inference;
            model.infer_first_output_f32(
                &[(model.input_name(), TensorInput::Array4(&input))],
                |shape, probs| {
                    let &[batch, steps, classes] = shape else {
                        return Err(OCRError::invalid_input("CTC recognizer expects [N,T,V]"));
                    };
                    if batch != ids.len()
                        || steps == 0
                        || classes != self.decoder.get_character_list().len()
                    {
                        return Err(OCRError::invalid_input(format!(
                            "unexpected CTC recognizer output {shape:?} for {} dictionary classes",
                            self.decoder.get_character_list().len()
                        )));
                    }
                    if decoding == CtcDecoding::Standard {
                        let view = ndarray::ArrayView3::from_shape((batch, steps, classes), probs)
                            .map_err(|e| OCRError::invalid_input(e.to_string()))?;
                        let standard = self.postprocess(&view, return_word_boxes);
                        for (j, ((&i, text), score)) in ids
                            .iter()
                            .zip(standard.texts)
                            .zip(standard.scores)
                            .enumerate()
                        {
                            result[i] = CtcReading {
                                text,
                                confidence: f64::from(score),
                                column_indices: standard
                                    .char_col_indices
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_default(),
                                sequence_length: steps,
                                width_fraction: width(i).min(w) as f32 / w as f32,
                                ..Default::default()
                            };
                        }
                        return Ok(());
                    }
                    for (&i, row) in ids.iter().zip(probs.chunks_exact(steps * classes)) {
                        result[i] = self
                            .decoder
                            .decode_frame_spans(row, width(i).min(w), w, steps);
                    }
                    Ok(())
                },
            )?;
        }
        Ok(result)
    }

    /// Lines are sorted by width and batched, because a line's reading depends on the padded width it runs at.
    fn read_gray_lines(
        &self,
        crops: &[impl std::borrow::Borrow<RgbImage>],
        batch_size: usize,
    ) -> Result<Vec<CtcReading>, OCRError> {
        const BUCKET: usize = 128;
        let height = GRAY_LINE_HEIGHT as usize;
        let lines: Vec<_> = crops
            .iter()
            .map(|img| gray_line(img.borrow(), 8, 1280))
            .collect();
        let mut order: Vec<_> = (0..lines.len()).collect();
        order.sort_by_key(|&i| lines[i].width());
        let chars = self.decoder.get_character_list();
        let mut result = vec![CtcReading::default(); lines.len()];
        for ids in order.chunks(batch_size.max(1)) {
            let widest = lines[*ids.last().unwrap()].width() as usize;
            let w = widest.div_ceil(BUCKET) * BUCKET;
            let mut input = Array4::from_elem((ids.len(), 1, height, w), 1f32);
            for (n, &i) in ids.iter().enumerate() {
                for (x, y, p) in lines[i].enumerate_pixels() {
                    input[[n, 0, y as usize, x as usize]] = (p[0] as f64 / 127.5 - 1.0) as f32;
                }
            }
            let model = &self.inference;
            let mut classes = None;
            let mut scores = None;
            for (name, output) in
                model.infer(&[(model.input_name(), TensorInput::Array4(&input))])?
            {
                match name.as_str() {
                    "ids" => classes = Some(output.try_into_array2_i64()?),
                    "conf" => scores = Some(output.try_into_array2_f32()?),
                    _ => {}
                }
            }
            let (Some(classes), Some(scores)) = (classes, scores) else {
                return Err(OCRError::invalid_input(
                    "recognizer has no `ids` and `conf` outputs",
                ));
            };
            let steps = classes.ncols();
            if classes.nrows() != ids.len() || scores.dim() != classes.dim() || steps == 0 {
                return Err(OCRError::invalid_input(format!(
                    "unexpected recognizer outputs {:?} and {:?} for {} lines",
                    classes.dim(),
                    scores.dim(),
                    ids.len()
                )));
            }
            // Input pixels per output timestep.
            let stride = (w / steps).max(1);
            for (n, &i) in ids.iter().enumerate() {
                let mut text = String::new();
                let (mut sum, mut count, mut previous) = (0f64, 0usize, 0);
                for t in 0..(lines[i].width() as usize / stride).min(steps) {
                    let class = classes[[n, t]];
                    if class != 0 && class != previous {
                        if let Some(&ch) = usize::try_from(class).ok().and_then(|c| chars.get(c)) {
                            text.push(ch);
                        }
                        sum += f64::from(scores[[n, t]]);
                        count += 1;
                    }
                    previous = class;
                }
                result[i] = CtcReading {
                    text,
                    // The mean is rounded to f32, the precision the scores carry.
                    confidence: if count == 0 {
                        0.
                    } else {
                        f64::from((sum / count as f64) as f32)
                    },
                    ..Default::default()
                };
            }
        }
        Ok(result)
    }
}
