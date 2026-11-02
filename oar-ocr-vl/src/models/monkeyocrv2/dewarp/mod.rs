//! MonkeyOCRv2's mask and point-grid page rectification.
//!
//! Follows modeling_preprocessor.py in MonkeyOCRv2 (Apache-2.0).

mod grid;

use crate::backbones::u2net::U2NetP;
use crate::runtime::{
    checkpoint::pytorch_builder,
    resampling::{cv2_linear_rgb, unwarp_rgb},
};
use crate::{Error, PagePreprocessor};
use candle_core::{Device, Tensor};
use image::RgbImage;
use std::collections::VecDeque;
use std::path::Path;

/// Full-page rectification using MonkeyOCRv2's mask and point-grid networks.
#[derive(Debug)]
pub struct MonkeyOcrV2Dewarp {
    mask: U2NetP,
    grid: grid::PointGrid,
    device: Device,
}
impl MonkeyOcrV2Dewarp {
    /// Load `preprocessor1.pth` and `preprocessor2.pth` in Float32.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        let dir = model_dir.as_ref();
        let mask = pytorch_builder(
            &dir.join("preprocessor1.pth"),
            "msk.stage1.rebnconvin.conv_s1.weight",
            &device,
        )
        .and_then(|vb| U2NetP::load(vb.pp("msk")))
        .map_err(|e| inference("load mask network", e))?;
        let grid = pytorch_builder(
            &dir.join("preprocessor2.pth"),
            "encoder.backbone.patch_embed.proj.weight",
            &device,
        )
        .and_then(grid::PointGrid::load)
        .map_err(|e| inference("load grid network", e))?;
        Ok(Self { mask, grid, device })
    }
    /// Rectify a page while keeping its original dimensions.
    /// Region coordinates describe the returned canvas.
    pub fn preprocess(&self, image: &RgbImage) -> Result<RgbImage, Error> {
        if image.width() == 0 || image.height() == 0 {
            return Err(Error::invalid_input("dewarp requires a nonempty page"));
        }
        self.run(image).map_err(|e| inference("rectify page", e))
    }
    fn run(&self, image: &RgbImage) -> candle_core::Result<RgbImage> {
        let input = cv2_linear_rgb(image, 512, 512, &self.device)?;
        let probability = self.mask.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        let mask = largest_mask(&probability, 512, 512);
        let mask = Tensor::from_vec(mask, (1, 1, 512, 512), &self.device)?;
        let grid = self.grid.forward(&input.broadcast_mul(&mask)?)?;
        unwarp_rgb(image, &grid)
    }
}
impl PagePreprocessor for MonkeyOcrV2Dewarp {
    fn preprocess(&self, image: &RgbImage) -> Result<RgbImage, Error> {
        MonkeyOcrV2Dewarp::preprocess(self, image)
    }
}
fn inference(context: &str, source: candle_core::Error) -> Error {
    crate::runtime::errors::candle_to_ocr_inference("MonkeyOCRv2 dewarp", context, source)
}

fn largest_mask(probability: &[f32], width: usize, height: usize) -> Vec<f32> {
    let foreground: Vec<_> = probability.iter().map(|&p| p > 0.8).collect();
    let mut seen = vec![false; foreground.len()];
    let mut largest = Vec::new();
    let mut largest_label = usize::MAX;
    for start in 0..foreground.len() {
        if seen[start] || !foreground[start] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        let mut label = usize::MAX;
        while let Some(i) = queue.pop_front() {
            component.push(i);
            let (x, y) = (i % width, i / width);
            // OpenCV's default 8-connected labels follow 2x2 block scan order.
            label = label.min((y / 2) * width.div_ceil(2) + x / 2);
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    let j = yy * width + xx;
                    if foreground[j] && !seen[j] {
                        seen[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        if component.len() > largest.len()
            || (component.len() == largest.len() && label < largest_label)
        {
            largest = component;
            largest_label = label;
        }
    }
    let mut mask = vec![0.; foreground.len()];
    for i in largest {
        mask[i] = 1.;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backbones::{swin_v2::SwinV2Block, u2net::ResidualUBlock};
    use candle_core::DType;

    fn reference() -> serde_json::Value {
        serde_json::from_str(include_str!("fixtures/reference.json")).unwrap()
    }
    fn floats(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect()
    }
    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/models/monkeyocrv2/dewarp/fixtures")
            .join(name)
    }

    #[test]
    fn mask_threshold_components_and_ties_match_opencv() {
        for case in reference()["masks"].as_array().unwrap() {
            let actual = largest_mask(
                &floats(&case["input"]),
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize,
            );
            assert_eq!(actual, floats(&case["output"]), "{}", case["name"]);
        }
    }

    #[test]
    fn float_resize_and_unwarp_match_cpu_reference_pixels() {
        let data = reference();
        let bytes = data["image"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let image = RgbImage::from_raw(5, 3, bytes).unwrap();
        for case in data["resize"].as_array().unwrap() {
            let actual = cv2_linear_rgb(
                &image,
                case["width"].as_u64().unwrap() as usize,
                case["height"].as_u64().unwrap() as usize,
                &Device::Cpu,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let expected = floats(&case["output"]);
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 2e-7)
            );
        }
        for case in data["sampling"].as_array().unwrap() {
            let grid = Tensor::from_vec(floats(&case["grid"]), (1, 2, 2, 2), &Device::Cpu).unwrap();
            let output = unwarp_rgb(&image, &grid).unwrap();
            let expected: Vec<_> = case["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            assert_eq!(output.as_raw(), &expected, "{}", case["name"]);
        }
    }

    #[test]
    fn synthetic_pickle_rsu_and_shifted_swin_match_reference() {
        let device = Device::Cpu;
        let plain = pytorch_builder(&fixture("synthetic_plain.pth"), "probe", &device).unwrap();
        assert_eq!(
            plain
                .get((2, 2), "probe")
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1., 2., 3., 4.]
        );
        let vb = pytorch_builder(
            &fixture("synthetic_blocks.pth"),
            "rsu.rebnconvin.conv_s1.weight",
            &device,
        )
        .unwrap();
        for case in reference()["networks"].as_array().unwrap() {
            let height = case["height"].as_u64().unwrap() as usize;
            let width = case["width"].as_u64().unwrap() as usize;
            let name = case["name"].as_str().unwrap();
            let actual = if name == "swin" {
                // The saved coordinate table takes precedence over this preset.
                let block = SwinV2Block::load(4, 2, 2, 1, 0, vb.pp("swin")).unwrap();
                block
                    .forward(
                        &Tensor::from_vec(floats(&case["input"]), (1, height * width, 4), &device)
                            .unwrap(),
                        height,
                        width,
                    )
                    .unwrap()
            } else {
                let block = ResidualUBlock::load(3, 2, 4, 4, name == "flat", vb.pp(name)).unwrap();
                block
                    .forward(
                        &Tensor::from_vec(floats(&case["input"]), (1, 3, height, width), &device)
                            .unwrap(),
                    )
                    .unwrap()
            };
            assert_eq!(actual.dtype(), DType::F32);
            let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let expected = floats(&case["output"]);
            let difference = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(difference < 3e-6, "{name} {height}x{width}: {difference}");
        }
    }
}
