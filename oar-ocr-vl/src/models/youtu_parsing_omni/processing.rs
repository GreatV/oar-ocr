use super::config::OmniConfig;
use crate::Error;
use crate::runtime::image::{image_to_chw, patchify_merge_grouped, pil_bicubic_resize};
use candle_core::{DType, Device, Tensor};
use image::{Rgb, RgbImage, imageops};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct ImageConfig {
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub image_min_num_tokens: usize,
    pub image_max_num_tokens: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImageConfig {
    pub fn load(path: &Path, omni: &OmniConfig) -> Result<Self, Error> {
        #[derive(Deserialize)]
        struct Processor {
            image_processor: ImageConfig,
        }
        let cfg = serde_json::from_slice::<Processor>(&std::fs::read(path)?)
            .map_err(|e| {
                Error::config(format!("invalid Youtu-Parsing-Omni processor config: {e}"))
            })?
            .image_processor;
        if cfg.patch_size != omni.patch_size
            || cfg.spatial_merge_size != omni.spatial_merge_size
            || cfg.temporal_patch_size != 1
            || cfg.image_min_num_tokens == 0
            || cfg.image_max_num_tokens < cfg.image_min_num_tokens
            || cfg
                .mean
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || cfg.std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::config(
                "Youtu-Parsing-Omni image processor does not match the encoder",
            ));
        }
        Ok(cfg)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ImagePlan {
    pub width: u32,
    pub height: u32,
    pub resized_width: u32,
    pub resized_height: u32,
}

pub(super) fn plan(
    width: u32,
    height: u32,
    cfg: &ImageConfig,
    max_pixels: Option<usize>,
) -> Result<ImagePlan, Error> {
    if width == 0 || height == 0 {
        return Err(Error::invalid_input(
            "Youtu-Parsing-Omni image dimensions must be positive",
        ));
    }
    let factor = cfg
        .patch_size
        .checked_mul(cfg.spatial_merge_size)
        .ok_or_else(|| Error::config("Youtu-Parsing-Omni image factor overflow"))?;
    let unit = factor
        .checked_mul(factor)
        .ok_or_else(|| Error::config("image pixel limit overflow"))?;
    let min_pixels = unit
        .checked_mul(cfg.image_min_num_tokens)
        .ok_or_else(|| Error::config("image pixel limit overflow"))?;
    let max_pixels = max_pixels.unwrap_or(
        unit.checked_mul(cfg.image_max_num_tokens)
            .ok_or_else(|| Error::config("image pixel limit overflow"))?,
    );
    if max_pixels < min_pixels || max_pixels > u32::MAX as usize || factor > u32::MAX as usize {
        return Err(Error::invalid_input(
            "Youtu-Parsing-Omni max_pixels must cover the minimum image size and fit in u32",
        ));
    }
    let factor = factor as u32;
    let area = f64::from(width) * f64::from(height);
    let (mut rw, mut rh) = (width, height);
    if area > max_pixels as f64 {
        let beta = (area / max_pixels as f64).sqrt();
        rw = (f64::from(width) / beta).floor().max(1.0) as u32;
        rh = (f64::from(height) / beta).floor().max(1.0) as u32;
    } else if area < min_pixels as f64 {
        let beta = (min_pixels as f64 / area).sqrt();
        rw = (f64::from(width) * beta).ceil().max(1.0) as u32;
        rh = (f64::from(height) * beta).ceil().max(1.0) as u32;
    }
    let (mut pw, mut ph) = (
        rw.div_ceil(factor).checked_mul(factor).ok_or_else(|| {
            Error::invalid_input("Youtu-Parsing-Omni padded image width overflow")
        })?,
        rh.div_ceil(factor).checked_mul(factor).ok_or_else(|| {
            Error::invalid_input("Youtu-Parsing-Omni padded image height overflow")
        })?,
    );
    if u64::from(pw) * u64::from(ph) > max_pixels as u64 {
        pw = (rw / factor * factor).max(factor);
        ph = (rh / factor * factor).max(factor);
        let scale = (f64::from(pw) / f64::from(rw)).min(f64::from(ph) / f64::from(rh));
        rw = (f64::from(rw) * scale)
            .round_ties_even()
            .max(1.0)
            .min(f64::from(pw)) as u32;
        rh = (f64::from(rh) * scale)
            .round_ties_even()
            .max(1.0)
            .min(f64::from(ph)) as u32;
    }
    if u64::from(pw) * u64::from(ph) > max_pixels as u64 {
        return Err(Error::invalid_input(
            "Youtu-Parsing-Omni image aspect ratio cannot fit max_pixels",
        ));
    }
    Ok(ImagePlan {
        width: pw,
        height: ph,
        resized_width: rw,
        resized_height: rh,
    })
}

pub(super) fn preprocess(
    image: &RgbImage,
    cfg: &ImageConfig,
    plan: ImagePlan,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, Error> {
    // The reference processor resizes with PIL's bicubic filter.
    let resized = pil_bicubic_resize(image, plan.resized_width, plan.resized_height)?;
    let mut canvas = RgbImage::from_pixel(
        plan.width,
        plan.height,
        Rgb(cfg.mean.map(|v| (v * 255.0) as u8)),
    );
    imageops::replace(&mut canvas, &resized, 0, 0);
    let chw = image_to_chw(&canvas, &cfg.mean, &cfg.std, Some(1.0 / 255.0));
    let (height, width) = (plan.height as usize, plan.width as usize);
    let gh = height / cfg.patch_size;
    let gw = width / cfg.patch_size;
    let patches = patchify_merge_grouped(
        &[&chw],
        3,
        height,
        width,
        1,
        gh,
        gw,
        cfg.patch_size,
        cfg.spatial_merge_size,
        1,
    );
    Ok(Tensor::from_vec(
        patches,
        (gh * gw, 3 * cfg.patch_size * cfg.patch_size),
        device,
    )?
    .to_dtype(dtype)?)
}

pub(super) fn prompt_tokens(
    tokenizer: &Tokenizer,
    plan: ImagePlan,
    merge_factor: usize,
    prompt: &str,
) -> Result<(Vec<u32>, Vec<usize>), Error> {
    let mut text = format!("<|vision_start|>{}*{}\n", plan.height, plan.width);
    let rows = plan.height as usize / merge_factor;
    let cols = plan.width as usize / merge_factor;
    for _ in 0..rows {
        for _ in 0..cols {
            text.push_str("<|image_pad|>");
        }
        text.push('\n');
    }
    text.push_str("<|vision_end|>");
    text.push_str(prompt);
    let ids = tokenizer
        .encode(text, false)
        .map_err(|e| Error::config(format!("Youtu-Parsing-Omni tokenization failed: {e}")))?
        .get_ids()
        .to_vec();
    let image_id = tokenizer
        .token_to_id("<|image_pad|>")
        .ok_or_else(|| Error::config("Youtu-Parsing-Omni tokenizer lacks <|image_pad|>"))?;
    let indices = ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id == image_id).then_some(i))
        .collect::<Vec<_>>();
    if indices.len() != rows * cols {
        return Err(Error::invalid_input(
            "Youtu-Parsing-Omni prompt must not contain image placeholder tokens",
        ));
    }
    Ok((ids, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_preserves_content_and_patch_order() {
        let cfg = ImageConfig {
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            image_min_num_tokens: 1,
            image_max_num_tokens: 32768,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let p = plan(65, 33, &cfg, None).unwrap();
        assert_eq!(
            (p.width, p.height, p.resized_width, p.resized_height),
            (96, 64, 65, 33)
        );
        let image = RgbImage::from_fn(65, 33, |x, y| Rgb([x as u8, y as u8, 255]));
        let pixels = preprocess(&image, &cfg, p, DType::F32, &Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(pixels.len(), 24);
        for (index, (x, y)) in [(0, (0, 0)), (1, (16, 0)), (2, (0, 16)), (4, (32, 0))] {
            assert!((pixels[index][0] - (x as f32 / 255.0 - 0.5) * 2.0).abs() < 1e-6);
            assert!((pixels[index][256] - (y as f32 / 255.0 - 0.5) * 2.0).abs() < 1e-6);
        }
        assert!((pixels[23][767] - (127.0 / 255.0 - 0.5) * 2.0).abs() < 1e-6);
        let p = plan(65, 33, &cfg, Some(2048)).unwrap();
        assert!(u64::from(p.width) * u64::from(p.height) <= 2048);
        assert!(p.resized_width <= p.width && p.resized_height <= p.height);
        assert!(plan(1, 100_000, &cfg, Some(1024)).is_err());
    }
}
