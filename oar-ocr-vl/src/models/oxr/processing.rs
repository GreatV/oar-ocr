//! Image patches and the image-first text protocol for OXR inference.

use super::config::ImageConfig;
use super::{OxrGenerationOptions, infer};
use crate::Error;
use crate::runtime::image::{patchify_merge_grouped, pil_bilinear_resize};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use tokenizers::Tokenizer;

/// Minimum image area from the upstream OXR vLLM processor profile.
pub const MIN_PIXELS: u32 = 200_704;
/// Maximum image area from the upstream OXR vLLM processor profile.
pub const MAX_PIXELS: u32 = 921_600;

pub(super) const PATCH_SIZE: usize = 16;
pub(super) const MERGE_SIZE: usize = 2;

pub(super) const IMAGE_BEGIN: i64 = -301;
pub(super) const IMAGE_ATOM: i64 = -300;
pub(super) const IMAGE_END: i64 = -302;

/// Inputs before the vision encoder, exposed for numerical alignment.
#[derive(Debug)]
pub struct OxrImageInputs {
    pub pixel_values: Tensor,
    pub grid_thw: [usize; 3],
}

/// Choose a 32-pixel-aligned image size under the OXR pixel-area contract.
///
/// The order of floating-point operations and ties-to-even rounding are part
/// of that contract, including at pixel-area boundaries.
pub fn smart_resize(
    height: u32,
    width: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), Error> {
    if height == 0 || width == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(Error::invalid_input(
            "OXR resize needs nonzero dimensions and ordered positive pixel bounds",
        ));
    }
    let factor = (PATCH_SIZE * MERGE_SIZE) as f64;
    let short = height.min(width) as f64;
    let long = height.max(width) as f64;
    let edges = if short < factor {
        [factor, (factor / short * long).round_ties_even()]
    } else {
        [short, long.min(200.0 * short)]
    };
    let area = edges[0] * edges[1];
    let rounded = edges.map(|edge| (edge / factor).round_ties_even() * factor);
    let rounded_area = rounded[0] * rounded[1];
    let aligned = if rounded_area > max_pixels as f64 {
        let scale = (area / max_pixels as f64).sqrt();
        edges.map(|edge| (edge / scale / factor).floor() * factor)
    } else if rounded_area < min_pixels as f64 {
        let scale = (min_pixels as f64 / area).sqrt();
        edges.map(|edge| (edge * scale / factor).ceil() * factor)
    } else {
        rounded
    };
    if aligned
        .iter()
        .any(|&edge| edge < factor || edge > u32::MAX as f64)
    {
        return Err(Error::invalid_input(
            "OXR smart resize produced unusable dimensions",
        ));
    }
    let [short, long] = aligned.map(|edge| edge as u32);
    Ok(if height <= width {
        (short, long)
    } else {
        (long, short)
    })
}

pub(super) fn preprocess(
    image: &RgbImage,
    config: &ImageConfig,
    options: &OxrGenerationOptions,
    device: &Device,
    dtype: DType,
) -> Result<OxrImageInputs, Error> {
    let (height, width) = smart_resize(
        image.height(),
        image.width(),
        options.min_pixels,
        options.max_pixels,
    )?;
    let resized = pil_bilinear_resize(image, width, height)?;
    let (h, w) = (height as usize, width as usize);
    let pixels = h * w;
    let mut chw = vec![0f32; pixels * 3];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            // The PIL backend first rescales in NumPy float64, then casts to
            // float32 before normalization. Do not fuse these operations.
            let scaled = (pixel[channel] as f64 * config.rescale_factor) as f32;
            chw[channel * pixels + index] =
                (scaled - config.image_mean[channel]) / config.image_std[channel];
        }
    }
    let (gh, gw) = (h / PATCH_SIZE, w / PATCH_SIZE);
    let patches = patchify_merge_grouped(&[&chw], 3, h, w, 1, gh, gw, PATCH_SIZE, MERGE_SIZE, 1);
    let pixel_values = Tensor::from_vec(patches, (gh * gw, 3 * PATCH_SIZE * PATCH_SIZE), device)
        .and_then(|x| x.to_dtype(dtype))
        .map_err(infer)?;
    Ok(OxrImageInputs {
        pixel_values,
        grid_thw: [1, gh, gw],
    })
}

/// Render the checkpoint chat template for an image followed by text.
pub fn build_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n<image>{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

pub(super) fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    atoms: usize,
) -> Result<Vec<i64>, Error> {
    expand_prompt(&build_prompt(prompt), atoms, |chunk| {
        tokenizer
            .encode(chunk, false)
            .map(|encoding| encoding.get_ids().iter().map(|&id| id as i64).collect())
            .map_err(|error| Error::invalid_input(format!("OXR tokenization failed: {error}")))
    })
}

fn expand_prompt(
    text: &str,
    atoms: usize,
    mut tokenize: impl FnMut(&str) -> Result<Vec<i64>, Error>,
) -> Result<Vec<i64>, Error> {
    if text.matches("<image>").count() != 1 || text.contains("<video>") {
        return Err(Error::invalid_input(
            "OXR requires exactly one image placeholder and no video placeholders",
        ));
    }
    let (before, after) = text.split_once("<image>").expect("one placeholder");
    let mut ids = tokenize(before)?;
    ids.push(IMAGE_BEGIN);
    ids.extend(std::iter::repeat_n(IMAGE_ATOM, atoms));
    ids.push(IMAGE_END);
    ids.extend(tokenize(after)?);
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_resize_and_merge_grouped_pixels() {
        assert_eq!(smart_resize(80, 112, 1024, MAX_PIXELS).unwrap(), (64, 128));
        assert_eq!(smart_resize(16, 40, 1024, MAX_PIXELS).unwrap(), (32, 64));
        assert_eq!(
            smart_resize(32, 8000, 1024, MAX_PIXELS).unwrap(),
            (32, 6400)
        );
        assert_eq!(
            smart_resize(1000, 1000, MIN_PIXELS, MAX_PIXELS).unwrap(),
            // 1000 / sqrt(1_000_000 / 921_600) / 32 is just below 30.
            (928, 928)
        );
        let image = RgbImage::from_fn(64, 32, |x, y| image::Rgb([x as u8, y as u8, 255]));
        let config: ImageConfig = serde_json::from_value(serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "rescale_factor": 1.0 / 255.0,
            "resample": 2, "do_resize": true, "do_rescale": true, "do_normalize": true
        })).unwrap();
        let inputs = preprocess(
            &image,
            &config,
            &OxrGenerationOptions::default().with_min_pixels(1024),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap();
        assert_eq!(inputs.grid_thw, [1, 2, 4]);
        let patches = inputs.pixel_values.to_vec2::<f32>().unwrap();
        for (index, (x, y)) in [
            (0, 0),
            (16, 0),
            (0, 16),
            (16, 16),
            (32, 0),
            (48, 0),
            (32, 16),
            (48, 16),
        ]
        .into_iter()
        .enumerate()
        {
            let norm = |v| ((v as f64 / 255.0) as f32 - 0.5) / 0.5;
            assert_eq!(patches[index][0], norm(x));
            assert_eq!(patches[index][256], norm(y));
            assert_eq!(patches[index][512], 1.0);
        }
    }

    #[test]
    fn chat_and_signed_visual_expansion() {
        let text = build_prompt(super::super::OxrTask::Text.prompt());
        assert_eq!(
            text,
            "<|im_start|>user\n<image>Recognize the content of the image.<|im_end|>\n<|im_start|>assistant\n"
        );
        let mut chunks = Vec::new();
        let ids = expand_prompt(&text, 2, |chunk| {
            chunks.push(chunk.to_owned());
            Ok(vec![chunks.len() as i64])
        })
        .unwrap();
        assert_eq!(ids, [1, IMAGE_BEGIN, IMAGE_ATOM, IMAGE_ATOM, IMAGE_END, 2]);
        assert_eq!(chunks[0], "<|im_start|>user\n");
        assert!(chunks[1].starts_with("Recognize"));
    }
}
