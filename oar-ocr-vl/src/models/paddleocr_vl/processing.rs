use super::config::PaddleOcrVlImageProcessorConfig;
use crate::error::Error;
use crate::utils::image::qwen_vl_resize;
pub use crate::utils::image::smart_resize;
use crate::utils::table::{convert_otsl_to_html, looks_like_table_tokens};
pub use crate::utils::text::strip_math_wrappers;
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct PaddleOcrVlImageInputs {
    pub pixel_values: Tensor,
    pub image_grid_thw: Vec<(usize, usize, usize)>,
}

pub fn postprocess_table_output(input: &str) -> String {
    let trimmed = input.trim();
    if !looks_like_table_tokens(trimmed) && !trimmed.contains("<table") {
        return trimmed.to_string();
    }
    convert_otsl_to_html(input)
}

/// `smart_resize` exactly as PaddleOCR-VL's reference image processor
/// computes it: a side shorter than `factor` is first scaled up to `factor`
/// (keeping the aspect ratio), and rounding follows Python's `round`, which
/// rounds halves to even.
pub(crate) fn reference_smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), Error> {
    if factor == 0 || height == 0 || width == 0 {
        return Err(Error::InvalidInput {
            message: format!(
                "PaddleOCR-VL smart_resize: invalid size {width}x{height} or factor {factor}"
            ),
        });
    }
    let factor = f64::from(factor);
    let (mut height, mut width) = (f64::from(height), f64::from(width));
    if height < factor {
        width = (width * factor / height).round_ties_even();
        height = factor;
    }
    if width < factor {
        height = (height * factor / width).round_ties_even();
        width = factor;
    }
    if height.max(width) / height.min(width) > 200.0 {
        return Err(Error::InvalidInput {
            message: format!(
                "PaddleOCR-VL smart_resize: absolute aspect ratio must be <= 200, got {:.3}",
                height.max(width) / height.min(width)
            ),
        });
    }
    let mut h_bar = (height / factor).round_ties_even() * factor;
    let mut w_bar = (width / factor).round_ties_even() * factor;
    if h_bar * w_bar > f64::from(max_pixels) {
        let beta = (height * width / f64::from(max_pixels)).sqrt();
        h_bar = (height / beta / factor).floor() * factor;
        w_bar = (width / beta / factor).floor() * factor;
    } else if h_bar * w_bar < f64::from(min_pixels) {
        let beta = (f64::from(min_pixels) / (height * width)).sqrt();
        h_bar = (height * beta / factor).ceil() * factor;
        w_bar = (width * beta / factor).ceil() * factor;
    }
    if h_bar < factor || w_bar < factor {
        return Err(Error::InvalidInput {
            message: format!(
                "PaddleOCR-VL smart_resize: {width}x{height} resizes below one patch group"
            ),
        });
    }
    Ok((h_bar as u32, w_bar as u32))
}

pub fn preprocess_images(
    images: &[RgbImage],
    cfg: &PaddleOcrVlImageProcessorConfig,
    device: &Device,
    dtype: DType,
) -> Result<PaddleOcrVlImageInputs, Error> {
    preprocess_images_with_max_pixels(images, cfg, device, dtype, cfg.max_pixels)
}

pub(crate) fn preprocess_images_with_max_pixels(
    images: &[RgbImage],
    cfg: &PaddleOcrVlImageProcessorConfig,
    device: &Device,
    dtype: DType,
    max_pixels: u32,
) -> Result<PaddleOcrVlImageInputs, Error> {
    cfg.validate()?;
    if images.is_empty() {
        return Err(Error::InvalidInput {
            message: "PaddleOCR-VL: no images provided".to_string(),
        });
    }

    let factor = (cfg.patch_size * cfg.merge_size) as u32;
    let patch = cfg.patch_size as u32;

    let mut all_patches: Vec<Tensor> = Vec::with_capacity(images.len());
    let mut grids = Vec::with_capacity(images.len());

    for img in images {
        let (h, w) = (img.height(), img.width());
        let (rh, rw) = if cfg.do_resize {
            reference_smart_resize(h, w, factor, cfg.min_pixels, max_pixels)?
        } else {
            (h, w)
        };

        // The reference processor resizes with PIL (bicubic for `resample: 3`
        // and by default), so use the PIL-exact resize.
        let resized = if rh != h || rw != w {
            qwen_vl_resize(img, rw, rh, cfg.resample)?
        } else {
            img.clone()
        };

        if rh % patch != 0 || rw % patch != 0 {
            return Err(Error::Config {
                message: format!(
                    "PaddleOCR-VL preprocess produced non-divisible dims: {rh}x{rw} not divisible by patch_size={patch}"
                ),
            });
        }

        let grid_h = (rh / patch) as usize;
        let grid_w = (rw / patch) as usize;
        let grid_t = 1usize;

        let num_patches = grid_t * grid_h * grid_w;
        let patch_elements = 3 * cfg.patch_size * cfg.patch_size;

        let mean = &cfg.image_mean;
        let std = &cfg.image_std;

        let raw_pixels = resized.as_raw();
        let stride = (rw * 3) as usize;
        let patch_u = cfg.patch_size;

        let mut patch_data = vec![0f32; num_patches * patch_elements];

        patch_data
            .par_chunks_mut(patch_elements)
            .enumerate()
            .for_each(|(i, chunk)| {
                let gh = i / grid_w;
                let gw = i % grid_w;
                let base_x = gw * patch_u;
                let base_y = gh * patch_u;

                let mut chunk_idx = 0;
                for c in 0..3usize {
                    for dy in 0..patch_u {
                        for dx in 0..patch_u {
                            let idx = (base_y + dy) * stride + (base_x + dx) * 3 + c;

                            let mut v = raw_pixels[idx] as f32;
                            if cfg.do_rescale {
                                v *= cfg.rescale_factor;
                            }
                            if cfg.do_normalize {
                                v = (v - mean[c]) / std[c];
                            }
                            chunk[chunk_idx] = v;
                            chunk_idx += 1;
                        }
                    }
                }
            });

        let patches = Tensor::from_vec(
            patch_data,
            (num_patches, 3usize, cfg.patch_size, cfg.patch_size),
            device,
        )
        .map_err(|e| Error::Processing {
            kind: crate::error::ProcessingStage::TensorOperation,
            context: "PaddleOCR-VL: failed to create pixel_values tensor".to_string(),
            source: Box::new(e),
        })?;

        all_patches.push(patches);
        grids.push((grid_t, grid_h, grid_w));
    }

    let pixel_values = Tensor::cat(&all_patches, 0).map_err(|e| Error::Processing {
        kind: crate::error::ProcessingStage::TensorOperation,
        context: "PaddleOCR-VL: failed to concatenate pixel_values tensors".to_string(),
        source: Box::new(e),
    })?;

    // Convert to the target dtype (e.g., BF16 for model inference)
    let pixel_values = pixel_values
        .to_dtype(dtype)
        .map_err(|e| Error::Processing {
            kind: crate::error::ProcessingStage::TensorOperation,
            context: "PaddleOCR-VL: failed to convert pixel_values to target dtype".to_string(),
            source: Box::new(e),
        })?;

    Ok(PaddleOcrVlImageInputs {
        pixel_values,
        image_grid_thw: grids,
    })
}

#[cfg(test)]
mod tests {
    use super::PaddleOcrVlImageProcessorConfig;
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn reference_smart_resize_matches_the_python_processor() {
        // Expected sizes from the reference processor's smart_resize
        // (PaddleOCR-VL-1.5 bounds): 350 / 28 = 12.5 rounds half to even,
        // and sides shorter than 28 px are scaled up before rounding.
        for ((height, width), expected) in [
            ((350, 630), (336, 616)),
            ((22, 500), (84, 1624)),
            ((18, 40), (252, 504)),
        ] {
            assert_eq!(
                reference_smart_resize(height, width, 28, 112_896, 1_003_520).unwrap(),
                expected,
                "{height}x{width}"
            );
        }
    }

    #[test]
    fn test_preprocess_outputs_expected_shapes() -> Result<(), Error> {
        let cfg = PaddleOcrVlImageProcessorConfig {
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            do_convert_rgb: true,
            rescale_factor: 1.0 / 255.0,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
            min_pixels: 28 * 28 * 130,
            max_pixels: 28 * 28 * 1280,
            resample: None,
            patch_size: 14,
            temporal_patch_size: 1,
            merge_size: 2,
        };

        let mut img = RgbImage::new(64, 64);
        for p in img.pixels_mut() {
            *p = Rgb([127, 127, 127]);
        }

        let device = Device::Cpu;
        let out = preprocess_images(&[img], &cfg, &device, DType::F32)?;
        assert_eq!(out.image_grid_thw.len(), 1);
        let (t, h, w) = out.image_grid_thw[0];
        assert_eq!(t, 1);
        assert!(h > 0);
        assert!(w > 0);
        let shape = out.pixel_values.dims();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 14);
        assert_eq!(shape[3], 14);
        Ok(())
    }
}
