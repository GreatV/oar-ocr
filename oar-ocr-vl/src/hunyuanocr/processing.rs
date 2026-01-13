use super::config::{HunyuanOcrImageProcessorConfig, HunyuanOcrVisionConfig};
use crate::utils::candle_to_ocr_processing;
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};
use oar_ocr_core::core::OCRError;

#[derive(Debug, Clone)]
pub struct HunyuanOcrImageInputs {
    pub pixel_values: Tensor,
    pub grid_thw_merged: (usize, usize, usize),
}

fn pil_resample_to_filter_type(resample: u32) -> Option<FilterType> {
    // Match PIL / transformers `PILImageResampling` integer values:
    // 0=NEAREST, 1=LANCZOS, 2=BILINEAR, 3=BICUBIC, 4=BOX, 5=HAMMING.
    match resample {
        0 => Some(FilterType::Nearest),
        1 => Some(FilterType::Lanczos3),
        2 => Some(FilterType::Triangle),
        3 => Some(FilterType::CatmullRom),
        4 => Some(FilterType::Triangle),
        5 => Some(FilterType::CatmullRom),
        _ => None,
    }
}

pub fn smart_resize_token_limited(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
    max_tokens: usize,
) -> Result<(u32, u32), OCRError> {
    if factor == 0 {
        return Err(OCRError::InvalidInput {
            message: "HunyuanOCR smart_resize_token_limited: factor must be > 0".to_string(),
        });
    }
    let (mut rh, mut rw) =
        crate::paddleocr_vl::smart_resize(height, width, factor, min_pixels, max_pixels)?;

    // Token count in HunYuanVL is based on merged grid with an extra newline token per row:
    // tokens = Hm * (Wm + 1)
    loop {
        let hm = (rh / factor) as usize;
        let wm = (rw / factor) as usize;
        let tokens = hm.saturating_mul(wm.saturating_add(1));
        if tokens <= max_tokens {
            break;
        }

        // Reduce the larger axis (in merged-grid units) first to keep aspect ratio roughly intact.
        if wm >= hm {
            if rw <= factor {
                return Err(OCRError::InvalidInput {
                    message: "HunyuanOCR smart_resize_token_limited: cannot satisfy max_tokens"
                        .to_string(),
                });
            }
            rw -= factor;
        } else {
            if rh <= factor {
                return Err(OCRError::InvalidInput {
                    message: "HunyuanOCR smart_resize_token_limited: cannot satisfy max_tokens"
                        .to_string(),
                });
            }
            rh -= factor;
        }
    }

    Ok((rh, rw))
}

fn clamp_to_max_image_size(
    height: u32,
    width: u32,
    factor: u32,
    max_image_size: usize,
) -> Result<(u32, u32), OCRError> {
    if factor == 0 {
        return Err(OCRError::InvalidInput {
            message: "HunyuanOCR clamp_to_max_image_size: factor must be > 0".to_string(),
        });
    }
    let max_image_size = u32::try_from(max_image_size).map_err(|_| OCRError::ConfigError {
        message: format!("HunyuanOCR max_image_size too large: {max_image_size}"),
    })?;
    if max_image_size == 0 {
        return Err(OCRError::ConfigError {
            message: "HunyuanOCR max_image_size must be > 0".to_string(),
        });
    }
    if max_image_size < factor {
        return Err(OCRError::ConfigError {
            message: format!(
                "HunyuanOCR max_image_size {max_image_size} must be >= factor={factor}"
            ),
        });
    }

    let max_dim = height.max(width);
    if max_dim <= max_image_size {
        return Ok((height, width));
    }

    let scale = max_image_size as f64 / max_dim as f64;
    let factor_f = factor as f64;

    let mut h = (((height as f64) * scale) / factor_f).floor() * factor_f;
    let mut w = (((width as f64) * scale) / factor_f).floor() * factor_f;

    // Ensure non-zero dims and factor divisibility.
    if h < factor_f {
        h = factor_f;
    }
    if w < factor_f {
        w = factor_f;
    }

    let h = h as u32;
    let w = w as u32;

    Ok((h, w))
}

pub fn preprocess_image(
    image: &RgbImage,
    cfg: &HunyuanOcrImageProcessorConfig,
    vision_cfg: &HunyuanOcrVisionConfig,
    device: &Device,
    dtype: DType,
) -> Result<HunyuanOcrImageInputs, OCRError> {
    cfg.validate()?;
    if vision_cfg.patch_size != cfg.patch_size {
        return Err(OCRError::ConfigError {
            message: format!(
                "HunyuanOCR patch_size mismatch: preprocessor {} != vision_config {}",
                cfg.patch_size, vision_cfg.patch_size
            ),
        });
    }
    if vision_cfg.spatial_merge_size != cfg.merge_size {
        return Err(OCRError::ConfigError {
            message: format!(
                "HunyuanOCR merge_size mismatch: preprocessor {} != vision_config.spatial_merge_size {}",
                cfg.merge_size, vision_cfg.spatial_merge_size
            ),
        });
    }

    let resize_filter = cfg
        .resample
        .and_then(pil_resample_to_filter_type)
        .unwrap_or(FilterType::CatmullRom);

    let (h, w) = (image.height(), image.width());
    let factor = (cfg.patch_size * cfg.merge_size) as u32;
    let (rh, rw) = smart_resize_token_limited(
        h,
        w,
        factor,
        cfg.min_pixels,
        cfg.max_pixels,
        vision_cfg.img_max_token_num,
    )?;
    let (rh, rw) = clamp_to_max_image_size(rh, rw, factor, vision_cfg.max_image_size)?;

    let resized = if rh != h || rw != w {
        image::imageops::resize(image, rw, rh, resize_filter)
    } else {
        image.clone()
    };

    if rh % factor != 0 || rw % factor != 0 {
        return Err(OCRError::ConfigError {
            message: format!(
                "HunyuanOCR preprocess produced non-divisible dims: {rh}x{rw} not divisible by factor={factor}"
            ),
        });
    }

    let t = 1usize;

    let hm = (rh / factor) as usize;
    let wm = (rw / factor) as usize;
    let grid_thw_merged = (t, hm, wm);

    let mean = &cfg.image_mean;
    let std = &cfg.image_std;

    let mut data = Vec::with_capacity(3 * (rh as usize) * (rw as usize));
    for c in 0..3usize {
        for y in 0..rh {
            for x in 0..rw {
                let p = resized.get_pixel(x, y);
                let v = p.0[c] as f32 / 255.0;
                let v = (v - mean[c]) / std[c];
                data.push(v);
            }
        }
    }

    let pixel_values = Tensor::from_vec(data, (1usize, 3usize, rh as usize, rw as usize), device)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: failed to create pixel_values tensor",
                e,
            )
        })?
        .to_dtype(dtype)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: failed to cast pixel_values dtype",
                e,
            )
        })?;

    // Sanity-check max image size constraint.
    if rh as usize > vision_cfg.max_image_size || rw as usize > vision_cfg.max_image_size {
        return Err(OCRError::InvalidInput {
            message: format!(
                "HunyuanOCR: resized image {rw}x{rh} exceeds max_image_size={}",
                vision_cfg.max_image_size
            ),
        });
    }

    Ok(HunyuanOcrImageInputs {
        pixel_values,
        grid_thw_merged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_clamp_to_max_image_size_keeps_factor_divisible() -> Result<(), OCRError> {
        let factor = 32;
        let (h, w) = clamp_to_max_image_size(3008, 512, factor, 2048)?;
        assert!(h <= 2048);
        assert!(w <= 2048);
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
        Ok(())
    }

    #[test]
    fn test_preprocess_image_clamps_to_max_image_size() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = HunyuanOcrImageProcessorConfig {
            min_pixels: 0,
            max_pixels: 100_000_000,
            patch_size: 16,
            resample: None,
            temporal_patch_size: 1,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
        };
        let vision_cfg = HunyuanOcrVisionConfig {
            hidden_size: 1,
            intermediate_size: 1,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            num_channels: 3,
            patch_size: 16,
            spatial_merge_size: 2,
            rms_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
            add_patchemb_bias: false,
            cat_extra_token: 0,
            max_vit_seq_len: 1,
            max_image_size: 2048,
            img_max_token_num: 100_000,
            interpolate_mode: "bilinear".to_string(),
        };

        let img = RgbImage::new(512, 3000);
        let out = preprocess_image(&img, &cfg, &vision_cfg, &Device::Cpu, DType::F32)?;
        let (_b, _c, rh, rw) = out.pixel_values.dims4()?;
        assert!(rh <= vision_cfg.max_image_size);
        assert!(rw <= vision_cfg.max_image_size);

        let factor = cfg.patch_size * cfg.merge_size;
        assert_eq!(rh % factor, 0);
        assert_eq!(rw % factor, 0);
        assert_eq!(out.grid_thw_merged, (1, rh / factor, rw / factor));
        Ok(())
    }
}
