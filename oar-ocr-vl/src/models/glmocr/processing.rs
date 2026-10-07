use super::config::{GlmOcrImageProcessorConfig, GlmOcrVisionConfig};
use crate::backbones::qwen_vl_processing::{self, QwenVlImageProcessorConfig};
use crate::error::Error;
use crate::utils::image::pil_resample_to_filter_type;
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};

#[derive(Debug, Clone)]
pub struct GlmOcrImageInputs {
    pub pixel_values: Tensor,
    pub grid_thw: (usize, usize, usize),
    pub num_image_tokens: usize,
}

fn smart_resize_glm(
    num_frames: usize,
    height: u32,
    width: u32,
    temporal_factor: usize,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), Error> {
    if num_frames < temporal_factor {
        return Err(Error::InvalidInput {
            message: format!(
                "GLM-OCR smart_resize: num_frames ({num_frames}) < temporal_factor ({temporal_factor})"
            ),
        });
    }
    if factor == 0 {
        return Err(Error::InvalidInput {
            message: "GLM-OCR smart_resize: factor must be > 0".to_string(),
        });
    }

    let mut height = height as f64;
    let mut width = width as f64;
    let factor_f = factor as f64;

    if height < factor_f {
        width = (width * factor_f / height).round();
        height = factor_f;
    }
    if width < factor_f {
        height = (height * factor_f / width).round();
        width = factor_f;
    }

    let max_dim = height.max(width);
    let min_dim = height.min(width);
    if min_dim > 0.0 && (max_dim / min_dim) > 200.0 {
        return Err(Error::InvalidInput {
            message: format!(
                "GLM-OCR smart_resize: absolute aspect ratio must be <= 200, got {:.3}",
                max_dim / min_dim
            ),
        });
    }

    let mut h_bar = (height / factor_f).round() * factor_f;
    let mut w_bar = (width / factor_f).round() * factor_f;
    let t_bar = (num_frames as f64 / temporal_factor as f64).round() * temporal_factor as f64;

    let volume = t_bar * h_bar * w_bar;
    if volume > max_pixels as f64 {
        let beta = ((num_frames as f64 * height * width) / max_pixels as f64).sqrt();
        h_bar = ((height / beta) / factor_f).floor() * factor_f;
        w_bar = ((width / beta) / factor_f).floor() * factor_f;
        if h_bar < factor_f {
            h_bar = factor_f;
        }
        if w_bar < factor_f {
            w_bar = factor_f;
        }
    } else if volume < min_pixels as f64 {
        let beta = (min_pixels as f64 / (num_frames as f64 * height * width)).sqrt();
        h_bar = ((height * beta) / factor_f).ceil() * factor_f;
        w_bar = ((width * beta) / factor_f).ceil() * factor_f;
    }

    Ok((h_bar as u32, w_bar as u32))
}

pub fn preprocess_image(
    image: &RgbImage,
    cfg: &GlmOcrImageProcessorConfig,
    vision_cfg: &GlmOcrVisionConfig,
    device: &Device,
    dtype: DType,
) -> Result<GlmOcrImageInputs, Error> {
    cfg.validate()?;
    if cfg.patch_size != vision_cfg.patch_size {
        return Err(Error::Config {
            message: format!(
                "GLM-OCR patch_size mismatch: preprocessor {} != vision_config {}",
                cfg.patch_size, vision_cfg.patch_size
            ),
        });
    }
    if cfg.temporal_patch_size != vision_cfg.temporal_patch_size {
        return Err(Error::Config {
            message: format!(
                "GLM-OCR temporal_patch_size mismatch: preprocessor {} != vision_config {}",
                cfg.temporal_patch_size, vision_cfg.temporal_patch_size
            ),
        });
    }
    if cfg.merge_size != vision_cfg.spatial_merge_size {
        return Err(Error::Config {
            message: format!(
                "GLM-OCR merge_size mismatch: preprocessor {} != vision_config {}",
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
    let (rh, rw) = if cfg.do_resize {
        smart_resize_glm(
            cfg.temporal_patch_size,
            h,
            w,
            cfg.temporal_patch_size,
            factor,
            cfg.size.shortest_edge,
            cfg.size.longest_edge,
        )?
    } else {
        (h, w)
    };

    let resized = if rh != h || rw != w {
        image::imageops::resize(image, rw, rh, resize_filter)
    } else {
        image.clone()
    };

    if rh % factor != 0 || rw % factor != 0 {
        return Err(Error::Config {
            message: format!(
                "GLM-OCR preprocess produced non-divisible dims: {rh}x{rw} not divisible by factor={factor}"
            ),
        });
    }

    // The resize above is GLM-OCR's temporal-volume-aware variant; the
    // normalize, temporal repetition, and patchify tail is the shared
    // Qwen-VL processing.
    let shared_cfg = QwenVlImageProcessorConfig::for_resized_frames(
        cfg.patch_size,
        cfg.temporal_patch_size,
        cfg.merge_size,
        cfg.image_mean.clone(),
        cfg.image_std.clone(),
        cfg.do_normalize,
        cfg.do_rescale.then_some(cfg.rescale_factor),
    );
    let inputs = qwen_vl_processing::preprocess_resized_frames(
        std::slice::from_ref(&resized),
        &shared_cfg,
        device,
        dtype,
        "GLM-OCR",
    )?;
    let grid_thw = inputs.image_grid_thw[0];
    let num_image_tokens = (grid_thw.1 * grid_thw.2) / (cfg.merge_size * cfg.merge_size);

    Ok(GlmOcrImageInputs {
        pixel_values: inputs.pixel_values,
        grid_thw,
        num_image_tokens,
    })
}
