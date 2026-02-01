use super::config::LightOnOcrImageProcessorConfig;
use crate::utils::{
    candle_to_ocr_processing,
    image::{image_to_chw, pil_resample_to_filter_type, round_up_to_multiple},
};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};
use oar_ocr_core::core::OCRError;

#[derive(Debug)]
pub struct LightOnOcrImageInputs {
    pub pixel_values: Tensor,
    pub grid_h: usize,
    pub grid_w: usize,
}

pub fn preprocess_image(
    image: &RgbImage,
    cfg: &LightOnOcrImageProcessorConfig,
    spatial_merge_size: usize,
    device: &Device,
    dtype: DType,
) -> Result<LightOnOcrImageInputs, OCRError> {
    cfg.validate()?;
    if spatial_merge_size == 0 {
        return Err(OCRError::ConfigError {
            message: "LightOnOCR spatial_merge_size must be > 0".to_string(),
        });
    }

    let (orig_h, orig_w) = (image.height(), image.width());
    let max_edge = cfg.size.longest_edge;
    let mut target_h = orig_h;
    let mut target_w = orig_w;

    if cfg.do_resize {
        let max_dim = orig_h.max(orig_w);
        if max_dim > max_edge {
            let ratio = max_dim as f32 / max_edge as f32;
            target_h = ((orig_h as f32) / ratio).ceil() as u32;
            target_w = ((orig_w as f32) / ratio).ceil() as u32;
        }
    }

    let factor = cfg.patch_size as u32;
    if cfg.do_resize {
        target_h = round_up_to_multiple(target_h, factor);
        target_w = round_up_to_multiple(target_w, factor);
    }

    let resize_filter = cfg
        .resample
        .and_then(pil_resample_to_filter_type)
        .unwrap_or(FilterType::CatmullRom);

    let resized = if target_h != orig_h || target_w != orig_w {
        image::imageops::resize(image, target_w, target_h, resize_filter)
    } else {
        image.clone()
    };

    if target_h % (cfg.patch_size as u32) != 0 || target_w % (cfg.patch_size as u32) != 0 {
        return Err(OCRError::ConfigError {
            message: format!(
                "LightOnOCR preprocess produced non-divisible dims: {target_h}x{target_w} not divisible by patch_size={}",
                cfg.patch_size
            ),
        });
    }

    let grid_h = (target_h / cfg.patch_size as u32) as usize;
    let grid_w = (target_w / cfg.patch_size as u32) as usize;
    let merged_h = grid_h / spatial_merge_size;
    let merged_w = grid_w / spatial_merge_size;
    if merged_h == 0 || merged_w == 0 {
        return Err(OCRError::ConfigError {
            message: format!(
                "LightOnOCR preprocess produced grid {}x{} too small for spatial_merge_size={}",
                grid_h, grid_w, spatial_merge_size
            ),
        });
    }

    let scale = if cfg.do_rescale {
        Some(cfg.rescale_factor)
    } else {
        None
    };

    let mean: &[f32] = if cfg.do_normalize {
        &cfg.image_mean
    } else {
        &[0.0, 0.0, 0.0]
    };
    let std: &[f32] = if cfg.do_normalize {
        &cfg.image_std
    } else {
        &[1.0, 1.0, 1.0]
    };

    let data = image_to_chw(&resized, mean, std, scale);

    let pixel_values = Tensor::from_vec(
        data,
        (1usize, 3usize, target_h as usize, target_w as usize),
        device,
    )
    .map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "LightOnOCR: failed to create pixel_values tensor",
            e,
        )
    })?;

    let pixel_values = pixel_values.to_dtype(dtype).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "LightOnOCR: failed to cast pixel_values dtype",
            e,
        )
    })?;

    Ok(LightOnOcrImageInputs {
        pixel_values,
        grid_h,
        grid_w,
    })
}
