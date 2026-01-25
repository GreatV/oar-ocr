use super::config::LightOnOcrImageProcessorConfig;
use crate::utils::candle_to_ocr_processing;
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

    let mut data = Vec::with_capacity(3 * target_h as usize * target_w as usize);
    let mean = &cfg.image_mean;
    let std = &cfg.image_std;

    for c in 0..3usize {
        for y in 0..target_h {
            for x in 0..target_w {
                let p = resized.get_pixel(x, y);
                let mut v = p.0[c] as f32;
                if cfg.do_rescale {
                    v *= cfg.rescale_factor;
                }
                if cfg.do_normalize {
                    v = (v - mean[c]) / std[c];
                }
                data.push(v);
            }
        }
    }

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

fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }
    value.div_ceil(multiple) * multiple
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
