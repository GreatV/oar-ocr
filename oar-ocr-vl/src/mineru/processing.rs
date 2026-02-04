use super::config::MinerUImageProcessorConfig;
use crate::utils::image::{image_to_chw, pil_resample_to_filter_type, smart_resize};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};
use oar_ocr_core::core::OCRError;

#[derive(Debug, Clone)]
pub struct MinerUImageInputs {
    pub pixel_values: Tensor,
    pub image_grid_thw: Vec<(usize, usize, usize)>,
}

pub fn preprocess_images(
    images: &[RgbImage],
    cfg: &MinerUImageProcessorConfig,
    device: &Device,
    dtype: DType,
) -> Result<MinerUImageInputs, OCRError> {
    cfg.validate()?;
    if images.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "MinerU2.5: no images provided".to_string(),
        });
    }

    let factor = (cfg.patch_size * cfg.merge_size) as u32;
    let patch = cfg.patch_size as u32;
    let merge = cfg.merge_size;
    let (min_pixels, max_pixels) = if cfg.do_resize {
        cfg.pixel_bounds()?
    } else {
        (0, 0)
    };
    let resize_filter = cfg
        .resample
        .and_then(pil_resample_to_filter_type)
        .unwrap_or(FilterType::CatmullRom);
    let default_mean = [0.0_f32; 3];
    let default_std = [1.0_f32; 3];
    let mean = if cfg.do_normalize {
        cfg.image_mean.as_slice()
    } else {
        &default_mean
    };
    let std = if cfg.do_normalize {
        cfg.image_std.as_slice()
    } else {
        &default_std
    };
    let rescale_factor = if cfg.do_rescale {
        Some(cfg.rescale_factor)
    } else {
        None
    };

    let mut all_patches: Vec<f32> = Vec::new();
    let mut grids: Vec<(usize, usize, usize)> = Vec::with_capacity(images.len());

    for img in images {
        let (h, w) = (img.height(), img.width());
        if cfg.do_resize && (h < factor || w < factor) {
            return Err(OCRError::InvalidInput {
                message: format!("MinerU2.5: height/width must be >= factor {factor}, got {h}x{w}"),
            });
        }
        let (rh, rw) = if cfg.do_resize {
            smart_resize(h, w, factor, min_pixels, max_pixels)?
        } else {
            (h, w)
        };

        let resized = if cfg.do_resize && (rh != h || rw != w) {
            image::imageops::resize(img, rw, rh, resize_filter)
        } else {
            img.clone()
        };

        if rh % patch != 0 || rw % patch != 0 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 preprocess produced non-divisible dims: {rh}x{rw} not divisible by patch_size={patch}"
                ),
            });
        }

        let grid_h = (rh / patch) as usize;
        let grid_w = (rw / patch) as usize;
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 preprocess produced grid not divisible by merge_size={merge}: {grid_h}x{grid_w}"
                ),
            });
        }

        let frame = image_to_chw(&resized, mean, std, rescale_factor);
        let frames: Vec<&[f32]> =
            std::iter::repeat_n(frame.as_slice(), cfg.temporal_patch_size).collect();

        let grid_t = frames.len() / cfg.temporal_patch_size;
        let channel = 3usize;
        let height = rh as usize;
        let width = rw as usize;
        let patch_dim = channel * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let num_patches = grid_t * grid_h * grid_w;

        let mut flat_patches: Vec<f32> = Vec::with_capacity(num_patches * patch_dim);

        let frame_area = height * width;
        for tt in 0..grid_t {
            for hb in 0..(grid_h / merge) {
                for wb in 0..(grid_w / merge) {
                    for h_inner in 0..merge {
                        for w_inner in 0..merge {
                            let base_y = (hb * merge + h_inner) * cfg.patch_size;
                            let base_x = (wb * merge + w_inner) * cfg.patch_size;

                            for c in 0..channel {
                                for t in 0..cfg.temporal_patch_size {
                                    let frame_idx = tt * cfg.temporal_patch_size + t;
                                    let frame_data = frames[frame_idx];
                                    for dy in 0..cfg.patch_size {
                                        let y = base_y + dy;
                                        let row_base = c * frame_area + y * width;
                                        for dx in 0..cfg.patch_size {
                                            let x = base_x + dx;
                                            flat_patches.push(frame_data[row_base + x]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if flat_patches.len() != num_patches * patch_dim {
            return Err(OCRError::Processing {
                kind: oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                context: format!(
                    "MinerU2.5: patch extraction mismatch, got {} expected {}",
                    flat_patches.len(),
                    num_patches * patch_dim
                ),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "patch extraction length mismatch",
                )),
            });
        }

        all_patches.extend(flat_patches);
        grids.push((grid_t, grid_h, grid_w));
    }

    let patch_dim = 3usize * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let total_patches = all_patches.len() / patch_dim;

    let pixel_values = Tensor::from_vec(all_patches, (total_patches, patch_dim), device)
        .map_err(|e| OCRError::Processing {
            kind: oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            context: "MinerU2.5: failed to create pixel_values tensor".to_string(),
            source: Box::new(e),
        })?
        .to_dtype(dtype)
        .map_err(|e| OCRError::Processing {
            kind: oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            context: "MinerU2.5: failed to convert pixel_values to target dtype".to_string(),
            source: Box::new(e),
        })?;

    Ok(MinerUImageInputs {
        pixel_values,
        image_grid_thw: grids,
    })
}
