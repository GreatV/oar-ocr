use crate::error::Error;
use crate::runtime::image::{
    image_to_chw, patchify_merge_grouped, pil_resample_to_filter_type, smart_resize,
};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};
use serde::Deserialize;
use std::path::Path;

/// Image processor shared by Qwen-VL-derived OCR checkpoints.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct MinerUImageProcessorConfig {
    #[serde(default)]
    pub min_pixels: Option<u32>,
    #[serde(default)]
    pub max_pixels: Option<u32>,
    #[serde(default)]
    pub size: Option<MinerUImageSize>,
    #[serde(default = "crate::runtime::checkpoint::default_true")]
    pub do_resize: bool,
    #[serde(default = "crate::runtime::checkpoint::default_true")]
    pub do_rescale: bool,
    #[serde(default = "crate::runtime::checkpoint::default_true")]
    pub do_normalize: bool,
    #[serde(default = "crate::runtime::checkpoint::default_true")]
    pub do_convert_rgb: bool,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    #[serde(default)]
    pub resample: Option<u32>,
    #[serde(default = "crate::runtime::checkpoint::default_rescale_factor")]
    pub rescale_factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MinerUImageSize {
    pub shortest_edge: u32,
    pub longest_edge: u32,
}

impl MinerUImageProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        crate::runtime::checkpoint::load_json_config(
            path,
            "Qwen-VL OCR",
            "preprocessor_config.json",
        )
    }

    pub fn pixel_bounds(&self) -> Result<(u32, u32), Error> {
        if let Some(size) = &self.size {
            if size.shortest_edge == 0 || size.longest_edge == 0 {
                return Err(Error::config(
                    "Qwen-VL OCR size.shortest_edge/longest_edge must be > 0",
                ));
            }
            return Ok((size.shortest_edge, size.longest_edge));
        }
        match (self.min_pixels, self.max_pixels) {
            (Some(min_pixels), Some(max_pixels)) => Ok((min_pixels, max_pixels)),
            _ => Err(Error::config(
                "Qwen-VL OCR preprocessor config is missing size or min/max pixels",
            )),
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.do_normalize {
            crate::runtime::checkpoint::validate_image_mean_std(
                "Qwen-VL OCR",
                &self.image_mean,
                &self.image_std,
            )?;
            if self.image_std.contains(&0.0) {
                return Err(Error::config(
                    "Qwen-VL OCR image_std values must be non-zero",
                ));
            }
        }
        crate::runtime::checkpoint::validate_patch_merge_temporal(
            "Qwen-VL OCR",
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
        )?;
        if self.do_resize {
            let (min_pixels, max_pixels) = self.pixel_bounds()?;
            if min_pixels == 0 || max_pixels == 0 {
                return Err(Error::config("Qwen-VL OCR min/max pixels must be > 0"));
            }
            if min_pixels > max_pixels {
                return Err(Error::config(format!(
                    "Qwen-VL OCR min_pixels ({min_pixels}) must be <= max_pixels ({max_pixels})"
                )));
            }
        }
        if self.do_rescale && self.rescale_factor <= 0.0 {
            return Err(Error::config("Qwen-VL OCR rescale_factor must be > 0"));
        }
        Ok(())
    }
}

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
) -> Result<MinerUImageInputs, Error> {
    cfg.validate()?;
    if images.is_empty() {
        return Err(Error::InvalidInput {
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
            return Err(Error::InvalidInput {
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
            return Err(Error::Config {
                message: format!(
                    "MinerU2.5 preprocess produced non-divisible dims: {rh}x{rw} not divisible by patch_size={patch}"
                ),
            });
        }

        let grid_h = (rh / patch) as usize;
        let grid_w = (rw / patch) as usize;
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(Error::Config {
                message: format!(
                    "MinerU2.5 preprocess produced grid not divisible by merge_size={merge}: {grid_h}x{grid_w}"
                ),
            });
        }

        let frame = image_to_chw(&resized, mean, std, rescale_factor);
        // For static document images, repeat the same frame to match the expected
        // temporal_patch_size dimension. This is correct behavior for image-only
        // models - the temporal dimension exists in the architecture but since
        // there's only one "frame" (the document image), it's repeated to match
        // the tensor shape expected by the vision encoder.
        let frames: Vec<&[f32]> =
            std::iter::repeat_n(frame.as_slice(), cfg.temporal_patch_size).collect();

        let grid_t = frames.len() / cfg.temporal_patch_size;
        let channel = 3usize;
        let height = rh as usize;
        let width = rw as usize;
        let patch_dim = channel * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let num_patches = grid_t * grid_h * grid_w;

        let flat_patches = patchify_merge_grouped(
            &frames,
            channel,
            height,
            width,
            grid_t,
            grid_h,
            grid_w,
            cfg.patch_size,
            merge,
            cfg.temporal_patch_size,
        );

        if flat_patches.len() != num_patches * patch_dim {
            return Err(Error::Processing {
                kind: crate::error::ProcessingStage::TensorOperation,
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
        .map_err(|e| Error::Processing {
            kind: crate::error::ProcessingStage::TensorOperation,
            context: "MinerU2.5: failed to create pixel_values tensor".to_string(),
            source: Box::new(e),
        })?
        .to_dtype(dtype)
        .map_err(|e| Error::Processing {
            kind: crate::error::ProcessingStage::TensorOperation,
            context: "MinerU2.5: failed to convert pixel_values to target dtype".to_string(),
            source: Box::new(e),
        })?;

    Ok(MinerUImageInputs {
        pixel_values,
        image_grid_thw: grids,
    })
}
