use crate::error::Error;
use crate::runtime::image::{image_to_merge_grouped_patches, qwen_vl_resize, smart_resize};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use serde::Deserialize;
use std::path::Path;

/// Image processor shared by Qwen-VL-derived OCR checkpoints.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct QwenVlImageProcessorConfig {
    #[serde(default)]
    pub min_pixels: Option<u32>,
    #[serde(default)]
    pub max_pixels: Option<u32>,
    #[serde(default)]
    pub size: Option<QwenVlImageSize>,
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
pub struct QwenVlImageSize {
    pub shortest_edge: u32,
    pub longest_edge: u32,
}

impl QwenVlImageProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        crate::runtime::checkpoint::load_json_config(
            path,
            "Qwen-VL OCR",
            "preprocessor_config.json",
        )
    }

    /// A config for frames a caller already resized: preprocessing then only
    /// normalizes, repeats the frame across the temporal dimension, and
    /// patchifies with this geometry. Models whose resize semantics differ
    /// from [`preprocess_images`](fn@preprocess_images) (different bounds or
    /// a temporal-volume-aware resize) resize themselves and delegate the
    /// rest through this.
    pub(crate) fn for_resized_frames(
        patch_size: usize,
        temporal_patch_size: usize,
        merge_size: usize,
        image_mean: Vec<f32>,
        image_std: Vec<f32>,
        do_normalize: bool,
        rescale: Option<f32>,
    ) -> Self {
        Self {
            min_pixels: None,
            max_pixels: None,
            size: None,
            do_resize: false,
            do_rescale: rescale.is_some(),
            do_normalize,
            do_convert_rgb: true,
            patch_size,
            temporal_patch_size,
            merge_size,
            image_mean,
            image_std,
            resample: None,
            rescale_factor: rescale.unwrap_or(1.0 / 255.0),
        }
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
        self.validate_geometry()?;
        if self.do_rescale && self.rescale_factor <= 0.0 {
            return Err(Error::config("Qwen-VL OCR rescale_factor must be > 0"));
        }
        Ok(())
    }

    /// Everything [`validate`](Self::validate) checks except the Qwen-VL
    /// rescale-factor rule. Configs from `for_resized_frames` carry another
    /// model's already-validated factor, whose rules differ.
    fn validate_geometry(&self) -> Result<(), Error> {
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
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct QwenVlImageInputs {
    pub pixel_values: Tensor,
    pub image_grid_thw: Vec<(usize, usize, usize)>,
}

pub fn preprocess_images(
    images: &[RgbImage],
    cfg: &QwenVlImageProcessorConfig,
    device: &Device,
    dtype: DType,
    model_name: &str,
) -> Result<QwenVlImageInputs, Error> {
    cfg.validate()?;
    preprocess_validated(images, cfg, device, dtype, model_name)
}

/// [`preprocess_images`] for frames the caller already resized with a
/// config from `for_resized_frames`. The caller's model has validated its
/// own rescale factor, so only the geometry checks run here.
pub(crate) fn preprocess_resized_frames(
    images: &[RgbImage],
    cfg: &QwenVlImageProcessorConfig,
    device: &Device,
    dtype: DType,
    model_name: &str,
) -> Result<QwenVlImageInputs, Error> {
    cfg.validate_geometry()?;
    preprocess_validated(images, cfg, device, dtype, model_name)
}

fn preprocess_validated(
    images: &[RgbImage],
    cfg: &QwenVlImageProcessorConfig,
    device: &Device,
    dtype: DType,
    model_name: &str,
) -> Result<QwenVlImageInputs, Error> {
    if images.is_empty() {
        return Err(Error::InvalidInput {
            message: format!("{model_name}: no images provided"),
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
                message: format!(
                    "{model_name}: height/width must be >= factor {factor}, got {h}x{w}"
                ),
            });
        }
        let (rh, rw) = if cfg.do_resize {
            smart_resize(h, w, factor, min_pixels, max_pixels)?
        } else {
            (h, w)
        };

        // Borrow the page when no resize happens — the tail-only configs
        // hand in already-resized frames, and cloning each page's RGB buffer
        // would copy tens of MiB per image for nothing. The Qwen-VL-family
        // processors resample with Pillow's bicubic, so the resize goes
        // through the PIL-exact path.
        let resized_on_heap;
        let resized: &image::RgbImage = if cfg.do_resize && (rh != h || rw != w) {
            resized_on_heap = qwen_vl_resize(img, rw, rh, cfg.resample)?;
            &resized_on_heap
        } else {
            img
        };

        if rh % patch != 0 || rw % patch != 0 {
            return Err(Error::Config {
                message: format!(
                    "{model_name} preprocess produced non-divisible dims: {rh}x{rw} not divisible by patch_size={patch}"
                ),
            });
        }

        let grid_h = (rh / patch) as usize;
        let grid_w = (rw / patch) as usize;
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(Error::Config {
                message: format!(
                    "{model_name} preprocess produced grid not divisible by merge_size={merge}: {grid_h}x{grid_w}"
                ),
            });
        }

        let grid_t = 1;
        let channel = 3usize;
        let patch_dim = channel * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let num_patches = grid_t * grid_h * grid_w;

        let flat_patches = image_to_merge_grouped_patches(
            resized,
            mean,
            std,
            rescale_factor,
            cfg.patch_size,
            merge,
            cfg.temporal_patch_size,
        );

        if flat_patches.len() != num_patches * patch_dim {
            return Err(Error::Processing {
                kind: crate::error::ProcessingStage::TensorOperation,
                context: format!(
                    "{model_name}: patch extraction mismatch, got {} expected {}",
                    flat_patches.len(),
                    num_patches * patch_dim
                ),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "patch extraction length mismatch",
                )),
            });
        }

        // The first image's patches move straight into the accumulator;
        // later images append. Single-image callers (the common case) thus
        // never copy their patch vector.
        if all_patches.is_empty() {
            all_patches = flat_patches;
        } else {
            all_patches.extend(flat_patches);
        }
        grids.push((grid_t, grid_h, grid_w));
    }

    let patch_dim = 3usize * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let total_patches = all_patches.len() / patch_dim;

    let pixel_values = Tensor::from_vec(all_patches, (total_patches, patch_dim), device)
        .map_err(|e| Error::Processing {
            kind: crate::error::ProcessingStage::TensorOperation,
            context: format!("{model_name}: failed to create pixel_values tensor"),
            source: Box::new(e),
        })?
        .to_dtype(dtype)
        .map_err(|e| Error::Processing {
            kind: crate::error::ProcessingStage::TensorOperation,
            context: format!("{model_name}: failed to convert pixel_values to target dtype"),
            source: Box::new(e),
        })?;

    Ok(QwenVlImageInputs {
        pixel_values,
        image_grid_thw: grids,
    })
}

#[cfg(test)]
mod tests {
    use super::{QwenVlImageProcessorConfig, preprocess_images};
    use candle_core::{DType, Device};
    use image::RgbImage;

    fn fixture_config() -> QwenVlImageProcessorConfig {
        QwenVlImageProcessorConfig {
            min_pixels: Some(65536),
            max_pixels: Some(16_777_216),
            size: None,
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            do_convert_rgb: true,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
            resample: None,
            rescale_factor: 1.0 / 255.0,
        }
    }

    #[test]
    fn direct_patches_match_chw_then_patchify_bits() -> Result<(), crate::Error> {
        use crate::runtime::image::{
            image_to_chw, patchify_merge_grouped, qwen_vl_resize, smart_resize,
        };

        let mut cfg = fixture_config();
        cfg.patch_size = 2;
        cfg.min_pixels = Some(64);
        cfg.max_pixels = Some(1024);
        cfg.image_mean = vec![0.48145466, 0.4578275, 0.40821073];
        cfg.image_std = vec![0.26862954, 0.2613026, 0.2757771];
        for (w, h) in [(7, 11), (37, 23), (59, 53), (16, 24)] {
            let image = RgbImage::from_fn(w, h, |x, y| {
                image::Rgb(std::array::from_fn(|c| {
                    ((x * 37 + y * 71 + c as u32 * 53) % 256) as u8
                }))
            });
            for (rescale, normalize) in [(false, false), (false, true), (true, false), (true, true)]
            {
                cfg.do_rescale = rescale;
                cfg.do_normalize = normalize;
                let (rh, rw) = smart_resize(h, w, 4, 64, 1024)?;
                let resized = qwen_vl_resize(&image, rw, rh, cfg.resample)?;
                let mean = if normalize {
                    cfg.image_mean.as_slice()
                } else {
                    &[0.0; 3]
                };
                let std = if normalize {
                    cfg.image_std.as_slice()
                } else {
                    &[1.0; 3]
                };
                let chw = image_to_chw(&resized, mean, std, rescale.then_some(cfg.rescale_factor));
                let (gh, gw) = (rh as usize / 2, rw as usize / 2);
                let expected = patchify_merge_grouped(
                    &[&chw, &chw],
                    3,
                    rh as usize,
                    rw as usize,
                    1,
                    gh,
                    gw,
                    2,
                    2,
                    2,
                );
                let actual = preprocess_images(
                    std::slice::from_ref(&image),
                    &cfg,
                    &Device::Cpu,
                    DType::F32,
                    "test",
                )?;
                assert_eq!(actual.image_grid_thw, vec![(1, gh, gw)]);
                let actual = actual.pixel_values.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(
                    actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    expected.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    "{w}x{h}, rescale={rescale}, normalize={normalize}",
                );
            }
        }
        Ok(())
    }

    #[test]
    fn resized_frames_without_normalization_keep_raw_rescaled_pixels() {
        // for_resized_frames callers pass frames they resized themselves; a
        // do_normalize=false config must leave rescaled pixels untouched
        // rather than applying default mean/std.
        let cfg = super::QwenVlImageProcessorConfig::for_resized_frames(
            16,
            2,
            2,
            vec![0.5; 3],
            vec![0.5; 3],
            false,
            Some(1.0 / 255.0),
        );
        let white = RgbImage::from_pixel(64, 64, image::Rgb([255, 255, 255]));
        let inputs = super::preprocess_resized_frames(
            std::slice::from_ref(&white),
            &cfg,
            &Device::Cpu,
            DType::F32,
            "FixtureModel",
        )
        .unwrap();
        assert_eq!(inputs.image_grid_thw, [(1, 4, 4)]);
        let values = inputs
            .pixel_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    }

    #[test]
    fn errors_name_the_calling_model() {
        let cfg = fixture_config();
        let device = Device::Cpu;

        let empty = preprocess_images(&[], &cfg, &device, DType::F32, "FixtureModel")
            .expect_err("empty input must fail");
        assert!(
            empty
                .to_string()
                .contains("FixtureModel: no images provided"),
            "unexpected error: {empty}"
        );

        // Below the patch*merge factor (16*2 = 32) on one side.
        let tiny = RgbImage::from_pixel(10, 200, image::Rgb([120, 140, 160]));
        let small = preprocess_images(
            std::slice::from_ref(&tiny),
            &cfg,
            &device,
            DType::F32,
            "FixtureModel",
        )
        .expect_err("sub-factor image must fail");
        assert!(
            small
                .to_string()
                .contains("FixtureModel: height/width must be >= factor 32"),
            "unexpected error: {small}"
        );
    }
}
