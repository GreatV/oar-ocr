//! Xiaomi-OCR-0 image preprocessing.
//!
//! Follows the official Qwen3-VL image processor: smart-resize the page to
//! patch-aligned dimensions inside the pixel-area bounds advertised by
//! `processor_config.json`, then patchify with the merged-grid grouping the
//! vision tower expects. The shared pieces (`smart_resize`, patchify,
//! normalization) come from [`crate::runtime::image`]; only the model glue
//! lives here. Unlike the shared Qwen-VL preprocessor, sub-factor crops are
//! upscaled rather than rejected, matching the OvisOCR2 path this mirrors.

use super::config::{XiaomiOcrProcessorConfig, XiaomiOcrVisionConfig};
use crate::error::Error;
use crate::utils::{
    candle_to_ocr_processing,
    image::{image_to_chw, patchify_merge_grouped, pil_resample_to_filter_type, smart_resize},
};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};

#[derive(Debug, Clone)]
pub struct XiaomiOcrImageInputs {
    pub pixel_values: Tensor,
    pub grid_thw: (usize, usize, usize),
    pub num_image_tokens: usize,
}

pub fn preprocess_image(
    image: &RgbImage,
    processor_cfg: &XiaomiOcrProcessorConfig,
    vision_cfg: &XiaomiOcrVisionConfig,
    device: &Device,
    dtype: DType,
) -> Result<XiaomiOcrImageInputs, Error> {
    processor_cfg.validate()?;
    vision_cfg.validate()?;
    processor_cfg.validate_vision_compatibility(vision_cfg)?;

    let image_processor = &processor_cfg.image_processor;
    let factor = image_processor
        .patch_size
        .checked_mul(image_processor.merge_size)
        .and_then(|factor| u32::try_from(factor).ok())
        .ok_or_else(|| Error::Config {
            message: "Xiaomi-OCR-0 patch/merge factor overflow".to_string(),
        })?;
    let (min_pixels, max_pixels) = processor_cfg.pixel_bounds()?;
    let (height, width) = (image.height(), image.width());
    if height == 0 || width == 0 {
        return Err(Error::InvalidInput {
            message: format!("Xiaomi-OCR-0 input image must be non-empty, got {width}x{height}"),
        });
    }
    let (resized_height, resized_width) = if image_processor.do_resize {
        smart_resize(height, width, factor, min_pixels, max_pixels)?
    } else {
        (height, width)
    };

    if resized_height % factor != 0 || resized_width % factor != 0 {
        return Err(Error::InvalidInput {
            message: format!(
                "Xiaomi-OCR-0 preprocessed dimensions {resized_height}x{resized_width} must be divisible by {factor}"
            ),
        });
    }

    // The checkpoint advertises PIL bicubic (`resample: 3`); fall back to the
    // equivalent CatmullRom when the field is absent, like the Qwen fast
    // processor does.
    let resize_filter = image_processor
        .resample
        .and_then(pil_resample_to_filter_type)
        .unwrap_or(FilterType::CatmullRom);
    let resized = if resized_height != height || resized_width != width {
        image::imageops::resize(image, resized_width, resized_height, resize_filter)
    } else {
        image.clone()
    };

    let default_mean = [0.0_f32; 3];
    let default_std = [1.0_f32; 3];
    let mean = if image_processor.do_normalize {
        image_processor.image_mean.as_slice()
    } else {
        &default_mean
    };
    let std = if image_processor.do_normalize {
        image_processor.image_std.as_slice()
    } else {
        &default_std
    };
    let rescale_factor = image_processor
        .do_rescale
        .then_some(image_processor.rescale_factor);
    let frame = image_to_chw(&resized, mean, std, rescale_factor);

    // A still image is one temporal grid cell whose frame is repeated to fill
    // the Conv3D temporal kernel.
    let frames: Vec<&[f32]> =
        std::iter::repeat_n(frame.as_slice(), image_processor.temporal_patch_size).collect();
    let grid_t = 1usize;
    let grid_h = resized_height as usize / image_processor.patch_size;
    let grid_w = resized_width as usize / image_processor.patch_size;
    if !grid_h.is_multiple_of(image_processor.merge_size)
        || !grid_w.is_multiple_of(image_processor.merge_size)
    {
        return Err(Error::Config {
            message: format!(
                "Xiaomi-OCR-0 patch grid {grid_h}x{grid_w} must be divisible by merge_size {}",
                image_processor.merge_size
            ),
        });
    }

    let flat = patchify_merge_grouped(
        &frames,
        vision_cfg.in_channels,
        resized_height as usize,
        resized_width as usize,
        grid_t,
        grid_h,
        grid_w,
        image_processor.patch_size,
        image_processor.merge_size,
        image_processor.temporal_patch_size,
    );
    let patch_dim = vision_cfg.in_channels
        * image_processor.temporal_patch_size
        * image_processor.patch_size
        * image_processor.patch_size;
    let num_patches = grid_t * grid_h * grid_w;
    if flat.len() != num_patches * patch_dim {
        return Err(Error::InvalidInput {
            message: format!(
                "Xiaomi-OCR-0 patch extraction produced {} values, expected {}",
                flat.len(),
                num_patches * patch_dim
            ),
        });
    }

    let pixel_values = Tensor::from_vec(flat, (num_patches, patch_dim), device)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "Xiaomi-OCR-0: create pixel_values",
                e,
            )
        })?
        .to_dtype(dtype)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "Xiaomi-OCR-0: cast pixel_values",
                e,
            )
        })?;
    let num_image_tokens = num_patches / (image_processor.merge_size * image_processor.merge_size);

    Ok(XiaomiOcrImageInputs {
        pixel_values,
        grid_thw: (grid_t, grid_h, grid_w),
        num_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn processor_config() -> XiaomiOcrProcessorConfig {
        serde_json::from_str(
            r#"{
              "image_processor": {
                "do_convert_rgb": true,
                "do_normalize": true,
                "do_rescale": true,
                "do_resize": true,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "merge_size": 2,
                "patch_size": 16,
                "resample": 3,
                "rescale_factor": 0.00392156862745098,
                "size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "temporal_patch_size": 2
              }
            }"#,
        )
        .unwrap()
    }

    fn vision_config() -> XiaomiOcrVisionConfig {
        XiaomiOcrVisionConfig {
            model_type: "qwen3_5_vision".to_string(),
            depth: 12,
            hidden_size: 768,
            intermediate_size: 3072,
            num_heads: 12,
            in_channels: 3,
            patch_size: 16,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 1024,
            num_position_embeddings: 2304,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            initializer_range: 0.02,
            deepstack_visual_indexes: Vec::new(),
        }
    }

    #[test]
    fn small_crops_upscale_to_the_advertised_minimum_area() -> Result<(), Box<dyn std::error::Error>>
    {
        let image = RgbImage::from_pixel(32, 32, Rgb([255, 255, 255]));
        let inputs = preprocess_image(
            &image,
            &processor_config(),
            &vision_config(),
            &Device::Cpu,
            DType::F32,
        )?;

        // 32x32 grows to the 256x256 (65536-pixel) advertised minimum: 16x16
        // patch grid, 8x8 merged grid, one token per merged cell.
        assert_eq!(inputs.grid_thw, (1, 16, 16));
        assert_eq!(inputs.num_image_tokens, 64);
        assert_eq!(inputs.pixel_values.dims(), &[256, 1536]);
        let first_patch = inputs.pixel_values.narrow(0, 0, 1)?.flatten_all()?;
        assert!(
            first_patch
                .to_vec1::<f32>()?
                .into_iter()
                .all(|value| (value - 1.0).abs() < 1e-6)
        );
        Ok(())
    }

    #[test]
    fn preprocess_rejects_processor_vision_mismatch() {
        let image = RgbImage::new(448, 448);
        let mut vision = vision_config();
        vision.patch_size = 14;
        let error = preprocess_image(
            &image,
            &processor_config(),
            &vision,
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err();
        assert!(error.to_string().contains("patch_size mismatch"));
    }

    #[test]
    fn preprocess_rejects_empty_image() {
        let error = preprocess_image(
            &RgbImage::new(0, 0),
            &processor_config(),
            &vision_config(),
            &Device::Cpu,
            DType::F32,
        )
        .unwrap_err();
        assert!(error.to_string().contains("must be non-empty"));
    }
}
