use super::config::MonkeyOcrV2ImageProcessorConfig;
use crate::backbones::qwen_vl_processing::{self, QwenVlImageProcessorConfig};
use crate::error::Error;
use crate::utils::image::{pil_resample_to_filter_type, smart_resize};
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};

#[derive(Debug, Clone)]
pub struct MonkeyOcrV2ImageInputs {
    pub pixel_values: Tensor,
    pub grid_thw: (usize, usize, usize),
    pub num_image_tokens: usize,
}

pub fn preprocess_image(
    image: &RgbImage,
    cfg: &MonkeyOcrV2ImageProcessorConfig,
    device: &Device,
    dtype: DType,
) -> Result<MonkeyOcrV2ImageInputs, Error> {
    cfg.validate()?;
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::InvalidInput {
            message: "MonkeyOCRv2 cannot process an empty image".to_string(),
        });
    }

    let factor = (cfg.patch_size * cfg.merge_size) as u32;
    let (height, width) = (image.height(), image.width());
    if !cfg.do_resize && (height < factor || width < factor) {
        return Err(Error::InvalidInput {
            message: format!(
                "MonkeyOCRv2 image dimensions must be at least {factor}x{factor}, got {width}x{height}"
            ),
        });
    }
    if !cfg.do_resize && (!height.is_multiple_of(factor) || !width.is_multiple_of(factor)) {
        return Err(Error::InvalidInput {
            message: format!(
                "MonkeyOCRv2 image dimensions must be divisible by {factor} when resizing is disabled, got {width}x{height}"
            ),
        });
    }
    let (resized_height, resized_width) = if cfg.do_resize {
        smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels)?
    } else {
        (height, width)
    };
    let filter = cfg
        .resample
        .and_then(pil_resample_to_filter_type)
        .unwrap_or(FilterType::CatmullRom);
    let resized_holder;
    let resized = if resized_height != height || resized_width != width {
        resized_holder = image::imageops::resize(image, resized_width, resized_height, filter);
        &resized_holder
    } else {
        image
    };

    // Resize above carries this model's own bounds; the normalize, temporal
    // repetition, and patchify tail is the shared Qwen-VL processing.
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
        std::slice::from_ref(resized),
        &shared_cfg,
        device,
        dtype,
        "MonkeyOCRv2",
    )?;
    let grid_thw = inputs.image_grid_thw[0];
    let num_image_tokens = (grid_thw.1 * grid_thw.2) / (cfg.merge_size * cfg.merge_size);
    Ok(MonkeyOcrV2ImageInputs {
        pixel_values: inputs.pixel_values,
        grid_thw,
        num_image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MonkeyOcrV2ImageProcessorConfig {
        MonkeyOcrV2ImageProcessorConfig {
            min_pixels: 28 * 28,
            max_pixels: 1024 * 1024,
            patch_size: 14,
            temporal_patch_size: 1,
            merge_size: 2,
            image_mean: vec![0.48145466, 0.4578275, 0.40821073],
            image_std: vec![0.26862954, 0.261_302_6, 0.275_777_1],
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            rescale_factor: 1.0 / 255.0,
            resample: Some(3),
        }
    }

    #[test]
    fn creates_merge_grouped_patches() {
        let image = RgbImage::new(84, 56);
        let inputs = preprocess_image(&image, &config(), &Device::Cpu, DType::F32).unwrap();
        assert_eq!(inputs.grid_thw, (1, 4, 6));
        assert_eq!(inputs.num_image_tokens, 6);
        assert_eq!(inputs.pixel_values.dims(), &[24, 588]);
    }

    #[test]
    fn upscales_small_crop_when_resize_is_enabled() {
        let image = RgbImage::new(84, 20);
        let inputs = preprocess_image(&image, &config(), &Device::Cpu, DType::F32).unwrap();
        assert_eq!(inputs.grid_thw, (1, 2, 6));
        assert_eq!(inputs.num_image_tokens, 3);
        assert_eq!(inputs.pixel_values.dims(), &[12, 588]);
    }

    #[test]
    fn rejects_too_small_image_when_resize_is_disabled() {
        let image = RgbImage::new(27, 28);
        let mut cfg = config();
        cfg.do_resize = false;

        let error = preprocess_image(&image, &cfg, &Device::Cpu, DType::F32).unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { message }
                if message.contains("at least 28x28, got 27x28")
        ));
    }

    #[test]
    fn rejects_unaligned_image_when_resize_is_disabled() {
        let image = RgbImage::new(85, 56);
        let mut cfg = config();
        cfg.do_resize = false;

        let error = preprocess_image(&image, &cfg, &Device::Cpu, DType::F32).unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { message }
                if message.contains("divisible by 28") && message.contains("85x56")
        ));
    }
}
