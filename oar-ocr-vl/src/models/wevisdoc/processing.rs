//! Image preprocessing for WeVisDoc (Qwen2VLImageProcessorFast semantics).

use super::config::Qwen3VlVisionConfig;
use crate::backbones::qwen_vl_processing::{MinerUImageProcessorConfig, preprocess_images};
use crate::error::Error;
use crate::runtime::checkpoint::load_json_config;
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use std::path::Path;

/// Preprocessed inputs for one page image.
#[derive(Debug, Clone)]
pub struct WeVisDocImageInputs {
    pub pixel_values: Tensor,
    pub grid_thw: (usize, usize, usize),
    pub num_image_tokens: usize,
}

pub(crate) fn load_image_processor_config(
    path: impl AsRef<Path>,
) -> Result<MinerUImageProcessorConfig, Error> {
    let cfg: MinerUImageProcessorConfig =
        load_json_config(path, "WeVisDoc", "preprocessor_config.json")?;
    cfg.validate()?;
    Ok(cfg)
}

pub(crate) fn validate_processor_vision_compatibility(
    cfg: &MinerUImageProcessorConfig,
    vision: &Qwen3VlVisionConfig,
) -> Result<(), Error> {
    if cfg.patch_size != vision.patch_size {
        return Err(Error::Config {
            message: format!(
                "WeVisDoc patch_size mismatch: processor {} != vision {}",
                cfg.patch_size, vision.patch_size
            ),
        });
    }
    if cfg.temporal_patch_size != vision.temporal_patch_size {
        return Err(Error::Config {
            message: format!(
                "WeVisDoc temporal_patch_size mismatch: processor {} != vision {}",
                cfg.temporal_patch_size, vision.temporal_patch_size
            ),
        });
    }
    if cfg.merge_size != vision.spatial_merge_size {
        return Err(Error::Config {
            message: format!(
                "WeVisDoc merge_size mismatch: processor {} != vision {}",
                cfg.merge_size, vision.spatial_merge_size
            ),
        });
    }
    if vision.in_channels != 3 {
        return Err(Error::Config {
            message: format!(
                "WeVisDoc image preprocessing supports three RGB channels, got {}",
                vision.in_channels
            ),
        });
    }
    Ok(())
}

/// Resize, rescale, normalize, and patchify one page image.
///
/// The processor's `size.shortest_edge`/`longest_edge` entries are minimum /
/// maximum pixel *areas* (65536..=16777216 for WeVisDoc), matching the
/// `Qwen2VLImageProcessorFast` checkpoint metadata.
pub fn preprocess_image(
    image: &RgbImage,
    cfg: &MinerUImageProcessorConfig,
    vision: &Qwen3VlVisionConfig,
    device: &Device,
    dtype: DType,
) -> Result<WeVisDocImageInputs, Error> {
    validate_processor_vision_compatibility(cfg, vision)?;
    // Document-parser crops can be narrower than the patch grid on one
    // side (for example a 10x200 rule). Pad the aspect ratio first, then
    // scale up to the patch grid: the other order lets a 1x4096 rule
    // upscale into 32x131072 before the pad multiplies it into ~86M
    // pixels, blowing up peak memory.
    let image = pad_to_ratio(image, SMART_RESIZE_MAX_RATIO);
    let min_edge = (cfg.merge_size * cfg.patch_size) as u32;
    let image = upscale_min_edge(&image, min_edge);
    let inputs = preprocess_images(std::slice::from_ref(&image), cfg, device, dtype, "WeVisDoc")?;
    let grid_thw = *inputs
        .image_grid_thw
        .first()
        .ok_or_else(|| Error::InvalidInput {
            message: "WeVisDoc preprocessing produced no image grid".to_string(),
        })?;
    let merge_group = cfg.merge_size * cfg.merge_size;
    let num_image_tokens = grid_thw.0 * grid_thw.1 * grid_thw.2 / merge_group;
    Ok(WeVisDocImageInputs {
        pixel_values: inputs.pixel_values,
        grid_thw,
        num_image_tokens,
    })
}

/// Pad the short side (centered, white) until the aspect ratio is within
/// the processor's limit, so extreme crops — a 1x4096 rule, say — survive
/// `smart_resize`'s ratio bound. Runs before any upscaling: padding the
/// already-upscaled image would multiply the pad into the upscale.
use image::Rgb;

/// `smart_resize` rejects aspect ratios above 200.
const SMART_RESIZE_MAX_RATIO: f32 = 200.0;

fn pad_to_ratio(image: &RgbImage, max_ratio: f32) -> RgbImage {
    let (w, h) = image.dimensions();
    if w == 0 || h == 0 {
        return image.clone();
    }
    let ratio = w.max(h) as f32 / w.min(h) as f32;
    if ratio <= max_ratio {
        return image.clone();
    }
    let (new_w, new_h) = if w > h {
        (w, ((w as f32 / max_ratio).ceil() as u32).max(1))
    } else {
        (((h as f32 / max_ratio).ceil() as u32).max(1), h)
    };
    let mut canvas = RgbImage::from_pixel(new_w, new_h, Rgb([255, 255, 255]));
    let x = ((new_w - w) / 2) as i64;
    let y = ((new_h - h) / 2) as i64;
    image::imageops::overlay(&mut canvas, image, x, y);
    canvas
}

/// Scale an image up until its shorter edge reaches `min_edge`, keeping the
/// aspect ratio. Images already at or above `min_edge` pass through
/// untouched.
fn upscale_min_edge(image: &RgbImage, min_edge: u32) -> RgbImage {
    let (w, h) = image.dimensions();
    let min_dim = w.min(h);
    if min_dim == 0 || min_dim >= min_edge {
        return image.clone();
    }
    let scale = min_edge as f32 / min_dim as f32;
    let new_w = ((w as f32 * scale).ceil() as u32).max(min_edge);
    let new_h = ((h as f32 * scale).ceil() as u32).max(min_edge);
    image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::CatmullRom)
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::CONFIG;
    use super::*;
    use image::Rgb;

    #[test]
    fn narrow_parser_crops_preprocess_instead_of_failing() {
        let config = super::super::config::tests::official_config();
        // The official fixture embeds the processor block; mirror the
        // loader's defaults for the fields it omits.
        let cfg = MinerUImageProcessorConfig {
            min_pixels: Some(65536),
            max_pixels: Some(16_777_216),
            size: None,
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            do_convert_rgb: true,
            patch_size: config.vision_config.patch_size,
            temporal_patch_size: config.vision_config.temporal_patch_size,
            merge_size: config.vision_config.spatial_merge_size,
            image_mean: vec![0.4814547, 0.4578275, 0.4082107],
            image_std: vec![0.2686295, 0.2613026, 0.2757771],
            resample: None,
            rescale_factor: 1.0 / 255.0,
        };
        let vision = &config.vision_config;
        for (w, h) in [(10u32, 200u32), (200, 10), (31, 31), (5, 4000), (4000, 5)] {
            let img = RgbImage::from_pixel(w, h, Rgb([120, 140, 160]));
            let inputs = preprocess_image(&img, &cfg, vision, &Device::Cpu, DType::F32)
                .unwrap_or_else(|e| panic!("{w}x{h} failed: {e}"));
            assert!(inputs.num_image_tokens > 0, "{w}x{h} produced no tokens");
        }
    }

    #[test]
    fn extreme_crops_pad_before_upscaling() {
        // 1x4096 is the codex-reported OOM shape: upscaling first turned it
        // into 32x131072, and the ratio pad then multiplied that into ~86M
        // pixels. Padding first keeps every intermediate bounded by the
        // ratio limit, and the final grid stays valid.
        let config = super::super::config::tests::official_config();
        let cfg = MinerUImageProcessorConfig {
            min_pixels: Some(65536),
            max_pixels: Some(16_777_216),
            size: None,
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            do_convert_rgb: true,
            patch_size: config.vision_config.patch_size,
            temporal_patch_size: config.vision_config.temporal_patch_size,
            merge_size: config.vision_config.spatial_merge_size,
            image_mean: vec![0.4814547, 0.4578275, 0.4082107],
            image_std: vec![0.2686295, 0.2613026, 0.2757771],
            resample: None,
            rescale_factor: 1.0 / 255.0,
        };
        let vision = &config.vision_config;
        let min_edge = (cfg.merge_size * cfg.patch_size) as u32;
        for (w, h) in [(1u32, 4096u32), (4096, 1)] {
            let img = RgbImage::from_pixel(w, h, Rgb([120, 140, 160]));
            let padded = pad_to_ratio(&img, SMART_RESIZE_MAX_RATIO);
            let upscaled = upscale_min_edge(&padded, min_edge);
            assert!(upscaled.width() >= min_edge && upscaled.height() >= min_edge);
            let padded_pixels = padded.width() as u64 * padded.height() as u64;
            let final_pixels = upscaled.width() as u64 * upscaled.height() as u64;
            // The upscale only ever grows the padded canvas by the short
            // edge factor; nothing between the two steps may explode.
            assert!(
                final_pixels < 4_000_000,
                "{w}x{h} upscaled to {final_pixels} pixels"
            );
            assert!(
                padded_pixels <= final_pixels && final_pixels <= padded_pixels * 4,
                "{w}x{h} padded {padded_pixels} vs final {final_pixels}"
            );
            let inputs = preprocess_image(&img, &cfg, vision, &Device::Cpu, DType::F32)
                .unwrap_or_else(|e| panic!("{w}x{h} failed: {e}"));
            assert!(inputs.num_image_tokens > 0, "{w}x{h} produced no tokens");
        }
    }

    /// Matches the official `preprocessor_config.json` (Tencent/WeVisDoc-2B).
    fn processor_config() -> MinerUImageProcessorConfig {
        serde_json::from_str(
            r#"{
              "size": {"longest_edge": 16777216, "shortest_edge": 65536},
              "patch_size": 16,
              "temporal_patch_size": 2,
              "merge_size": 2,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "processor_class": "Qwen3VLProcessor",
              "image_processor_type": "Qwen2VLImageProcessorFast"
            }"#,
        )
        .unwrap()
    }

    fn vision_config() -> Qwen3VlVisionConfig {
        let cfg: Qwen3VlVisionConfig =
            serde_json::from_str::<super::super::config::WeVisDocConfig>(CONFIG)
                .unwrap()
                .vision_config;
        cfg
    }

    #[test]
    fn processor_config_parses_official_bounds() {
        let cfg = processor_config();
        cfg.validate().unwrap();
        assert_eq!(cfg.pixel_bounds().unwrap(), (65536, 16_777_216));
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.merge_size, 2);
        assert_eq!(cfg.image_mean, vec![0.5; 3]);
    }

    #[test]
    fn preprocess_keeps_in_range_image_unchanged() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = processor_config();
        let image = RgbImage::from_pixel(512, 512, Rgb([255, 255, 255]));
        let inputs = preprocess_image(&image, &cfg, &vision_config(), &Device::Cpu, DType::F32)?;
        // 512x512 (262144 pixels) is inside [65536, 16777216]; the smart
        // resize keeps it unchanged. Factor 32: 512/16 = 32 patches per side.
        assert_eq!(inputs.grid_thw, (1, 32, 32));
        assert_eq!(inputs.num_image_tokens, 256);
        assert_eq!(inputs.pixel_values.dims(), &[1024, 1536]);
        Ok(())
    }

    #[test]
    fn preprocess_respects_minimum_area() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = processor_config();
        // 64x64 = 4096 pixels is far below the 65536-pixel floor; smart resize
        // scales the area up to at least 65536 and rounds to factor 32.
        let image = RgbImage::from_pixel(64, 64, Rgb([0, 0, 0]));
        let inputs = preprocess_image(&image, &cfg, &vision_config(), &Device::Cpu, DType::F32)?;
        let (grid_t, grid_h, grid_w) = inputs.grid_thw;
        assert_eq!(grid_t, 1);
        assert!(grid_h as u32 * 16 * (grid_w as u32 * 16) >= 65536);
        assert!(grid_h.is_multiple_of(2) && grid_w.is_multiple_of(2));
        Ok(())
    }

    #[test]
    fn rejects_processor_vision_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let cfg = processor_config();
        let mut vision = vision_config();
        vision.patch_size = 14;
        let image = RgbImage::from_pixel(512, 512, Rgb([255, 255, 255]));
        let error = preprocess_image(&image, &cfg, &vision, &Device::Cpu, DType::F32).unwrap_err();
        assert!(error.to_string().contains("patch_size mismatch"));
        Ok(())
    }
}
