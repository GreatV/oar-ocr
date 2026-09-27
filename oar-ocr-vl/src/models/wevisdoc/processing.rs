//! Image preprocessing for WeVisDoc (Qwen2VLImageProcessorFast semantics).

use super::config::Qwen3VlVisionConfig;
use crate::backbones::qwen_vl_processing::{QwenVlImageProcessorConfig, preprocess_images};
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

// Test probe: device MiB observed at the pixel-value upload point on
// this thread's most recent `preprocess_image` call. Thread-local so
// parallel GPU tests cannot read each other's readings; read it with
// `take_last_upload_probe_mib` (a doc comment cannot attach to the
// thread_local! invocation, hence plain comments).
#[cfg(all(test, feature = "cuda"))]
std::thread_local! {
    static LAST_UPLOAD_PROBE_MIB: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-only accessor: the last upload-point reading stored on this
/// thread, resetting it to zero.
#[cfg(all(test, feature = "cuda"))]
pub(crate) fn take_last_upload_probe_mib() -> u64 {
    LAST_UPLOAD_PROBE_MIB.with(|probe| probe.replace(0))
}

pub(crate) fn load_image_processor_config(
    path: impl AsRef<Path>,
) -> Result<QwenVlImageProcessorConfig, Error> {
    let cfg: QwenVlImageProcessorConfig =
        load_json_config(path, "WeVisDoc", "preprocessor_config.json")?;
    cfg.validate()?;
    Ok(cfg)
}

/// CPU-only planning for one page image: mirrors the upscale + resize the
/// real preprocessing performs and returns the resulting image-token count.
/// Lets callers know the decode bucket — and release stale fixed KV —
/// before anything is uploaded to the device.
pub fn plan_num_image_tokens(
    image: &RgbImage,
    cfg: &QwenVlImageProcessorConfig,
) -> Result<usize, Error> {
    cfg.validate()?;
    // Mirror preprocess_image's exact sequence through the pure dims
    // functions: pad to the ratio limit, upscale the short edge, then
    // resize. No pixel buffers are touched.
    let factor = (cfg.merge_size * cfg.patch_size) as u32;
    let (w, h) = {
        let (pw, ph) = padded_dims(image.width(), image.height(), SMART_RESIZE_MAX_RATIO);
        upscaled_dims(pw, ph, factor)
    };
    let (min_pixels, max_pixels) = if cfg.do_resize {
        cfg.pixel_bounds()?
    } else {
        (0, 0)
    };
    let (rh, rw) = if cfg.do_resize {
        crate::runtime::image::smart_resize(h, w, factor, min_pixels, max_pixels)?
    } else {
        (h, w)
    };
    if rh % cfg.patch_size as u32 != 0 || rw % cfg.patch_size as u32 != 0 {
        return Err(Error::InvalidInput {
            message: format!(
                "WeVisDoc preprocessing plan: {rh}x{rw} not divisible by patch {}",
                cfg.patch_size
            ),
        });
    }
    let grid_h = rh / cfg.patch_size as u32;
    let grid_w = rw / cfg.patch_size as u32;
    let merge_group = cfg.merge_size * cfg.merge_size;
    Ok(grid_h as usize * grid_w as usize / merge_group)
}

pub(crate) fn validate_processor_vision_compatibility(
    cfg: &QwenVlImageProcessorConfig,
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
    cfg: &QwenVlImageProcessorConfig,
    vision: &Qwen3VlVisionConfig,
    device: &Device,
    dtype: DType,
) -> Result<WeVisDocImageInputs, Error> {
    validate_processor_vision_compatibility(cfg, vision)?;
    // Test-only production-entry probe: always armed in cuda test builds
    // (thread-local, so parallel tests stay isolated) and records device
    // memory at the upload point, before the pixel values are uploaded.
    #[cfg(all(test, feature = "cuda"))]
    if let Device::Cuda(cuda) = device {
        cuda.cuda_stream().synchronize().unwrap();
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output();
        let mib = match output {
            Ok(out) => String::from_utf8_lossy(&out.stdout)
                .trim()
                .lines()
                .next()
                .and_then(|line| line.trim().parse().ok())
                .unwrap_or(0),
            Err(_) => 0,
        };
        LAST_UPLOAD_PROBE_MIB.with(|probe| probe.set(mib));
        eprintln!("DBGM9 pre-upload used={mib}MiB");
    }
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

/// `smart_resize` rejects aspect ratios above 200.
const SMART_RESIZE_MAX_RATIO: f32 = 200.0;

/// Target dims after padding the short side so the aspect ratio is within
/// the processor's limit. Pure arithmetic — no pixel buffers touched.
fn padded_dims(w: u32, h: u32, max_ratio: f32) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (w, h);
    }
    let ratio = w.max(h) as f32 / w.min(h) as f32;
    if ratio <= max_ratio {
        return (w, h);
    }
    if w > h {
        (w, ((w as f32 / max_ratio).ceil() as u32).max(1))
    } else {
        (((h as f32 / max_ratio).ceil() as u32).max(1), h)
    }
}

/// Target dims after scaling until the shorter edge reaches `min_edge`.
/// Pure arithmetic — no pixel buffers touched.
fn upscaled_dims(w: u32, h: u32, min_edge: u32) -> (u32, u32) {
    let min_dim = w.min(h);
    if min_dim == 0 || min_dim >= min_edge {
        return (w, h);
    }
    let scale = min_edge as f32 / min_dim as f32;
    (
        ((w as f32 * scale).ceil() as u32).max(min_edge),
        ((h as f32 * scale).ceil() as u32).max(min_edge),
    )
}

/// Pad the short side (centered, white) until the aspect ratio is within
/// the processor's limit, so extreme crops — a 1x4096 rule, say — survive
/// `smart_resize`'s ratio bound. Runs before any upscaling: padding the
/// already-upscaled image would multiply the pad into the upscale.
use image::Rgb;

fn pad_to_ratio(image: &RgbImage, max_ratio: f32) -> RgbImage {
    let (w, h) = image.dimensions();
    let (new_w, new_h) = padded_dims(w, h, max_ratio);
    if (new_w, new_h) == (w, h) {
        return image.clone();
    }
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
    let (new_w, new_h) = upscaled_dims(w, h, min_edge);
    if (new_w, new_h) == (w, h) {
        return image.clone();
    }
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
        let cfg = QwenVlImageProcessorConfig {
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
        for (w, h) in [(10u32, 200u32), (200, 10), (31, 31), (5, 400), (400, 5)] {
            let img = RgbImage::from_pixel(w, h, Rgb([120, 140, 160]));
            let inputs = preprocess_image(&img, &cfg, vision, &Device::Cpu, DType::F32)
                .unwrap_or_else(|e| panic!("{w}x{h} failed: {e}"));
            assert!(inputs.num_image_tokens > 0, "{w}x{h} produced no tokens");
        }
    }

    #[test]
    fn dims_arithmetic_matches_pixel_helpers() {
        // The plan path uses these pure functions; they must agree with
        // what the pixel-carrying helpers actually do.
        let cases = [
            (1u32, 400u32),
            (400, 1),
            (10, 200),
            (200, 10),
            (80, 60),
            (5, 400),
        ];
        for (w, h) in cases {
            let img = RgbImage::from_pixel(w, h, Rgb([1, 2, 3]));
            let (pw, ph) = padded_dims(w, h, SMART_RESIZE_MAX_RATIO);
            let padded = pad_to_ratio(&img, SMART_RESIZE_MAX_RATIO);
            assert_eq!((pw, ph), (padded.width(), padded.height()), "pad {w}x{h}");
            let (uw, uh) = upscaled_dims(pw, ph, 32);
            let upscaled = upscale_min_edge(&padded, 32);
            assert_eq!(
                (uw, uh),
                (upscaled.width(), upscaled.height()),
                "scale {w}x{h}"
            );
            assert!(uw >= 32 && uh >= 32, "{w}x{h} below factor");
        }
    }

    #[test]
    fn plan_matches_actual_token_counts() {
        // plan_num_image_tokens mirrors preprocess_image's resize math;
        // any drift between the two shows up here.
        let config = super::super::config::tests::official_config();
        let cfg = QwenVlImageProcessorConfig {
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
        // Small images with the same aspect extremes as the real
        // parser crops: the plan/preprocess agreement is scale-free,
        // and tiny canvases keep the run at millisecond cost.
        for (w, h) in [
            (80u32, 60u32),
            (100, 100),
            (64, 128),
            (10, 200),
            (200, 10),
            (1, 400),
            (400, 1),
            (5, 400),
            (400, 5),
        ] {
            let img = RgbImage::from_pixel(w, h, Rgb([120, 140, 160]));
            let planned = plan_num_image_tokens(&img, &cfg)
                .unwrap_or_else(|e| panic!("plan {w}x{h} failed: {e}"));
            let actual = preprocess_image(&img, &cfg, vision, &Device::Cpu, DType::F32)
                .unwrap_or_else(|e| panic!("preprocess {w}x{h} failed: {e}"))
                .num_image_tokens;
            assert_eq!(
                planned, actual,
                "plan/preprocess drift at {w}x{h}: {planned} vs {actual}"
            );
        }
    }

    #[test]
    fn extreme_crops_pad_before_upscaling() {
        // The codex-reported OOM shape was a 1xN strip: upscaling first
        // multiplied it into tens of millions of pixels. Padding first
        // keeps every intermediate bounded by the ratio limit, and the
        // final grid stays valid. The bound is ratio-driven, so a small
        // strip (1x400) exercises the same math at millisecond cost.
        let config = super::super::config::tests::official_config();
        let cfg = QwenVlImageProcessorConfig {
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
        for (w, h) in [(1u32, 400u32), (400, 1)] {
            let img = RgbImage::from_pixel(w, h, Rgb([120, 140, 160]));
            let padded = pad_to_ratio(&img, SMART_RESIZE_MAX_RATIO);
            let upscaled = upscale_min_edge(&padded, min_edge);
            assert!(upscaled.width() >= min_edge && upscaled.height() >= min_edge);
            let padded_pixels = padded.width() as u64 * padded.height() as u64;
            let final_pixels = upscaled.width() as u64 * upscaled.height() as u64;
            // The codex OOM was ~86M pixels; nothing between the two
            // steps may approach that. (The padded-to-final *ratio*
            // depends on how far below `min_edge` the padded short edge
            // sits, so only the absolute bound is scale-free.)
            assert!(
                final_pixels < 4_000_000,
                "{w}x{h} upscaled to {final_pixels} pixels"
            );
            assert!(
                padded_pixels <= final_pixels,
                "{w}x{h} padded {padded_pixels} vs final {final_pixels}"
            );
            let inputs = preprocess_image(&img, &cfg, vision, &Device::Cpu, DType::F32)
                .unwrap_or_else(|e| panic!("{w}x{h} failed: {e}"));
            assert!(inputs.num_image_tokens > 0, "{w}x{h} produced no tokens");
        }
    }

    /// Matches the official `preprocessor_config.json` (Tencent/WeVisDoc-2B).
    fn processor_config() -> QwenVlImageProcessorConfig {
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
