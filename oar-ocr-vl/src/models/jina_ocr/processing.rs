//! Image preprocessing for jina-ocr-v1 (DeepseekOCRProcessor semantics).

use crate::error::Error;
use crate::runtime::checkpoint::load_json_config;
use crate::runtime::errors::candle_to_ocr_inference;
use candle_core::{DType, Device, Tensor};
use image::{Rgb, imageops::FilterType};
use serde::Deserialize;
use std::path::Path;

const PATCH_SIZE: u32 = 16;
const DOWNSAMPLE_RATIO: u32 = 4;
/// Side of the global view (`DS_OCR_BASE_SIZE`).
pub const BASE_SIZE: u32 = 1024;
/// Side of each local tile (`DS_OCR_TILE_SIZE`).
pub const TILE_SIZE: u32 = 640;
/// `dynamic_preprocess` tile-count bounds.
const MIN_TILES: u32 = 2;
const MAX_TILES: u32 = 9;
/// Pad color: the normalization mean mapped back to 0..255.
const PAD_GRAY: u8 = 127;

/// The official OCR instruction (`DEFAULT_OCR_PROMPT`).
pub const DEFAULT_OCR_PROMPT: &str = "Transcribe the provided document image into a clean Markdown format, preserving the natural reading order.";

#[derive(Debug, Clone, Deserialize)]
pub struct JinaOcrProcessorConfig {
    #[serde(default = "default_true")]
    pub crop_mode: bool,
    #[serde(default = "default_base_size")]
    pub base_size: u32,
    #[serde(default = "default_tile_size")]
    pub image_size: u32,
}

fn default_true() -> bool {
    true
}

fn default_base_size() -> u32 {
    BASE_SIZE
}

fn default_tile_size() -> u32 {
    TILE_SIZE
}

impl JinaOcrProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let cfg: Self = load_json_config(path, "JinaOCR", "processor_config.json")?;
        if cfg.base_size != BASE_SIZE {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR base_size must be {BASE_SIZE} (the checkpoint's global-view resolution), got {}",
                    cfg.base_size
                ),
            });
        }
        if cfg.image_size != TILE_SIZE {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR image_size must be {TILE_SIZE}, got {}",
                    cfg.image_size
                ),
            });
        }
        Ok(cfg)
    }
}

/// Preprocessed inputs for one page image.
#[derive(Debug, Clone)]
pub struct JinaOcrImageInputs {
    /// Global view, `(1, 3, base_size, base_size)`.
    pub global_view: Tensor,
    /// Local tiles, `(t, 3, image_size, image_size)`; empty when the image
    /// fits in one tile (or crop mode is off).
    pub tiles: Tensor,
    /// Tile grid as `(width_tiles, height_tiles)`.
    pub tile_grid: (usize, usize),
    /// Total image-placeholder tokens: global + local layout.
    #[allow(dead_code)]
    pub num_image_tokens: usize,
    /// Queries per tile side after the 4× downsample.
    pub tile_queries: usize,
    /// Queries per global-view side after the 4× downsample.
    pub global_queries: usize,
}

/// `ImageOps.pad`: aspect-preserving resize to fit `size × size`, then pad
/// with the normalization mean color.
pub fn pad_to_square(image: &image::RgbImage, size: u32) -> image::RgbImage {
    let (width, height) = (image.width(), image.height());
    let scale = f64::from(size) / f64::from(width.max(height));
    let new_w = ((f64::from(width) * scale).round() as u32).max(1).min(size);
    let new_h = ((f64::from(height) * scale).round() as u32)
        .max(1)
        .min(size);
    let resized = image::imageops::resize(image, new_w, new_h, FilterType::CatmullRom);
    let mut canvas = image::RgbImage::from_pixel(size, size, Rgb([PAD_GRAY, PAD_GRAY, PAD_GRAY]));
    let x = size.saturating_sub(new_w) / 2;
    let y = size.saturating_sub(new_h) / 2;
    image::imageops::overlay(&mut canvas, &resized, x as i64, y as i64);
    canvas
}

/// Aspect-ratio tile grid whose layout best matches the image
/// (`find_closest_aspect_ratio` + `dynamic_preprocess`, 2..=9 tiles).
pub fn dynamic_tile_grid(width: u32, height: u32) -> (usize, usize) {
    let mut ratios: Vec<(u32, u32)> = Vec::new();
    for n in MIN_TILES..=MAX_TILES {
        for i in 1..=n {
            for j in 1..=n {
                if i * j <= MAX_TILES && i * j >= MIN_TILES {
                    ratios.push((i, j));
                }
            }
        }
    }
    ratios.sort_by_key(|(i, j)| i * j);
    let aspect = f64::from(width) / f64::from(height);
    let area = f64::from(width) * f64::from(height);
    let mut best = (1u32, 1u32);
    let mut best_diff = f64::INFINITY;
    for &(i, j) in &ratios {
        let diff = (aspect - f64::from(i) / f64::from(j)).abs();
        if diff < best_diff
            || (diff == best_diff
                && area
                    > 0.5
                        * f64::from(TILE_SIZE)
                        * f64::from(TILE_SIZE)
                        * f64::from(i)
                        * f64::from(j))
        {
            best_diff = diff;
            best = (i, j);
        }
    }
    (best.0 as usize, best.1 as usize)
}

/// Queries per tile side: `ceil((size / patch) / downsample)`.
pub fn compute_queries(size: u32) -> usize {
    ((size / PATCH_SIZE).div_ceil(DOWNSAMPLE_RATIO)) as usize
}

fn normalize_to_tensor(image: &image::RgbImage, device: &Device) -> Result<Tensor, Error> {
    // ToTensor (x/255) then Normalize(0.5, 0.5): x*2 - 1, CHW.
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut data = vec![0f32; 3 * height * width];
    for (index, pixel) in image.pixels().enumerate() {
        let [r, g, b] = pixel.0;
        data[index] = (f32::from(r) / 255.0 - 0.5) / 0.5;
        data[height * width + index] = (f32::from(g) / 255.0 - 0.5) / 0.5;
        data[2 * height * width + index] = (f32::from(b) / 255.0 - 0.5) / 0.5;
    }
    Tensor::from_vec(data, (1, 3, height, width), device)
        .map_err(|e| candle_to_ocr_inference("JinaOCR", "normalize image", e))
}

/// Build the flat image-placeholder token layout for one image.
pub(crate) fn image_tokens(image_token_id: u32, inputs: &JinaOcrImageInputs) -> Vec<u32> {
    std::iter::repeat_n(image_token_id, image_token_count(inputs)).collect()
}

/// Total placeholder count: global rows (`q + 1` per row, plus a view
/// separator) followed by the tile rows (`q * width_tiles + 1` per row).
pub(crate) fn image_token_count(inputs: &JinaOcrImageInputs) -> usize {
    let mut tokens = inputs.global_queries * (inputs.global_queries + 1) + 1;
    let (width_tiles, height_tiles) = inputs.tile_grid;
    if width_tiles > 1 || height_tiles > 1 {
        let row = inputs.tile_queries * width_tiles + 1;
        tokens += row * inputs.tile_queries * height_tiles;
    }
    tokens
}

/// Resize/normalize/patchify one page image.
pub fn preprocess_image(
    image: &image::RgbImage,
    processor: &JinaOcrProcessorConfig,
    device: &Device,
    dtype: DType,
) -> Result<JinaOcrImageInputs, Error> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::InvalidInput {
            message: "JinaOCR input image must be non-empty".to_string(),
        });
    }
    let global_queries = compute_queries(processor.base_size);
    let tile_queries = compute_queries(processor.image_size);

    let mut tile_grid = (1usize, 1usize);
    let mut tiles: Vec<Tensor> = Vec::new();
    if processor.crop_mode
        && (image.width() > processor.image_size || image.height() > processor.image_size)
    {
        tile_grid = dynamic_tile_grid(image.width(), image.height());
        let target_w = processor.image_size * tile_grid.0 as u32;
        let target_h = processor.image_size * tile_grid.1 as u32;
        let resized = image::imageops::resize(image, target_w, target_h, FilterType::CatmullRom);
        for row in 0..tile_grid.1 {
            for col in 0..tile_grid.0 {
                let tile = image::imageops::crop_imm(
                    &resized,
                    col as u32 * processor.image_size,
                    row as u32 * processor.image_size,
                    processor.image_size,
                    processor.image_size,
                );
                tiles.push(normalize_to_tensor(&tile.to_image(), device)?);
            }
        }
    }
    let global_view = normalize_to_tensor(&pad_to_square(image, processor.base_size), device)?;

    let tiles_tensor = if tiles.is_empty() {
        Tensor::zeros(
            (
                0usize,
                3,
                processor.image_size as usize,
                processor.image_size as usize,
            ),
            DType::F32,
            device,
        )?
    } else {
        let refs: Vec<&Tensor> = tiles.iter().collect();
        Tensor::cat(&refs, 0)?
    };
    let inputs = JinaOcrImageInputs {
        global_view: global_view.to_dtype(dtype)?,
        tiles: tiles_tensor.to_dtype(dtype)?,
        tile_grid,
        num_image_tokens: 0,
        tile_queries,
        global_queries,
    };
    let num_image_tokens = image_token_count(&inputs);
    Ok(JinaOcrImageInputs {
        num_image_tokens,
        ..inputs
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn queries_match_the_reference_formula() {
        assert_eq!(compute_queries(1024), 16);
        assert_eq!(compute_queries(640), 10);
    }

    #[test]
    fn small_images_skip_tiling() -> Result<(), Box<dyn std::error::Error>> {
        let image = RgbImage::from_pixel(512, 512, Rgb([255, 255, 255]));
        let processor = JinaOcrProcessorConfig {
            crop_mode: true,
            base_size: BASE_SIZE,
            image_size: TILE_SIZE,
        };
        let inputs = preprocess_image(&image, &processor, &Device::Cpu, DType::F32)?;
        assert_eq!(inputs.tile_grid, (1, 1));
        assert_eq!(inputs.tiles.dims(), &[0, 3, 640, 640]);
        // 16*17 + 1 = 273 global tokens only.
        assert_eq!(inputs.num_image_tokens, 273);
        Ok(())
    }

    #[test]
    fn tall_images_tile_with_a_global_view() -> Result<(), Box<dyn std::error::Error>> {
        // 640x1280 = 1x2 grid.
        let image = RgbImage::from_pixel(640, 1280, Rgb([0, 0, 0]));
        let processor = JinaOcrProcessorConfig {
            crop_mode: true,
            base_size: BASE_SIZE,
            image_size: TILE_SIZE,
        };
        let inputs = preprocess_image(&image, &processor, &Device::Cpu, DType::F32)?;
        assert_eq!(inputs.tile_grid, (1, 2));
        assert_eq!(inputs.tiles.dims(), &[2, 3, 640, 640]);
        let row = 10 + 1;
        assert_eq!(inputs.num_image_tokens, 273 + row * 10 * 2);
        Ok(())
    }

    #[test]
    fn pad_to_square_centers_the_image() {
        let image = RgbImage::from_pixel(100, 50, Rgb([255, 255, 255]));
        let padded = pad_to_square(&image, 64);
        assert_eq!((padded.width(), padded.height()), (64, 64));
        // Center rows contain the resized white image; corners are gray pad.
        assert_eq!(padded.get_pixel(0, 0).0, [127, 127, 127]);
        assert_eq!(padded.get_pixel(32, 32).0, [255, 255, 255]);
    }

    #[test]
    fn tile_grid_prefers_the_closest_aspect_ratio() {
        assert_eq!(dynamic_tile_grid(640, 1280), (1, 2));
        assert_eq!(dynamic_tile_grid(1280, 640), (2, 1));
        // Only called for images exceeding one tile; a near-square page picks
        // the smallest grid covering it.
        assert_eq!(dynamic_tile_grid(700, 700), (2, 2));
        assert_eq!(dynamic_tile_grid(1920, 640), (3, 1));
    }
}
