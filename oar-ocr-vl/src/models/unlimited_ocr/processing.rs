//! Page images and the `<PAGE><image>` prompt layout of Unlimited-OCR-2,
//! matching the reference `_uocr_build_inputs` (the vLLM processor).

use super::infer;
use crate::Error;
use crate::runtime::image::pil_bicubic_resize;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use image::{Rgb, RgbImage, imageops};
use tokenizers::Tokenizer;

/// Global view side: every page is letterboxed into this square.
pub const BASE_SIZE: u32 = 1024;
/// Tile side of the dynamic ("gundam") tiling.
pub const TILE_SIZE: u32 = 640;
/// Tile-count bounds of the dynamic tiling.
pub const MIN_TILES: u32 = 2;
pub const MAX_TILES: u32 = 16;
/// Default instruction; the layout markers are prepended per page.
pub const DEFAULT_PROMPT: &str = "Multi page parsing.";

const PATCH_SIZE: u32 = 16;
const DOWNSAMPLE_RATIO: u32 = 4;
const PAD_GRAY: u8 = 127;

/// How a page is encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnlimitedOcrViewMode {
    /// The reference default: dynamic tiles plus the global view for a
    /// single page ("gundam"), the global view alone for several pages.
    Auto,
    /// Dynamic tiles plus the global view for every page.
    Tiled,
    /// The global view alone for every page ("base").
    Global,
}

/// One preprocessed page.
#[derive(Debug)]
pub struct UnlimitedOcrPage {
    /// Letterboxed global view `(1, 3, 1024, 1024)`.
    pub global: Tensor,
    /// Row-major tiles `(w * h, 3, 640, 640)`, or `None` without tiling.
    pub tiles: Option<Tensor>,
    /// Tile grid `(columns, rows)`; `(1, 1)` without tiling.
    pub grid: (usize, usize),
}

/// Visual queries per side of a view: `ceil((size / patch) / downsample)`.
fn queries(size: u32) -> usize {
    (size / PATCH_SIZE).div_ceil(DOWNSAMPLE_RATIO) as usize
}

impl UnlimitedOcrPage {
    /// Placeholder tokens for this page: global rows (`q + 1` each) plus the
    /// view separator, then the tile rows.
    pub fn num_tokens(&self) -> usize {
        let global = queries(BASE_SIZE);
        let mut tokens = global * (global + 1) + 1;
        if self.tiles.is_some() {
            let tile = queries(TILE_SIZE);
            let (columns, rows) = self.grid;
            tokens += (tile * columns + 1) * tile * rows;
        }
        tokens
    }
}

/// `find_closest_aspect_ratio` over the `min..=max` tile grids, sorted by
/// tile count; later equal-difference grids win when the image is large.
pub fn tile_grid(width: u32, height: u32, max_tiles: u32) -> (usize, usize) {
    let mut ratios: Vec<(u32, u32)> = Vec::new();
    for i in 1..=max_tiles {
        for j in 1..=max_tiles {
            if (MIN_TILES..=max_tiles).contains(&(i * j)) {
                ratios.push((i, j));
            }
        }
    }
    ratios.sort_by_key(|(i, j)| i * j);
    let aspect = f64::from(width) / f64::from(height);
    let area = f64::from(width) * f64::from(height);
    let tile_area = f64::from(TILE_SIZE) * f64::from(TILE_SIZE);
    let mut best = (1u32, 1u32);
    let mut best_diff = f64::INFINITY;
    for (i, j) in ratios {
        let diff = (aspect - f64::from(i) / f64::from(j)).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (i, j);
        } else if diff == best_diff && area > 0.5 * tile_area * f64::from(i * j) {
            best = (i, j);
        }
    }
    (best.0 as usize, best.1 as usize)
}

/// `ImageOps.pad(image, (size, size), color=127)`: a Pillow bicubic
/// `contain` fit, which always resizes, then a centred paste whose offset
/// rounds half to even like Python's `round`.
pub fn letterbox(image: &RgbImage, size: u32) -> Result<RgbImage, Error> {
    let (width, height) = image.dimensions();
    let round = |value: f64| (value.round_ties_even() as u32).max(1);
    let (new_width, new_height) = match width.cmp(&height) {
        std::cmp::Ordering::Equal => (size, size),
        std::cmp::Ordering::Greater => (
            size,
            round(f64::from(height) / f64::from(width) * f64::from(size)),
        ),
        std::cmp::Ordering::Less => (
            round(f64::from(width) / f64::from(height) * f64::from(size)),
            size,
        ),
    };
    let resized = pil_bicubic_resize(image, new_width, new_height)?;
    if (new_width, new_height) == (size, size) {
        return Ok(resized);
    }
    let mut canvas = RgbImage::from_pixel(size, size, Rgb([PAD_GRAY; 3]));
    let offset = |free: u32| (f64::from(free) * 0.5).round_ties_even() as i64;
    if new_width != size {
        imageops::overlay(&mut canvas, &resized, offset(size - new_width), 0);
    } else {
        imageops::overlay(&mut canvas, &resized, 0, offset(size - new_height));
    }
    Ok(canvas)
}

/// `ToTensor` + `Normalize(0.5, 0.5)`, rounded through bfloat16 like the
/// reference, which always hands the encoder bf16 pixels.
fn normalize(images: &[RgbImage], device: &Device, dtype: DType) -> Result<Tensor, Error> {
    let (width, height) = images[0].dimensions();
    let plane = (width * height) as usize;
    let mut data = vec![0f32; images.len() * 3 * plane];
    for (image, out) in images.iter().zip(data.chunks_exact_mut(3 * plane)) {
        for (index, pixel) in image.pixels().enumerate() {
            for channel in 0..3 {
                let value = (f32::from(pixel.0[channel]) / 255.0 - 0.5) / 0.5;
                out[channel * plane + index] = bf16::from_f32(value).to_f32();
            }
        }
    }
    Tensor::from_vec(
        data,
        (images.len(), 3, height as usize, width as usize),
        device,
    )
    .and_then(|t| t.to_dtype(dtype))
    .map_err(infer)
}

/// Encode one page; `tiled` selects the dynamic tiling, which pages no
/// larger than one tile never use.
pub fn preprocess_page(
    image: &RgbImage,
    tiled: bool,
    device: &Device,
    dtype: DType,
) -> Result<UnlimitedOcrPage, Error> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(Error::invalid_input(
            "Unlimited-OCR page image has a zero dimension",
        ));
    }
    let global = normalize(&[letterbox(image, BASE_SIZE)?], device, dtype)?;
    if !tiled || (width <= TILE_SIZE && height <= TILE_SIZE) {
        return Ok(UnlimitedOcrPage {
            global,
            tiles: None,
            grid: (1, 1),
        });
    }
    let (columns, rows) = tile_grid(width, height, MAX_TILES);
    let resized = pil_bicubic_resize(image, TILE_SIZE * columns as u32, TILE_SIZE * rows as u32)?;
    let tiles: Vec<RgbImage> = (0..rows * columns)
        .map(|index| {
            let (row, column) = ((index / columns) as u32, (index % columns) as u32);
            imageops::crop_imm(
                &resized,
                column * TILE_SIZE,
                row * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            )
            .to_image()
        })
        .collect();
    Ok(UnlimitedOcrPage {
        global,
        tiles: Some(normalize(&tiles, device, dtype)?),
        grid: (columns, rows),
    })
}

/// `[BOS] + ("<PAGE>" + image placeholders) per page + instruction`.
pub fn prompt_ids(
    tokenizer: &Tokenizer,
    pages: &[UnlimitedOcrPage],
    instruction: &str,
    bos_id: u32,
    page_token_id: u32,
    image_token_id: u32,
) -> Result<Vec<u32>, Error> {
    let tail = tokenizer
        .encode(instruction, false)
        .map_err(|e| Error::invalid_input(format!("Unlimited-OCR tokenization failed: {e}")))?;
    let mut ids = vec![bos_id];
    for page in pages {
        ids.push(page_token_id);
        ids.extend(std::iter::repeat_n(image_token_id, page.num_tokens()));
    }
    ids.extend_from_slice(tail.get_ids());
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiling_and_token_counts_follow_the_reference() {
        // An A4-like portrait page picks 2 columns x 3 rows.
        assert_eq!(tile_grid(1240, 1754, MAX_TILES), (2, 3));
        // A wide page reaches past jina's nine-tile cap.
        assert_eq!(tile_grid(6000, 1000, MAX_TILES), (6, 1));
        let page = UnlimitedOcrPage {
            global: Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            tiles: Some(Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap()),
            grid: (2, 3),
        };
        assert_eq!(page.num_tokens(), 273 + 21 * 30);
        // A near-square page whose short side rounds to the full size is
        // still resized (Pillow's `contain`), not pasted uncropped.
        let near_square = RgbImage::from_fn(100, 99, |x, y| Rgb([x as u8, y as u8, 7]));
        assert_eq!(
            letterbox(&near_square, 32).unwrap(),
            pil_bicubic_resize(&near_square, 32, 32).unwrap()
        );
    }
}
