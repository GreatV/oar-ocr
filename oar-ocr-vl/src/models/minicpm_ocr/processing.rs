//! Image slicing and prompt layout matching `MiniCPMOcrImageProcessor` and
//! `MiniCPMOcrProcessor` in Transformers, with the default
//! `small_image_upsize="always"` and `downsample_mode="4x"` settings.

use super::config::MiniCpmOcrImageProcessorConfig;
use super::infer;
use crate::Error;
use crate::runtime::image::{image_to_chw, pil_bicubic_resize};
use candle_core::{DType, Device, Tensor};
use image::RgbImage;
use image::imageops;

/// Side of the patch block one visual token covers in `4x` mode.
pub(super) const MERGE_SIZE: usize = 2;

/// One slice (the overview or a grid cell) ready for the vision tower.
#[derive(Debug)]
pub struct MiniCpmOcrSlice {
    /// Flattened patches `(height * width, channels * patch * patch)` in
    /// raster order, each laid out as `[channel][row][column]`.
    pub pixel_values: Tensor,
    /// Patch grid `(height, width)`.
    pub grid_hw: (usize, usize),
}

impl MiniCpmOcrSlice {
    /// Visual tokens this slice contributes after the 2x2 merge.
    pub fn num_tokens(&self) -> usize {
        self.grid_hw.0 * self.grid_hw.1 / (MERGE_SIZE * MERGE_SIZE)
    }
}

/// Preprocessed image: the resized overview first, then the grid cells in
/// row-major order.
#[derive(Debug)]
pub struct MiniCpmOcrImageInputs {
    pub slices: Vec<MiniCpmOcrSlice>,
    /// Slice grid `(rows, columns)`, or `None` for an unsliced image.
    pub grid: Option<(usize, usize)>,
}

impl MiniCpmOcrImageInputs {
    /// The text that replaces the image placeholder, including the
    /// `<image_id>0</image_id>` prefix the processor adds by default.
    pub fn placeholder(&self, use_image_id: bool) -> String {
        let pads = |count: usize| "<|image_pad|>".repeat(count);
        let mut text = String::new();
        if use_image_id {
            text.push_str("<image_id>0</image_id>");
        }
        text.push_str("<image>");
        text.push_str(&pads(self.slices[0].num_tokens()));
        text.push_str("</image>");
        if let Some((_, columns)) = self.grid {
            let rows: Vec<String> = self.slices[1..]
                .chunks(columns)
                .map(|row| {
                    row.iter()
                        .map(|slice| format!("<slice>{}</slice>", pads(slice.num_tokens())))
                        .collect()
                })
                .collect();
            text.push_str(&rows.join("\n"));
        }
        text
    }

    pub fn num_tokens(&self) -> usize {
        self.slices.iter().map(MiniCpmOcrSlice::num_tokens).sum()
    }
}

/// `max(round(length / divisor) * divisor, divisor)` with Python's
/// ties-to-even `round`.
fn ensure_divide(length: f64, divisor: u32) -> u32 {
    let divisor_f = divisor as f64;
    (((length / divisor_f).round_ties_even() * divisor_f) as u32).max(divisor)
}

/// Patch-aligned size for one slice; a slice larger than
/// `scale_resolution` squared, or any slice when `allow_upscale`, is first
/// rescaled to that area while keeping its aspect ratio.
fn find_best_resize(
    height: f64,
    width: f64,
    scale_resolution: u32,
    patch_size: u32,
    allow_upscale: bool,
) -> (u32, u32) {
    let (mut height, mut width) = (height, width);
    let resolution = scale_resolution as f64;
    if height * width > resolution * resolution || allow_upscale {
        let aspect_ratio = width / height;
        height = (resolution / aspect_ratio.sqrt()).trunc();
        width = (height * aspect_ratio).trunc();
    }
    let divisor = patch_size * MERGE_SIZE as u32;
    (
        ensure_divide(height, divisor),
        ensure_divide(width, divisor),
    )
}

fn get_refine_size(
    height: u32,
    width: u32,
    grid: (usize, usize),
    scale_resolution: u32,
    patch_size: u32,
) -> (u32, u32) {
    let (rows, columns) = (grid.0 as u32, grid.1 as u32);
    let refine_width = ensure_divide(width as f64, columns);
    let refine_height = ensure_divide(height as f64, rows);
    let (best_height, best_width) = find_best_resize(
        refine_height as f64 / rows as f64,
        refine_width as f64 / columns as f64,
        scale_resolution,
        patch_size,
        true,
    );
    (best_height * rows, best_width * columns)
}

/// Slice grid `(rows, columns)` whose aspect ratio best matches the image,
/// or `None` when the image fits in one slice.
pub fn sliced_grid(
    height: u32,
    width: u32,
    max_slice_nums: usize,
    scale_resolution: u32,
) -> Option<(usize, usize)> {
    let log_ratio = (width as f64 / height as f64).ln();
    let ratio = width as f64 * height as f64 / (scale_resolution as f64).powi(2);
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best = (1, 1);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple - 1, multiple, multiple + 1] {
        if num_slices == 1 || num_slices > max_slice_nums {
            continue;
        }
        for rows in (1..=num_slices).filter(|rows| num_slices.is_multiple_of(*rows)) {
            let columns = num_slices / rows;
            let error = (log_ratio - (columns as f64 / rows as f64).ln()).abs();
            if error < min_error || (error == min_error && rows > best.0) {
                best = (rows, columns);
                min_error = error;
            }
        }
    }
    Some(best)
}

fn to_slice(
    image: &RgbImage,
    config: &MiniCpmOcrImageProcessorConfig,
    device: &Device,
    dtype: DType,
) -> Result<MiniCpmOcrSlice, Error> {
    let patch = config.patch_size as usize;
    let (width, height) = (image.width() as usize, image.height() as usize);
    let (grid_h, grid_w) = (height / patch, width / patch);
    let chw = image_to_chw(
        image,
        &config.image_mean,
        &config.image_std,
        Some(1.0 / 255.0),
    );
    let patch_dim = 3 * patch * patch;
    let mut patches = vec![0f32; grid_h * grid_w * patch_dim];
    for (index, out) in patches.chunks_exact_mut(patch_dim).enumerate() {
        let (row, column) = (index / grid_w, index % grid_w);
        for channel in 0..3 {
            for y in 0..patch {
                let source = channel * height * width + (row * patch + y) * width + column * patch;
                let target = (channel * patch + y) * patch;
                out[target..target + patch].copy_from_slice(&chw[source..source + patch]);
            }
        }
    }
    let pixel_values = Tensor::from_vec(patches, (grid_h * grid_w, patch_dim), device)
        .and_then(|t| t.to_dtype(dtype))
        .map_err(infer)?;
    Ok(MiniCpmOcrSlice {
        pixel_values,
        grid_hw: (grid_h, grid_w),
    })
}

/// Resize and slice an image into the overview plus up to
/// `max_slice_nums` grid cells.
pub fn preprocess(
    image: &RgbImage,
    config: &MiniCpmOcrImageProcessorConfig,
    max_slice_nums: usize,
    device: &Device,
    dtype: DType,
) -> Result<MiniCpmOcrImageInputs, Error> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(Error::invalid_input(
            "MiniCPM-OCR image has a zero dimension",
        ));
    }
    if max_slice_nums == 0 {
        return Err(Error::invalid_input(
            "MiniCPM-OCR max_slice_nums must be positive",
        ));
    }
    let grid = config
        .slice_mode
        .then(|| sliced_grid(height, width, max_slice_nums, config.scale_resolution))
        .flatten();
    let (source_h, source_w) = find_best_resize(
        height as f64,
        width as f64,
        config.scale_resolution,
        config.patch_size,
        grid.is_none(),
    );
    let source = pil_bicubic_resize(image, source_w, source_h)?;
    let mut slices = vec![to_slice(&source, config, device, dtype)?];
    if let Some((rows, columns)) = grid {
        let (refine_h, refine_w) = get_refine_size(
            height,
            width,
            (rows, columns),
            config.scale_resolution,
            config.patch_size,
        );
        let refined = pil_bicubic_resize(image, refine_w, refine_h)?;
        let (cell_h, cell_w) = (refine_h / rows as u32, refine_w / columns as u32);
        for row in 0..rows as u32 {
            for column in 0..columns as u32 {
                let cell =
                    imageops::crop_imm(&refined, column * cell_w, row * cell_h, cell_w, cell_h)
                        .to_image();
                slices.push(to_slice(&cell, config, device, dtype)?);
            }
        }
    }
    Ok(MiniCpmOcrImageInputs { slices, grid })
}

/// Chat-template prompt with the image placeholder before the instruction,
/// the thinking block closed (`enable_thinking=False`).
pub fn build_prompt(placeholder: &str, instruction: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        format!("{placeholder}\n{instruction}").trim()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicing_follows_the_reference_grid_and_sizes() {
        // A 1000x1414 portrait page: 8 slices requested; 3x3 matches its aspect best.
        assert_eq!(sliced_grid(1414, 1000, 9, 448), Some((3, 3)));
        assert_eq!(find_best_resize(1414.0, 1000.0, 448, 14, false), (532, 364));
        assert_eq!(get_refine_size(1414, 1000, (3, 3), 448, 14), (1596, 1092));
        // Small crops are always upscaled to the slice budget.
        assert_eq!(sliced_grid(40, 300, 9, 448), None);
        assert_eq!(find_best_resize(40.0, 300.0, 448, 14, true), (168, 1232));
    }
}
