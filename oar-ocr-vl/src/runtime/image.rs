use crate::api::error::Error;
use image::RgbImage;
use image::imageops::{self, FilterType};
use rayon::prelude::*;

/// Convert an RGB image to a CHW tensor buffer with normalization and parallel processing.
///
/// Output layout: [R0...Rn, G0...Gn, B0...Bn]
pub fn image_to_chw(
    image: &RgbImage,
    mean: &[f32],
    std: &[f32],
    rescale_factor: Option<f32>,
) -> Vec<f32> {
    // RGB layout of the source buffer: 3 interleaved channels per pixel.
    const RGB_CHANNELS: usize = 3;
    const R: usize = 0;
    const G: usize = 1;
    const B: usize = 2;

    let width = image.width() as usize;
    let height = image.height() as usize;
    let num_pixels = width * height;
    let scale = rescale_factor.unwrap_or(1.0);

    let mut output = vec![0.0f32; num_pixels * RGB_CHANNELS];
    let raw_pixels = image.as_raw();

    // Split the output into the three contiguous channel planes. Each plane is
    // a disjoint `&mut [f32]`, so writing them concurrently is safe — no raw
    // pointers or `UnsafeSlice` needed.
    let (r_plane, rest) = output.split_at_mut(num_pixels);
    let (g_plane, b_plane) = rest.split_at_mut(num_pixels);

    // Parallel iteration over pixels: each iteration writes one element in each
    // plane at the same index, and the three planes are non-overlapping.
    // `par_chunks_exact` walks `raw_pixels` in `RGB_CHANNELS`-sized strides so
    // the compiler can elide bounds checks on the source indexing.
    raw_pixels
        .par_chunks_exact(RGB_CHANNELS)
        .zip(r_plane.par_iter_mut())
        .zip(g_plane.par_iter_mut())
        .zip(b_plane.par_iter_mut())
        .for_each(|(((chunk, r_out), g_out), b_out)| {
            let r = chunk[R] as f32 * scale;
            let g = chunk[G] as f32 * scale;
            let b = chunk[B] as f32 * scale;

            *r_out = (r - mean[0]) / std[0];
            *g_out = (g - mean[1]) / std[1];
            *b_out = (b - mean[2]) / std[2];
        });

    output
}

/// Rearrange CHW frame buffers into Qwen2-VL style merge-grouped flat patches.
///
/// This is the shared "patchify" used by GLM-OCR and MinerU2.5 preprocessing:
/// it flattens per-frame CHW pixel buffers into the `pixel_values` layout
/// `(num_patches, patch_dim)` expected by the vision encoder, where
/// `patch_dim = channel * temporal_patch * patch_size * patch_size`.
///
/// Patch ordering follows the spatial-merge convention: iterate
/// `(temporal, h-block, w-block, h-in-block, w-in-block)`, and each patch is
/// laid out as `[channel][temporal][ph][pw]`.
///
/// # Arguments
///
/// * `frames` - `grid_t * temporal_patch` CHW buffers, each of length
///   `channel * height * width`. Frame `t * temporal_patch + tp` is the `tp`-th
///   temporal sample of grid-time `t`.
/// * `channel`, `height`, `width` - dimensions of each CHW frame buffer.
/// * `grid_t`, `grid_h`, `grid_w` - patch grid dimensions. `grid_h` and
///   `grid_w` must be multiples of `merge_size` (callers validate this).
/// * `patch_size`, `merge_size`, `temporal_patch` - patch/merge configuration.
///
/// # Returns
///
/// A flat `Vec<f32>` of length `grid_t * grid_h * grid_w * patch_dim`, ready to
/// be wrapped as a `(num_patches, patch_dim)` tensor.
#[allow(clippy::too_many_arguments)]
pub fn patchify_merge_grouped(
    frames: &[&[f32]],
    channel: usize,
    height: usize,
    width: usize,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    patch_size: usize,
    merge_size: usize,
    temporal_patch: usize,
) -> Vec<f32> {
    debug_assert_eq!(
        frames.len(),
        grid_t * temporal_patch,
        "patchify_merge_grouped: expected {} frames (grid_t * temporal_patch), got {}",
        grid_t * temporal_patch,
        frames.len()
    );
    debug_assert!(
        frames.iter().all(|f| f.len() == channel * height * width),
        "patchify_merge_grouped: every frame must have length channel * height * width"
    );

    let patch_dim = channel * temporal_patch * patch_size * patch_size;
    let num_patches = grid_t * grid_h * grid_w;
    let mut flat = Vec::with_capacity(num_patches * patch_dim);

    let channel_stride = height * width;
    let grid_h_blocks = grid_h / merge_size;
    let grid_w_blocks = grid_w / merge_size;

    for t in 0..grid_t {
        for hb in 0..grid_h_blocks {
            for wb in 0..grid_w_blocks {
                for hm in 0..merge_size {
                    for wm in 0..merge_size {
                        let patch_row = hb * merge_size + hm;
                        let patch_col = wb * merge_size + wm;
                        for c in 0..channel {
                            for tp in 0..temporal_patch {
                                let frame = frames[t * temporal_patch + tp];
                                let base = c * channel_stride;
                                for ph in 0..patch_size {
                                    let h_idx = patch_row * patch_size + ph;
                                    let row_base = base + h_idx * width;
                                    for pw in 0..patch_size {
                                        let w_idx = patch_col * patch_size + pw;
                                        flat.push(frame[row_base + w_idx]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    flat
}

/// Convert PIL/Transformers integer resampling constants to `image::FilterType`.
///
/// Mappings:
/// 0 => Nearest
/// 1 => Lanczos3
/// 2 => Triangle (Bilinear)
/// 3 => CatmullRom (Bicubic)
/// 4 => Box
/// 5 => Hamming
pub fn pil_resample_to_filter_type(resample: u32) -> Option<FilterType> {
    match resample {
        0 => Some(FilterType::Nearest),
        1 => Some(FilterType::Lanczos3),
        2 => Some(FilterType::Triangle),
        3 => Some(FilterType::CatmullRom),
        4 => Some(FilterType::Triangle), // Box map to Triangle as approx or specific if available
        5 => Some(FilterType::CatmullRom), // Hamming map to CatmullRom as approx
        _ => None,
    }
}

/// Fractional bits of PIL's 8-bit-per-channel resampling coefficients
/// (`PRECISION_BITS = 32 - 8 - 2` in `Resample.c`).
const PIL_RESAMPLE_PRECISION_BITS: u32 = 22;

/// A Pillow resampling filter: kernel support (in source pixels at unit
/// scale) and the kernel itself.
#[derive(Clone, Copy)]
struct PilFilter {
    support: f64,
    kernel: fn(f64) -> f64,
}

const PIL_BICUBIC: PilFilter = PilFilter {
    support: 2.0,
    kernel: pil_bicubic_kernel,
};

const PIL_BILINEAR: PilFilter = PilFilter {
    support: 1.0,
    kernel: |t| (1.0 - t.abs()).max(0.0),
};

const PIL_LANCZOS: PilFilter = PilFilter {
    support: 3.0,
    kernel: pil_lanczos_kernel,
};

/// PIL's `bicubic` filter kernel: Keys a = -0.5 with support 2.
fn pil_bicubic_kernel(t: f64) -> f64 {
    const A: f64 = -0.5;
    let t = t.abs();
    if t < 1.0 {
        ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
        (((t - 5.0) * t + 8.0) * t - 4.0) * A
    } else {
        0.0
    }
}

fn pil_sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

/// PIL's `lanczos` filter kernel: sinc windowed by sinc over `[-3, 3)`.
fn pil_lanczos_kernel(t: f64) -> f64 {
    if (-3.0..3.0).contains(&t) {
        pil_sinc(t) * pil_sinc(t / 3.0)
    } else {
        0.0
    }
}

/// Fixed-point tap weights of one resampling axis, per output sample.
struct PilResampleAxis {
    /// First source index per output sample.
    bases: Vec<usize>,
    /// Quantized weights for output sample `i` (PIL's clamped window).
    weights: Vec<Vec<i32>>,
}

impl PilResampleAxis {
    /// Reproduce PIL's coefficient construction for one axis: the tap window
    /// spans `int(center - support + 0.5)` .. `int(center + support + 0.5)`
    /// clamped to the source, weights are the filter kernel normalized to
    /// unit sum and quantized with C `(int)` truncation at
    /// [`PIL_RESAMPLE_PRECISION_BITS`] fractional bits.
    fn new(in_dim: u32, out_dim: u32, filter: PilFilter) -> Self {
        let scale = in_dim as f64 / out_dim as f64;
        let filterscale = scale.max(1.0);
        let support = filter.support * filterscale;
        let ksize = (support.ceil() as usize) * 2 + 1;
        let ss = 1.0 / filterscale;
        let quantize = |weight: f64| -> i32 {
            let scaled = weight * (1u64 << PIL_RESAMPLE_PRECISION_BITS) as f64;
            // C `(int)(value ± 0.5)` truncates toward zero.
            let rounded = if scaled >= 0.0 {
                scaled + 0.5
            } else {
                scaled - 0.5
            };
            rounded as i64 as i32
        };
        let mut bases = Vec::with_capacity(out_dim as usize);
        let mut weights = Vec::with_capacity(out_dim as usize);
        for index in 0..out_dim as i64 {
            let center = (index as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).trunc() as i64).max(0) as usize;
            let xmax = ((center + support + 0.5).trunc() as i64).min(in_dim as i64) as usize;
            let count = (xmax - xmin).min(ksize);
            let mut row = vec![0f64; count];
            for (tap, weight) in row.iter_mut().enumerate() {
                let distance = (xmin + tap) as f64 - center + 0.5;
                *weight = (filter.kernel)(distance * ss);
            }
            let sum: f64 = row.iter().sum();
            if sum != 0.0 {
                for weight in &mut row {
                    *weight /= sum;
                }
            }
            bases.push(xmin);
            weights.push(row.iter().map(|w| quantize(*w)).collect());
        }
        Self { bases, weights }
    }
}

/// Shift a fixed-point accumulator back to 8 bits with PIL's rounding bias
/// and clamp, mirroring `clip8`.
fn pil_fixed_to_u8(value: i64) -> u8 {
    let rounded =
        (value + (1i64 << (PIL_RESAMPLE_PRECISION_BITS - 1))) >> PIL_RESAMPLE_PRECISION_BITS;
    rounded.clamp(0, 255) as u8
}

/// Resize for pipelines whose upstream is the transformers Qwen-VL-family
/// processor (Qwen2VLImageProcessor and its Fast/Qwen3VL/Glm46V variants):
/// those resample with Pillow's bicubic — the family default and the
/// meaning of an explicit `resample=3` — so bicubic goes through the
/// PIL-exact path. An explicit non-bicubic `resample` keeps the `image`
/// crate approximation, as before.
pub fn qwen_vl_resize(
    image: &RgbImage,
    width: u32,
    height: u32,
    resample: Option<u32>,
) -> Result<RgbImage, Error> {
    match resample {
        None | Some(3) => pil_bicubic_resize(image, width, height),
        Some(other) => {
            let filter = pil_resample_to_filter_type(other).unwrap_or(FilterType::CatmullRom);
            Ok(imageops::resize(image, width, height, filter))
        }
    }
}

/// PIL-exact bicubic resize of an RGB image, reproducing Pillow's
/// `Resample.c` 8-bit pipeline: horizontal pass first with the fixed-point
/// coefficients and a rounded uint8 intermediate, then the vertical pass.
/// The `image` crate's `CatmullRom` is a close but different kernel
/// discipline — it differs from PIL's output by up to ~24/255 at sharp
/// edges, enough to shift vision-model logits — while this variant matches
/// Pillow bit-for-bit.
pub fn pil_bicubic_resize(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage, Error> {
    pil_resize(image, width, height, PIL_BICUBIC)
}

/// Pillow-exact bilinear RGB resize, including the rounded uint8 intermediate.
pub(crate) fn pil_bilinear_resize(
    image: &RgbImage,
    width: u32,
    height: u32,
) -> Result<RgbImage, Error> {
    pil_resize(image, width, height, PIL_BILINEAR)
}

/// PIL-exact `Image.LANCZOS` resize of an RGB image; see
/// [`pil_bicubic_resize`].
pub(crate) fn pil_lanczos_resize(
    image: &RgbImage,
    width: u32,
    height: u32,
) -> Result<RgbImage, Error> {
    pil_resize(image, width, height, PIL_LANCZOS)
}

fn pil_resize(
    image: &RgbImage,
    width: u32,
    height: u32,
    filter: PilFilter,
) -> Result<RgbImage, Error> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidInput {
            message: format!("pil_resize: invalid target {width}x{height}"),
        });
    }
    let (in_w, in_h) = image.dimensions();
    if width == in_w && height == in_h {
        return Ok(image.clone());
    }
    // Horizontal pass into a rounded uint8 intermediate, like PIL.
    let mid = if width != in_w {
        let axis = PilResampleAxis::new(in_w, width, filter);
        let src = image.as_raw();
        let src_stride = in_w as usize * 3;
        let out_stride = width as usize * 3;
        let mut buffer = vec![0u8; out_stride * in_h as usize];
        buffer
            .par_chunks_exact_mut(out_stride)
            .enumerate()
            .for_each(|(y, out_row)| {
                let src_row = &src[y * src_stride..(y + 1) * src_stride];
                for x in 0..width as usize {
                    let base = axis.bases[x];
                    let weights = &axis.weights[x];
                    let mut acc = [0i64; 3];
                    for (tap, weight) in weights.iter().enumerate() {
                        let weight = *weight as i64;
                        let pixel = &src_row[(base + tap) * 3..(base + tap) * 3 + 3];
                        for channel in 0..3 {
                            acc[channel] += weight * pixel[channel] as i64;
                        }
                    }
                    for channel in 0..3 {
                        out_row[x * 3 + channel] = pil_fixed_to_u8(acc[channel]);
                    }
                }
            });
        RgbImage::from_raw(width, in_h, buffer).ok_or_else(|| Error::InvalidInput {
            message: "pil_resize: intermediate buffer size mismatch".to_string(),
        })?
    } else {
        image.clone()
    };
    // Vertical pass over the intermediate.
    if height != in_h {
        let axis = PilResampleAxis::new(in_h, height, filter);
        let mid_raw = mid.as_raw();
        let stride = width as usize * 3;
        let mut buffer = vec![0u8; stride * height as usize];
        buffer
            .par_chunks_exact_mut(stride)
            .enumerate()
            .for_each(|(y, out_row)| {
                let base = axis.bases[y];
                let weights = &axis.weights[y];
                let mut acc = vec![0i64; stride];
                for (tap, weight) in weights.iter().enumerate() {
                    let weight = *weight as i64;
                    let src_row = &mid_raw[(base + tap) * stride..(base + tap + 1) * stride];
                    for (accumulator, sample) in acc.iter_mut().zip(src_row) {
                        *accumulator += weight * *sample as i64;
                    }
                }
                for (sample, accumulator) in out_row.iter_mut().zip(acc) {
                    *sample = pil_fixed_to_u8(accumulator);
                }
            });
        RgbImage::from_raw(width, height, buffer).ok_or_else(|| Error::InvalidInput {
            message: "pil_resize: output buffer size mismatch".to_string(),
        })
    } else {
        Ok(mid)
    }
}

/// Round up a value to the next multiple of a factor.
pub fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }
    value.div_ceil(multiple) * multiple
}

/// Smart resize calculating new dimensions based on pixel constraints and factor alignment.
///
/// Mirrors the Qwen2-VL-family `smart_resize` in transformers, including
/// Python's `round` (halves round to even).
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), Error> {
    if factor == 0 {
        return Err(Error::InvalidInput {
            message: "smart_resize: factor must be > 0".to_string(),
        });
    }

    let height = height as f64;
    let width = width as f64;
    let factor_f = factor as f64;

    let max_dim = height.max(width);
    let min_dim = height.min(width);
    if min_dim > 0.0 && (max_dim / min_dim) > 200.0 {
        return Err(Error::InvalidInput {
            message: format!(
                "smart_resize: absolute aspect ratio must be <= 200, got {:.3}",
                max_dim / min_dim
            ),
        });
    }

    let mut h_bar = (height / factor_f).round_ties_even() * factor_f;
    let mut w_bar = (width / factor_f).round_ties_even() * factor_f;

    let area = h_bar * w_bar;
    if area > max_pixels as f64 {
        let beta = ((height * width) / max_pixels as f64).sqrt();
        h_bar = ((height / beta) / factor_f).floor() * factor_f;
        w_bar = ((width / beta) / factor_f).floor() * factor_f;
        if h_bar < factor_f {
            h_bar = factor_f;
        }
        if w_bar < factor_f {
            w_bar = factor_f;
        }
        // After scaling down, ensure we don't violate min_pixels due to quantization
        if (h_bar * w_bar) < min_pixels as f64 {
            return Err(Error::InvalidInput {
                message: format!(
                    "smart_resize: cannot satisfy both min_pixels={} and max_pixels={} constraints after quantization",
                    min_pixels, max_pixels
                ),
            });
        }
    } else if area < min_pixels as f64 {
        let beta = (min_pixels as f64 / (height * width)).sqrt();
        h_bar = ((height * beta) / factor_f).ceil() * factor_f;
        w_bar = ((width * beta) / factor_f).ceil() * factor_f;
        // Ensure minimum dimensions are at least factor_f
        if h_bar < factor_f {
            h_bar = factor_f;
        }
        if w_bar < factor_f {
            w_bar = factor_f;
        }
        // After scaling up, ensure we don't violate max_pixels due to quantization
        if (h_bar * w_bar) > max_pixels as f64 {
            return Err(Error::InvalidInput {
                message: format!(
                    "smart_resize: cannot satisfy both min_pixels={} and max_pixels={} constraints after quantization",
                    min_pixels, max_pixels
                ),
            });
        }
    }

    Ok((h_bar as u32, w_bar as u32))
}

/// Clamp dimensions to a maximum image size while maintaining aspect ratio and factor divisibility.
///
/// Used by HunyuanOCR.
pub fn clamp_to_max_image_size(
    height: u32,
    width: u32,
    factor: u32,
    max_image_size: usize,
) -> Result<(u32, u32), Error> {
    if factor == 0 {
        return Err(Error::InvalidInput {
            message: "clamp_to_max_image_size: factor must be > 0".to_string(),
        });
    }
    let max_image_size = u32::try_from(max_image_size).map_err(|_| Error::Config {
        message: format!("max_image_size too large: {max_image_size}"),
    })?;
    if max_image_size == 0 {
        return Err(Error::Config {
            message: "max_image_size must be > 0".to_string(),
        });
    }
    if max_image_size < factor {
        return Err(Error::Config {
            message: format!("max_image_size {max_image_size} must be >= factor={factor}"),
        });
    }

    let max_dim = height.max(width);
    if max_dim <= max_image_size {
        return Ok((height, width));
    }

    let scale = max_image_size as f64 / max_dim as f64;
    let factor_f = factor as f64;

    let mut h = (((height as f64) * scale) / factor_f).floor() * factor_f;
    let mut w = (((width as f64) * scale) / factor_f).floor() * factor_f;

    // Ensure non-zero dims and factor divisibility.
    if h < factor_f {
        h = factor_f;
    }
    if w < factor_f {
        w = factor_f;
    }

    let h = h as u32;
    let w = w as u32;

    Ok((h, w))
}

/// Resize an image for MinerU2.5 inference.
///
/// Handles two cases:
/// 1. If the aspect ratio exceeds `max_aspect_ratio`, the image is padded with
///    white to bring it within bounds.
/// 2. If the minimum edge is below `min_edge`, the image is scaled up.
pub fn resize_for_mineru(image: &RgbImage, min_edge: u32, max_aspect_ratio: f32) -> RgbImage {
    use image::{Rgb, imageops};
    use std::borrow::Cow;

    let (mut w, mut h) = image.dimensions();
    let mut out = Cow::Borrowed(image);

    // Handle extreme aspect ratios by padding
    let ratio = (w.max(h) as f32) / (w.min(h).max(1) as f32);
    if ratio > max_aspect_ratio {
        let (new_w, new_h) = if w > h {
            (w, (w as f32 / max_aspect_ratio).ceil() as u32)
        } else {
            ((h as f32 / max_aspect_ratio).ceil() as u32, h)
        };
        let mut canvas = RgbImage::from_pixel(new_w, new_h, Rgb([255, 255, 255]));
        let x = ((new_w - w) / 2) as i64;
        let y = ((new_h - h) / 2) as i64;
        imageops::overlay(&mut canvas, &*out, x, y);
        out = Cow::Owned(canvas);
        (w, h) = out.dimensions();
    }

    // Scale up if minimum edge is too small
    let min_dim = w.min(h);
    if min_dim < min_edge {
        let scale = min_edge as f32 / min_dim as f32;
        let new_w = (w as f32 * scale).ceil() as u32;
        let new_h = (h as f32 * scale).ceil() as u32;
        out = Cow::Owned(imageops::resize(
            &*out,
            new_w,
            new_h,
            imageops::FilterType::CatmullRom,
        ));
    }

    out.into_owned()
}

/// Loads an image from disk and converts it to RGB.
///
/// The format comes from the file's contents, not its name, so extensionless
/// and mislabelled files load. `image::open` would pick the decoder from the
/// path suffix and reject both.
pub fn load_image(path: impl AsRef<std::path::Path>) -> Result<RgbImage, crate::Error> {
    let file = std::io::BufReader::new(std::fs::File::open(path.as_ref())?);
    let decoded = image::ImageReader::new(file)
        .with_guessed_format()?
        .decode()?;
    Ok(decoded.to_rgb8())
}

/// Decodes an image from memory and converts it to RGB.
pub fn load_image_from_memory(bytes: &[u8]) -> Result<RgbImage, crate::Error> {
    Ok(image::load_from_memory(bytes)?.to_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exactness against Pillow is the entire point of the PIL resize:
    /// the golden pairs below were produced with Pillow 12.1.0
    /// (`Image.resize(..., Image.BICUBIC)`) from a deterministic 12x9 RGB
    /// pattern, covering a downscale (7x5) and an upscale (13x10) of both
    /// axes.
    #[test]
    fn pil_bicubic_resize_matches_pillow() {
        let src = [
            0, 0, 0, 23, 11, 3, 46, 22, 6, 69, 33, 9, 92, 44, 12, 115, 55, 15, 138, 66, 18, 161,
            77, 21, 184, 88, 24, 207, 99, 27, 230, 110, 30, 253, 121, 33, 5, 29, 61, 28, 40, 64,
            51, 51, 67, 74, 62, 70, 97, 73, 73, 120, 84, 76, 143, 95, 79, 166, 106, 82, 189, 117,
            85, 212, 128, 88, 235, 139, 91, 2, 150, 94, 10, 58, 122, 33, 69, 125, 56, 80, 128, 79,
            91, 131, 102, 102, 134, 125, 113, 137, 148, 124, 140, 171, 135, 143, 194, 146, 146,
            217, 157, 149, 240, 168, 152, 7, 179, 155, 15, 87, 183, 38, 98, 186, 61, 109, 189, 84,
            120, 192, 107, 131, 195, 130, 142, 198, 153, 153, 201, 176, 164, 204, 199, 175, 207,
            222, 186, 210, 245, 197, 213, 12, 208, 216, 20, 116, 244, 43, 127, 247, 66, 138, 250,
            89, 149, 253, 112, 160, 0, 135, 171, 3, 158, 182, 6, 181, 193, 9, 204, 204, 12, 227,
            215, 15, 250, 226, 18, 17, 237, 21, 25, 145, 49, 48, 156, 52, 71, 167, 55, 94, 178, 58,
            117, 189, 61, 140, 200, 64, 163, 211, 67, 186, 222, 70, 209, 233, 73, 232, 244, 76,
            255, 255, 79, 22, 10, 82, 30, 174, 110, 53, 185, 113, 76, 196, 116, 99, 207, 119, 122,
            218, 122, 145, 229, 125, 168, 240, 128, 191, 251, 131, 214, 6, 134, 237, 17, 137, 4,
            28, 140, 27, 39, 143, 35, 203, 171, 58, 214, 174, 81, 225, 177, 104, 236, 180, 127,
            247, 183, 150, 2, 186, 173, 13, 189, 196, 24, 192, 219, 35, 195, 242, 46, 198, 9, 57,
            201, 32, 68, 204, 40, 232, 232, 63, 243, 235, 86, 254, 238, 109, 9, 241, 132, 20, 244,
            155, 31, 247, 178, 42, 250, 201, 53, 253, 224, 64, 0, 247, 75, 3, 14, 86, 6, 37, 97, 9,
        ];
        let expected = [
            11, 17, 27, 49, 35, 32, 89, 55, 37, 129, 74, 43, 168, 92, 48, 213, 112, 53, 178, 130,
            58, 20, 66, 140, 58, 84, 145, 98, 104, 150, 138, 123, 156, 177, 141, 161, 227, 161,
            166, 104, 184, 171, 29, 120, 188, 67, 138, 196, 107, 159, 102, 147, 182, 59, 186, 202,
            67, 239, 229, 72, 124, 206, 77, 38, 173, 94, 76, 195, 99, 116, 216, 104, 156, 199, 110,
            199, 167, 118, 207, 91, 130, 49, 62, 135, 47, 228, 207, 85, 211, 212, 125, 106, 217,
            165, 28, 226, 210, 40, 183, 195, 51, 95, 16, 76, 102,
        ];
        let src = RgbImage::from_raw(12, 9, src.to_vec()).expect("source buffer");
        let out = pil_bicubic_resize(&src, 7, 5).expect("resize");
        assert_eq!(out.dimensions(), (7, 5));
        assert_eq!(out.as_raw(), &expected[..], "must match Pillow bit-for-bit");
        // Upscale golden from the same source, also Pillow 12.1.0.
        let expected_up = [
            0, 0, 0, 20, 9, 2, 42, 19, 4, 63, 29, 7, 84, 39, 10, 105, 49, 13, 127, 60, 16, 148, 70,
            18, 169, 80, 21, 190, 90, 24, 211, 100, 27, 233, 110, 29, 255, 120, 32, 4, 24, 51, 24,
            34, 54, 46, 44, 56, 67, 54, 59, 88, 64, 62, 109, 74, 65, 131, 85, 68, 152, 95, 70, 173,
            105, 73, 194, 115, 76, 219, 125, 79, 219, 135, 81, 29, 145, 84, 9, 51, 107, 29, 61,
            110, 51, 71, 112, 72, 81, 115, 93, 91, 118, 114, 101, 121, 136, 112, 124, 157, 122,
            126, 178, 132, 129, 199, 142, 132, 224, 152, 135, 220, 162, 137, 0, 172, 140, 13, 77,
            162, 33, 87, 165, 55, 97, 167, 76, 107, 169, 97, 117, 186, 118, 127, 195, 140, 138,
            198, 161, 148, 200, 182, 158, 203, 203, 168, 206, 228, 178, 209, 225, 188, 211, 6, 198,
            214, 18, 103, 234, 38, 113, 237, 60, 123, 239, 81, 133, 244, 102, 143, 134, 123, 153,
            85, 145, 164, 90, 166, 174, 92, 187, 184, 95, 208, 194, 98, 233, 204, 101, 230, 215,
            103, 11, 242, 106, 22, 129, 159, 42, 139, 162, 64, 149, 164, 85, 159, 169, 106, 169,
            59, 127, 179, 11, 149, 190, 15, 170, 199, 17, 191, 215, 20, 212, 235, 23, 240, 246, 26,
            248, 244, 28, 15, 136, 31, 27, 155, 51, 47, 165, 54, 69, 175, 56, 90, 185, 59, 111,
            195, 75, 132, 211, 84, 154, 227, 87, 175, 240, 89, 196, 218, 92, 219, 162, 95, 229,
            178, 98, 163, 170, 100, 21, 0, 103, 31, 181, 125, 51, 191, 128, 73, 201, 130, 94, 216,
            133, 115, 230, 136, 136, 203, 139, 158, 188, 142, 179, 209, 144, 200, 141, 149, 229, 0,
            156, 202, 8, 159, 0, 20, 161, 29, 47, 164, 36, 208, 181, 56, 218, 184, 78, 228, 186,
            99, 215, 189, 120, 229, 192, 141, 86, 195, 163, 0, 198, 184, 9, 201, 205, 23, 194, 233,
            43, 177, 210, 53, 180, 1, 63, 182, 33, 73, 185, 40, 233, 233, 60, 243, 236, 82, 255,
            238, 103, 59, 241, 124, 0, 244, 145, 24, 247, 167, 38, 250, 188, 48, 255, 209, 58, 174,
            237, 68, 0, 214, 78, 0, 5, 88, 1, 37, 98, 4,
        ];
        let up = pil_bicubic_resize(&src, 13, 10).expect("upscale");
        assert_eq!(up.dimensions(), (13, 10));
        assert_eq!(
            up.as_raw(),
            &expected_up[..],
            "upscale must match Pillow too"
        );
        let identity = pil_bicubic_resize(&src, 12, 9).expect("identity");
        assert_eq!(identity.as_raw(), src.as_raw());
    }

    /// PaddleOCR-VL's spotting upscale: Pillow 12.1.0
    /// `Image.resize((10, 8), Image.LANCZOS)` of the 5x4 pattern below.
    #[test]
    fn pil_lanczos_resize_matches_pillow() {
        let src: Vec<u8> = (0..4u32)
            .flat_map(|y| (0..5u32).flat_map(move |x| (0..3u32).map(move |c| (x, y, c))))
            .map(|(x, y, c)| ((x * 67 + y * 131 + c * 29) * 53 % 256) as u8)
            .collect();
        let expected = [
            3, 3, 0, 39, 40, 51, 157, 158, 193, 240, 241, 255, 209, 210, 222, 173, 174, 169, 163,
            164, 162, 143, 144, 146, 126, 127, 128, 118, 119, 120, 0, 0, 7, 52, 53, 39, 184, 185,
            137, 255, 255, 209, 221, 222, 206, 177, 178, 188, 168, 169, 174, 151, 152, 151, 134,
            135, 135, 125, 126, 127, 0, 0, 32, 72, 73, 20, 211, 212, 44, 255, 255, 94, 235, 236,
            179, 188, 189, 220, 182, 183, 199, 168, 169, 164, 151, 152, 151, 142, 143, 144, 15, 16,
            57, 69, 70, 13, 160, 161, 0, 221, 221, 27, 255, 255, 193, 234, 234, 255, 214, 215, 232,
            183, 184, 178, 167, 168, 167, 161, 162, 163, 57, 58, 71, 45, 46, 30, 47, 48, 0, 119,
            120, 62, 245, 246, 227, 255, 255, 255, 243, 244, 250, 191, 192, 191, 179, 180, 180,
            176, 177, 178, 88, 89, 84, 51, 52, 61, 12, 13, 40, 47, 48, 75, 151, 152, 162, 212, 212,
            208, 238, 239, 238, 216, 217, 219, 203, 204, 205, 191, 192, 193, 96, 97, 95, 79, 80,
            85, 56, 57, 70, 32, 33, 47, 43, 44, 49, 109, 110, 109, 214, 215, 215, 246, 246, 247,
            228, 229, 231, 201, 202, 203, 96, 97, 100, 97, 98, 97, 93, 94, 88, 36, 36, 29, 0, 0, 0,
            46, 47, 50, 197, 198, 199, 255, 255, 255, 241, 242, 243, 206, 207, 208,
        ];
        let src = RgbImage::from_raw(5, 4, src).expect("source buffer");
        let out = pil_lanczos_resize(&src, 10, 8).expect("resize");
        assert_eq!(out.as_raw(), &expected[..], "must match Pillow bit-for-bit");
    }

    #[test]
    fn pil_bilinear_resize_matches_pillow() {
        // Pillow Image.resize(..., Image.Resampling.BILINEAR), RGB uint8.
        let src = RgbImage::from_fn(5, 4, |x, y| {
            image::Rgb([
                ((x * 71 + y * 13) % 256) as u8,
                ((x * 31 + y * 83) % 256) as u8,
                ((x * 109 + y * 47) % 256) as u8,
            ])
        });
        let down = pil_bilinear_resize(&src, 3, 2).unwrap();
        assert_eq!(
            down.as_raw(),
            &[
                39, 72, 81, 151, 113, 113, 116, 134, 146, 60, 156, 154, 172, 118, 126, 137, 81,
                115,
            ]
        );
        let up = pil_bilinear_resize(&src, 7, 6).unwrap();
        assert_eq!(
            up.as_raw(),
            &[
                0, 0, 0, 41, 18, 62, 91, 40, 140, 142, 62, 218, 193, 84, 113, 134, 106, 118, 28,
                124, 180, 7, 42, 24, 48, 60, 86, 98, 82, 127, 149, 104, 114, 200, 126, 100, 141,
                148, 142, 35, 166, 204, 15, 97, 55, 56, 115, 117, 106, 137, 122, 157, 159, 17, 208,
                150, 95, 149, 160, 154, 43, 178, 192, 24, 152, 86, 65, 170, 148, 115, 192, 153,
                166, 214, 48, 217, 84, 126, 158, 45, 113, 52, 63, 53, 33, 208, 118, 74, 152, 180,
                124, 120, 185, 175, 142, 80, 226, 72, 158, 167, 58, 126, 61, 76, 42, 39, 249, 141,
                80, 120, 203, 130, 33, 208, 181, 55, 103, 232, 77, 181, 173, 99, 149, 67, 117, 65,
            ]
        );
    }

    /// Callers pass paths straight from a CLI, where an extensionless or
    /// mislabelled file is ordinary. The format must come from the bytes.
    #[test]
    fn loads_images_whose_name_does_not_match_their_format() {
        let dir = std::env::temp_dir().join("oar_vl_load_image_by_content");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let mut png = Vec::new();
        RgbImage::from_pixel(3, 2, ::image::Rgb([1, 2, 3]))
            .write_with_encoder(::image::codecs::png::PngEncoder::new(&mut png))
            .expect("encode png");

        for name in ["no_extension", "actually_a_png.jpg"] {
            let path = dir.join(name);
            std::fs::write(&path, &png).expect("write fixture");
            let loaded = load_image(&path).expect("load by content");
            assert_eq!(loaded.dimensions(), (3, 2));
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn test_patchify_merge_grouped_ordering() {
        // channel=2, height=width=4, patch_size=2, merge_size=2 => one 2x2
        // merge block of 2x2-pixel patches, temporal=1, grid_t=1.
        let channel = 2usize;
        let (height, width) = (4usize, 4usize);
        let (patch_size, merge_size, temporal) = (2usize, 2usize, 1usize);
        let (grid_t, grid_h, grid_w) = (1usize, 2usize, 2usize);

        // frame[c*16 + y*4 + x] = c*100 + y*10 + x
        let mut frame = vec![0.0f32; channel * height * width];
        for c in 0..channel {
            for y in 0..height {
                for x in 0..width {
                    frame[c * height * width + y * width + x] = (c * 100 + y * 10 + x) as f32;
                }
            }
        }
        let frames: Vec<&[f32]> = vec![frame.as_slice()];

        let flat = patchify_merge_grouped(
            &frames, channel, height, width, grid_t, grid_h, grid_w, patch_size, merge_size,
            temporal,
        );

        // Expected order: (hm, wm) over the merge block, then channel, then the
        // 2x2 pixels of each patch laid out row-major.
        let expected: Vec<f32> = vec![
            0., 1., 10., 11., 100., 101., 110., 111., // hm0,wm0: c0 then c1
            2., 3., 12., 13., 102., 103., 112., 113., // hm0,wm1
            20., 21., 30., 31., 120., 121., 130., 131., // hm1,wm0
            22., 23., 32., 33., 122., 123., 132., 133., // hm1,wm1
        ];
        assert_eq!(flat, expected);
        // num_patches * patch_dim = (1*2*2) * (2*1*2*2) = 4 * 8 = 32
        assert_eq!(flat.len(), 32);
    }

    #[test]
    fn test_patchify_merge_grouped_merge1_is_raster() {
        // merge_size=1 => patches are emitted in raster (gh, gw) order, each
        // patch being a contiguous channel-major block.
        let channel = 1usize;
        let (height, width) = (2usize, 4usize);
        let (patch_size, merge_size, temporal) = (2usize, 1usize, 1usize);
        let (grid_t, grid_h, grid_w) = (1usize, 1usize, 2usize);

        let mut frame = vec![0.0f32; channel * height * width];
        for y in 0..height {
            for x in 0..width {
                frame[y * width + x] = (y * width + x) as f32;
            }
        }
        let frames: Vec<&[f32]> = vec![frame.as_slice()];

        let flat = patchify_merge_grouped(
            &frames, channel, height, width, grid_t, grid_h, grid_w, patch_size, merge_size,
            temporal,
        );

        // Patch (0,0): x in {0,1}; patch (0,1): x in {2,3}
        let expected: Vec<f32> = vec![
            0., 1., 4., 5., // first patch
            2., 3., 6., 7., // second patch
        ];
        assert_eq!(flat, expected);
    }

    #[test]
    fn test_smart_resize_matches_the_python_reference() -> Result<(), Error> {
        // 70 / 28 = 2.5 rounds to 2 like Python's `round`.
        assert_eq!(smart_resize(70, 210, 28, 3_136, 12_845_056)?, (56, 224));
        assert_eq!(smart_resize(350, 630, 28, 3_136, 12_845_056)?, (336, 616));
        Ok(())
    }

    #[test]
    fn test_clamp_to_max_image_size_keeps_factor_divisible() -> Result<(), Error> {
        let factor = 32;
        let (h, w) = clamp_to_max_image_size(3008, 512, factor, 2048)?;
        assert!(h <= 2048);
        assert!(w <= 2048);
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
        Ok(())
    }
}
