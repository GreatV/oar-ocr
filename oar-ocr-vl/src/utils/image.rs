use image::RgbImage;
use image::imageops::FilterType;
use oar_ocr_core::core::OCRError;
use rayon::prelude::*;

struct UnsafeSlice<T> {
    slice: *mut [T],
}

unsafe impl<T: Send> Send for UnsafeSlice<T> {}
unsafe impl<T: Sync> Sync for UnsafeSlice<T> {}

impl<T> UnsafeSlice<T> {
    fn new(slice: &mut [T]) -> Self {
        let slice = slice as *mut [T];
        Self { slice }
    }

    /// SAFETY: The caller must ensure that `idx` is in bounds and that
    /// no other thread is accessing the same index.
    unsafe fn write(&self, idx: usize, value: T) {
        // SAFETY: dereferencing the raw pointer is unsafe, but guaranteed by
        // the type invariant and caller contract.
        unsafe {
            let slice = &mut *self.slice;
            slice[idx] = value;
        }
    }
}

/// Convert an RGB image to a CHW tensor buffer with normalization and parallel processing.
///
/// Output layout: [R0...Rn, G0...Gn, B0...Bn]
pub fn image_to_chw(
    image: &RgbImage,
    mean: &[f32],
    std: &[f32],
    rescale_factor: Option<f32>,
) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let num_pixels = width * height;
    let scale = rescale_factor.unwrap_or(1.0);

    #[allow(clippy::uninit_vec)]
    let mut output = Vec::with_capacity(num_pixels * 3);
    // SAFETY: We will overwrite all elements in the parallel loop.
    // The loop covers 0..num_pixels, writing to i, i+num_pixels, i+2*num_pixels.
    // Total written: 3 * num_pixels.
    #[allow(clippy::uninit_vec)]
    unsafe {
        output.set_len(num_pixels * 3)
    };

    let output_slice = UnsafeSlice::new(&mut output);
    let raw_pixels = image.as_raw();

    // Parallel iteration over pixels
    (0..num_pixels).into_par_iter().for_each(|i| {
        let r = raw_pixels[3 * i] as f32 * scale;
        let g = raw_pixels[3 * i + 1] as f32 * scale;
        let b = raw_pixels[3 * i + 2] as f32 * scale;

        // Normalize
        let r_norm = (r - mean[0]) / std[0];
        let g_norm = (g - mean[1]) / std[1];
        let b_norm = (b - mean[2]) / std[2];

        // Write to CHW planes
        unsafe {
            output_slice.write(i, r_norm);
            output_slice.write(num_pixels + i, g_norm);
            output_slice.write(2 * num_pixels + i, b_norm);
        }
    });

    output
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

/// Round up a value to the next multiple of a factor.
pub fn round_up_to_multiple(value: u32, multiple: u32) -> u32 {
    if multiple == 0 {
        return value;
    }
    value.div_ceil(multiple) * multiple
}

/// Smart resize calculating new dimensions based on pixel constraints and factor alignment.
///
/// Used by PaddleOCR-VL and HunyuanOCR.
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), OCRError> {
    if factor == 0 {
        return Err(OCRError::InvalidInput {
            message: "smart_resize: factor must be > 0".to_string(),
        });
    }

    let height = height as f64;
    let width = width as f64;
    let factor_f = factor as f64;

    let max_dim = height.max(width);
    let min_dim = height.min(width);
    if min_dim > 0.0 && (max_dim / min_dim) > 200.0 {
        return Err(OCRError::InvalidInput {
            message: format!(
                "smart_resize: absolute aspect ratio must be <= 200, got {:.3}",
                max_dim / min_dim
            ),
        });
    }

    let mut h_bar = (height / factor_f).round() * factor_f;
    let mut w_bar = (width / factor_f).round() * factor_f;

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
            return Err(OCRError::InvalidInput {
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
            return Err(OCRError::InvalidInput {
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
) -> Result<(u32, u32), OCRError> {
    if factor == 0 {
        return Err(OCRError::InvalidInput {
            message: "clamp_to_max_image_size: factor must be > 0".to_string(),
        });
    }
    let max_image_size = u32::try_from(max_image_size).map_err(|_| OCRError::ConfigError {
        message: format!("max_image_size too large: {max_image_size}"),
    })?;
    if max_image_size == 0 {
        return Err(OCRError::ConfigError {
            message: "max_image_size must be > 0".to_string(),
        });
    }
    if max_image_size < factor {
        return Err(OCRError::ConfigError {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smart_resize_factor_divisibility() -> Result<(), OCRError> {
        let (h, w) = smart_resize(100, 200, 28, 147_384, 2_822_400)?;
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        Ok(())
    }

    #[test]
    fn test_clamp_to_max_image_size_keeps_factor_divisible() -> Result<(), OCRError> {
        let factor = 32;
        let (h, w) = clamp_to_max_image_size(3008, 512, factor, 2048)?;
        assert!(h <= 2048);
        assert!(w <= 2048);
        assert_eq!(h % factor, 0);
        assert_eq!(w % factor, 0);
        Ok(())
    }
}
