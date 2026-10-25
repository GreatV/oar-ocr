//! RGB8 specialization of image 0.25.10's separable sampler (MIT/Apache-2.0).
//! Keep its f32 tap order, normalization and final rounding; only independent
//! output rows run in parallel. In particular, do not replace sums with FMA.

use image::{GenericImageView, RgbImage, imageops::FilterType};
use rayon::prelude::*;

struct Taps {
    start: usize,
    weights: Vec<f32>,
}

fn kernel(filter: FilterType, x: f32) -> f32 {
    match filter {
        FilterType::Nearest => 1.0,
        FilterType::Triangle => {
            if x.abs() < 1.0 {
                1.0 - x.abs()
            } else {
                0.0
            }
        }
        FilterType::CatmullRom => cubic(x, 0.0, 0.5),
        FilterType::Gaussian => {
            let r = 0.5f32;
            ((2.0 * std::f32::consts::PI).sqrt() * r).recip()
                * (-x.powi(2) / (2.0 * r.powi(2))).exp()
        }
        FilterType::Lanczos3 => {
            if x.abs() < 3.0 {
                sinc(x) * sinc(x / 3.0)
            } else {
                0.0
            }
        }
    }
}

fn sinc(x: f32) -> f32 {
    let a = x * std::f32::consts::PI;
    if x == 0.0 { 1.0 } else { a.sin() / a }
}

fn cubic(x: f32, b: f32, c: f32) -> f32 {
    let a = x.abs();
    let k = if a < 1.0 {
        (12.0 - 9.0 * b - 6.0 * c) * a.powi(3)
            + (-18.0 + 12.0 * b + 6.0 * c) * a.powi(2)
            + (6.0 - 2.0 * b)
    } else if a < 2.0 {
        (-b - 6.0 * c) * a.powi(3)
            + (6.0 * b + 30.0 * c) * a.powi(2)
            + (-12.0 * b - 48.0 * c) * a
            + (8.0 * b + 24.0 * c)
    } else {
        0.0
    };
    k / 6.0
}

fn taps(input: u32, output: u32, filter: FilterType) -> Vec<Taps> {
    let support = match filter {
        FilterType::Nearest => 0.0,
        FilterType::Triangle => 1.0,
        FilterType::CatmullRom => 2.0,
        FilterType::Gaussian | FilterType::Lanczos3 => 3.0,
    };
    let ratio = input as f32 / output as f32;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let support = support * sratio;
    (0..output)
        .map(|out| {
            let center = (out as f32 + 0.5) * ratio;
            let start = ((center - support).floor() as i64).clamp(0, i64::from(input) - 1) as u32;
            let end = ((center + support).ceil() as i64)
                .clamp(i64::from(start) + 1, i64::from(input)) as u32;
            let center = center - 0.5;
            let mut sum = 0.0;
            let mut weights = Vec::with_capacity((end - start) as usize);
            for i in start..end {
                let weight = kernel(filter, (i as f32 - center) / sratio);
                weights.push(weight);
                sum += weight;
            }
            for weight in &mut weights {
                *weight /= sum;
            }
            Taps {
                start: start as usize,
                weights,
            }
        })
        .collect()
}

pub(super) fn resize_rgb8(
    image: &RgbImage,
    width: u32,
    height: u32,
    filter: FilterType,
) -> RgbImage {
    if image.width() == 0
        || image.height() == 0
        || width == 0
        || height == 0
        || image.dimensions() == (width, height)
    {
        return image::imageops::resize(image, width, height, filter);
    }
    let source_row = image.width() as usize * 3;
    let vertical = taps(image.height(), height, filter);
    let horizontal = taps(image.width(), width, filter);
    // RGB channels are independent; the generic sampler's fourth channel is
    // unused. Preserve f32 intermediates until the horizontal pass finishes.
    let mut intermediate = vec![0f32; source_row * height as usize];
    intermediate
        .par_chunks_mut(source_row)
        .zip(vertical.par_iter())
        .for_each(|(row, taps)| {
            for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let mut value = [0.0; 3];
                for (i, weight) in taps.weights.iter().enumerate() {
                    let offset = (taps.start + i) * source_row + x * 3;
                    for (channel, value) in value.iter_mut().enumerate() {
                        *value += image.as_raw()[offset + channel] as f32 * weight;
                    }
                }
                pixel.copy_from_slice(&value);
            }
        });
    // This also preserves the source image's color-space metadata.
    let mut out: RgbImage = image.buffer_with_dimensions(width, height);
    out.as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let source = &intermediate[y * source_row..(y + 1) * source_row];
            for (pixel, taps) in row.as_chunks_mut::<3>().0.iter_mut().zip(&horizontal) {
                let mut value = [0.0f32; 3];
                for (i, weight) in taps.weights.iter().enumerate() {
                    let offset = (taps.start + i) * 3;
                    for (channel, value) in value.iter_mut().enumerate() {
                        *value += source[offset + channel] * weight;
                    }
                }
                for (pixel, value) in pixel.iter_mut().zip(value) {
                    *pixel = value.clamp(0.0, 255.0).round() as u8;
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_resize_matches_image_bytes() {
        for (w, h, out_w, out_h) in [
            (13, 17, 31, 5),
            (31, 29, 7, 11),
            (17, 19, 23, 29),
            (1, 17, 7, 3),
            (19, 1, 3, 7),
            (1, 1, 13, 11),
            (19, 23, 19, 23),
            (0, 7, 3, 9),
            (7, 0, 9, 3),
            (9, 7, 0, 3),
            (9, 7, 3, 0),
        ] {
            let input = RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([
                    ((x * 37 + y * 61) % 256) as u8,
                    if (x + y) % 3 == 0 { 255 } else { 0 },
                    ((x * y + 97) % 256) as u8,
                ])
            });
            for filter in [
                FilterType::Nearest,
                FilterType::Triangle,
                FilterType::CatmullRom,
                FilterType::Gaussian,
                FilterType::Lanczos3,
            ] {
                let actual = resize_rgb8(&input, out_w, out_h, filter);
                let expected = image::imageops::resize(&input, out_w, out_h, filter);
                assert_eq!(actual, expected, "{w}x{h} -> {out_w}x{out_h}, {filter:?}");
            }
        }
    }
}
