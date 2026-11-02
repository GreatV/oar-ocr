use image::RgbImage;
use rayon::prelude::*;

/// OpenCV's generic uint8 `INTER_CUBIC` path, without antialiasing.
/// Optimized OpenCV SIMD/IPP paths can round some pixels differently.
pub(super) fn cv2_cubic_resize(image: &RgbImage, width: u32, height: u32) -> Option<RgbImage> {
    if image.width() == 0 || image.height() == 0 || width == 0 || height == 0 {
        return None;
    }
    if image.dimensions() == (width, height) {
        return Some(image.clone());
    }
    let horizontal = cv2_cubic_taps(image.width(), width);
    let vertical = cv2_cubic_taps(image.height(), height);
    let stride = width as usize * 3;
    let mut rows = vec![0i32; stride * image.height() as usize];
    rows.par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, tap) in horizontal.iter().enumerate() {
                for channel in 0..3 {
                    row[x * 3 + channel] = tap
                        .indices
                        .iter()
                        .zip(tap.weights)
                        .map(|(&sx, weight)| {
                            image.get_pixel(sx as u32, y as u32)[channel] as i32 * weight
                        })
                        .sum();
                }
            }
        });
    let mut output = RgbImage::new(width, height);
    output
        .as_mut()
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(y, row)| {
            let tap = &vertical[y];
            for (index, value) in row.iter_mut().enumerate() {
                let sum: i32 = tap
                    .indices
                    .iter()
                    .zip(tap.weights)
                    .map(|(&sy, weight)| rows[sy * stride + index] * weight)
                    .sum();
                *value = ((sum + (1 << 21)) >> 22).clamp(0, 255) as u8;
            }
        });
    Some(output)
}

struct Cv2CubicTap {
    indices: [usize; 4],
    weights: [i32; 4],
}

fn cv2_cubic_taps(input: u32, output: u32) -> Vec<Cv2CubicTap> {
    // Adapted from OpenCV 4.13.0 imgproc/src/resize.cpp (Apache-2.0):
    // interpolateCubic, HResizeCubic and FixedPtCast. Coefficients are f32,
    // quantized to 11 bits with cvRound's ties-to-even; pixels round at 22 bits.
    let scale = 1. / (output as f64 / input as f64);
    (0..output)
        .map(|position| {
            let center = ((position as f64 + 0.5) * scale - 0.5) as f32;
            let base = center.floor() as i64;
            let x = center - base as f32;
            let a = -0.75f32;
            let c0 = ((a * (x + 1.) - 5. * a) * (x + 1.) + 8. * a) * (x + 1.) - 4. * a;
            let c1 = ((a + 2.) * x - (a + 3.)) * x * x + 1.;
            let c2 = ((a + 2.) * (1. - x) - (a + 3.)) * (1. - x) * (1. - x) + 1.;
            Cv2CubicTap {
                indices: std::array::from_fn(|k| {
                    (base + k as i64 - 1).clamp(0, input as i64 - 1) as usize
                }),
                weights: [c0, c1, c2, 1. - c0 - c1 - c2]
                    .map(|weight| (weight * 2048.).round_ties_even() as i32),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_opencv_generic_cubic_pixels() {
        // cv2 4.13.0 INTER_CUBIC, setUseOptimized(False): borders, clipping,
        // coefficient quantization and downsampling without an antialias pass.
        let sample = |width, height| {
            RgbImage::from_fn(width, height, |x, y| {
                image::Rgb(std::array::from_fn(|channel| {
                    let i = ((y * width + x) * 3) as usize + channel;
                    ((i * 37 + i / 11 * 19) % 256) as u8
                }))
            })
        };
        let up = cv2_cubic_resize(&sample(4, 3), 7, 5).unwrap();
        assert_eq!(
            up.as_raw(),
            &[
                0, 4, 68, 19, 74, 127, 103, 153, 191, 198, 68, 101, 223, 0, 21, 138, 61, 112, 67,
                130, 190, 83, 120, 27, 81, 128, 87, 90, 134, 164, 168, 109, 156, 202, 86, 123, 105,
                97, 141, 25, 108, 158, 221, 255, 14, 151, 188, 56, 65, 102, 124, 118, 153, 214,
                170, 208, 245, 84, 134, 171, 14, 72, 109, 208, 245, 153, 129, 165, 125, 33, 69, 99,
                67, 128, 177, 146, 193, 230, 162, 109, 145, 165, 38, 73, 168, 205, 255, 97, 134,
                187, 11, 48, 87, 36, 99, 134, 133, 154, 191, 229, 80, 117, 255, 18, 55,
            ]
        );
        let down = cv2_cubic_resize(&sample(12, 9), 3, 2).unwrap();
        assert_eq!(
            down.as_raw(),
            &[
                139, 142, 101, 91, 141, 177, 195, 55, 129, 115, 124, 161, 164, 90, 112, 117, 154,
                74,
            ]
        );
    }
}
