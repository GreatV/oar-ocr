//! The OpenCV operations Indra OCR's reference pipeline uses, on grayscale
//! planes: the gray conversion of `imread`, `resize` with `INTER_LINEAR` and `INTER_AREA`,
//! `warpAffine` rotations, quarter turns, and 4-connected components.
//!
//! The models were trained on crops made by OpenCV, so these reproduce its
//! arithmetic (fixed-point linear coefficients, area tables, round half to
//! even) rather than approximating it with the `image` crate's filters.

use image::{GrayImage, RgbImage};
use rayon::prelude::*;

/// RGB to gray as the reference reads a color PNG (`cv2.imread` with
/// `IMREAD_GRAYSCALE`, which lets libpng convert): 0.299 / 0.587 / 0.114 in
/// 15-bit fixed point, truncated, with gray pixels passed through.
pub(crate) fn rgb_to_gray(image: &RgbImage) -> GrayImage {
    let (w, h) = image.dimensions();
    let data = image
        .as_raw()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[r, g, b]| {
            if r == g && r == b {
                r
            } else {
                ((r as u32 * 9797 + g as u32 * 19234 + b as u32 * 3737) >> 15) as u8
            }
        })
        .collect();
    GrayImage::from_raw(w, h, data).expect("buffer matches dimensions")
}

pub(crate) use oar_ocr_core::processors::raster::{Interp, resize, rotation_matrix};
fn sat_u8(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// `cv2.invertAffineTransform`.
fn invert_affine(m: &[f64; 6]) -> [f64; 6] {
    let d = m[0] * m[4] - m[1] * m[3];
    let d = if d != 0.0 { 1.0 / d } else { 0.0 };
    let (a11, a22, a12, a21) = (m[4] * d, m[0] * d, -m[1] * d, -m[3] * d);
    [
        a11,
        a12,
        -a11 * m[2] - a12 * m[5],
        a21,
        a22,
        -a21 * m[2] - a22 * m[5],
    ]
}

/// A plane element [`warp`] samples.
trait Pixel: Copy + Default + Send + Sync {
    fn get(self) -> f32;
    fn put(v: f32) -> Self;
}

impl Pixel for u8 {
    fn get(self) -> f32 {
        self as f32
    }
    fn put(v: f32) -> Self {
        sat_u8(v)
    }
}

/// Bilinear `cv2.warpAffine` of a plane onto a same-size plane, with
/// `BORDER_REPLICATE` (`replicate`) or a constant zero border. Rows are
/// computed in parallel.
fn warp<T: Pixel>(src: &[T], w: usize, h: usize, m: &[f64; 6], replicate: bool) -> Vec<T> {
    let im = invert_affine(m).map(|v| v as f32);
    let mut out = vec![T::default(); w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        warp_row(src, w, h, &im, y, replicate, |x, value| row[x] = value);
    });
    out
}

// Both sinks consume the same rounded pixels: coordinate arithmetic,
// interpolation order, border handling and saturation stay in one place.
fn warp_row<T: Pixel>(
    src: &[T],
    w: usize,
    h: usize,
    im: &[f32; 6],
    y: usize,
    replicate: bool,
    mut put: impl FnMut(usize, T),
) {
    let at = |x: i64, y: i64| -> f32 {
        if replicate {
            T::get(src[y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize])
        } else if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            0.0
        } else {
            T::get(src[y as usize * w + x as usize])
        }
    };
    let yf = y as f32;
    let (bx, by) = (yf * im[1] + im[2], yf * im[4] + im[5]);
    for x in 0..w {
        // OpenCV 5's float-coordinate warp: a fused multiply-add per
        // coordinate and lerp-form interpolation.
        let xf = x as f32;
        let sx = xf.mul_add(im[0], bx);
        let sy = xf.mul_add(im[3], by);
        let (x0, y0) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - x0, sy - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let (p00, p01, p10, p11) = if x0 >= 0 && y0 >= 0 && x0 + 1 < w as i64 && y0 + 1 < h as i64 {
            let i = y0 as usize * w + x0 as usize;
            (
                T::get(src[i]),
                T::get(src[i + 1]),
                T::get(src[i + w]),
                T::get(src[i + w + 1]),
            )
        } else {
            (
                at(x0, y0),
                at(x0 + 1, y0),
                at(x0, y0 + 1),
                at(x0 + 1, y0 + 1),
            )
        };
        let top = p00 + fx * (p01 - p00);
        let bottom = p10 + fx * (p11 - p10);
        put(x, T::put(top + fy * (bottom - top)));
    }
}

/// Integer row sums of the zero-bordered rotation, without storing its pixels.
pub(super) fn rotated_row_sums(src: &GrayImage, degrees: f64) -> Vec<i64> {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let m = rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, degrees);
    let im = invert_affine(&m).map(|v| v as f32);
    let mut rows = vec![0i64; h];
    rows.par_iter_mut().enumerate().for_each(|(y, sum)| {
        warp_row(src.as_raw(), w, h, &im, y, false, |_, value| {
            *sum += value as i64;
        });
    });
    rows
}

/// An 8-bit plane rotated by `degrees` counter-clockwise about its center,
/// zero outside (`cv2.warpAffine` defaults).
#[cfg(test)]
pub(crate) fn rotate_u8_zero(src: &GrayImage, degrees: f64) -> GrayImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let m = rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, degrees);
    let data = warp(src.as_raw(), w, h, &m, false);
    GrayImage::from_raw(w as u32, h as u32, data).expect("buffer matches dimensions")
}

/// An 8-bit page rotated by `degrees` counter-clockwise about its center,
/// edges replicated.
pub(crate) fn rotate_u8(src: &GrayImage, degrees: f64) -> GrayImage {
    let (w, h) = (src.width() as usize, src.height() as usize);
    let m = rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, degrees);
    let data = warp(src.as_raw(), w, h, &m, true);
    GrayImage::from_raw(w as u32, h as u32, data).expect("buffer matches dimensions")
}

/// `np.rot90(img, -k)`: `k` quarter turns clockwise.
pub(crate) fn rot90_cw(src: &GrayImage, k: u32) -> GrayImage {
    match k % 4 {
        1 => image::imageops::rotate90(src),
        2 => image::imageops::rotate180(src),
        3 => image::imageops::rotate270(src),
        _ => src.clone(),
    }
}

/// A 4-connected component of a binary mask.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Component {
    pub left: usize,
    pub right: usize,
    pub area: usize,
}

/// `cv2.connectedComponentsWithStats(mask, connectivity=4)`: per-pixel labels
/// (0 = background, components numbered 1.. in raster order of their first
/// pixel) and the stats of component `i` at index `i - 1`.
pub(crate) fn connected_components(
    mask: &[bool],
    w: usize,
    h: usize,
) -> (Vec<u32>, Vec<Component>) {
    fn find(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut labels = vec![0u32; w * h];
    let mut parent = vec![0u32];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] {
                continue;
            }
            let up = if y > 0 { labels[i - w] } else { 0 };
            let left = if x > 0 { labels[i - 1] } else { 0 };
            labels[i] = match (up, left) {
                (0, 0) => {
                    let l = parent.len() as u32;
                    parent.push(l);
                    l
                }
                (a, 0) | (0, a) => a,
                (a, b) => {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    // The smaller root wins, so numbering stays in raster order.
                    let (lo, hi) = (ra.min(rb), ra.max(rb));
                    parent[hi as usize] = lo;
                    lo
                }
            };
        }
    }
    let mut final_label = vec![0u32; parent.len()];
    let mut n = 0u32;
    for i in 1..parent.len() {
        let r = find(&mut parent, i as u32);
        if r == i as u32 {
            n += 1;
            final_label[i] = n;
        } else {
            final_label[i] = final_label[r as usize];
        }
    }
    let mut stats = vec![
        Component {
            left: usize::MAX,
            right: 0,
            area: 0
        };
        n as usize
    ];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if labels[i] == 0 {
                continue;
            }
            let l = final_label[labels[i] as usize];
            labels[i] = l;
            let c = &mut stats[l as usize - 1];
            c.left = c.left.min(x);
            c.right = c.right.max(x + 1);
            c.area += 1;
        }
    }
    (labels, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_sum_sink_matches_materialized_rotation() {
        let patterned = GrayImage::from_fn(37, 29, |x, y| {
            image::Luma([if x == 0 || y == 28 || (y + x / 4) % 7 < 2 {
                255
            } else {
                ((x * 31 + y * 17) % 256) as u8
            }])
        });
        for ink in [
            GrayImage::new(13, 9),
            GrayImage::from_pixel(13, 9, image::Luma([127])),
            GrayImage::from_pixel(1, 7, image::Luma([255])),
            GrayImage::from_pixel(7, 1, image::Luma([255])),
            patterned,
        ] {
            for a in [-45.0, -3.5, -0.3, 0.0, 0.1, 2.5, 45.0] {
                let rotated = rotate_u8_zero(&ink, a);
                let expected: Vec<i64> = rotated
                    .as_raw()
                    .chunks_exact(ink.width() as usize)
                    .map(|row| row.iter().map(|&v| i64::from(v)).sum())
                    .collect();
                assert_eq!(rotated_row_sums(&ink, a), expected, "angle {a}");
            }
        }
    }

    /// Outputs of `cv2.resize` (OpenCV 5.0) on a fixed ramp; the models
    /// expect crops made with OpenCV's exact arithmetic.
    #[test]
    fn resize_matches_opencv() {
        let src: Vec<u8> = (0..35u32).map(|i| ((i * 37 + 11) % 256) as u8).collect();
        let a = GrayImage::from_raw(7, 5, src).unwrap();
        let b = image::imageops::crop_imm(&a, 0, 0, 4, 3).to_image();
        let cases: [(&GrayImage, u32, u32, Interp, &[u8]); 4] = [
            (&a, 3, 2, Interp::Linear, &[38, 124, 211, 45, 132, 218]),
            (&a, 3, 2, Interp::Area, &[40, 124, 209, 47, 132, 216]),
            (
                &b,
                6,
                5,
                Interp::Linear,
                &[
                    11, 29, 54, 79, 103, 122, 12, 30, 55, 80, 105, 123, 14, 33, 57, 82, 107, 125,
                    16, 34, 59, 83, 108, 127, 17, 35, 60, 85, 109, 128,
                ],
            ),
            (
                &b,
                6,
                4,
                Interp::Area,
                &[
                    11, 30, 48, 85, 104, 122, 13, 31, 50, 87, 105, 124, 15, 33, 52, 89, 107, 126,
                    17, 36, 54, 91, 110, 128,
                ],
            ),
        ];
        for (src, w, h, interp, expected) in cases {
            assert_eq!(
                resize(src, w, h, interp).as_raw(),
                expected,
                "{w}x{h} {interp:?}"
            );
        }
    }
}
