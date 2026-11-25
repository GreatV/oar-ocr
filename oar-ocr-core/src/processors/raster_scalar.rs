//! OpenCV's portable scalar code paths.
//!
//! OpenCV rounds its fixed-point sums differently in its scalar and SIMD builds. The parent module follows the SIMD paths; these follow the scalar ones, which is the arithmetic LiteParse's ocria-nano runtime feeds the OCRia models. Adapted from LiteParse (Apache-2.0). See the NOTICE beside this file.
use super::Interp;
use image::{GrayImage, ImageBuffer, Luma, Pixel, RgbImage};
use std::sync::LazyLock;

/// OpenCV's `INTER_RESIZE_COEF_BITS`.
const RESIZE_ONE: i32 = 1 << 11;

/// `cvRound` clamped to a `short`.
fn sat_short(v: f32) -> i32 {
    (v as f64)
        .round_ties_even()
        .clamp(i16::MIN as f64, i16::MAX as f64) as i32
}

/// `cv2.cvtColor(.., COLOR_RGB2GRAY)` with 14-bit weights.
pub fn grayscale(image: &RgbImage) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let [r, g, b] = image[(x, y)].0.map(u32::from);
        Luma([((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14) as u8])
    })
}

/// `cv2.resize` of an 8-bit image to `w` x `h`.
pub fn resize<P>(
    src: &ImageBuffer<P, Vec<u8>>,
    w: u32,
    h: u32,
    interp: Interp,
) -> ImageBuffer<P, Vec<u8>>
where
    P: Pixel<Subpixel = u8>,
{
    if src.dimensions() == (w, h) {
        return src.clone();
    }
    let ch = P::CHANNEL_COUNT as usize;
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let (w, h) = (w as usize, h as usize);
    let (scale_x, scale_y) = (sw as f64 / w as f64, sh as f64 / h as f64);
    let data = if interp == Interp::Linear || scale_x < 1.0 || scale_y < 1.0 {
        resize_linear(src.as_raw(), (sw, sh), (w, h), ch, interp == Interp::Area)
    } else {
        let (kx, ky) = (scale_x.round(), scale_y.round());
        if (scale_x - kx).abs() < f64::EPSILON && (scale_y - ky).abs() < f64::EPSILON {
            resize_area_fast(src.as_raw(), sw, (w, h), ch, (kx as usize, ky as usize))
        } else {
            resize_area(src.as_raw(), (sw, sh), (w, h), ch, (scale_x, scale_y))
        }
    };
    ImageBuffer::from_raw(w as u32, h as u32, data).expect("buffer matches dimensions")
}

/// Two-tap source index and fixed-point weights per output index; `area` selects the coefficients OpenCV uses to emulate `INTER_AREA` when upscaling.
fn linear_taps(src: usize, dst: usize, area: bool) -> Vec<(usize, i32, i32)> {
    let inv_scale = dst as f64 / src as f64;
    let scale = 1.0 / inv_scale;
    (0..dst)
        .map(|d| {
            let (mut sx, mut f) = if area {
                let sx = (d as f64 * scale).floor() as i64;
                let f = ((d + 1) as f64 - (sx + 1) as f64 * inv_scale) as f32;
                (sx, if f <= 0.0 { 0.0 } else { f - f.floor() })
            } else {
                let fx = ((d as f64 + 0.5) * scale - 0.5) as f32;
                let sx = fx.floor();
                (sx as i64, fx - sx)
            };
            if sx < 0 {
                sx = 0;
                f = 0.0;
            }
            if sx >= src as i64 - 1 {
                sx = src as i64 - 1;
                f = 0.0;
            }
            (
                sx as usize,
                sat_short((1.0 - f) * RESIZE_ONE as f32),
                sat_short(f * RESIZE_ONE as f32),
            )
        })
        .collect()
}

fn resize_linear(
    src: &[u8],
    (sw, sh): (usize, usize),
    (w, h): (usize, usize),
    ch: usize,
    area: bool,
) -> Vec<u8> {
    let xt = linear_taps(sw, w, area);
    let yt = linear_taps(sh, h, area);
    let hresize = |row: usize, dst: &mut [i32]| {
        let line = &src[row * sw * ch..(row + 1) * sw * ch];
        for (dx, &(sx, a0, a1)) in xt.iter().enumerate() {
            for c in 0..ch {
                let s0 = line[sx * ch + c] as i32;
                dst[dx * ch + c] = if sx + 1 < sw {
                    s0 * a0 + line[(sx + 1) * ch + c] as i32 * a1
                } else {
                    s0 * RESIZE_ONE
                };
            }
        }
    };
    let mut out = vec![0u8; w * h * ch];
    let (mut r0, mut r1) = (vec![0i32; w * ch], vec![0i32; w * ch]);
    let (mut row0, mut row1) = (usize::MAX, usize::MAX);
    for (dy, &(sy, b0, b1)) in yt.iter().enumerate() {
        let sy1 = (sy + 1).min(sh - 1);
        if row0 != sy {
            if row1 == sy {
                std::mem::swap(&mut r0, &mut r1);
                row1 = usize::MAX;
            } else {
                hresize(sy, &mut r0);
            }
            row0 = sy;
        }
        if row1 != sy1 {
            hresize(sy1, &mut r1);
            row1 = sy1;
        }
        for (o, (&a, &b)) in out[dy * w * ch..(dy + 1) * w * ch]
            .iter_mut()
            .zip(r0.iter().zip(&r1))
        {
            let v = (a as i64 * b0 as i64 + b as i64 * b1 as i64 + (1 << 21)) >> 22;
            *o = v.clamp(0, 255) as u8;
        }
    }
    out
}

/// Block mean for integer downscale factors.
fn resize_area_fast(
    src: &[u8],
    sw: usize,
    (w, h): (usize, usize),
    ch: usize,
    (kx, ky): (usize, usize),
) -> Vec<u8> {
    let area = (kx * ky) as u32;
    let mut out = vec![0u8; w * h * ch];
    for dy in 0..h {
        for dx in 0..w {
            for c in 0..ch {
                let mut sum = 0u32;
                for y in dy * ky..(dy + 1) * ky {
                    for x in dx * kx..(dx + 1) * kx {
                        sum += src[(y * sw + x) * ch + c] as u32;
                    }
                }
                let v = if area == 4 {
                    (sum + 2) >> 2
                } else {
                    (sum as f64 * (1.0f32 / area as f32) as f64).round_ties_even() as u32
                };
                out[(dy * w + dx) * ch + c] = v.min(255) as u8;
            }
        }
    }
    out
}

/// `computeResizeAreaTab`: (source index, weight) runs per output index.
fn area_tab(src: usize, dst: usize, scale: f64) -> Vec<Vec<(usize, f32)>> {
    (0..dst)
        .map(|d| {
            let fs1 = d as f64 * scale;
            let fs2 = fs1 + scale;
            let cell = scale.min(src as f64 - fs1);
            let s2 = (fs2.floor() as i64).min(src as i64 - 1);
            let s1 = (fs1.ceil() as i64).min(s2);
            let mut taps = Vec::new();
            if s1 as f64 - fs1 > 1e-3 {
                taps.push(((s1 - 1) as usize, ((s1 as f64 - fs1) / cell) as f32));
            }
            for s in s1..s2 {
                taps.push((s as usize, (1.0 / cell) as f32));
            }
            if fs2 - s2 as f64 > 1e-3 {
                taps.push((
                    s2 as usize,
                    ((fs2 - s2 as f64).min(1.0).min(cell) / cell) as f32,
                ));
            }
            taps
        })
        .collect()
}

/// Area-weighted downscale for non-integer factors.
fn resize_area(
    src: &[u8],
    (sw, sh): (usize, usize),
    (w, h): (usize, usize),
    ch: usize,
    (scale_x, scale_y): (f64, f64),
) -> Vec<u8> {
    let xt: Vec<(usize, usize, f32)> = area_tab(sw, w, scale_x)
        .into_iter()
        .enumerate()
        .flat_map(|(dx, taps)| taps.into_iter().map(move |(sx, a)| (dx * ch, sx * ch, a)))
        .collect();
    let yt = area_tab(sh, h, scale_y);
    let row_len = sw * ch;
    let mut out = vec![0u8; w * h * ch];
    // Adjacent output rows share a boundary source row, so its horizontal sums are kept.
    let (mut buf, mut buf_row) = (vec![0f32; w * ch], usize::MAX);
    let mut sum = vec![0f32; w * ch];
    for (dy, taps) in yt.iter().enumerate() {
        sum.fill(0.0);
        for &(sy, beta) in taps {
            if buf_row != sy {
                buf.fill(0.0);
                let row = &src[sy * row_len..(sy + 1) * row_len];
                for &(d, s, alpha) in &xt {
                    for c in 0..ch {
                        buf[d + c] += row[s + c] as f32 * alpha;
                    }
                }
                buf_row = sy;
            }
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s += b * beta;
            }
        }
        for (o, s) in out[dy * w * ch..(dy + 1) * w * ch].iter_mut().zip(&sum) {
            *o = (*s as f64).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// `cv2.getPerspectiveTransform(src, dst)`: the row-major homography taking `src` corners to `dst`.
pub fn perspective_transform(src: &[[f32; 2]; 4], dst: &[[f32; 2]; 4]) -> [f64; 9] {
    let mut a = [[0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (src[i][0] as f64, src[i][1] as f64);
        let (u, v) = (dst[i][0] as f64, dst[i][1] as f64);
        a[i] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, u];
        a[i + 4] = [0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, v];
    }
    // Gauss-Jordan elimination with partial pivoting.
    for col in 0..8 {
        let mut pivot = col;
        for row in col + 1..8 {
            // The last of equal pivots, as `Iterator::max_by` picks.
            if a[row][col].abs() >= a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        let d = a[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        let pivot_row = a[col];
        for (r, row) in a.iter_mut().enumerate() {
            if r != col {
                let f = row[col] / d;
                if f != 0.0 {
                    for k in col..9 {
                        row[k] -= f * pivot_row[k];
                    }
                }
            }
        }
    }
    let mut m = [0f64; 9];
    for i in 0..8 {
        m[i] = if a[i][i] != 0.0 {
            a[i][8] / a[i][i]
        } else {
            0.0
        };
    }
    m[8] = 1.0;
    m
}

fn invert3(m: &[f64; 9]) -> [f64; 9] {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det == 0.0 {
        return [0.0; 9];
    }
    let d = 1.0 / det;
    [
        (m[4] * m[8] - m[5] * m[7]) * d,
        (m[2] * m[7] - m[1] * m[8]) * d,
        (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d,
        (m[0] * m[8] - m[2] * m[6]) * d,
        (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d,
        (m[1] * m[6] - m[0] * m[7]) * d,
        (m[0] * m[4] - m[1] * m[3]) * d,
    ]
}

const INTER_BITS: i64 = 5;
const INTER_TAB: i64 = 1 << INTER_BITS;
const REMAP_SCALE: i32 = 1 << 15;

/// OpenCV `interpolateCubic` (A = -0.75).
fn cubic(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

/// OpenCV `initInterTab2D(INTER_CUBIC)`: 32 x 32 sub-pixel positions with 16 weights each, corrected to sum to exactly 2^15.
static CUBIC: LazyLock<Vec<[i32; 16]>> = LazyLock::new(|| {
    let one_d: Vec<[f32; 4]> = (0..INTER_TAB)
        .map(|i| cubic(i as f32 / INTER_TAB as f32))
        .collect();
    let mut tab = Vec::with_capacity((INTER_TAB * INTER_TAB) as usize);
    for vy in &one_d {
        for vx in &one_d {
            let mut t: [i32; 16] =
                std::array::from_fn(|k| sat_short(vy[k / 4] * vx[k % 4] * REMAP_SCALE as f32));
            let diff = t.iter().sum::<i32>() - REMAP_SCALE;
            if diff != 0 {
                let (mut min, mut max) = (10, 10);
                for k in [10, 11, 14, 15] {
                    if t[k] < t[min] {
                        min = k;
                    } else if t[k] > t[max] {
                        max = k;
                    }
                }
                t[if diff < 0 { max } else { min }] -= diff;
            }
            tab.push(t);
        }
    }
    tab
});

/// `cv2.warpPerspective(img, m, (w, h), flags=INTER_CUBIC, borderMode=BORDER_REPLICATE)`.
pub fn warp_perspective_cubic(img: &RgbImage, m: &[f64; 9], w: u32, h: u32) -> RgbImage {
    let inv = invert3(m);
    let (sw, sh) = (img.width() as i64, img.height() as i64);
    let src = img.as_raw();
    RgbImage::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as f64, y as f64);
        let x0 = inv[0] * xf + inv[1] * yf + inv[2];
        let y0 = inv[3] * xf + inv[4] * yf + inv[5];
        let den = inv[6] * xf + inv[7] * yf + inv[8];
        let scale = if den != 0.0 {
            INTER_TAB as f64 / den
        } else {
            0.0
        };
        let fixed = |v: f64| {
            (v * scale)
                .clamp(i32::MIN as f64, i32::MAX as f64)
                .round_ties_even() as i64
        };
        let (xi, yi) = (fixed(x0), fixed(y0));
        let (sx, sy) = (xi >> INTER_BITS, yi >> INTER_BITS);
        let weights =
            &CUBIC[((yi & (INTER_TAB - 1)) * INTER_TAB + (xi & (INTER_TAB - 1))) as usize];
        let mut sum = [0i64; 3];
        for (k, &weight) in weights.iter().enumerate() {
            // The 4 x 4 source window is clamped to the image.
            let xx = (sx - 1 + (k % 4) as i64).clamp(0, sw - 1);
            let yy = (sy - 1 + (k / 4) as i64).clamp(0, sh - 1);
            let p = ((yy * sw + xx) * 3) as usize;
            for c in 0..3 {
                sum[c] += src[p + c] as i64 * weight as i64;
            }
        }
        image::Rgb(sum.map(|s| ((s + (1 << 14)) >> 15).clamp(0, 255) as u8))
    })
}
