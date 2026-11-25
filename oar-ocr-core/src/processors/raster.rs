//! OpenCV-compatible image operations for OCR preprocessing.
use image::{GrayImage, Luma, Rgb, RgbImage};

#[path = "raster_scalar.rs"]
pub mod scalar;

/// Interpolation of [`resize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    /// Bilinear interpolation.
    Linear,
    /// Area resampling.
    Area,
}

/// Saturating round half to even, as OpenCV's `saturate_cast<uchar>(float)`.
fn sat_u8(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// `cv2.resize` of an 8-bit plane to `w` x `h`.
pub fn resize(src: &GrayImage, w: u32, h: u32, interp: Interp) -> GrayImage {
    let (sw, sh) = src.dimensions();
    if (sw, sh) == (w, h) {
        return src.clone();
    }
    // OpenCV derives the scale from its inverse, which can differ in the
    // last bit from `sw / w`.
    let scale_x = 1.0 / (w as f64 / sw as f64);
    let scale_y = 1.0 / (h as f64 / sh as f64);
    let iscale_x = scale_x.round_ties_even() as u32;
    let iscale_y = scale_y.round_ties_even() as u32;
    let area_fast = (scale_x - iscale_x as f64).abs() < f64::EPSILON
        && (scale_y - iscale_y as f64).abs() < f64::EPSILON;
    let mut interp = interp;
    // OpenCV turns an exact 2x linear downscale into an area average.
    if interp == Interp::Linear && area_fast && iscale_x == 2 && iscale_y == 2 {
        interp = Interp::Area;
    }
    let data = if interp == Interp::Area && scale_x >= 1.0 && scale_y >= 1.0 {
        if area_fast {
            resize_area_fast(
                src,
                w as usize,
                h as usize,
                iscale_x as usize,
                iscale_y as usize,
            )
        } else {
            resize_area(src, w as usize, h as usize, scale_x, scale_y)
        }
    } else {
        resize_linear(src, w as usize, h as usize, interp == Interp::Area)
    };
    GrayImage::from_raw(w, h, data).expect("buffer matches dimensions")
}

/// Fixed-point bilinear resize (`INTER_RESIZE_COEF_BITS` = 11); with
/// `area_mode`, the coefficients OpenCV uses to emulate `INTER_AREA` when
/// upscaling.
fn resize_linear(src: &GrayImage, w: usize, h: usize, area_mode: bool) -> Vec<u8> {
    const SCALE: f32 = 2048.0;
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let s = src.as_raw();
    let (inv_x, inv_y) = (w as f64 / sw as f64, h as f64 / sh as f64);
    let coef = |d: usize, inv: f64, size: usize, clamp: bool| -> (i64, i32, i32) {
        let scale = 1.0 / inv;
        let (mut sx, mut f) = if area_mode {
            let sx = (d as f64 * scale).floor() as i64;
            let f = ((d + 1) as f64 - (sx + 1) as f64 * inv) as f32;
            (sx, if f <= 0.0 { 0.0 } else { f - f.floor() })
        } else {
            let f = ((d as f64 + 0.5) * scale - 0.5) as f32;
            let sx = f.floor() as i64;
            (sx, f - sx as f32)
        };
        if clamp {
            if sx < 0 {
                f = 0.0;
                sx = 0;
            }
            if sx >= size as i64 - 1 {
                f = 0.0;
                sx = size as i64 - 1;
            }
        }
        let a0 = ((1.0 - f) * SCALE).round_ties_even() as i32;
        let a1 = (f * SCALE).round_ties_even() as i32;
        (sx, a0, a1)
    };
    let xs: Vec<_> = (0..w).map(|dx| coef(dx, inv_x, sw, true)).collect();
    let mut out = vec![0u8; w * h];
    let mut r0 = vec![0i32; w];
    let mut r1 = vec![0i32; w];
    let hresize = |row: usize, dst: &mut [i32]| {
        let line = &s[row * sw..(row + 1) * sw];
        for (d, &(sx, a0, a1)) in dst.iter_mut().zip(&xs) {
            let sx = sx as usize;
            *d = if sx + 1 < sw {
                line[sx] as i32 * a0 + line[sx + 1] as i32 * a1
            } else {
                line[sx] as i32 * 2048
            };
        }
    };
    for dy in 0..h {
        // Rows are clamped to the image instead of zeroing the weight.
        let (sy, b0, b1) = coef(dy, inv_y, sh, false);
        let y0 = sy.clamp(0, sh as i64 - 1) as usize;
        let y1 = (sy + 1).clamp(0, sh as i64 - 1) as usize;
        hresize(y0, &mut r0);
        hresize(y1, &mut r1);
        let dst = &mut out[dy * w..(dy + 1) * w];
        for x in 0..w {
            // OpenCV's SIMD rows keep 16 bits: (S >> 4) * b >> 16, then >> 2.
            let v = (((r0[x] >> 4) * b0) >> 16) + (((r1[x] >> 4) * b1) >> 16);
            dst[x] = ((v + 2) >> 2).clamp(0, 255) as u8;
        }
    }
    out
}

/// Block average for integer downscale factors.
fn resize_area_fast(src: &GrayImage, w: usize, h: usize, kx: usize, ky: usize) -> Vec<u8> {
    let sw = src.width() as usize;
    let s = src.as_raw();
    let scale = 1.0f32 / (kx * ky) as f32;
    let mut out = vec![0u8; w * h];
    for dy in 0..h {
        for dx in 0..w {
            let mut sum = 0u32;
            for y in dy * ky..(dy + 1) * ky {
                for x in dx * kx..(dx + 1) * kx {
                    sum += s[y * sw + x] as u32;
                }
            }
            out[dy * w + dx] = if kx == 2 && ky == 2 {
                ((sum + 2) >> 2) as u8
            } else {
                sat_u8(sum as f32 * scale)
            };
        }
    }
    out
}

/// `computeResizeAreaTab`: (destination index, source index, weight).
fn area_tab(ssize: usize, dsize: usize, scale: f64) -> Vec<(usize, usize, f32)> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for d in 0..dsize {
        let fs1 = d as f64 * scale;
        let fs2 = fs1 + scale;
        let cell = scale.min(ssize as f64 - fs1);
        let mut s2 = fs2.floor() as i64;
        s2 = s2.min(ssize as i64 - 1);
        let s1 = (fs1.ceil() as i64).min(s2);
        if s1 as f64 - fs1 > 1e-3 {
            tab.push((d, (s1 - 1) as usize, ((s1 as f64 - fs1) / cell) as f32));
        }
        for s in s1..s2 {
            tab.push((d, s as usize, (1.0 / cell) as f32));
        }
        if fs2 - s2 as f64 > 1e-3 {
            tab.push((
                d,
                s2 as usize,
                ((fs2 - s2 as f64).min(1.0).min(cell) / cell) as f32,
            ));
        }
    }
    tab
}

/// Area-weighted downscale for non-integer factors.
fn resize_area(src: &GrayImage, w: usize, h: usize, scale_x: f64, scale_y: f64) -> Vec<u8> {
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let s = src.as_raw();
    let xtab = area_tab(sw, w, scale_x);
    let ytab = area_tab(sh, h, scale_y);
    let mut out = vec![0u8; w * h];
    let mut buf = vec![0f32; w];
    let mut sum = vec![0f32; w];
    let mut prev_dy: Option<usize> = None;
    for &(dy, sy, beta) in &ytab {
        buf.fill(0.0);
        let line = &s[sy * sw..(sy + 1) * sw];
        for &(dx, sx, alpha) in &xtab {
            buf[dx] += line[sx] as f32 * alpha;
        }
        if prev_dy != Some(dy) {
            if let Some(p) = prev_dy {
                for (o, v) in out[p * w..(p + 1) * w].iter_mut().zip(&sum) {
                    *o = sat_u8(*v);
                }
            }
            for (v, b) in sum.iter_mut().zip(&buf) {
                *v = beta * b;
            }
            prev_dy = Some(dy);
        } else {
            for (v, b) in sum.iter_mut().zip(&buf) {
                *v += beta * b;
            }
        }
    }
    if let Some(p) = prev_dy {
        for (o, v) in out[p * w..(p + 1) * w].iter_mut().zip(&sum) {
            *o = sat_u8(*v);
        }
    }
    out
}

/// `cv2.getRotationMatrix2D(center, degrees, 1.0)`.
pub fn rotation_matrix(cx: f64, cy: f64, degrees: f64) -> [f64; 6] {
    let a = degrees.to_radians();
    let (alpha, beta) = (a.cos(), a.sin());
    [
        alpha,
        beta,
        (1.0 - alpha) * cx - beta * cy,
        -beta,
        alpha,
        beta * cx + (1.0 - alpha) * cy,
    ]
}
/// Resize an RGB image using the same fixed-point interpolation on each plane.
pub fn resize_rgb(img: &RgbImage, w: u32, h: u32) -> RgbImage {
    if img.dimensions() == (w, h) {
        return img.clone();
    }
    let planes: [_; 3] = std::array::from_fn(|c| {
        let plane = GrayImage::from_fn(img.width(), img.height(), |x, y| Luma([img[(x, y)][c]]));
        resize(&plane, w, h, Interp::Linear)
    });
    RgbImage::from_fn(w, h, |x, y| {
        Rgb(std::array::from_fn(|c| planes[c][(x, y)][0]))
    })
}

/// RGB to grayscale with the rounded fixed-point coefficients used by cvtColor.
pub fn grayscale(image: &RgbImage) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let [r, g, b] = image[(x, y)].0.map(u32::from);
        Luma([((9798 * r + 19235 * g + 3735 * b + 16384) >> 15) as u8])
    })
}

/// Subtract smoothed column backgrounds and stretch contrast to grayscale.
pub fn flatten(gray: &GrayImage) -> GrayImage {
    let (w, h) = gray.dimensions();
    let margin = (h / 6).max(2).min(h);
    let mut sample = Vec::with_capacity((margin * 2) as usize);
    let bg: Vec<_> = (0..w)
        .map(|x| {
            sample.clear();
            for y in 0..margin {
                sample.push(gray[(x, y)][0]);
                sample.push(gray[(x, h - 1 - y)][0]);
            }
            sample.sort_unstable();
            let mid = sample.len() / 2;
            u8::midpoint(sample[mid], sample[mid - 1])
        })
        .collect();
    let radius = ((h | 1).clamp(3, 255) / 2) as i64;
    let bg: Vec<_> = (0..w)
        .map(|x| {
            sample.clear();
            sample.extend(
                (-radius..=radius).map(|dx| bg[(x as i64 + dx).clamp(0, w as i64 - 1) as usize]),
            );
            *sample.select_nth_unstable(radius as usize).1
        })
        .collect();
    let mut hist = [0_u64; 256];
    let mut d = GrayImage::from_fn(w, h, |x, y| {
        let value = gray[(x, y)][0].abs_diff(bg[x as usize]);
        hist[value as usize] += 1;
        Luma([value])
    });
    let target = (0.98 * (w as f64 * h as f64)).ceil() as u64;
    let mut acc = 0;
    let hi = hist
        .iter()
        .position(|&n| {
            acc += n;
            acc >= target
        })
        .unwrap_or(255)
        .max(24);
    let scale = 255.0 / hi as f32;
    for pixel in d.as_mut() {
        *pixel = (255. - (*pixel as f32 * scale).min(255.)) as u8;
    }
    d
}
