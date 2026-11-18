//! Raster operations for the OpenThai detector and line recognizers.
use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Vector2, Vector3};
use std::sync::LazyLock;

use oar_ocr_core::processors::raster;

pub(super) type Point = Vector2<f32>;
pub(super) type Quad = [Point; 4];
pub(super) type Transform = Matrix3<f64>;

pub(super) fn median(mut xs: Vec<f32>) -> f32 {
    if xs.is_empty() {
        0.
    } else {
        crate::indra::median_f32(&mut xs)
    }
}

pub(super) fn line_height(boxes: &[Quad]) -> f32 {
    let heights: Vec<_> = boxes.iter().map(|b| (b[0] - b[3]).norm().max(1.)).collect();
    let lines: Vec<_> = boxes
        .iter()
        .zip(&heights)
        .filter(|(b, h)| (b[0] - b[1]).norm() >= 2.0 * **h)
        .map(|(_, h)| *h)
        .collect();
    if boxes.is_empty() {
        1.
    } else if lines.len() >= 3 {
        median(lines)
    } else {
        median(heights)
    }
}

pub(super) fn skew(boxes: &[Quad]) -> f64 {
    let mut angles: Vec<_> = boxes
        .iter()
        .filter_map(|b| {
            let edge = b[1] - b[0];
            (edge.norm() >= 3. * (b[3] - b[0]).norm().max(1.))
                .then(|| (edge.y as f64).atan2(edge.x as f64).to_degrees())
        })
        .collect();
    if angles.is_empty() {
        0.
    } else {
        crate::indra::median_f64(&mut angles)
    }
}

fn cubic(f: f32) -> [f32; 4] {
    let a = -0.75;
    let x = f + 1.;
    let c0 = ((a * x - 5. * a) * x + 8. * a) * x - 4. * a;
    let c1 = ((a + 2.) * f - (a + 3.)) * f * f + 1.;
    let x = 1. - f;
    let c2 = ((a + 2.) * x - (a + 3.)) * x * x + 1.;
    [c0, c1, c2, 1. - c0 - c1 - c2]
}

// OpenCV quantizes source positions to 1/32 pixel and cubic weights to 15 fractional bits.
static CUBIC: LazyLock<Vec<[i32; 16]>> = LazyLock::new(|| {
    (0..1024)
        .map(|i| {
            let (cx, cy) = (cubic((i % 32) as f32 / 32.), cubic((i / 32) as f32 / 32.));
            let mut v: [i32; 16] =
                std::array::from_fn(|j| (cy[j / 4] * cx[j % 4] * 32768.).round_ties_even() as i32);
            let diff = 32768 - v.iter().sum::<i32>();
            let mut k = 10;
            for j in [10, 11, 14, 15] {
                if (diff > 0 && v[j] > v[k]) || (diff < 0 && v[j] < v[k]) {
                    k = j;
                }
            }
            v[k] += diff;
            v
        })
        .collect()
});

fn warp(
    img: &RgbImage,
    w: u32,
    h: u32,
    back: Transform,
    fill: Option<Rgb<u8>>,
    affine: bool,
) -> RgbImage {
    let sample = |x: i64, y: i64| {
        if let Some(p) = img.get_pixel_checked(x as u32, y as u32) {
            return *p;
        }
        fill.unwrap_or_else(|| {
            img[(
                x.clamp(0, img.width() as i64 - 1) as u32,
                y.clamp(0, img.height() as i64 - 1) as u32,
            )]
        })
    };
    // warpPerspective restarts its homogeneous coordinates at each 32×32-workspace tile.
    let tile_width = (1024 / h.clamp(1, 16)).min(w).max(1);
    RgbImage::from_fn(w, h, |x, y| {
        let tile_x = f64::from(x / tile_width * tile_width);
        let (x, y) = (x as f64, y as f64);
        let projected = |row: usize| {
            let base = (back[(row, 0)] * tile_x + back[(row, 1)] * y) + back[(row, 2)];
            base + back[(row, 0)] * (x - tile_x)
        };
        let den = projected(2);
        let coord = |row| {
            let a: f64 = back[(row, 0)] * x;
            let b: f64 = back[(row, 1)] * y + back[(row, 2)];
            if affine {
                (((a * 1024.).round_ties_even() as i64)
                    + ((b * 1024.).round_ties_even() as i64)
                    + 16)
                    >> 5
            } else if den != 0. {
                (projected(row) * (32. / den)).round_ties_even() as i64
            } else {
                0
            }
        };
        let (u, v) = (coord(0usize), coord(1usize));
        let (sx, sy) = (u.div_euclid(32), v.div_euclid(32));
        if u % 32 == 0 && v % 32 == 0 {
            return sample(sx, sy);
        }
        let weights = &CUBIC[((v.rem_euclid(32) * 32) + u.rem_euclid(32)) as usize];
        let mut sum = [16384; 3];
        for (j, &weight) in weights.iter().enumerate() {
            let p = sample(sx + (j % 4) as i64 - 1, sy + (j / 4) as i64 - 1);
            for c in 0..3 {
                sum[c] += weight * p[c] as i32;
            }
        }
        Rgb(sum.map(|n| (n >> 15).clamp(0, 255) as u8))
    })
}

// The forward four-corner solve and the subsequent inversion both affect half-step ties
// on OpenCV's 1/32-pixel sampling grid, including for axis-aligned rectangles.
fn crop_inverse(q: &Quad, width: u32, height: u32) -> Transform {
    let (w, h) = (width as f32, height as f32);
    let target = [(0., 0.), (w, 0.), (w, h), (0., h)];
    let mut equations = [[0_f64; 9]; 8];
    for (i, (p, (u, v))) in q.iter().zip(target).enumerate() {
        let (x, y) = (f64::from(p.x), f64::from(p.y));
        // getPerspectiveTransform multiplies its Point2f coordinates before widening.
        equations[i] = [
            x,
            y,
            1.,
            0.,
            0.,
            0.,
            f64::from(-p.x * u),
            f64::from(-p.y * u),
            f64::from(u),
        ];
        equations[i + 4] = [
            0.,
            0.,
            0.,
            x,
            y,
            1.,
            f64::from(-p.x * v),
            f64::from(-p.y * v),
            f64::from(v),
        ];
    }
    for i in 0..8 {
        let mut pivot = i;
        for j in i + 1..8 {
            if equations[j][i].abs() > equations[pivot][i].abs() {
                pivot = j;
            }
        }
        if equations[pivot][i].abs() < 100. * f64::EPSILON {
            return Transform::zeros();
        }
        equations.swap(i, pivot);
        let pivot_row = equations[i];
        let inv = -1. / pivot_row[i];
        for row in &mut equations[i + 1..] {
            let factor = row[i] * inv;
            for k in i + 1..9 {
                row[k] += factor * pivot_row[k];
            }
        }
    }
    let mut coefficients = [0.; 8];
    for i in (0..8).rev() {
        let mut value = equations[i][8];
        for (k, &coefficient) in coefficients.iter().enumerate().skip(i + 1) {
            value -= equations[i][k] * coefficient;
        }
        coefficients[i] = value / equations[i][i];
    }
    let [a, b, c, d, e, f, g, h] = coefficients;
    let det = a * (e - f * h) - b * (d - f * g) + c * (d * h - e * g);
    if det == 0. {
        return Transform::zeros();
    }
    Transform::new(
        e - f * h,
        c * h - b,
        b * f - c * e,
        f * g - d,
        a - c * g,
        c * d - a * f,
        d * h - e * g,
        b * g - a * h,
        a * e - b * d,
    ) * (1. / det)
}

pub(super) fn crop(img: &RgbImage, q: &Quad) -> RgbImage {
    let w = (q[0] - q[1]).norm().max((q[2] - q[3]).norm()).max(1.) as u32;
    let h = (q[0] - q[3]).norm().max((q[1] - q[2]).norm()).max(1.) as u32;
    let back = crop_inverse(q, w, h);
    warp(img, w, h, back, None, false)
}

pub(super) fn deskew(img: &RgbImage, degrees: f64) -> (RgbImage, Transform) {
    let (w, h) = (img.width() as f64, img.height() as f64);
    let [a, b, mut tx, c, d, mut ty] = raster::rotation_matrix(w / 2., h / 2., degrees);
    let nw = (h * b.abs() + w * a.abs()).round_ties_even();
    let nh = (h * a.abs() + w * b.abs()).round_ties_even();
    tx += (nw - w) / 2.;
    ty += (nh - h) / 2.;
    let forward = Transform::new(a, b, tx, c, d, ty, 0., 0., 1.);
    let fill = Rgb(std::array::from_fn(|c| {
        let edges = (0..img.width())
            .flat_map(|x| [img[(x, 0)][c], img[(x, img.height() - 1)][c]])
            .chain((0..img.height()).flat_map(|y| [img[(0, y)][c], img[(img.width() - 1, y)][c]]));
        median(edges.map(f32::from).collect()) as u8
    }));
    (
        warp(
            img,
            nw as u32,
            nh as u32,
            forward.try_inverse().expect("rotation is invertible"),
            Some(fill),
            true,
        ),
        forward,
    )
}

pub(super) fn back_quad(q: [Vector2<f64>; 4], back: &Transform) -> [[f64; 2]; 4] {
    q.map(|p| {
        let v = back * Vector3::new(p.x, p.y, 1.);
        [v.x, v.y].map(|n| (n * 10.).round_ties_even() / 10.)
    })
}

pub(super) fn horizontal_slice<T: nalgebra::RealField + Copy>(
    q: &[Vector2<T>; 4],
    left: T,
    right: T,
) -> [Vector2<T>; 4] {
    let top = q[1] - q[0];
    let bottom = q[2] - q[3];
    [
        q[0] + top * left,
        q[0] + top * right,
        q[3] + bottom * right,
        q[3] + bottom * left,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perspective_crop_keeps_opencv_half_step_ties() {
        // Adjacent samples on the same output row straddle a 1/32-pixel rounding boundary.
        let image = RgbImage::from_fn(451, 614, |_, y| Rgb([if y == 603 { 255 } else { 0 }; 3]));
        let (left, right) = (421.75003, 448.2625);
        let (top, bottom) = (589.96875, 612.39844);
        let q = [
            Point::new(left, top),
            Point::new(right, top),
            Point::new(right, bottom),
            Point::new(left, bottom),
        ];
        let cropped = crop(&image, &q);
        assert_eq!(cropped.dimensions(), (26, 22));
        assert!(cropped[(9, 12)][0] > cropped[(8, 12)][0]);
    }
}
