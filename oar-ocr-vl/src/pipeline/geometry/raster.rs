//! Integer polygon filling following OpenCV's LINE_8 and 16-bit edge arithmetic.
//!
//! Adapted from OpenCV 4.13 drawing.cpp, Copyright Intel Corporation and
//! contributors. The license is in LICENSE-OpenCV in this directory.

use crate::document::geometry::BoundingBox;

fn clip_line(
    mut a: [i64; 2],
    mut b: [i64; 2],
    width: i64,
    height: i64,
) -> (bool, [i64; 2], [i64; 2]) {
    let code = |p: [i64; 2]| {
        i32::from(p[0] < 0)
            + 2 * i32::from(p[0] >= width)
            + 4 * i32::from(p[1] < 0)
            + 8 * i32::from(p[1] >= height)
    };
    let (mut ca, mut cb) = (code(a), code(b));
    if ca & cb == 0 && ca | cb != 0 {
        if ca & 12 != 0 {
            let edge = if ca < 8 { 0 } else { height - 1 };
            a[0] += ((edge - a[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            a[1] = edge;
            ca = code(a) & 3;
        }
        if cb & 12 != 0 {
            let edge = if cb < 8 { 0 } else { height - 1 };
            b[0] += ((edge - b[1]) as f64 * (b[0] - a[0]) as f64 / (b[1] - a[1]) as f64) as i64;
            b[1] = edge;
            cb = code(b) & 3;
        }
        if ca & cb == 0 && ca | cb != 0 {
            if ca != 0 {
                let edge = if ca == 1 { 0 } else { width - 1 };
                a[1] += ((edge - a[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                a[0] = edge;
                ca = 0;
            }
            if cb != 0 {
                let edge = if cb == 1 { 0 } else { width - 1 };
                b[1] += ((edge - b[0]) as f64 * (b[1] - a[1]) as f64 / (b[0] - a[0]) as f64) as i64;
                b[0] = edge;
                cb = 0;
            }
        }
    }
    (ca | cb == 0, a, b)
}

fn line(mask: &mut [bool], width: usize, mut a: [i64; 2], mut b: [i64; 2]) {
    if a[0] > b[0] {
        std::mem::swap(&mut a, &mut b);
    }
    let (dx, dy) = (b[0] - a[0], (b[1] - a[1]).abs());
    let sy = if a[1] <= b[1] { 1 } else { -1 };
    let vertical = dy > dx;
    let (major, minor) = if vertical { (dy, dx) } else { (dx, dy) };
    let mut error = major - 2 * minor;
    for _ in 0..=major {
        mask[a[1] as usize * width + a[0] as usize] = true;
        let diagonal = error < 0;
        error -= 2 * minor;
        if diagonal {
            error += 2 * major;
        }
        if vertical {
            a[1] += sy;
            if diagonal {
                a[0] += 1;
            }
        } else {
            a[0] += 1;
            if diagonal {
                a[1] += sy;
            }
        }
    }
}

pub(crate) fn mask_crop(image: &mut image::RgbImage, polygon: &BoundingBox, offset: (u32, u32)) {
    if polygon.points.len() < 3 || image.width() == 0 || image.height() == 0 {
        return;
    }
    let points: Vec<_> = polygon
        .points
        .iter()
        .map(|p| {
            [
                i64::from(p.x as i32) - i64::from(offset.0),
                i64::from(p.y as i32) - i64::from(offset.1),
            ]
        })
        .collect();
    let (width, height) = (image.width() as usize, image.height() as usize);
    let mut mask = vec![false; width * height];
    let mut edges = Vec::new();
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        let (inside, clipped_a, clipped_b) = clip_line(a, b, width as i64, height as i64);
        if inside {
            line(&mut mask, width, clipped_a, clipped_b);
        }
        if a[1] == b[1] {
            continue;
        }
        let (ay, by) = if clipped_a[1] != clipped_b[1] {
            (clipped_a[1], clipped_b[1])
        } else {
            (a[1], b[1])
        };
        let step = ((clipped_b[0] - clipped_a[0]) << 16) / (by - ay);
        let (y0, y1, x) = if a[1] < b[1] {
            (a[1], b[1], (clipped_a[0] << 16) + (a[1] - ay) * step)
        } else {
            (b[1], a[1], (clipped_b[0] << 16) + (b[1] - by) * step)
        };
        edges.push((y0, y1, x, step));
    }
    for y in 0..height {
        let mut intersections: Vec<_> = edges
            .iter()
            .filter(|&&(y0, y1, _, _)| y as i64 >= y0 && (y as i64) < y1)
            .map(|&(y0, _, x, step)| x + (y as i64 - y0) * step)
            .collect();
        intersections.sort_unstable();
        for &[a, b] in intersections.as_chunks::<2>().0 {
            let start = ((a + 65535) >> 16).max(0);
            let end = (b >> 16).min(width as i64 - 1);
            for x in start..=end {
                mask[y * width + x as usize] = true;
            }
        }
    }
    for (i, pixel) in image.pixels_mut().enumerate() {
        if !mask[i] {
            *pixel = image::Rgb([255; 3]);
        }
    }
}
