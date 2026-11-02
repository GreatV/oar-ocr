//! Minimum-area rectangle arithmetic following OpenCV 4.13 rotcalipers.cpp.
//!
//! Adapted from OpenCV, Copyright Intel Corporation and contributors.
//! The license is in LICENSE-OpenCV in this directory.

use crate::document::geometry::{BoundingBox, Point};

pub(super) fn minimum_quad(poly: &BoundingBox) -> Option<BoundingBox> {
    let mut sorted = poly.points.clone();
    sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    sorted.dedup();
    if sorted.len() < 3 {
        return None;
    }
    let turn = |a: Point, b: Point, c: Point| {
        (b.x as f64 - a.x as f64) * (c.y as f64 - b.y as f64)
            - (b.y as f64 - a.y as f64) * (c.x as f64 - b.x as f64)
    };
    let mut hull = Vec::new();
    for &p in &sorted {
        while hull.len() >= 2 && turn(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0. {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len();
    for &p in sorted.iter().rev().skip(1) {
        while hull.len() > lower && turn(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0. {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    let n = hull.len();
    if n < 3 {
        return None;
    }
    // Preserve cyclic source-index order when the hull follows the input curve.
    let ids: Vec<_> = hull
        .iter()
        .map(|p| poly.points.iter().position(|v| v == p).unwrap_or(0))
        .collect();
    for ascending in [true, false] {
        let start = if ascending {
            (0..n).min_by_key(|&i| ids[i])
        } else {
            (0..n).max_by_key(|&i| ids[i])
        }
        .unwrap_or(0);
        if (1..n).all(|i| (ids[(start + i - 1) % n] < ids[(start + i) % n]) == ascending) {
            hull.rotate_left(start);
            break;
        }
    }
    let mut extremes = [0; 4];
    let mut vectors = Vec::with_capacity(n);
    let mut inverse = Vec::with_capacity(n);
    for i in 0..n {
        if hull[i].y < hull[extremes[0]].y {
            extremes[0] = i;
        }
        if hull[i].x > hull[extremes[1]].x {
            extremes[1] = i;
        }
        if hull[i].y > hull[extremes[2]].y {
            extremes[2] = i;
        }
        if hull[i].x < hull[extremes[3]].x {
            extremes[3] = i;
        }
        let (dx, dy) = (
            hull[(i + 1) % n].x - hull[i].x,
            hull[(i + 1) % n].y - hull[i].y,
        );
        vectors.push([dx, dy]);
        inverse.push((1. / ((dx as f64) * (dx as f64) + (dy as f64) * (dy as f64)).sqrt()) as f32);
    }
    let mut best = f32::MAX;
    let mut selected = None;
    for _ in 0..n {
        let [a, b, c, d] = extremes.map(|i| vectors[i]);
        let rotations = [a, [b[1], -b[0]], [-c[0], -c[1]], [-d[1], d[0]]];
        let mut side = 0;
        for i in 1..4 {
            if rotations[i][1] * rotations[side][0] - rotations[i][0] * rotations[side][1] < 0. {
                side = i;
            }
        }
        let index = extremes[side];
        let (x, y) = (
            vectors[index][0] * inverse[index],
            vectors[index][1] * inverse[index],
        );
        let [u, v] = match side {
            0 => [x, y],
            1 => [y, -x],
            2 => [-x, -y],
            _ => [-y, x],
        };
        extremes[side] = (extremes[side] + 1) % n;
        let dx = hull[extremes[1]].x - hull[extremes[3]].x;
        let dy = hull[extremes[1]].y - hull[extremes[3]].y;
        let width = dx * u + dy * v;
        let dx = hull[extremes[2]].x - hull[extremes[0]].x;
        let dy = hull[extremes[2]].y - hull[extremes[0]].y;
        let height = -dx * v + dy * u;
        let area = width * height;
        if area <= best {
            best = area;
            selected = Some((u, v, width, height, hull[extremes[3]], hull[extremes[0]]));
        }
    }
    let (u, v, width, height, left, bottom) = selected?;
    let c1 = u * left.x + left.y * v;
    let c2 = -v * bottom.x + bottom.y * u;
    let determinant = 1. / (u * u + v * v);
    let origin = [
        (c1 * u - c2 * v) * determinant,
        (u * c2 + v * c1) * determinant,
    ];
    let e1 = [u * width, v * width];
    let e2 = [-v * height, u * height];
    let center = [
        origin[0] + (e1[0] + e2[0]) * 0.5,
        origin[1] + (e1[1] + e2[1]) * 0.5,
    ];
    let mut w = ((e2[0] as f64).powi(2) + (e2[1] as f64).powi(2)).sqrt() as f32;
    let mut h = ((e1[0] as f64).powi(2) + (e1[1] as f64).powi(2)).sqrt() as f32;
    let angle = if e1[0] == 0. && e1[1] > 0. {
        std::mem::swap(&mut w, &mut h);
        -90.
    } else {
        (-(e1[0] as f64).atan2(e1[1] as f64)).to_degrees() as f32
    };
    let radians = (angle as f64).to_radians();
    let a = radians.sin() as f32 * 0.5;
    let b = radians.cos() as f32 * 0.5;
    let p0 = Point::new(center[0] - a * h - b * w, center[1] + b * h - a * w);
    let p1 = Point::new(center[0] + a * h - b * w, center[1] - b * h - a * w);
    let mut quad = vec![
        p0,
        p1,
        Point::new(2. * center[0] - p0.x, 2. * center[1] - p0.y),
        Point::new(2. * center[0] - p1.x, 2. * center[1] - p1.y),
    ];
    let mean = Point::new(
        quad.iter().map(|p| p.x).sum::<f32>() * 0.25,
        quad.iter().map(|p| p.y).sum::<f32>() * 0.25,
    );
    quad.sort_by(|a, b| {
        (a.y - mean.y)
            .atan2(a.x - mean.x)
            .total_cmp(&(b.y - mean.y).atan2(b.x - mean.x))
    });
    let start = (0..4)
        .min_by(|&a, &b| (quad[a].x + quad[a].y).total_cmp(&(quad[b].x + quad[b].y)))
        .unwrap_or(0);
    quad.rotate_left(start);
    Some(BoundingBox::new(quad))
}
