//! Minimum-area rectangles with the f32 evaluation order of OpenCV 4.12.
//! The detector truncates rectangle vertices before its round offset, so a single ulp can change a crop.
//! Adapted from OpenCV's rotcalipers.cpp and convhull.cpp. See NOTICE.
use super::{Point, Quad};

/// Keep the contour's cyclic order when the hull indices form a monotone sequence,
/// which determines the first support vertex on an axis-aligned edge.
fn hull(points: &[Point]) -> Vec<Point> {
    let mut ids: Vec<_> = (0..points.len()).collect();
    ids.sort_by(|&a, &b| {
        points[a]
            .x
            .total_cmp(&points[b].x)
            .then(points[a].y.total_cmp(&points[b].y))
    });
    ids.dedup_by(|a, b| points[*a] == points[*b]);
    if ids.len() < 3 {
        return Vec::new();
    }
    let chain = |indices: &[usize]| {
        let mut out: Vec<usize> = Vec::new();
        for &i in indices {
            while let [.., a, b] = out.as_slice() {
                let ab = points[*b] - points[*a];
                let ac = points[i] - points[*a];
                if ab.perp(&ac) > 0. {
                    break;
                }
                out.pop();
            }
            out.push(i);
        }
        out.pop();
        out
    };
    let mut lower = chain(&ids);
    ids.reverse();
    let mut indices = chain(&ids);
    indices.append(&mut lower);
    if indices.len() < 3 {
        return Vec::new();
    }
    let n = indices.len();
    let rises = (0..n)
        .filter(|&i| indices[i] < indices[(i + 1) % n])
        .count();
    let next = if rises == n - 1 {
        indices.iter().enumerate().min_by_key(|(_, v)| *v)
    } else if rises == 1 {
        indices.iter().enumerate().max_by_key(|(_, v)| *v)
    } else {
        None
    };
    if let Some((i, _)) = next {
        indices.rotate_left(i);
    }
    indices.into_iter().map(|i| points[i]).collect()
}

pub(super) fn mini_box(points: &[Point]) -> (Quad, f32) {
    let hull = hull(points);
    if hull.is_empty() {
        return ([Point::zeros(); 4], 0.);
    }
    let n = hull.len();
    let mut support = [0; 4]; // bottom, right, top, left in Cartesian coordinates
    let mut edges = Vec::with_capacity(n);
    for (i, p) in hull.iter().enumerate() {
        if p.y < hull[support[0]].y {
            support[0] = i;
        }
        if p.x > hull[support[1]].x {
            support[1] = i;
        }
        if p.y > hull[support[2]].y {
            support[2] = i;
        }
        if p.x < hull[support[3]].x {
            support[3] = i;
        }
        edges.push(hull[(i + 1) % n] - p);
    }
    let mut best_area = f32::INFINITY;
    let mut best = None;
    for _ in 0..n {
        let [b, r, t, l] = support.map(|i| edges[i]);
        let candidates = [b, Point::new(r.y, -r.x), -t, Point::new(-l.y, l.x)];
        let mut side = 0;
        for i in 1..4 {
            if candidates[i].perp(&candidates[side]) > 0. {
                side = i;
            }
        }
        let edge = candidates[side];
        let along = edge * (1. / edge.cast::<f64>().norm()) as f32;
        let across = Point::new(-along.y, along.x);
        support[side] = (support[side] + 1) % n;
        let [bottom, right, top, left] = support.map(|i| hull[i]);
        let w = (right - left).dot(&along);
        let h = (top - bottom).dot(&across);
        let area = w * h;
        if area <= best_area {
            best_area = area;
            best = Some((along, across, w, h, left, bottom));
        }
    }
    let (along, across, w, h, left, bottom) = best.expect("nondegenerate convex hull");
    let (u, v) = (along.dot(&left), across.dot(&bottom));
    let inv_det = 1. / along.perp(&across);
    let origin = Point::new(u * across.y - v * along.y, along.x * v - across.x * u) * inv_det;
    let (dx, dy) = (along * w, across * h);
    let center = origin + (dx + dy) * 0.5;
    let (width, height) = (
        dx.cast::<f64>().norm() as f32,
        dy.cast::<f64>().norm() as f32,
    );
    let angle = (dx.y as f64).atan2(dx.x as f64) as f32;
    let degrees = (f64::from(angle * 180.) / std::f64::consts::PI) as f32;
    let (sin, cos) = f64::from(degrees).to_radians().sin_cos();
    let (a, b) = (sin as f32 * 0.5, cos as f32 * 0.5);
    let p = Point::new(
        center.x - a * height - b * width,
        center.y + b * height - a * width,
    );
    let t = Point::new(
        center.x + a * height - b * width,
        center.y - b * height - a * width,
    );
    let mut q = [p, t, 2. * center - p, 2. * center - t];
    q.sort_by(|a, b| a.x.total_cmp(&b.x));
    if q[0].y > q[1].y {
        q.swap(0, 1);
    }
    if q[2].y > q[3].y {
        q.swap(2, 3);
    }
    ([q[0], q[2], q[3], q[1]], width.min(height))
}
