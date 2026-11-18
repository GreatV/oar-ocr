//! OpenCV accumulation and integer offsets reproduce DB models trained and evaluated with truncation before polygon expansion.
mod calipers;
use super::{BoundingBox, DBPostProcess, DBPostProcessConfig};
use calipers::mini_box;
use image::{GrayImage, Luma};
use imageproc::contours::find_contours;
type Point = nalgebra::Vector2<f32>;
type Quad = [Point; 4];

impl DBPostProcess {
    pub(crate) fn opencv_quads(
        &self,
        prob: &[f32],
        map: (usize, usize),
        source: (u32, u32),
        pad: u32,
        thresholds: &DBPostProcessConfig,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let (pw, ph) = map;
        let (w, h) = source;
        let (padded_w, padded_h) = (w + 2 * pad, h + 2 * pad);
        let mut scores = Vec::new();
        // findContours adds a zero border, including when the detector mask touches the page edge.
        let mut bitmap = GrayImage::new(pw as u32 + 2, ph as u32 + 2);
        for (y, row) in prob.chunks_exact(pw).enumerate() {
            for (x, &p) in row.iter().enumerate() {
                bitmap[(x as u32 + 1, y as u32 + 1)] = Luma([u8::from(p > thresholds.thresh)]);
            }
        }
        let mut boxes = Vec::new();
        for contour in find_contours::<i32>(&bitmap)
            .into_iter()
            .rev()
            .take(self.max_candidates)
        {
            let points = contour
                .points
                .iter()
                .map(|p| Point::new((p.x - 1) as f32, (p.y - 1) as f32))
                .collect::<Vec<_>>();
            let (q, side) = mini_box(&points);
            if side < self.min_size {
                continue;
            }
            let confidence = score(&q, prob, pw, ph);
            if confidence < f64::from(thresholds.box_thresh) {
                continue;
            }
            let (mut q, side) = mini_box(&unclip(&q, thresholds.unclip_ratio));
            if side < self.min_size + 2. {
                continue;
            }
            for p in &mut q {
                p.x = (p.x * (padded_w as f32 / pw as f32) - pad as f32).clamp(0., (w - 1) as f32);
                p.y = (p.y * (padded_h as f32 / ph as f32) - pad as f32).clamp(0., (h - 1) as f32);
            }
            let span = |a: usize| {
                q.iter().map(|p| p[a]).fold(f32::NEG_INFINITY, f32::max)
                    - q.iter().map(|p| p[a]).fold(f32::INFINITY, f32::min)
            };
            if span(0) >= 2. && span(1) >= 2. {
                boxes.push(BoundingBox::new(
                    q.map(|p| super::Point::new(p.x, p.y)).to_vec(),
                ));
                scores.push(confidence as f32);
            }
        }
        (boxes, scores)
    }
}
fn score(q: &Quad, prob: &[f32], w: usize, h: usize) -> f64 {
    let bound = |axis: usize, n: usize| {
        let lo = q
            .iter()
            .map(|p| p[axis])
            .fold(f32::INFINITY, f32::min)
            .floor()
            .clamp(0., (n - 1) as f32) as u32;
        let hi = q
            .iter()
            .map(|p| p[axis])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .clamp(0., (n - 1) as f32) as u32;
        (lo, hi)
    };
    let (x0, x1) = bound(0, w);
    let (y0, y1) = bound(1, h);
    // A convex polygon covers a single inclusive run on each scanline. Trace the
    // integer LINE_8 boundary used by fillPoly, including its half-step tie rule.
    let mut spans = vec![(i32::MAX, i32::MIN); (y1 - y0 + 1) as usize];
    let vertices = q.map(|p| {
        (
            (f64::from(p.x) - f64::from(x0)) as i32,
            (f64::from(p.y) - f64::from(y0)) as i32,
        )
    });
    let xmax = (x1 - x0) as i32;
    let ymax = (y1 - y0) as i32;
    for i in 0..4 {
        let (mut a, mut b) = (vertices[i], vertices[(i + 1) % 4]);
        if !clip_edge(&mut a, &mut b, xmax, ymax) {
            continue;
        }
        if a.0 > b.0 {
            std::mem::swap(&mut a, &mut b);
        }
        let dx = b.0 - a.0;
        let dy = (b.1 - a.1).abs();
        let sy = (b.1 - a.1).signum();
        let steep = dy > dx;
        let major = dx.max(dy);
        let minor = dx.min(dy);
        let mut error = major - 2 * minor;
        let (mut x, mut y) = a;
        for _ in 0..=major {
            let (lo, hi) = &mut spans[y as usize];
            *lo = (*lo).min(x);
            *hi = (*hi).max(x);
            let diagonal = error < 0;
            if !steep || diagonal {
                x += 1;
            }
            if steep || diagonal {
                y += sy;
            }
            if diagonal {
                error += 2 * major;
            }
            error -= 2 * minor;
        }
    }
    // A rectangle may cross both sides of the probability map. Interior scanlines
    // still extend to the image edge even when an outline segment was clipped off.
    for i in 0..4 {
        let (mut a, mut b) = (vertices[i], vertices[(i + 1) % 4]);
        if a.1 == b.1 {
            continue;
        }
        if a.1 > b.1 {
            std::mem::swap(&mut a, &mut b);
        }
        let dx = (i64::from(b.0 - a.0) << 16) / i64::from(b.1 - a.1);
        for y in a.1.max(0)..b.1.min(ymax + 1) {
            let x = (i64::from(a.0) << 16) + i64::from(y - a.1) * dx;
            let (lo, hi) = &mut spans[y as usize];
            *lo = (*lo).min(((x + 65535) >> 16) as i32);
            *hi = (*hi).max((x >> 16) as i32);
        }
    }
    let mut sum = 0.;
    let mut count = 0;
    for (y, (lo, hi)) in spans.into_iter().enumerate() {
        let (lo, hi) = (lo.max(0), hi.min(xmax));
        if hi < lo {
            continue;
        }
        let start = (y + y0 as usize) * w + x0 as usize + lo as usize;
        let len = (hi - lo + 1) as usize;
        sum += prob[start..start + len]
            .iter()
            .map(|&x| f64::from(x))
            .sum::<f64>();
        count += len;
    }
    if count == 0 { 0. } else { sum / count as f64 }
}

// Clip the endpoints before stepping along the major axis, as in OpenCV's integer line iterator.
fn clip_edge(a: &mut (i32, i32), b: &mut (i32, i32), xmax: i32, ymax: i32) -> bool {
    let code = |(x, y): (i32, i32)| {
        (x < 0) as u8
            | (((x > xmax) as u8) << 1)
            | (((y < 0) as u8) << 2)
            | (((y > ymax) as u8) << 3)
    };
    if code(*a) & code(*b) != 0 {
        return false;
    }
    for first in [true, false] {
        let (p, q) = if first { (&mut *a, *b) } else { (&mut *b, *a) };
        if p.1 < 0 || p.1 > ymax {
            let y = p.1.clamp(0, ymax);
            p.0 += (f64::from(y - p.1) * f64::from(q.0 - p.0) / f64::from(q.1 - p.1)) as i32;
            p.1 = y;
        }
    }
    if code(*a) & code(*b) != 0 {
        return false;
    }
    for first in [true, false] {
        let (p, q) = if first { (&mut *a, *b) } else { (&mut *b, *a) };
        if p.0 < 0 || p.0 > xmax {
            let x = p.0.clamp(0, xmax);
            p.1 += (f64::from(x - p.0) * f64::from(q.1 - p.1) / f64::from(q.0 - p.0)) as i32;
            p.0 = x;
        }
    }
    true
}
// The reference computes the offset from the float rectangle, then passes integer-truncated vertices to Clipper's round join.
fn unclip(q: &Quad, ratio: f32) -> Vec<Point> {
    let vertices = q.map(|p| p.cast::<f64>());
    let mut area = 0.;
    let mut perimeter = 0.;
    for i in 0..4 {
        let a = vertices[i];
        let b = vertices[(i + 1) % 4];
        area += a.x * b.y - a.y * b.x;
        perimeter += (q[(i + 1) % 4] - q[i]).norm() as f64;
    }
    // The reference uses the decimal configuration value in double precision.
    let ratio = if ratio == 1.4 {
        1.4_f64
    } else {
        f64::from(ratio)
    };
    let delta = area.abs() * 0.5 * ratio / perimeter;
    if delta <= 0. || !delta.is_finite() {
        return Vec::new();
    }
    let vertices = vertices.map(|p| p.map(f64::trunc));
    let normals: [_; 4] = std::array::from_fn(|i| {
        let d = vertices[(i + 1) % 4] - vertices[i];
        if d.x == 0. && d.y == 0. {
            nalgebra::Vector2::zeros()
        } else {
            nalgebra::Vector2::new(d.y, -d.x) * (1. / d.norm())
        }
    });
    let steps = (std::f64::consts::PI / (1. - 0.25_f64.min(delta * 0.25) / delta).acos())
        .min(delta * std::f64::consts::PI);
    let (sin, cos) = (std::f64::consts::TAU / steps).sin_cos();
    let mut out = Vec::new();
    for i in 0..4 {
        let mut n = normals[(i + 3) % 4];
        let last = normals[i];
        let angle = (n.x * last.y - n.y * last.x).atan2(n.dot(&last));
        let count = ((steps / std::f64::consts::TAU) * angle.abs())
            .round()
            .max(1.) as usize;
        let mut push = |v: nalgebra::Vector2<f64>| {
            let p = vertices[i] + delta * v;
            out.push(Point::new(p.x.round() as f32, p.y.round() as f32));
        };
        for _ in 0..count {
            push(n);
            n = nalgebra::Vector2::new(n.x * cos - n.y * sin, n.x * sin + n.y * cos);
        }
        push(last);
    }
    out.dedup();
    // Clipper emits the union's exterior ring after its rightmost top vertex.
    // The cyclic start breaks ties between equal-area caliper rectangles.
    if let Some((top, _)) = out
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.y.total_cmp(&b.y).then_with(|| b.x.total_cmp(&a.x)))
    {
        let start = (top + 1) % out.len();
        out.rotate_left(start);
    }
    out
}
