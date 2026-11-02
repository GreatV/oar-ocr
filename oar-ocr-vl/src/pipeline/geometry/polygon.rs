//! Polygon geometry for layout masks and region crops.
//!
//! Mask selection and Auto geometry follow PaddleX layout_analysis/processors.py
//! (Copyright PaddlePaddle Authors, Apache-2.0).

use super::calipers::minimum_quad;
use crate::document::geometry::{BoundingBox, Point};
use crate::pipeline::layout::LayoutShapeMode;
use std::collections::VecDeque;

type Xy = [f64; 2];
fn sub(a: Xy, b: Xy) -> Xy {
    [a[0] - b[0], a[1] - b[1]]
}
fn cross(a: Xy, b: Xy) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn dot(a: Xy, b: Xy) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn norm(a: Xy) -> f64 {
    dot(a, a).sqrt()
}
fn points(b: &BoundingBox) -> Vec<Xy> {
    b.points.iter().map(|p| [p.x as f64, p.y as f64]).collect()
}
fn bounding(points: Vec<Xy>) -> BoundingBox {
    BoundingBox::new(
        points
            .into_iter()
            .map(|p| Point::new(p[0] as f32, p[1] as f32))
            .collect(),
    )
}
fn area(p: &[Xy]) -> f64 {
    (0..p.len())
        .map(|i| cross(p[i], p[(i + 1) % p.len()]))
        .sum::<f64>()
        .abs()
        * 0.5
}

fn segments(p: &[Xy]) -> impl Iterator<Item = (Xy, Xy)> + '_ {
    (0..p.len()).map(|i| (p[i], p[(i + 1) % p.len()]))
}

/// Integrate overlap between polygon interiors over vertical slabs.
/// Edge crossings split slabs so interval endpoints remain linear within each.
fn intersection_area(a: &[Xy], b: &[Xy]) -> (f64, f64, f64) {
    let edges: Vec<_> = segments(a).chain(segments(b)).collect();
    let mut xs: Vec<_> = a.iter().chain(b).map(|p| p[0]).collect();
    for (i, &(p, q)) in edges.iter().enumerate() {
        let r = sub(q, p);
        for &(s, t) in &edges[i + 1..] {
            let d = sub(t, s);
            let denominator = cross(r, d);
            if denominator.abs() < 1e-12 {
                continue;
            }
            let u = cross(sub(s, p), d) / denominator;
            let v = cross(sub(s, p), r) / denominator;
            if (0. ..=1.).contains(&u) && (0. ..=1.).contains(&v) {
                xs.push(p[0] + u * r[0]);
            }
        }
    }
    xs.sort_by(f64::total_cmp);
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    let intervals = |poly: &[Xy], x: f64| {
        let orientation = if segments(poly).map(|(a, b)| cross(a, b)).sum::<f64>() > 0. {
            1
        } else {
            -1
        };
        let mut crossings: Vec<_> = segments(poly)
            .filter(|&(a, b)| x > a[0].min(b[0]) && x < a[0].max(b[0]))
            .map(|(a, b)| {
                (
                    a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]),
                    if b[0] > a[0] { 1 } else { -1 },
                )
            })
            .collect();
        crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut winding = 0;
        let mut previous = 0.;
        let mut intervals = Vec::new();
        for (y, direction) in crossings {
            if winding * orientation > 0 && y > previous {
                intervals.push([previous, y]);
            }
            winding += direction;
            previous = y;
        }
        intervals
    };
    let mut areas = (0., 0., 0.);
    for w in xs.windows(2) {
        let x = (w[0] + w[1]) * 0.5;
        let ys_a = intervals(a, x);
        let ys_b = intervals(b, x);
        let length: f64 = ys_a
            .iter()
            .flat_map(|a| {
                ys_b.iter()
                    .map(move |b| (a[1].min(b[1]) - a[0].max(b[0])).max(0.))
            })
            .sum();
        let width = w[1] - w[0];
        areas.0 += length * width;
        areas.1 += ys_a.iter().map(|a| a[1] - a[0]).sum::<f64>() * width;
        areas.2 += ys_b.iter().map(|b| b[1] - b[0]).sum::<f64>() * width;
    }
    areas
}

pub(crate) fn polygon_overlap(a: &BoundingBox, b: &BoundingBox, union: bool) -> f64 {
    let (a, b) = (points(a), points(b));
    if a.len() < 3 || b.len() < 3 {
        return 0.;
    }
    let (intersection, area_a, area_b) = intersection_area(&a, &b);
    let reference = if union {
        area_a + area_b - intersection
    } else {
        area_a.min(area_b)
    };
    if reference > 0. {
        (intersection / reference).clamp(0., 1.)
    } else {
        0.
    }
}

pub(crate) fn normalize_polygon(
    bbox: [f32; 4],
    polygon: Option<BoundingBox>,
    mode: LayoutShapeMode,
    previous: Option<&BoundingBox>,
) -> BoundingBox {
    let bbox = bbox.map(f32::trunc);
    let rect = BoundingBox::from_coords(bbox[0], bbox[1], bbox[2], bbox[3]);
    let Some(poly) = polygon.filter(|p| p.points.len() >= 4) else {
        return rect;
    };
    match mode {
        LayoutShapeMode::Rect => rect,
        LayoutShapeMode::Poly => poly,
        LayoutShapeMode::Quad => minimum_quad(&poly).unwrap_or(rect),
        LayoutShapeMode::Auto => {
            if let Some(quad) = minimum_quad(&poly) {
                if polygon_overlap(&rect, &quad, true) >= 0.95 {
                    return rect;
                }
                if polygon_overlap(&poly, &quad, true) >= 0.8
                    && previous.is_none_or(|p| polygon_overlap(p, &rect, false) < 0.01)
                {
                    return quad;
                }
            }
            poly
        }
    }
}

fn outer_contour(mask: &[u8], width: usize, height: usize) -> Vec<Xy> {
    let directions: [(i32, i32); 8] = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ];
    let at = |x: i32, y: i32| {
        x >= 0
            && y >= 0
            && (x as usize) < width
            && (y as usize) < height
            && mask[y as usize * width + x as usize] != 0
    };
    let mut seen = vec![false; mask.len()];
    let mut best = Vec::new();
    let mut best_area = -1.;
    for start in 0..mask.len() {
        if mask[start] == 0 || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut count = 0;
        while let Some(i) = queue.pop_front() {
            count += 1;
            let (x, y) = ((i % width) as i32, (i / width) as i32);
            for (dx, dy) in directions {
                let (xx, yy) = (x + dx, y + dy);
                if at(xx, yy) {
                    let j = yy as usize * width + xx as usize;
                    if !seen[j] {
                        seen[j] = true;
                        queue.push_back(j);
                    }
                }
            }
        }
        let initial = ((start % width) as i32, (start / width) as i32);
        let mut current = initial;
        let mut back = (current.0 - 1, current.1);
        let mut contour = Vec::new();
        let mut first_next = None;
        for _ in 0..count * 8 + 1 {
            let d = directions
                .iter()
                .position(|&(dx, dy)| (current.0 + dx, current.1 + dy) == back)
                .unwrap_or(0);
            let found = (0..8).find_map(|step| {
                let k = (d + step) % 8;
                let (dx, dy) = directions[k];
                let next = (current.0 + dx, current.1 + dy);
                at(next.0, next.1).then_some((k, next))
            });
            let Some((k, next)) = found else {
                contour.push([current.0 as f64, current.1 as f64]);
                break;
            };
            if first_next.is_some_and(|p| current == initial && next == p) {
                break;
            }
            first_next.get_or_insert(next);
            contour.push([current.0 as f64, current.1 as f64]);
            let (dx, dy) = directions[(k + 7) % 8];
            back = (current.0 + dx, current.1 + dy);
            current = next;
        }
        let simple: Vec<_> = (0..contour.len())
            .filter_map(|i| {
                let a = sub(contour[i], contour[(i + contour.len() - 1) % contour.len()]);
                let b = sub(contour[(i + 1) % contour.len()], contour[i]);
                (a != b || contour.len() == 1).then_some(contour[i])
            })
            .collect();
        let candidate_area = area(&simple);
        if candidate_area >= best_area {
            best_area = candidate_area;
            best = simple;
        }
    }
    best
}

// Closed-curve initialization and cleanup follow OpenCV 4.13 approx.cpp.
// Copyright Intel Corporation and contributors; see LICENSE-OpenCV.
fn approximate_closed(poly: &[Xy]) -> Vec<Xy> {
    if poly.len() <= 2 {
        return poly.to_vec();
    }
    let epsilon = segments(poly)
        .map(|(a, b)| f64::from(norm(sub(a, b)) as f32))
        .sum::<f64>()
        * 0.004;
    let squared_epsilon = epsilon * epsilon;
    let mut start = 0;
    let mut end = 0;
    for _ in 0..3 {
        start = end;
        let mut maximum = 0.;
        for offset in 1..poly.len() {
            let i = (start + offset) % poly.len();
            let delta = sub(poly[i], poly[start]);
            let distance = dot(delta, delta);
            if distance > maximum {
                maximum = distance;
                end = i;
            }
        }
        if maximum <= squared_epsilon {
            return vec![poly[start]];
        }
    }
    let mut result = Vec::new();
    let mut stack = vec![(end, start), (start, end)];
    while let Some((a, b)) = stack.pop() {
        let delta = sub(poly[b], poly[a]);
        let squared = dot(delta, delta);
        let mut farthest = a;
        let mut maximum = 0.;
        let mut i = (a + 1) % poly.len();
        while i != b {
            let p = sub(poly[i], poly[a]);
            let projection = dot(p, delta);
            let distance = if projection < 0. {
                dot(p, p) * squared
            } else if projection > squared {
                let p = sub(poly[i], poly[b]);
                dot(p, p) * squared
            } else {
                cross(p, delta).powi(2)
            };
            if distance > maximum {
                maximum = distance;
                farthest = i;
            }
            i = (i + 1) % poly.len();
        }
        if maximum > squared_epsilon * squared {
            stack.push((farthest, b));
            stack.push((a, farthest));
        } else {
            result.push(poly[a]);
        }
    }
    let count = result.len();
    if count <= 2 {
        return result;
    }
    let mut remaining = count;
    let mut position = 1;
    let mut write = 0;
    let mut start = result[count - 1];
    let mut point = result[0];
    let mut i = 0;
    while i < count && remaining > 2 {
        let end = result[position];
        position = (position + 1) % count;
        let delta = sub(end, start);
        let distance = cross(sub(point, start), delta).abs();
        let successive = dot(sub(point, start), sub(end, point));
        if distance * distance <= 0.5 * squared_epsilon * dot(delta, delta)
            && delta[0] != 0.
            && delta[1] != 0.
            && successive >= 0.
        {
            remaining -= 1;
            start = end;
            result[write] = start;
            write = (write + 1) % count;
            point = result[position];
            position = (position + 1) % count;
            i += 2;
        } else {
            start = point;
            result[write] = start;
            write = (write + 1) % count;
            point = end;
            i += 1;
        }
    }
    result.truncate(remaining);
    result
}

fn custom_vertices(poly: &[Xy], max_distance: f64) -> Vec<Xy> {
    let n = poly.len();
    if n < 3 {
        return poly.to_vec();
    }
    let info: Vec<_> = (0..n)
        .map(|i| {
            let (a, b) = (
                sub(poly[(i + n - 1) % n], poly[i]),
                sub(poly[(i + 1) % n], poly[i]),
            );
            let angle = (dot(a, b) / (norm(a) * norm(b)))
                .clamp(-1., 1.)
                .acos()
                .to_degrees();
            (cross([-a[0], -a[1]], b) < 0., angle, a, b)
        })
        .collect();
    let concave: Vec<_> = (0..n).filter(|&i| !info[i].0).collect();
    let mut groups = Vec::new();
    let mut group = Vec::new();
    for i in concave.iter().copied() {
        if group.last().is_some_and(|&previous| i != previous + 1) {
            if group.len() >= 2 {
                groups.extend(&group);
            }
            group.clear();
        }
        group.push(i);
    }
    if group.len() >= 2 {
        groups.extend(group);
    }
    if concave.first() == Some(&0)
        && concave.last() == Some(&(n - 1))
        && !(groups.contains(&0) && groups.contains(&(n - 1)))
    {
        groups.clear();
    }
    let kept: Vec<_> = (0..n)
        .filter(|&i| info[i].0 || (groups.contains(&i) && info[i].1 >= 120.))
        .collect();
    let mut selected = vec![false; n];
    for (pos, &i) in kept.iter().enumerate() {
        selected[i] = true;
        let j = kept[(pos + 1) % kept.len()];
        let distance = norm(sub(poly[i], poly[j]));
        let limit = max_distance * 0.3;
        if distance > limit && limit > 0. {
            let intermediate: Vec<_> = (1..(j + n - i) % n)
                .map(|offset| (i + offset) % n)
                .collect();
            let needed = (distance / limit).ceil() as usize - 1;
            if intermediate.len() <= needed {
                for k in intermediate {
                    selected[k] = true;
                }
            } else {
                for k in 0..needed {
                    selected[intermediate[k * intermediate.len() / needed]] = true;
                }
            }
        }
    }
    (0..n)
        .filter(|&i| selected[i])
        .map(|i| {
            let (convex, angle, a, b) = info[i];
            if convex && (angle - 45.).abs() < 1. {
                let direction = [
                    a[0] / norm(a) + b[0] / norm(b),
                    a[1] / norm(a) + b[1] / norm(b),
                ];
                let distance = (norm(a) + norm(b)) * 0.5;
                [
                    poly[i][0] + direction[0] / norm(direction) * distance,
                    poly[i][1] + direction[1] / norm(direction) * distance,
                ]
            } else {
                poly[i]
            }
        })
        .collect()
}

pub(crate) fn mask_polygon(
    mask: &[u8],
    width: usize,
    height: usize,
    bbox: [f32; 4],
    page_size: (u32, u32),
    max_box_width: f64,
) -> Option<BoundingBox> {
    if width == 0
        || height == 0
        || mask.len() != width * height
        || page_size.0 == 0
        || page_size.1 == 0
    {
        return None;
    }
    let bounds = bbox.map(|p| p.trunc() as i32);
    let w = bounds[2] - bounds[0];
    let h = bounds[3] - bounds[1];
    if w <= 0 || h <= 0 {
        return None;
    }
    let (w, h) = (w as usize, h as usize);
    let coordinates = [
        (bounds[0] as f64 * width as f64 / page_size.0 as f64)
            .round_ties_even()
            .clamp(0., width as f64) as usize,
        (bounds[1] as f64 * height as f64 / page_size.1 as f64)
            .round_ties_even()
            .clamp(0., height as f64) as usize,
        (bounds[2] as f64 * width as f64 / page_size.0 as f64)
            .round_ties_even()
            .clamp(0., width as f64) as usize,
        (bounds[3] as f64 * height as f64 / page_size.1 as f64)
            .round_ties_even()
            .clamp(0., height as f64) as usize,
    ];
    if coordinates[2] <= coordinates[0] || coordinates[3] <= coordinates[1] {
        return None;
    }
    let mut resized = vec![0; w.checked_mul(h)?];
    for y in 0..h {
        for x in 0..w {
            let sx = coordinates[0] + x * (coordinates[2] - coordinates[0]) / w;
            let sy = coordinates[1] + y * (coordinates[3] - coordinates[1]) / h;
            resized[y * w + x] = mask[sy * width + sx];
        }
    }
    let contour = outer_contour(&resized, w, h);
    let approximated = approximate_closed(&contour);
    let max_distance = if w as f64 > max_box_width * 0.6 {
        w as f64
    } else {
        max_box_width
    };
    let polygon = custom_vertices(&approximated, max_distance);
    if polygon.len() < 4 {
        return None;
    }
    Some(bounding(
        polygon
            .into_iter()
            .map(|p| [p[0] + bounds[0] as f64, p[1] + bounds[1] as f64])
            .collect(),
    ))
}

pub(crate) use super::raster::mask_crop;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_auto_geometry_and_crop_pixels_match_reference() {
        // Tiny synthetic inputs evaluated with PaddleX 3.7.1 and OpenCV 4.13.
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/layout_reference.json")).unwrap();
        let mut failures = Vec::new();
        for case in reference["masks"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as usize;
            let height = case["height"].as_u64().unwrap() as usize;
            let mask: Vec<_> = case["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let bbox = [0., 0., width as f32, height as f32];
            let mode = match case["mode"].as_str().unwrap() {
                "rect" => LayoutShapeMode::Rect,
                "quad" => LayoutShapeMode::Quad,
                "poly" => LayoutShapeMode::Poly,
                _ => LayoutShapeMode::Auto,
            };
            let polygon = mask_polygon(
                &mask,
                width,
                height,
                bbox,
                (width as u32, height as u32),
                width as f64,
            );
            let actual = normalize_polygon(bbox, polygon, mode, None);
            let expected: Vec<Point> = case["polygon"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| Point::new(p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32))
                .collect();
            if actual.points.len() != expected.len()
                || actual
                    .points
                    .iter()
                    .zip(&expected)
                    .any(|(a, b)| (a.x - b.x).abs() > 1e-4 || (a.y - b.y).abs() > 1e-4)
            {
                failures.push(format!(
                    "{} {}: actual {:?}; expected {:?}",
                    case["name"], case["mode"], actual.points, expected
                ));
            }
        }
        for case in reference["crops"].as_array().unwrap() {
            let width = case["width"].as_u64().unwrap() as u32;
            let height = case["height"].as_u64().unwrap() as u32;
            let polygon = BoundingBox::new(
                case["polygon"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        Point::new(p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32)
                    })
                    .collect(),
            );
            let mut crop = image::RgbImage::new(width, height);
            mask_crop(&mut crop, &polygon, (0, 0));
            let actual: Vec<_> = crop.pixels().map(|p| u8::from(p.0 == [0; 3])).collect();
            let expected: Vec<_> = case["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            if actual != expected {
                failures.push(format!(
                    "crop {}: actual {:?}; expected {:?}",
                    case["name"], actual, expected
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));

        let bowtie = bounding(vec![[0., 0.], [2., 2.], [0., 2.], [2., 0.]]);
        let lower = BoundingBox::from_coords(0., 0., 2., 1.);
        assert!((polygon_overlap(&bowtie, &lower, true) - 0.5).abs() < 1e-10);
        let reversed = BoundingBox::new(bowtie.points.iter().rev().copied().collect());
        assert_eq!(polygon_overlap(&reversed, &lower, false), 0.);
        let concave = bounding(vec![
            [0., 0.],
            [8., 0.],
            [8., 2.],
            [2., 2.],
            [2., 8.],
            [0., 8.],
        ]);
        let rect = BoundingBox::from_coords(1., 1., 4., 4.);
        assert!((polygon_overlap(&concave, &rect, false) - 5. / 9.).abs() < 1e-10);
    }
}
