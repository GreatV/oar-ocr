//! Line-box stages before recognition: upright crops, sideways-page tests and column-gutter splits.
//!
//! Adapted from LiteParse's ocria-nano runtime (Apache-2.0). See the NOTICE beside this file.
use super::geometry::{Point, Quad};
use image::{GrayImage, Luma, RgbImage, imageops};
use oar_ocr_core::processors::raster::scalar;
use std::collections::HashSet;

fn points(q: &Quad) -> [[f32; 2]; 4] {
    q.map(|p| [p.x, p.y])
}

fn distance(a: Point, b: Point) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// Perspective-crop a quad and turn a crop at least 1.5 times taller than wide a quarter turn counter-clockwise.
pub(super) fn crop(img: &RgbImage, q: &Quad) -> RgbImage {
    let w = (distance(q[0], q[1]).max(distance(q[2], q[3])) as u32).max(2);
    let h = (distance(q[0], q[3]).max(distance(q[1], q[2])) as u32).max(2);
    let (fw, fh) = (w as f32, h as f32);
    let target = [[0.0, 0.0], [fw, 0.0], [fw, fh], [0.0, fh]];
    let m = scalar::perspective_transform(&points(q), &target);
    let crop = scalar::warp_perspective_cubic(img, &m, w, h);
    if f64::from(h) / f64::from(w) >= 1.5 {
        imageops::rotate270(&crop)
    } else {
        crop
    }
}

/// Left, top, right and bottom of a quad.
fn extent(q: &Quad) -> [f64; 4] {
    let fold = |axis: usize, init: f64, f: fn(f64, f64) -> f64| {
        q.iter().map(|p| f64::from(p[axis])).fold(init, f)
    };
    [
        fold(0, f64::INFINITY, f64::min),
        fold(1, f64::INFINITY, f64::min),
        fold(0, f64::NEG_INFINITY, f64::max),
        fold(1, f64::NEG_INFINITY, f64::max),
    ]
}

/// Boxes that are clearly horizontal text lines: at least 30 px wide and three times wider than tall.
pub(super) fn long_lines(boxes: &[Quad]) -> usize {
    boxes
        .iter()
        .filter(|q| {
            let [x0, y0, x1, y1] = extent(q);
            x1 - x0 >= 30.0 && x1 - x0 >= 3.0 * (y1 - y0)
        })
        .count()
}

/// Detector boxes of a page scanned on its side: many boxes, almost none of them horizontal lines.
pub(super) fn looks_sideways(boxes: &[Quad]) -> bool {
    boxes.len() >= 15 && (long_lines(boxes) as f64) < 0.25 * boxes.len() as f64
}

/// The first level maximising Otsu's between-class variance.
fn otsu(values: impl Iterator<Item = u8>) -> u8 {
    let mut hist = [0f64; 256];
    let mut n = 0f64;
    for v in values {
        hist[v as usize] += 1.0;
        n += 1.0;
    }
    let n = n.max(1.0);
    let (mut w, mut mu) = (0f64, 0f64);
    let mut cum = [(0f64, 0f64); 256];
    for i in 0..256 {
        w += hist[i] / n;
        mu += hist[i] / n * i as f64;
        cum[i] = (w, mu);
    }
    let total = cum[255].1;
    let mut best = (f64::NEG_INFINITY, 0u8);
    for (i, &(w, mu)) in cum.iter().enumerate() {
        let s = (total * w - mu).powi(2) / (w * (1.0 - w));
        if s.is_finite() && s > best.0 {
            best = (s, i as u8);
        }
    }
    best.1
}

/// Minimum blank-run width of a column gutter, in line heights.
const GUTTER_MIN: f64 = 0.4;

/// Cut near-horizontal boxes at blank vertical runs that line up with other boxes' edges and stay clear in the neighbouring lines.
pub(super) fn split_gutters(img: &RgbImage, boxes: &[Quad]) -> Vec<Quad> {
    if boxes.len() < 3 {
        return boxes.to_vec();
    }
    // The channel mean, truncated.
    let gray = GrayImage::from_fn(img.width(), img.height(), |x, y| {
        Luma([(img[(x, y)].0.iter().map(|&v| u32::from(v)).sum::<u32>() / 3) as u8])
    });
    let (gw, gh) = (i64::from(gray.width()), i64::from(gray.height()));
    let px = |x: i64, y: i64| gray.as_raw()[(y * gw + x) as usize];
    let extents: Vec<[f64; 4]> = boxes.iter().map(extent).collect();
    let mut out = Vec::with_capacity(boxes.len());
    for (i, q) in boxes.iter().enumerate() {
        let [x0, y0, x1, y1] = extents[i];
        let h = y1 - y0;
        let tilt = (f64::from(q[1].y) - f64::from(q[0].y)).abs();
        if h < 6.0 || x1 - x0 < 4.0 * h || tilt > 0.25 * h {
            out.push(*q);
            continue;
        }
        let ya = ((y0 + 0.15 * h) as i64).max(0).min(gh);
        let yb = ((y1 - 0.15 * h).ceil() as i64).clamp(0, gh);
        let xa = (x0 as i64).max(0);
        let xb = (x1.ceil() as i64).min(gw);
        if yb <= ya || xb <= xa {
            out.push(*q);
            continue;
        }
        let window = move || (ya..yb).flat_map(move |y| (xa..xb).map(move |x| (x, y)));
        let (lo, hi) = window().fold((255u8, 0u8), |(lo, hi), (x, y)| {
            (lo.min(px(x, y)), hi.max(px(x, y)))
        });
        if i32::from(hi) - i32::from(lo) < 60 {
            out.push(*q);
            continue;
        }
        let threshold = otsu(window().map(|(x, y)| px(x, y)));
        // A printed column rule belongs to the gutter, not the text.
        let rule = rule_columns(&px, (xa, xb), (y0, y1), (gw, gh), threshold);
        // Ink columns relative to `xa`.
        let ink: Vec<i64> = (xa..xb)
            .filter(|&x| !rule.contains(&x) && (ya..yb).any(|y| px(x, y) < threshold))
            .map(|x| x - xa)
            .collect();
        if ink.len() < 2 {
            out.push(*q);
            continue;
        }
        let (cy, tolerance) = ((y0 + y1) / 2.0, 0.6 * h);
        let line_height = |e: &[f64; 4]| 0.5 * h < e[3] - e[1] && e[3] - e[1] < 2.0 * h;
        let mut cuts: Vec<(f64, f64)> = Vec::new();
        for pair in ink.windows(2) {
            if ((pair[1] - pair[0]) as f64) <= GUTTER_MIN * h {
                continue;
            }
            // Blank columns [a, b).
            let (a, b) = ((xa + pair[0] + 1) as f64, (xa + pair[1]) as f64);
            let (mut left, mut right) = (0, 0);
            for (j, e) in extents.iter().enumerate() {
                if j != i && ((e[1] + e[3]) / 2.0 - cy).abs() < 25.0 * h && line_height(e) {
                    left += usize::from((e[0] - b).abs() < tolerance);
                    right += usize::from((e[2] - a).abs() < tolerance);
                }
            }
            // Both sides need a column of edges; a hanging-indent marker has only one.
            if left < 1 || right < 1 || left + right < 3 || a - x0 < 3.0 * h {
                continue;
            }
            // The gutter stays clear in the neighbouring lines, judged per line.
            let margin = (((b - a) * 0.25) as i64).max(1) as f64;
            let (sa, sb) = ((a + margin) as i64, (b - margin) as i64);
            if sb <= sa {
                continue;
            }
            let (mut clear, mut inked) = (0, 0);
            for (j, e) in extents.iter().enumerate() {
                if j == i || ((e[1] + e[3]) / 2.0 - cy).abs() > 4.0 * h || !line_height(e) {
                    continue;
                }
                if e[2] <= sa as f64 || e[0] >= sb as f64 {
                    clear += 1;
                    continue;
                }
                let eh = e[3] - e[1];
                let rows = ((e[1] + 0.15 * eh) as i64).clamp(0, gh)
                    ..((e[3] - 0.15 * eh).ceil() as i64).clamp(0, gh);
                let columns = sa.clamp(0, gw)..sb.clamp(0, gw);
                let mut cells = rows.flat_map(|y| columns.clone().map(move |x| (x, y)));
                if cells.any(|(x, y)| !rule.contains(&x) && px(x, y) < threshold) {
                    inked += 1;
                } else {
                    clear += 1;
                }
            }
            if clear >= 2 && (inked == 0 || (inked == 1 && clear >= 3)) {
                cuts.push((a, b));
            }
        }
        if cuts.is_empty() {
            out.push(*q);
            continue;
        }
        let pad = 0.15 * h;
        let mut edges = vec![x0];
        for &(a, b) in &cuts {
            edges.push(a + pad);
            edges.push(b - pad);
        }
        edges.push(x1);
        for pair in edges.chunks(2) {
            let (start, end) = (pair[0], pair[1]);
            if end - start > 0.5 * h {
                let (s, e, top, bottom) = (start as f32, end as f32, y0 as f32, y1 as f32);
                out.push([
                    Point::new(s, top),
                    Point::new(e, top),
                    Point::new(e, bottom),
                    Point::new(s, bottom),
                ]);
            }
        }
    }
    out
}

/// Columns of a printed vertical rule crossing the line box `y0..y1`.
///
/// A rule is a run at most about 0.15 line heights wide that is inked down the whole line and through bands of 0.35 line heights above and below it, with blank columns on both sides. The run is widened by its anti-aliased edges. Glyph stems stop at the line.
fn rule_columns(
    px: &impl Fn(i64, i64) -> u8,
    (xa, xb): (i64, i64),
    (y0, y1): (f64, f64),
    (gw, gh): (i64, i64),
    threshold: u8,
) -> HashSet<i64> {
    let h = y1 - y0;
    let band = (0.35 * h).ceil() as i64;
    let (top, bottom) = (y0 as i64, y1.ceil() as i64);
    let inked_share = |x: i64, ya: i64, yb: i64| -> f64 {
        let (ya, yb) = (ya.clamp(0, gh), yb.clamp(0, gh));
        if yb <= ya || x < 0 || x >= gw {
            return 0.0;
        }
        (ya..yb).filter(|&y| px(x, y) < threshold).count() as f64 / (yb - ya) as f64
    };
    let solid: Vec<bool> = (xa..xb)
        .map(|x| {
            inked_share(x, top, bottom) >= 0.9
                && inked_share(x, top - band, top) >= 0.8
                && inked_share(x, bottom, bottom + band) >= 0.8
        })
        .collect();
    let max_width = ((0.15 * h).ceil() as i64).max(3);
    // On a dark panel every gap between light letters is inked top to bottom too, so a rule needs a light ground beside it.
    let flank = ((0.25 * h).ceil() as i64).max(3);
    let blank_side = |from: i64, to: i64| {
        (from..to)
            .filter(|&x| inked_share(x, top, bottom) < 0.1)
            .count() as i64
            * 2
            >= to - from
    };
    let edge = ((0.06 * h).ceil() as i64).max(2);
    let mut out = HashSet::new();
    let mut i = 0;
    while i < solid.len() {
        if !solid[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < solid.len() && solid[i] {
            i += 1;
        }
        let (run_start, run_end) = (xa + start as i64, xa + i as i64);
        if (i - start) as i64 <= max_width
            && blank_side(run_start - flank, run_start)
            && blank_side(run_end, run_end + flank)
        {
            out.extend(run_start - edge..run_end + edge);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn a_box_across_a_gutter_other_boxes_end_at_is_cut() {
        let mut img = RgbImage::from_pixel(400, 100, Rgb([255; 3]));
        let line = |x0: f32, x1: f32, top: f32| {
            [
                Point::new(x0, top),
                Point::new(x1, top),
                Point::new(x1, top + 20.),
                Point::new(x0, top + 20.),
            ]
        };
        let mut boxes = Vec::new();
        for row in 0..3u32 {
            let top = 10 + row * 30;
            for y in top + 4..top + 16 {
                // Two text blocks of dark strokes with grey edges, 4 px apart, either side of an 80 px gutter.
                for x in (20..160).chain(240..380) {
                    match x % 4 {
                        0 => img.put_pixel(x, y, Rgb([0; 3])),
                        1 => img.put_pixel(x, y, Rgb([60; 3])),
                        _ => {}
                    }
                }
            }
            if row == 0 {
                boxes.push(line(18., 382., top as f32));
            } else {
                boxes.push(line(18., 162., top as f32));
                boxes.push(line(238., 382., top as f32));
            }
        }
        let split = split_gutters(&img, &boxes);
        assert_eq!(split.len(), 6);
        // The cut leaves 0.15 line heights of margin beside the ink on each side.
        assert_eq!((split[0][0].x, split[0][1].x), (18., 160.));
        assert_eq!((split[1][0].x, split[1][1].x), (237., 382.));
        assert_eq!(split[2..], boxes[1..]);
    }
}
