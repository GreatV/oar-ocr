//! Line quads from a detector that predicts shrunk text-line kernels.
//!
//! Each connected component of the probability map is a line's kernel: its minimum-area rectangle is grown back by the shrink ratio the training targets used, then collinear neighbours are joined. Adapted from LiteParse's ocria-nano detector (Apache-2.0). See the NOTICE beside this file.
use super::{BoundingBox, Point};

type Quad = [[f32; 2]; 4];

/// Kernel shrink ratio the detector targets were built with.
const SHRINK: f32 = 0.3;
/// Collinear boxes are joined across gaps under this many line heights.
const MERGE_GAP: f32 = 0.6;
/// Regions above this probability that touch no line are faint candidates.
const FAINT_THRESHOLD: f32 = 0.1;
/// Most faint candidates returned for one page, strongest first.
const MAX_FAINT: usize = 24;

/// Lines of one probability map, in image pixels.
pub(crate) struct LineKernels {
    /// Line quads ordered by 8-pixel row band, then left edge.
    pub lines: Vec<BoundingBox>,
    /// Lowest mean kernel probability among the components of each line.
    pub scores: Vec<f32>,
    /// Short-token candidates the detector scored under its threshold.
    pub faint: Vec<BoundingBox>,
}

/// How a probability map relates to the image it was predicted from.
pub(crate) struct MapGeometry {
    /// Probability map width and height.
    pub map: (usize, usize),
    /// Detector input width and height.
    pub input: (usize, usize),
    /// Source image width and height.
    pub source: (u32, u32),
    /// Margin added on every side of the source before resizing.
    pub pad: u32,
}

pub(crate) fn detect(
    prob: &[f32],
    geometry: &MapGeometry,
    threshold: f32,
    min_score: f32,
    with_faint: bool,
) -> LineKernels {
    let (pw, ph) = geometry.map;
    let (w, h) = geometry.source;
    let pad = geometry.pad as f32;
    let sx = geometry.input.0 as f64 / f64::from(w + 2 * geometry.pad);
    let sy = geometry.input.1 as f64 / f64::from(h + 2 * geometry.pad);
    let stride = (geometry.input.1 / ph) as f32;
    let mask: Vec<bool> = prob.iter().map(|&p| p > threshold).collect();
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    for comp in components4(&mask, pw, ph) {
        if comp.pixels.len() < 2 {
            continue;
        }
        let mean = comp
            .pixels
            .iter()
            .map(|&(x, y)| prob[y * pw + x] as f64)
            .sum::<f64>()
            / comp.pixels.len() as f64;
        if (mean as f32) < min_score {
            continue;
        }
        // The hull of the component's cells needs only each row's leftmost and rightmost cell.
        let mut span = vec![(usize::MAX, 0usize); comp.y1 - comp.y0 + 1];
        for &(x, y) in &comp.pixels {
            let r = &mut span[y - comp.y0];
            r.0 = r.0.min(x);
            r.1 = r.1.max(x);
        }
        let mut points = Vec::with_capacity(span.len() * 4);
        for (i, &(x0, x1)) in span.iter().enumerate() {
            if x0 == usize::MAX {
                continue;
            }
            let y = (comp.y0 + i) as f32;
            for (px, py) in [
                (x0 as f32, y),
                (x0 as f32, y + 1.0),
                (x1 as f32 + 1.0, y),
                (x1 as f32 + 1.0, y + 1.0),
            ] {
                points.push((px * stride, py * stride));
            }
        }
        let mut rect = min_area_rect(&points);
        // Line height is the side nearer vertical: a single digit's kernel is taller than wide.
        let w_is_horizontal = ((rect.angle + 90.0).rem_euclid(180.0) - 90.0).abs() < 45.0;
        let kernel_h = if w_is_horizontal { rect.h } else { rect.w };
        let grow = SHRINK * (kernel_h / (1.0 - 2.0 * SHRINK));
        rect.w += 2.0 * grow;
        rect.h += 2.0 * grow;
        let mut quad = box_points(&rect);
        for p in &mut quad {
            p[0] = ((p[0] as f64 / sx) as f32 - pad).clamp(0.0, w as f32);
            p[1] = ((p[1] as f64 / sy) as f32 - pad).clamp(0.0, h as f32);
        }
        boxes.push(order_quad(&quad));
        scores.push(mean as f32);
    }
    let (boxes, scores) = merge_collinear(&boxes, &scores, MERGE_GAP);
    let key = |q: &Quad| {
        (
            (extreme(q, 1, false) as f64 / 8.0).round_ties_even(),
            extreme(q, 0, false),
        )
    };
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        let (ka, kb) = (key(&boxes[a]), key(&boxes[b]));
        ka.0.total_cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
    });
    let boxes: Vec<Quad> = order.iter().map(|&i| boxes[i]).collect();
    let faint = if with_faint {
        let to_image = |x: usize, y: usize| {
            (
                ((x as f32 * stride) as f64 / sx) as f32 - pad,
                ((y as f32 * stride) as f64 / sy) as f32 - pad,
            )
        };
        faint_boxes(prob, &mask, (pw, ph), to_image, &boxes, (w, h))
    } else {
        Vec::new()
    };
    let bbox = |q: &Quad| BoundingBox::new(q.iter().map(|p| Point::new(p[0], p[1])).collect());
    LineKernels {
        lines: boxes.iter().map(bbox).collect(),
        scores: order.iter().map(|&i| scores[i]).collect(),
        faint: faint.iter().map(bbox).collect(),
    }
}

fn extreme(q: &Quad, axis: usize, max: bool) -> f32 {
    let values = q.iter().map(|p| p[axis]);
    if max {
        values.fold(f32::NEG_INFINITY, f32::max)
    } else {
        values.fold(f32::INFINITY, f32::min)
    }
}

fn norm(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn center(q: &Quad) -> [f32; 2] {
    [
        (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4.0,
        (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4.0,
    ]
}

/// Faint short tokens beside detected lines.
///
/// A lone narrow glyph in a table cell looks like a rule fragment, which the detector learned to suppress. Regions above [`FAINT_THRESHOLD`] that touch no detected line become candidates when they sit in the middle of a level line's row and are no taller than that line.
fn faint_boxes(
    prob: &[f32],
    mask: &[bool],
    (pw, ph): (usize, usize),
    to_image: impl Fn(usize, usize) -> (f32, f32),
    boxes: &[Quad],
    (w, h): (u32, u32),
) -> Vec<Quad> {
    // A candidate keeps the side margin a detected line has around its text.
    const PAD: f32 = 0.35;
    let lines: Vec<[f32; 4]> = boxes
        .iter()
        .map(|q| {
            [
                extreme(q, 0, false),
                extreme(q, 1, false),
                extreme(q, 0, true),
                extreme(q, 1, true),
            ]
        })
        .collect();
    // Lines that run level, so their box is the row's band.
    let rows: Vec<[f32; 4]> = boxes
        .iter()
        .zip(&lines)
        .filter(|(q, l)| (q[1][1] - q[0][1]).abs() <= 0.1 * (l[2] - l[0]))
        .map(|(_, &l)| l)
        .collect();
    if rows.is_empty() {
        return Vec::new();
    }
    let weak: Vec<bool> = prob.iter().map(|&p| p > FAINT_THRESHOLD).collect();
    let mut candidates: Vec<(f32, Quad)> = Vec::new();
    for comp in components4(&weak, pw, ph) {
        if comp.pixels.iter().any(|&(x, y)| mask[y * pw + x]) {
            continue;
        }
        let (x0, y0) = to_image(comp.x0, comp.y0);
        let (x1, y1) = to_image(comp.x1 + 1, comp.y1 + 1);
        let cy = (y0 + y1) / 2.0;
        // The row is a level line whose middle half holds the region's centre.
        let Some(&[_, ry0, _, ry1]) = rows
            .iter()
            .filter(|r| {
                let h = r[3] - r[1];
                cy >= r[1] + 0.25 * h && cy <= r[3] - 0.25 * h
            })
            .min_by(|a, b| {
                (a[1] + a[3] - 2.0 * cy)
                    .abs()
                    .total_cmp(&(b[1] + b[3] - 2.0 * cy).abs())
            })
        else {
            continue;
        };
        let row_h = ry1 - ry0;
        // A short token's core is lower than its row's line box and at most a few glyphs wide.
        if y1 - y0 > 0.8 * row_h || x1 - x0 > 1.5 * row_h {
            continue;
        }
        let (bx0, bx1) = (
            (x0 - PAD * row_h).max(0.0),
            (x1 + PAD * row_h).min(w as f32),
        );
        let (by0, by1) = (ry0.max(0.0), ry1.min(h as f32));
        let area = (bx1 - bx0) * (by1 - by0);
        // A gap between words, an accent or a descender is part of a detected line.
        let covered = lines.iter().any(|l| {
            let ix = (bx1.min(l[2]) - bx0.max(l[0])).max(0.0);
            let iy = (by1.min(l[3]) - by0.max(l[1])).max(0.0);
            ix * iy > 0.2 * area
        });
        if covered {
            continue;
        }
        let peak = comp
            .pixels
            .iter()
            .map(|&(x, y)| prob[y * pw + x])
            .fold(0f32, f32::max);
        candidates.push((peak, [[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]]));
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(MAX_FAINT);
    candidates.into_iter().map(|(_, q)| q).collect()
}

/// Corners as [top-left, top-right, bottom-right, bottom-left] along the line's long axis.
fn order_quad(q: &Quad) -> Quad {
    let c = center(q);
    let e1 = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
    let e2 = [q[2][0] - q[1][0], q[2][1] - q[1][1]];
    let mut u = if norm(e1) >= norm(e2) { e1 } else { e2 };
    let n = norm(u) + 1e-6;
    u = [u[0] / n, u[1] / n];
    if u[0] < 0.0 {
        u = [-u[0], -u[1]];
    }
    let v = [-u[1], u[0]];
    let a = q.map(|p| (p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1]);
    let b = q.map(|p| (p[0] - c[0]) * v[0] + (p[1] - c[1]) * v[1]);
    // The first extreme index, as np.argmin and np.argmax return.
    let pick = |f: &dyn Fn(usize) -> f32, max: bool| {
        let mut best = 0;
        for i in 1..4 {
            if (max && f(i) > f(best)) || (!max && f(i) < f(best)) {
                best = i;
            }
        }
        best
    };
    let tl = pick(&|i| a[i] + b[i], false);
    let tr = pick(&|i| a[i] - b[i], true);
    let br = pick(&|i| a[i] + b[i], true);
    let bl = pick(&|i| a[i] - b[i], false);
    [q[tl], q[tr], q[br], q[bl]]
}

/// Join near-horizontal line quads that continue one another.
///
/// Each box links to its nearest qualifying right neighbour when that neighbour's nearest left neighbour is the same box. Each chain becomes the rectangle spanning it along its longest member's axis, scored by its weakest member.
fn merge_collinear(boxes: &[Quad], scores: &[f32], gap: f32) -> (Vec<Quad>, Vec<f32>) {
    const MAX_TILT: f32 = 0.35;
    const MAX_PERP: f32 = 0.35;
    const MIN_H_RATIO: f32 = 0.6;
    const MAX_DANGLE: f32 = 0.07;
    let n = boxes.len();
    if n < 2 {
        return (boxes.to_vec(), scores.to_vec());
    }
    let c: Vec<[f32; 2]> = boxes.iter().map(center).collect();
    let mut u: Vec<[f32; 2]> = Vec::with_capacity(n);
    let mut len: Vec<f32> = Vec::with_capacity(n);
    let mut hgt: Vec<f32> = Vec::with_capacity(n);
    for q in boxes {
        let d = [q[1][0] - q[0][0], q[1][1] - q[0][1]];
        let l = norm(d).max(1e-3);
        u.push([d[0] / l, d[1] / l]);
        len.push(l);
        hgt.push(norm([q[3][0] - q[0][0], q[3][1] - q[0][1]]).max(1e-3));
    }
    let v: Vec<[f32; 2]> = u.iter().map(|u| [-u[1], u[0]]).collect();
    let flat: Vec<bool> = (0..n)
        .map(|i| u[i][1].abs() <= MAX_TILT && len[i] >= 0.5 * hgt[i])
        .collect();
    let cos_min = 1.0 - MAX_DANGLE * MAX_DANGLE / 2.0;
    let mut right = vec![None; n];
    let mut left_best: Vec<(f32, Option<usize>)> = vec![(f32::INFINITY, None); n];
    for a in 0..n {
        let mut best = (f32::INFINITY, None);
        if flat[a] {
            for b in 0..n {
                if a == b || !flat[b] {
                    continue;
                }
                let d = [c[b][0] - c[a][0], c[b][1] - c[a][1]];
                let along = d[0] * u[a][0] + d[1] * u[a][1];
                let perp = (d[0] * v[a][0] + d[1] * v[a][1]).abs();
                let hm = (hgt[a] + hgt[b]) / 2.0;
                let g = along - (len[a] + len[b]) / 2.0;
                let ok = along > 0.0
                    && hgt[a].min(hgt[b]) / hgt[a].max(hgt[b]) >= MIN_H_RATIO
                    && (u[a][0] * u[b][0] + u[a][1] * u[b][1]).abs() >= cos_min
                    && perp <= MAX_PERP * hm
                    && g <= gap * hm
                    && g >= -0.5 * len[a].min(len[b]);
                if ok {
                    if g < best.0 {
                        best = (g, Some(b));
                    }
                    // The first box reaching each neighbour's minimum gap.
                    if g < left_best[b].0 {
                        left_best[b] = (g, Some(a));
                    }
                }
            }
        }
        right[a] = best.1;
    }
    let next: Vec<Option<usize>> = (0..n)
        .map(|a| right[a].filter(|&b| left_best[b].1 == Some(a)))
        .collect();
    let mut has_prev = vec![false; n];
    for &b in next.iter().flatten() {
        has_prev[b] = true;
    }
    let mut out = Vec::new();
    let mut out_scores = Vec::new();
    let mut seen = vec![false; n];
    for a in 0..n {
        if has_prev[a] || seen[a] {
            continue;
        }
        let mut chain = vec![a];
        seen[a] = true;
        while let Some(b) = next[chain[chain.len() - 1]].filter(|&b| !seen[b]) {
            chain.push(b);
            seen[b] = true;
        }
        out_scores.push(
            chain
                .iter()
                .map(|&i| scores[i])
                .fold(f32::INFINITY, f32::min),
        );
        if chain.len() == 1 {
            out.push(boxes[a]);
            continue;
        }
        let points: Vec<[f32; 2]> = chain.iter().flat_map(|&i| boxes[i]).collect();
        // The chain runs along its longest member, the first one on ties.
        let mut k = chain[0];
        for &i in &chain[1..] {
            if len[i] > len[k] {
                k = i;
            }
        }
        let count = points.len() as f32;
        let ca = [
            points.iter().map(|p| p[0]).sum::<f32>() / count,
            points.iter().map(|p| p[1]).sum::<f32>() / count,
        ];
        let (ua, va) = (u[k], v[k]);
        let mut along = (f32::INFINITY, f32::NEG_INFINITY);
        let mut across = (f32::INFINITY, f32::NEG_INFINITY);
        for p in &points {
            let (dx, dy) = (p[0] - ca[0], p[1] - ca[1]);
            let pa = dx * ua[0] + dy * ua[1];
            let pb = dx * va[0] + dy * va[1];
            along = (along.0.min(pa), along.1.max(pa));
            across = (across.0.min(pb), across.1.max(pb));
        }
        let at = |s: f32, t: f32| [ca[0] + ua[0] * s + va[0] * t, ca[1] + ua[1] * s + va[1] * t];
        out.push(order_quad(&[
            at(along.0, across.0),
            at(along.1, across.0),
            at(along.1, across.1),
            at(along.0, across.1),
        ]));
    }
    // Boxes left on a cycle of links stay as they are.
    for i in 0..n {
        if !seen[i] {
            out.push(boxes[i]);
            out_scores.push(scores[i]);
        }
    }
    (out, out_scores)
}

/// One 4-connected component of a binary mask: its bounding box and pixels.
struct Component {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    pixels: Vec<(usize, usize)>,
}

/// 4-connected components of a row-major mask, in raster order of each component's first pixel.
fn components4(mask: &[bool], w: usize, h: usize) -> Vec<Component> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut comp = Component {
            x0: usize::MAX,
            y0: usize::MAX,
            x1: 0,
            y1: 0,
            pixels: Vec::new(),
        };
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            comp.x0 = comp.x0.min(x);
            comp.y0 = comp.y0.min(y);
            comp.x1 = comp.x1.max(x);
            comp.y1 = comp.y1.max(y);
            comp.pixels.push((x, y));
            let mut visit = |q: usize| {
                if mask[q] && !seen[q] {
                    seen[q] = true;
                    stack.push(q);
                }
            };
            if x > 0 {
                visit(p - 1);
            }
            if x + 1 < w {
                visit(p + 1);
            }
            if y > 0 {
                visit(p - w);
            }
            if y + 1 < h {
                visit(p + w);
            }
        }
        out.push(comp);
    }
    out
}

/// A rotated rectangle: centre, side lengths along `angle` (w) and its normal (h), and the angle in degrees.
#[derive(Clone, Copy)]
struct RotatedRect {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    angle: f32,
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// The smallest-area rectangle around `points`: one candidate per convex hull edge, the first on ties.
fn min_area_rect(points: &[(f32, f32)]) -> RotatedRect {
    let mut p: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    p.dedup();
    let point = RotatedRect {
        cx: p[0].0 as f32,
        cy: p[0].1 as f32,
        w: 0.0,
        h: 0.0,
        angle: 0.0,
    };
    if p.len() == 1 {
        return point;
    }
    // Monotone chain.
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(p.len() * 2);
    for &q in &p {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
            hull.pop();
        }
        hull.push(q);
    }
    let lower = hull.len() + 1;
    for &q in p.iter().rev().skip(1) {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
            hull.pop();
        }
        hull.push(q);
    }
    hull.pop();
    let mut best: Option<(f64, RotatedRect)> = None;
    for i in 0..hull.len() {
        let (a, b) = (hull[i], hull[(i + 1) % hull.len()]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = dx.hypot(dy);
        if len == 0.0 {
            continue;
        }
        let (ux, uy) = (dx / len, dy / len);
        let (vx, vy) = (-uy, ux);
        let mut u = (f64::INFINITY, f64::NEG_INFINITY);
        let mut v = (f64::INFINITY, f64::NEG_INFINITY);
        for &q in &hull {
            let pu = q.0 * ux + q.1 * uy;
            let pv = q.0 * vx + q.1 * vy;
            u = (u.0.min(pu), u.1.max(pu));
            v = (v.0.min(pv), v.1.max(pv));
        }
        let area = (u.1 - u.0) * (v.1 - v.0);
        if best.as_ref().is_none_or(|(least, _)| area < *least) {
            let (cu, cv) = ((u.0 + u.1) / 2.0, (v.0 + v.1) / 2.0);
            best = Some((
                area,
                RotatedRect {
                    cx: (cu * ux + cv * vx) as f32,
                    cy: (cu * uy + cv * vy) as f32,
                    w: (u.1 - u.0) as f32,
                    h: (v.1 - v.0) as f32,
                    angle: uy.atan2(ux).to_degrees() as f32,
                },
            ));
        }
    }
    best.map_or(point, |b| b.1)
}

/// `cv2.boxPoints`: the rectangle's four corners.
fn box_points(r: &RotatedRect) -> Quad {
    let a = (r.angle as f64).to_radians();
    let (b, c) = ((a.cos() * 0.5) as f32, (a.sin() * 0.5) as f32);
    let p0 = [r.cx - c * r.h - b * r.w, r.cy + b * r.h - c * r.w];
    let p1 = [r.cx + c * r.h - b * r.w, r.cy - b * r.h - c * r.w];
    let p2 = [2.0 * r.cx - p0[0], 2.0 * r.cy - p0[1]];
    let p3 = [2.0 * r.cx - p1[0], 2.0 * r.cy - p1[1]];
    [p0, p1, p2, p3]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 60 x 20 probability map at one image pixel per cell with one detected line, whose box is 20 px tall.
    fn page() -> (Vec<f32>, Vec<bool>, Vec<Quad>) {
        let mut prob = vec![0f32; 60 * 20];
        for y in 8..12 {
            for x in 2..20 {
                prob[y * 60 + x] = 0.9;
            }
        }
        let mask = prob.iter().map(|&p| p > 0.55).collect();
        let line = [[0.0, 0.0], [22.0, 0.0], [22.0, 20.0], [0.0, 20.0]];
        (prob, mask, vec![line])
    }

    fn faint(prob: &[f32], mask: &[bool], boxes: &[Quad]) -> Vec<Quad> {
        let to_image = |x: usize, y: usize| (x as f32, y as f32);
        faint_boxes(prob, mask, (60, 20), to_image, boxes, (60, 20))
    }

    #[test]
    fn faint_token_in_a_line_row_is_a_candidate() {
        let (mut prob, mask, boxes) = page();
        // A weak "1" in the same row, well right of the line.
        for y in 8..12 {
            prob[y * 60 + 40] = 0.13;
        }
        let found = faint(&prob, &mask, &boxes);
        assert_eq!(found.len(), 1);
        // The row's band, with a side margin around the region.
        assert_eq!((found[0][0][1], found[0][2][1]), (0.0, 20.0));
        assert!(found[0][0][0] < 40.0 && found[0][1][0] > 41.0);
    }

    #[test]
    fn faint_regions_off_the_row_touching_a_line_or_too_tall_are_not() {
        let (mut prob, mask, boxes) = page();
        // Centred below the line's middle half.
        prob[18 * 60 + 40] = 0.2;
        prob[19 * 60 + 40] = 0.2;
        // Touching the detected line.
        prob[10 * 60 + 20] = 0.2;
        // Taller than 0.8 of the row.
        for y in 0..20 {
            prob[y * 60 + 50] = 0.2;
        }
        assert!(faint(&prob, &mask, &boxes).is_empty());
    }

    #[test]
    fn collinear_words_merge_into_one_line_and_a_distant_word_stays() {
        let word = |x: f32, y: f32| [[x, y], [x + 40.0, y], [x + 40.0, y + 10.0], [x, y + 10.0]];
        // The second word starts 4 px after the first, the third 30 px after the second.
        let boxes = [word(0.0, 0.0), word(44.0, 0.0), word(114.0, 0.0)];
        let (merged, scores) = merge_collinear(&boxes, &[0.9, 0.7, 0.8], MERGE_GAP);
        assert_eq!(merged.len(), 2);
        assert_eq!(
            merged[0],
            [[0.0, 0.0], [84.0, 0.0], [84.0, 10.0], [0.0, 10.0]]
        );
        assert_eq!(merged[1], boxes[2]);
        assert_eq!(scores, [0.7, 0.8]);
    }
}
