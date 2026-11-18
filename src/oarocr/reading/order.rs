//! Text-conditioned column cuts and row assignment.
use super::{
    Reading,
    geometry::{Quad, line_height, median},
    text::digits,
};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct Extent {
    x0: f32,
    x1: f32,
    y0: f32,
    y1: f32,
}
impl Extent {
    fn width(self) -> f32 {
        self.x1 - self.x0
    }
    fn cy(self) -> f32 {
        (self.y0 + self.y1) * 0.5
    }
    fn cx(self) -> f32 {
        (self.x0 + self.x1) * 0.5
    }
}
fn extents(boxes: &[Quad], degrees: f64) -> Vec<Extent> {
    let (s, c) = (-degrees.to_radians()).sin_cos();
    let (s, c) = (s as f32, c as f32);
    boxes
        .iter()
        .map(|q| {
            let mut e = Extent {
                x0: f32::INFINITY,
                y0: f32::INFINITY,
                x1: f32::NEG_INFINITY,
                y1: f32::NEG_INFINITY,
            };
            for p in q {
                let x = p.x * c - p.y * s;
                let y = p.x * s + p.y * c;
                e.x0 = e.x0.min(x);
                e.x1 = e.x1.max(x);
                e.y0 = e.y0.min(y);
                e.y1 = e.y1.max(y);
            }
            e
        })
        .collect()
}
fn gaps(mut intervals: Vec<(f32, f32)>, min: f32) -> Vec<(f32, f32)> {
    intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut out = Vec::new();
    let Some(&(mut reach, _)) = intervals.first() else {
        return out;
    };
    for (lo, hi) in intervals {
        if lo - reach >= min {
            out.push((reach, lo));
        }
        reach = reach.max(hi);
    }
    out
}

struct Layout<'a> {
    ext: &'a [Extent],
    text: &'a [Reading],
    height: f32,
}
impl Layout<'_> {
    fn gutters(&self, ids: &[usize]) -> Vec<(f32, f32)> {
        gaps(
            ids.iter()
                .map(|&i| (self.ext[i].x0, self.ext[i].x1))
                .collect(),
            0.8 * self.height,
        )
    }
    fn prose(&self, ids: &[usize]) -> bool {
        if ids.len() < 12 {
            return false;
        }
        let span = ids
            .iter()
            .map(|&i| self.ext[i].x1)
            .fold(f32::NEG_INFINITY, f32::max)
            - ids
                .iter()
                .map(|&i| self.ext[i].x0)
                .fold(f32::INFINITY, f32::min);
        let widths: Vec<_> = ids.iter().map(|&i| self.ext[i].width()).collect();
        let full = widths.iter().filter(|&&w| w >= 0.75 * span).count();
        if median(widths) < 7. * self.height || (full as f64) < 0.35 * ids.len() as f64 {
            return false;
        }
        let mut n = 0;
        let mut numeric = 0;
        let mut unique = HashSet::new();
        for &i in ids {
            let s = &self.text[i].text;
            unique.insert(s.trim());
            for c in s.chars().filter(|c| !c.is_whitespace()) {
                n += 1;
                numeric += usize::from(digits(c));
            }
        }
        numeric as f64 <= 0.3 * n as f64 && unique.len() as f64 >= 0.8 * ids.len() as f64
    }
    fn list_starts(&self, col: &[usize]) -> f64 {
        let mut counts = HashMap::new();
        let mut count = 0;
        for s in col
            .iter()
            .map(|&i| self.text[i].text.trim())
            .filter(|s| !s.is_empty())
        {
            let mut digit = false;
            let mut start = String::new();
            for c in s.chars() {
                if digits(c) {
                    if digit {
                        continue;
                    }
                    start.push('D');
                    digit = true;
                } else {
                    start.push(c);
                    digit = false;
                }
                if start.chars().count() == 3 {
                    break;
                }
            }
            *counts.entry(start).or_insert(0usize) += 1;
            count += 1;
        }
        let mut counts: Vec<_> = counts.into_values().collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts.into_iter().take(2).sum::<usize>() as f64 / count.max(1) as f64
    }
    fn separate(&self, left: &[usize], right: &[usize]) -> bool {
        if !self.prose(left)
            || !self.prose(right)
            || (self.list_starts(left) + self.list_starts(right)) * 0.5 >= 0.6
        {
            return false;
        }
        let facing = left
            .iter()
            .filter(|&&i| {
                enumerator(&self.text[i].text).is_some_and(|n| {
                    right.iter().any(|&j| {
                        (self.ext[i].cy() - self.ext[j].cy()).abs() <= 0.5 * self.height
                            && enumerator(&self.text[j].text) == Some(n + 1)
                    })
                })
            })
            .count();
        2 * facing < left.len().min(right.len())
    }
    fn blocks(&self, ids: Vec<usize>, depth: u8) -> Vec<Vec<usize>> {
        if ids.len() < 6 || depth > 12 {
            return vec![ids];
        }
        let gutters = self.gutters(&ids);
        let mut candidates: Vec<_> = gutters.to_vec();
        candidates.sort_by(|a, b| (b.1 - b.0).total_cmp(&(a.1 - a.0)));
        for (x0, x1) in candidates {
            let left: Vec<_> = ids
                .iter()
                .copied()
                .filter(|&i| self.ext[i].x1 <= x0)
                .collect();
            let right: Vec<_> = ids
                .iter()
                .copied()
                .filter(|&i| self.ext[i].x0 >= x1)
                .collect();
            let lmin = self
                .gutters(&left)
                .last()
                .map_or(f32::NEG_INFINITY, |g| g.1);
            let rmax = self.gutters(&right).first().map_or(f32::INFINITY, |g| g.0);
            let l = left
                .iter()
                .copied()
                .filter(|&i| self.ext[i].x0 >= lmin)
                .collect::<Vec<_>>();
            let r = right
                .iter()
                .copied()
                .filter(|&i| self.ext[i].x1 <= rmax)
                .collect::<Vec<_>>();
            if self.separate(&l, &r) {
                let mut blocks = self.blocks(left, depth + 1);
                blocks.extend(self.blocks(right, depth + 1));
                return blocks;
            }
        }
        let mut ys = gaps(
            ids.iter()
                .map(|&i| (self.ext[i].y0, self.ext[i].y1))
                .collect(),
            0.8 * self.height,
        );
        ys.sort_by(|a, b| (b.1 - b.0).total_cmp(&(a.1 - a.0)));
        if let Some(&(lo, _)) = ys.first() {
            let (top, bottom) = ids.into_iter().partition(|&i| self.ext[i].y1 <= lo);
            let mut blocks = self.blocks(top, depth + 1);
            blocks.extend(self.blocks(bottom, depth + 1));
            return blocks;
        }
        self.banded(&ids, depth).unwrap_or_else(|| vec![ids])
    }
    fn banded(&self, ids: &[usize], depth: u8) -> Option<Vec<Vec<usize>>> {
        let lo = ids
            .iter()
            .map(|&i| self.ext[i].x0)
            .fold(f32::INFINITY, f32::min);
        let hi = ids
            .iter()
            .map(|&i| self.ext[i].x1)
            .fold(f32::NEG_INFINITY, f32::max);
        let span = hi - lo;
        let (mut wide, rest): (Vec<_>, Vec<_>) = ids
            .iter()
            .copied()
            .partition(|&i| self.ext[i].width() >= 0.55 * span);
        if wide.len() as f64 > 0.15 * ids.len() as f64 || rest.len() < 24 {
            return None;
        }
        let origin = (lo + 0.15 * span) as i32;
        let n = (0.7 * span) as usize + 1;
        let mut changes = vec![0i32; n + 1];
        for &i in &rest {
            let e = self.ext[i];
            let a = (e.x0 as i32 - origin).clamp(0, n as i32) as usize;
            let b = (e.x1.ceil() as i32 - origin).clamp(0, n as i32) as usize;
            if a < b {
                changes[a] += 1;
                changes[b] -= 1;
            }
        }
        let mut holes = Vec::new();
        let mut start = None;
        let mut count = 0;
        for (x, &d) in changes.iter().enumerate() {
            count += d;
            if x < n && count <= 1.max((0.03 * rest.len() as f64) as i32) {
                start.get_or_insert(x);
            } else if let Some(a) = start.take()
                && (x - a) as f32 >= 0.4 * self.height
            {
                holes.push(((origin + a as i32) as f32, (origin + x as i32) as f32));
            }
        }
        let mut best = None;
        for (k, &(a, b)) in holes.iter().enumerate() {
            let lower = k.checked_sub(1).map_or(f32::NEG_INFINITY, |j| holes[j].1);
            let upper = holes.get(k + 1).map_or(f32::INFINITY, |g| g.0);
            let l: Vec<_> = rest
                .iter()
                .copied()
                .filter(|&i| self.ext[i].x1 <= a && self.ext[i].x0 >= lower)
                .collect();
            let r: Vec<_> = rest
                .iter()
                .copied()
                .filter(|&i| self.ext[i].x0 >= b && self.ext[i].x1 <= upper)
                .collect();
            if self.separate(&l, &r) && best.is_none_or(|(x, y)| b - a > y - x) {
                best = Some((a, b));
            }
        }
        let (a, b) = best?;
        let (mut crossing, body): (Vec<_>, Vec<_>) = rest
            .into_iter()
            .partition(|&i| self.ext[i].x0 < a && self.ext[i].x1 > b);
        wide.append(&mut crossing);
        wide.sort_by(|&i, &j| self.ext[i].cy().total_cmp(&self.ext[j].cy()));
        let mut blocks = Vec::new();
        let mut low = f32::NEG_INFINITY;
        for s in wide.into_iter().map(Some).chain(std::iter::once(None)) {
            let high = s.map_or(f32::INFINITY, |i| self.ext[i].cy());
            let (left, right): (Vec<_>, Vec<_>) = body
                .iter()
                .copied()
                .filter(|&i| low < self.ext[i].cy() && self.ext[i].cy() <= high)
                .partition(|&i| self.ext[i].cx() < (a + b) * 0.5);
            for col in [left, right] {
                if !col.is_empty() {
                    blocks.extend(self.blocks(col, depth + 1));
                }
            }
            if let Some(i) = s {
                blocks.push(vec![i]);
            }
            low = high;
        }
        Some(blocks)
    }
}

fn enumerator(s: &str) -> Option<u32> {
    let mut chars = s
        .trim_start()
        .strip_prefix('(')
        .unwrap_or(s.trim_start())
        .chars()
        .peekable();
    let mut n = 0;
    let mut k = 0;
    while let Some(&c) = chars.peek() {
        if !digits(c) {
            break;
        }
        k += 1;
        if k > 4 {
            return None;
        }
        n = n * 10
            + if c.is_ascii_digit() {
                c as u32 - '0' as u32
            } else {
                c as u32 - '๐' as u32
            };
        chars.next();
    }
    (k > 0 && matches!(chars.find(|c| !c.is_whitespace()), Some('.' | ')'))).then_some(n)
}

pub(super) fn rows(boxes: &[Quad], texts: &[Reading], angle: f64) -> Vec<Vec<usize>> {
    let heights: Vec<_> = boxes.iter().map(|q| (q[0] - q[3]).norm().max(1.)).collect();
    let height = line_height(boxes);
    let ext = extents(boxes, angle);
    let partition = |e: &[Extent]| {
        Layout {
            ext: e,
            text: texts,
            height,
        }
        .blocks((0..boxes.len()).collect(), 0)
    };
    let mut blocks = partition(&ext);
    if angle != 0. {
        let alt = partition(&extents(boxes, 0.));
        if alt.len() > blocks.len() {
            blocks = alt;
        }
    }
    let mut out = Vec::new();
    for ids in blocks {
        let (mut full, mut small): (Vec<_>, Vec<_>) =
            ids.into_iter().partition(|&i| heights[i] >= 0.6 * height);
        if full.is_empty() {
            std::mem::swap(&mut full, &mut small);
        }
        full.sort_by(|&i, &j| ext[i].cy().total_cmp(&ext[j].cy()));
        let mut rows: Vec<(f64, Vec<usize>)> = Vec::new();
        for i in full {
            let y = ext[i].cy() as f64;
            if let Some((center, r)) = rows.last_mut()
                && (*center - y).abs() <= height as f64 * 0.5
            {
                *center = (*center * r.len() as f64 + y) / (r.len() + 1) as f64;
                r.push(i);
            } else {
                rows.push((y, vec![i]));
            }
        }
        for i in small {
            let y = ext[i].cy() as f64;
            if let Some((cy, r)) = rows
                .iter_mut()
                .min_by(|(a, _), (b, _)| (a - y).abs().total_cmp(&(b - y).abs()))
                && (*cy - y).abs() <= (height + heights[i]) as f64 * 0.5
            {
                r.push(i);
            } else {
                rows.push((y, vec![i]));
            }
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, mut r) in rows {
            r.sort_by(|&i, &j| ext[i].cx().total_cmp(&ext[j].cx()));
            out.push(r);
        }
    }
    out
}
