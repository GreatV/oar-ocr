//! Reattach displaced Thai marks to the nearest eligible line before final filtering.
use super::{
    OAROCR, Reading,
    geometry::{Quad, line_height},
    text,
};
use image::RgbImage;
use nalgebra::Vector2;
use oar_ocr_core::core::OCRError;

#[derive(Clone, Copy)]
struct Frame {
    origin: Vector2<f64>,
    along: Vector2<f64>,
    across: Vector2<f64>,
    w: f64,
    h: f64,
}
impl Frame {
    fn project(self, b: &Quad) -> [f64; 4] {
        let s = b.map(|p| (p.cast::<f64>() - self.origin).dot(&self.along));
        let t = b.map(|p| (p.cast::<f64>() - self.origin).dot(&self.across));
        [
            s.into_iter().fold(f64::INFINITY, f64::min),
            s.into_iter().fold(f64::NEG_INFINITY, f64::max),
            t.into_iter().fold(f64::INFINITY, f64::min),
            t.into_iter().fold(f64::NEG_INFINITY, f64::max),
        ]
    }
}
fn extents(q: &Quad) -> [f32; 4] {
    [
        q.iter().map(|p| p.x).fold(f32::INFINITY, f32::min),
        q.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max),
        q.iter().map(|p| p.y).fold(f32::INFINITY, f32::min),
        q.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max),
    ]
}
impl OAROCR {
    pub(super) fn attach_marks(
        &self,
        img: &RgbImage,
        boxes: &mut [Quad],
        reads: &mut [Reading],
    ) -> Result<Vec<bool>, OCRError> {
        let normal = line_height(boxes);
        let heights: Vec<_> = boxes.iter().map(|b| (b[0] - b[3]).norm()).collect();
        let ext: Vec<_> = boxes.iter().map(extents).collect();
        let mut fragments: Vec<_> = (0..boxes.len())
            .map(|i| {
                if boxes.len() < 3
                    || heights[i] >= 0.35 * normal
                    || reads[i].text.chars().filter(|&c| c != ' ').count() > 3
                {
                    return false;
                }
                let [x0, x1, y0, y1] = ext[i];
                ext.iter()
                    .enumerate()
                    .any(|(j, &[left, right, top, bottom])| {
                        j != i
                            && heights[j] >= 0.6 * normal
                            && left <= x1
                            && x0 <= right
                            && (y0 - bottom).abs().min((top - y1).abs()) <= 0.5 * normal
                    })
            })
            .collect();
        let junk: Vec<_> = reads
            .iter()
            .enumerate()
            .map(|(i, r)| {
                fragments[i]
                    || text::drop_reason(
                        &text::canonical(&r.text),
                        r.confidence,
                        self.rec_score_threshold,
                        Some(heights[i] / normal),
                    )
                    .is_some()
            })
            .collect();
        if !self.stages.reattach_marks {
            return Ok(fragments);
        }
        if junk.iter().all(|&v| v) || !junk.iter().any(|&v| v) {
            return Ok(fragments);
        }
        let kept: Vec<_> = boxes
            .iter()
            .enumerate()
            .filter(|(i, _)| !junk[*i])
            .map(|(_, q)| *q)
            .collect();
        let height = line_height(&kept);
        let frames: Vec<_> = boxes
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                let b = b.map(|p| p.cast::<f64>());
                let w = (b[1] - b[0]).norm();
                let h = (b[3] - b[0]).norm();
                (!junk[i] && w >= 1. && h >= 0.5 * f64::from(height)).then(|| {
                    (
                        i,
                        Frame {
                            origin: b[0],
                            along: (b[1] - b[0]) / w,
                            across: (b[3] - b[0]) / h,
                            w,
                            h,
                        },
                    )
                })
            })
            .collect();
        let mut owners = Vec::new();
        let mut reach = vec![[0f64; 2]; boxes.len()];
        for (j, b) in boxes.iter().enumerate().filter(|(j, _)| junk[*j]) {
            let mut best = None;
            for &(i, f) in &frames {
                let [s0, s1, t0, t1] = f.project(b);
                if t1 - t0 > 0.9 * f.h || s1.min(f.w) - s0.max(0.) < 0.5 * (s1 - s0) {
                    continue;
                }
                let (gap, side, extension) = if t1 <= 0.5 * f.h {
                    ((-t1).max(0.), 0, -t0)
                } else if t0 >= 0.5 * f.h {
                    ((t0 - f.h).max(0.), 1, t1 - f.h)
                } else {
                    continue;
                };
                if gap <= 0.35 * f.h
                    && extension <= 0.9 * f.h
                    && best.is_none_or(|(g, s, _, _)| (gap, side) < (g, s))
                {
                    best = Some((gap, side, i, extension));
                }
            }
            if let Some((_, side, i, r)) = best {
                owners.push((j, i));
                reach[i][side] = reach[i][side].max(r);
            }
        }
        if owners.is_empty() {
            return Ok(fragments);
        }
        let mut floating = false;
        for side in [0, 1] {
            let mut ratios: Vec<_> = frames
                .iter()
                .filter(|(i, _)| reach[*i][side] > 0.)
                .map(|(i, f)| reach[*i][side] / f.h)
                .collect();
            if ratios.len() < 3 || ratios.len() * 4 < frames.len() {
                continue;
            }
            floating = true;
            let typical = crate::indra::median_f64(&mut ratios);
            for &(i, f) in &frames {
                if reach[i][side] > 0. {
                    continue;
                }
                let mut room = typical * f.h;
                for &(j, _) in &frames {
                    if i == j {
                        continue;
                    }
                    let [s0, s1, t0, t1] = f.project(&boxes[j]);
                    if s1.min(f.w) <= s0.max(0.) {
                        continue;
                    }
                    if side == 0 && t1 <= 0.5 * f.h {
                        room = room.min((-t1 * 0.9).max(0.));
                    }
                    if side == 1 && t0 >= 0.5 * f.h {
                        room = room.min(((t0 - f.h) * 0.9).max(0.));
                    }
                }
                reach[i][side] = room;
            }
        }
        let mut ids = Vec::new();
        let mut grown = Vec::new();
        for (i, f) in frames {
            let [up, down] = reach[i];
            if up <= 0. && down <= 0. {
                continue;
            }
            ids.push(i);
            let mut b = boxes[i];
            for (k, p) in b.iter_mut().enumerate() {
                *p = (p.cast::<f64>() + f.across * if k < 2 { -up } else { down }).cast();
                p.x = p.x.clamp(0., (img.width() - 1) as f32);
                p.y = p.y.clamp(0., (img.height() - 1) as f32);
            }
            grown.push(b);
        }
        let new = self.read_boxes(img, &grown)?;
        let mut done = vec![false; boxes.len()];
        for ((i, b), r) in ids.into_iter().zip(grown).zip(new) {
            let new = text::canonical(&r.text);
            let old = text::canonical(&reads[i].text);
            let number = |s: &str| {
                s.chars()
                    .filter(|c| crate::indra::is_digit(*c))
                    .collect::<String>()
            };
            if text::drop_reason(&new, r.confidence, self.rec_score_threshold, None).is_some()
                || (if floating {
                    number(&old) != number(&new)
                } else {
                    text::base(&old) != text::base(&new)
                })
            {
                continue;
            }
            boxes[i] = b;
            reads[i] = r;
            done[i] = true;
        }
        for (j, i) in owners {
            fragments[j] |= done[i];
        }
        Ok(fragments)
    }
}
