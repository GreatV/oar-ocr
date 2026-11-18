//! Plan split candidates per page, then recognize crops from all pages in bounded batches.
use super::{
    CropStrategy, OAROCR, PageRead, Reading, ReadingOrder, as_bbox,
    geometry::{self, Point, Quad},
    order, recognition,
};
use crate::oarocr::{EdgeProcessor, TextCroppingProcessor};
use image::RgbImage;
use oar_ocr_core::{
    core::OCRError,
    processors::{BoundingBox, quad_box_order},
};
use std::{ops::Range, sync::Arc};

pub(super) fn quad(b: &BoundingBox) -> Quad {
    if let [a, b, c, d] = b.points.as_slice() {
        [a, b, c, d].map(|p| Point::new(p.x, p.y))
    } else {
        let (l, t, r, b) = (b.x_min(), b.y_min(), b.x_max(), b.y_max());
        [
            Point::new(l, t),
            Point::new(r, t),
            Point::new(r, b),
            Point::new(l, b),
        ]
    }
}

fn standard_order(boxes: &[BoundingBox], seal: bool) -> Vec<usize> {
    if !seal {
        return quad_box_order(boxes);
    }
    let mut ids: Vec<_> = (0..boxes.len()).collect();
    ids.sort_by(|&i, &j| {
        boxes[i]
            .y_min()
            .partial_cmp(&boxes[j].y_min())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ids
}

fn standard_rows(boxes: &[BoundingBox], seal: bool) -> Vec<Vec<usize>> {
    standard_order(boxes, seal)
        .into_iter()
        .map(|index| vec![index])
        .collect()
}

struct Candidates {
    pieces: Range<usize>,
    whole: Option<usize>,
}

struct PagePlan {
    boxes: Vec<BoundingBox>,
    lines: Vec<Candidates>,
}

impl OAROCR {
    pub(in crate::oarocr) fn line_image_fallback(
        &self,
        img: &RgbImage,
        mut boxes: Vec<BoundingBox>,
    ) -> Vec<BoundingBox> {
        let (w, h) = img.dimensions();
        if self.stages.recognize_line_images
            && boxes.is_empty()
            && h > 0
            && w > 0
            && h <= 200
            && f64::from(w) >= 1.5 * f64::from(h)
        {
            boxes.push(BoundingBox::from_coords(
                0.,
                0.,
                (w - 1) as f32,
                (h - 1) as f32,
            ));
        }
        boxes
    }

    fn plan_page(&self, img: &RgbImage, boxes: &[BoundingBox]) -> Result<PagePlan, OCRError> {
        // Keep the classic stable tie order for equal-aspect crops in the recognition pool.
        let ids = if self.stages.reading_order == ReadingOrder::Standard {
            standard_order(
                boxes,
                self.text_type
                    .as_deref()
                    .is_some_and(|s| s.eq_ignore_ascii_case("seal")),
            )
        } else {
            (0..boxes.len()).collect()
        };
        let mut candidates = Vec::with_capacity(boxes.len());
        let mut whole = Vec::new();
        let mut lines = Vec::with_capacity(boxes.len());
        for i in ids {
            let bbox = &boxes[i];
            let start = candidates.len();
            let pieces = if self.stages.split_line_gaps && bbox.points.len() == 4 {
                let q = quad(bbox);
                recognition::split(&self.crop_line(img, &q)?, &q)
            } else {
                None
            };
            let original = if let Some(pieces) = pieces {
                candidates.extend(pieces.iter().map(as_bbox));
                whole.push(bbox.clone());
                Some(whole.len() - 1)
            } else {
                candidates.push(bbox.clone());
                None
            };
            lines.push(Candidates {
                pieces: start..candidates.len(),
                whole: original,
            });
        }
        for line in &mut lines {
            if let Some(i) = &mut line.whole {
                *i += candidates.len();
            }
        }
        candidates.extend(whole);
        Ok(PagePlan {
            boxes: candidates,
            lines,
        })
    }

    pub(in crate::oarocr) fn read_batch(
        &self,
        inputs: &[(&Arc<RgbImage>, Vec<BoundingBox>)],
    ) -> Result<Vec<PageRead>, OCRError> {
        const MAX_POOLED_CROPS: usize = 4096;
        let plans: Vec<_> = inputs
            .iter()
            .map(|(img, boxes)| self.plan_page(img, boxes))
            .collect::<Result<_, _>>()?;
        let mut readings: Vec<_> = plans
            .iter()
            .map(|p| vec![Reading::default(); p.boxes.len()])
            .collect();
        let mut pool = Vec::new();
        let mut slots = Vec::new();
        let flush = |pool: &mut Vec<Arc<RgbImage>>,
                     slots: &mut Vec<(usize, usize, Option<f32>)>,
                     readings: &mut [Vec<Reading>]|
         -> Result<(), OCRError> {
            let result = self.read_pool(std::mem::take(pool))?;
            for ((page, index, angle), mut read) in slots.drain(..).zip(result) {
                read.orientation = angle;
                readings[page][index] = read;
            }
            Ok(())
        };
        for (page, (plan, (image, _))) in plans.iter().zip(inputs).enumerate() {
            let mut start = 0;
            while start < plan.boxes.len() {
                let end = (start + MAX_POOLED_CROPS - pool.len()).min(plan.boxes.len());
                let boxes = &plan.boxes[start..end];
                let crops = match self.stages.crop_strategy {
                    CropStrategy::Standard => TextCroppingProcessor::new(true)
                        .process((Arc::clone(image), boxes.to_vec()))?,
                    CropStrategy::PerspectiveCubic => boxes
                        .iter()
                        .map(|b| Some(Arc::new(geometry::crop(image, &quad(b)))))
                        .collect(),
                };
                let (indices, mut valid): (Vec<_>, Vec<_>) = crops
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, c)| c.map(|img| (i, img)))
                    .unzip();
                let angles = self.orient_crops(&mut valid)?;
                slots.extend(
                    indices
                        .into_iter()
                        .zip(angles)
                        .map(|(i, a)| (page, start + i, a)),
                );
                pool.extend(valid);
                if pool.len() == MAX_POOLED_CROPS {
                    flush(&mut pool, &mut slots, &mut readings)?;
                }
                start = end;
            }
        }
        if !pool.is_empty() {
            flush(&mut pool, &mut slots, &mut readings)?;
        }
        Ok(plans
            .into_iter()
            .zip(readings)
            .map(|(plan, mut readings)| {
                let mut polygons = Vec::new();
                let mut reads = Vec::new();
                for line in plan.lines {
                    let mut keep = line.pieces;
                    if let Some(i) = line.whole {
                        let count = |r: &Reading| r.text.chars().filter(|&c| c != ' ').count();
                        let parts = &readings[keep.clone()];
                        if parts.iter().any(|r| {
                            !r.available || r.confidence < f64::from(self.rec_score_threshold)
                        }) || (parts.iter().map(count).sum::<usize>() as f64)
                            < 0.9 * count(&readings[i]) as f64
                        {
                            keep = i..i + 1;
                        }
                    }
                    for i in keep {
                        polygons.push(plan.boxes[i].clone());
                        reads.push(std::mem::take(&mut readings[i]));
                    }
                }
                let boxes: Vec<_> = polygons.iter().map(quad).collect();
                let rows = match self.stages.reading_order {
                    ReadingOrder::TextColumns => {
                        order::rows(&boxes, &reads, geometry::skew(&boxes))
                    }
                    ReadingOrder::Standard => standard_rows(
                        &polygons,
                        self.text_type
                            .as_deref()
                            .is_some_and(|s| s.eq_ignore_ascii_case("seal")),
                    ),
                };
                PageRead {
                    polygons,
                    boxes,
                    reads,
                    rows,
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_order_keeps_each_sorted_region_on_its_own_line() {
        let boxes = [(60., 21.), (0., 52.), (0., 25.)]
            .map(|(x, y)| BoundingBox::from_coords(x, y, x + 30., y + 12.));
        assert_eq!(standard_rows(&boxes, false), [vec![2], vec![0], vec![1]]);
    }

    #[test]
    fn per_region_options_keep_cross_image_pooling() {
        let mut stages = super::super::options::PageStages {
            text_postprocessing: super::super::TextPostprocessing::ThaiPrinted,
            split_line_gaps: true,
            recognition_preprocessing: super::super::RecognitionPreprocessing::Flatten,
            reading_order: ReadingOrder::TextColumns,
            ..Default::default()
        };
        assert!(!stages.needs_page_loop());
        stages.reattach_marks = true;
        assert!(stages.needs_page_loop());
    }
}
