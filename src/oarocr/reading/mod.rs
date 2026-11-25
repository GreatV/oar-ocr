//! Page-level OCR stages that use recognition evidence for orientation, rereads, and reading order.
mod geometry;
mod lines;
mod marks;
pub(super) mod options;
mod order;
mod pool;
mod recognition;
pub(super) mod scripts;
use super::{
    CropStrategy, CtcDecoding, DbGeometry, OAROCR, PageOrientation, ReadingOrder,
    RecognitionPreprocessing, SecondaryRouting, TextPostprocessing,
};
use crate::oarocr::{TextRegion, processors::scripts::is_junk, processors::thai as text};
use geometry::{Quad, Transform};
use image::{RgbImage, imageops};
use oar_ocr_core::{
    core::OCRError,
    processors::{BoundingBox, CtcFrame, Point as BoxPoint},
};
use std::sync::Arc;
/// Label of a region read from a faint short-token candidate.
pub(in crate::oarocr) const FAINT_LABEL: &str = "faint";

fn box_from_pixels(points: [[f64; 2]; 4]) -> BoundingBox {
    BoundingBox::new(
        points
            .map(|[x, y]| BoxPoint::new(x as f32, y as f32))
            .to_vec(),
    )
}
fn as_bbox(q: &Quad) -> BoundingBox {
    BoundingBox::new(q.map(|p| BoxPoint::new(p.x, p.y)).to_vec())
}
#[derive(Clone)]
struct WordPositions {
    columns: Vec<usize>,
    positions: Vec<f32>,
    steps: usize,
    ratio: f32,
    max_ratio: f32,
}
#[derive(Clone, Default)]
struct Reading {
    available: bool,
    positions: Option<WordPositions>,
    text: String,
    confidence: f64,
    frames: Option<Vec<CtcFrame>>,
    orientation: Option<f32>,
    /// Read from a region the detector scored under its threshold.
    faint: bool,
}
pub(super) struct PageRead {
    polygons: Vec<BoundingBox>,
    boxes: Vec<Quad>,
    reads: Vec<Reading>,
    rows: Vec<Vec<usize>>,
}
fn axis(q: &Quad) -> i8 {
    let w = (q[1] - q[0]).norm();
    let h = (q[3] - q[0]).norm();
    if h >= 2. * w.max(1.) {
        1
    } else if w >= 2. * h.max(1.) {
        -1
    } else {
        0
    }
}
impl OAROCR {
    fn sideways(&self, img: &RgbImage, boxes: &[Quad]) -> Result<u16, OCRError> {
        let mut vert: Vec<_> = boxes.iter().filter(|q| axis(q) == 1).collect();
        if vert.len() < 3 || vert.len() <= boxes.iter().filter(|q| axis(q) == -1).count() {
            return Ok(0);
        }
        vert.sort_by(|a, b| (b[0] - b[3]).norm().total_cmp(&(a[0] - a[3]).norm()));
        let crops: Vec<_> = vert
            .into_iter()
            .take(8)
            .map(|q| self.crop_line(img, q))
            .collect::<Result<_, _>>()?;
        let turned: Vec<_> = crops
            .iter()
            .map(|c| Arc::new(imageops::rotate270(c)))
            .chain(crops.iter().map(|c| Arc::new(imageops::rotate90(c))))
            .collect();
        let r = self.recognize(&turned, &self.pipeline.text_recognition_adapter)?;
        let score = |rs: &[Reading]| rs.iter().map(|r| r.confidence).sum::<f64>();
        Ok(if score(&r[..crops.len()]) > score(&r[crops.len()..]) {
            270
        } else {
            90
        })
    }
    fn upside_down(&self, img: &RgbImage, page: &PageRead) -> Result<bool, OCRError> {
        let mut scores: Vec<_> = page
            .reads
            .iter()
            .filter(|r| r.text.chars().count() >= 2)
            .map(|r| r.confidence)
            .collect();
        if scores.len() < 3 || crate::indra::median_f64(&mut scores) >= 0.8 {
            return Ok(false);
        }
        let mut ids: Vec<_> = (0..page.boxes.len())
            .filter(|&i| axis(&page.boxes[i]) == -1)
            .collect();
        let width = |i: usize| (page.boxes[i][0] - page.boxes[i][1]).norm();
        ids.sort_by(|&a, &b| width(b).total_cmp(&width(a)));
        ids.truncate(8);
        if ids.len() < 3 {
            return Ok(false);
        }
        let crops: Vec<_> = ids
            .iter()
            .map(|&i| {
                self.crop_line(img, &page.boxes[i])
                    .map(|crop| Arc::new(imageops::rotate180(&crop)))
            })
            .collect::<Result<_, _>>()?;
        let score: f64 = self
            .recognize(&crops, &self.pipeline.text_recognition_adapter)?
            .iter()
            .map(|r| r.confidence)
            .sum();
        Ok(score
            > ids.iter().map(|&i| page.reads[i].confidence).sum::<f64>() + 0.15 * ids.len() as f64)
    }
    fn read_page(
        &self,
        img: &Arc<RgbImage>,
        boxes: Vec<BoundingBox>,
    ) -> Result<PageRead, OCRError> {
        Ok(self
            .read_batch(&[(img, boxes)])?
            .pop()
            .expect("one input page"))
    }
    pub(super) fn read_working_page(
        &self,
        original: &Arc<RgbImage>,
        initial: Vec<BoundingBox>,
        mut faint: Vec<BoundingBox>,
    ) -> Result<(Vec<TextRegion>, Vec<usize>, u16), OCRError> {
        let mut img = original.clone();
        let (width, height) = img.dimensions();
        let mut boxes = initial;
        let quads = |boxes: &[BoundingBox]| boxes.iter().map(pool::quad).collect::<Vec<_>>();
        let vote = self.stages.page_orientation == PageOrientation::RecognitionVote;
        let mut rotation = if vote {
            self.sideways(&img, &quads(&boxes))?
        } else {
            0
        };
        let mut forward = Transform::identity();
        if rotation != 0 {
            if rotation == 90 {
                forward = Transform::new(0., -1., (height - 1) as f64, 1., 0., 0., 0., 0., 1.);
                img = Arc::new(imageops::rotate90(img.as_ref()));
            } else {
                forward = Transform::new(0., 1., 0., -1., 0., (width - 1) as f64, 0., 0., 1.);
                img = Arc::new(imageops::rotate270(img.as_ref()));
            }
            boxes = self.detect_text_boxes(&img)?;
        }
        if self.stages.page_orientation == PageOrientation::LineShapes
            && lines::looks_sideways(&quads(&boxes))
        {
            let turned = Arc::new(imageops::rotate270(img.as_ref()));
            let turned_boxes = self.detect_text_boxes(&turned)?;
            if lines::long_lines(&quads(&turned_boxes)) > 2 * lines::long_lines(&quads(&boxes)) + 5
            {
                forward = Transform::new(0., 1., 0., -1., 0., (width - 1) as f64, 0., 0., 1.);
                img = turned;
                boxes = turned_boxes;
                faint.clear();
                rotation = 270;
            }
        }
        let skew = geometry::skew(&quads(&boxes));
        if self
            .stages
            .deskew_threshold
            .is_some_and(|v| skew.abs() >= v)
        {
            let (upright, m) = geometry::deskew(&img, skew);
            img = Arc::new(upright);
            forward = m * forward;
            boxes = self.detect_text_boxes(&img)?;
        }
        if self.stages.split_column_gutters {
            boxes = lines::split_gutters(&img, &quads(&boxes))
                .iter()
                .map(as_bbox)
                .collect();
        }
        let mut page = if let Some(scripts) = &self.pipeline.scripts {
            let (mut page, upside_down) = self.read_script_page(scripts, &img, boxes, faint)?;
            if upside_down {
                // Corners run in reading order, so an upside-down line starts at its bottom-right.
                for (polygon, q) in page.polygons.iter_mut().zip(&mut page.boxes) {
                    if polygon.points.len() == 4 {
                        polygon.points.rotate_left(2);
                    }
                    q.rotate_left(2);
                }
                rotation = (rotation + 180) % 360;
            }
            page
        } else {
            self.read_page(&img, boxes)?
        };
        if vote && rotation == 0 && self.upside_down(&img, &page)? {
            forward = Transform::new(
                -1.,
                0.,
                (img.width() - 1) as f64,
                0.,
                -1.,
                (img.height() - 1) as f64,
                0.,
                0.,
                1.,
            ) * forward;
            img = Arc::new(imageops::rotate180(img.as_ref()));
            rotation = 180;
            page = self.read_page(&img, self.detect_text_boxes(&img)?)?;
        }
        let back = forward
            .try_inverse()
            .expect("page rotations are invertible");
        let (regions, rows) = self.finish_page(&img, page, &back)?;
        Ok((regions, rows, rotation))
    }
    pub(super) fn finish_page(
        &self,
        img: &RgbImage,
        mut page: PageRead,
        back: &Transform,
    ) -> Result<(Vec<TextRegion>, Vec<usize>), OCRError> {
        let thai = self.stages.text_postprocessing == TextPostprocessing::ThaiPrinted;
        let frag = if self.stages.reattach_marks || thai {
            self.attach_marks(img, &mut page.boxes, &mut page.reads)?
        } else {
            vec![false; page.boxes.len()]
        };
        let normal = geometry::line_height(&page.boxes);
        // Script-routed pages drop detector false positives instead of returning them empty.
        let drop_junk = self.pipeline.scripts.is_some();
        let mut regions = Vec::new();
        let mut row_ids = Vec::new();
        for (row, ids) in page.rows.into_iter().enumerate() {
            for i in ids {
                if frag[i] {
                    continue;
                }
                let (r, q) = (&page.reads[i], &page.boxes[i]);
                let value = if thai {
                    text::canonical(&r.text)
                } else if r.confidence < f64::from(self.rec_score_threshold) {
                    String::new()
                } else {
                    r.text.clone()
                };
                if thai
                    && text::drop_reason(
                        &value,
                        r.confidence,
                        self.rec_score_threshold,
                        Some((q[0] - q[3]).norm() / normal),
                    )
                    .is_some()
                {
                    continue;
                }
                if drop_junk && (value.trim().is_empty() || is_junk(&value, r.confidence as f32)) {
                    continue;
                }
                let polygon = &page.polygons[i];
                let bbox = if polygon.points.len() == 4
                    && (self.stages.db_geometry == DbGeometry::OpenCv || self.stages.reattach_marks)
                {
                    box_from_pixels(geometry::back_quad(q.map(|p| p.cast()), back))
                } else {
                    map_polygon(polygon, back)
                };
                let mut region = r.region(bbox, value);
                if r.available && self.return_word_box {
                    region.word_boxes = if let Some(pos) = &r.positions {
                        let words = if !pos.columns.is_empty() && pos.steps > 0 {
                            Some(Self::ctc_word_boxes(
                                polygon,
                                &r.text,
                                &pos.columns,
                                pos.steps,
                                pos.ratio,
                                pos.max_ratio,
                            ))
                        } else if !pos.positions.is_empty() {
                            Some(Self::char_positions_to_word_boxes(
                                polygon,
                                &pos.positions,
                                r.text.chars().count(),
                            ))
                        } else {
                            None
                        };
                        words.map(|ws| ws.iter().map(|w| map_polygon(w, back)).collect())
                    } else {
                        Some(recognition::words(r, q, back, thai))
                    };
                }
                regions.push(region);
                row_ids.push(match self.stages.reading_order {
                    ReadingOrder::Standard | ReadingOrder::Detector => row_ids.len(),
                    ReadingOrder::TextColumns => row,
                });
            }
        }
        Ok((regions, row_ids))
    }
}

fn map_polygon(bbox: &BoundingBox, back: &Transform) -> BoundingBox {
    if *back == Transform::identity() {
        return bbox.clone();
    }
    BoundingBox::new(
        bbox.points
            .iter()
            .map(|p| {
                let v = back * nalgebra::Vector3::new(f64::from(p.x), f64::from(p.y), 1.);
                BoxPoint::new(v.x as f32, v.y as f32)
            })
            .collect(),
    )
}

impl Reading {
    fn region(&self, bbox: BoundingBox, value: String) -> TextRegion {
        let mut region = TextRegion::new(bbox.clone());
        region.dt_poly = Some(bbox.clone());
        if self.available {
            region.text = Some(Arc::from(value));
            region.confidence = Some(self.confidence as f32);
            region.rec_poly = Some(bbox);
            region.orientation_angle = self.orientation;
            if self.faint {
                region.label = Some(FAINT_LABEL.into());
            }
        }
        region
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detections_without_a_recognition_result_are_kept_without_text() {
        let bbox = BoundingBox::from_coords(0., 10., 10., 15.);
        let region = Reading::default().region(bbox.clone(), String::new());
        assert_eq!(region.text, None);
        assert_eq!(region.confidence, None);
        assert_eq!(region.bounding_box.points, bbox.points);
        assert_eq!(region.dt_poly.unwrap().points, bbox.points);
        assert!(region.rec_poly.is_none());
    }
}
