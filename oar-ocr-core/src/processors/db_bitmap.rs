use crate::processors::geometry::{BoundingBox, Point};
use crate::processors::types::ScoreMode;
use clipper2::{EndType, JoinType, Path as ClipperPath};
use image::GrayImage;
use imageproc::contours::find_contours;

use super::DBPostProcess;

impl DBPostProcess {
    pub(super) fn polygons_from_bitmap(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        bitmap: &GrayImage,
        dest_width: u32,
        dest_height: u32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let height = bitmap.height() as usize;
        let width = bitmap.width() as usize;
        let width_scale = dest_width as f32 / width as f32;
        let height_scale = dest_height as f32 / height as f32;

        let contours = find_contours::<u32>(bitmap);
        let mut boxes = Vec::new();
        let mut scores = Vec::new();

        for contour in contours.into_iter().take(self.max_candidates) {
            if contour.points.len() < 4 {
                continue;
            }

            let bbox = BoundingBox::from_contour(&contour);
            let epsilon = 0.002 * bbox.perimeter();
            let approx = bbox.approx_poly_dp(epsilon);

            if approx.points.len() < 4 {
                continue;
            }

            let score = self.box_score_fast(pred, &approx);
            if score < box_thresh {
                continue;
            }

            let unclipped_points = if approx.points.len() > 2 {
                let unclipped = self.unclip(&approx, unclip_ratio);
                if unclipped.points.is_empty() {
                    continue;
                }
                unclipped.points
            } else {
                continue;
            };

            let min_rect = BoundingBox::new(unclipped_points.clone()).get_min_area_rect();
            if min_rect.min_side() < self.min_size + 2.0 {
                continue;
            }

            let scaled_points: Vec<Point> = unclipped_points
                .iter()
                .map(|point| {
                    let x = (point.x * width_scale)
                        .round()
                        .clamp(0.0, dest_width as f32);
                    let y = (point.y * height_scale)
                        .round()
                        .clamp(0.0, dest_height as f32);
                    Point::new(x, y)
                })
                .collect();

            boxes.push(BoundingBox::new(scaled_points));
            scores.push(score);
        }

        (boxes, scores)
    }

    pub(super) fn boxes_from_bitmap(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        bitmap: &GrayImage,
        dest_width: u32,
        dest_height: u32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let height = bitmap.height() as usize;
        let width = bitmap.width() as usize;
        let width_scale = dest_width as f32 / width as f32;
        let height_scale = dest_height as f32 / height as f32;

        let contours = find_contours::<u32>(bitmap);
        let mut boxes = Vec::new();
        let mut scores = Vec::new();

        for contour in contours.into_iter().take(self.max_candidates) {
            let bbox = BoundingBox::from_contour(&contour);
            let min_rect = bbox.get_min_area_rect();

            if min_rect.min_side() < self.min_size {
                continue;
            }

            let score = match self.score_mode {
                ScoreMode::Fast => self.box_score_fast(pred, &bbox),
                ScoreMode::Slow => self.box_score_slow(pred, &contour),
            };

            if score < box_thresh {
                continue;
            }

            let unclipped = self.unclip(&bbox, unclip_ratio);
            let final_rect = unclipped.get_min_area_rect();

            if final_rect.min_side() < self.min_size + 2.0 {
                continue;
            }

            let box_points = final_rect.get_box_points();
            let scaled_points: Vec<Point> = box_points
                .iter()
                .map(|point| {
                    let x = (point.x * width_scale)
                        .round()
                        .clamp(0.0, dest_width as f32);
                    let y = (point.y * height_scale)
                        .round()
                        .clamp(0.0, dest_height as f32);
                    Point::new(x, y)
                })
                .collect();

            boxes.push(BoundingBox::new(scaled_points));
            scores.push(score);
        }

        (boxes, scores)
    }

    fn unclip(&self, bbox: &BoundingBox, unclip_ratio: f32) -> BoundingBox {
        if bbox.points.len() < 3 {
            return bbox.clone();
        }

        let clipper_path: ClipperPath = bbox
            .points
            .iter()
            .map(|point| (point.x as f64, point.y as f64))
            .collect::<Vec<_>>()
            .into();

        if clipper_path.len() < 3 {
            return BoundingBox::new(Vec::new());
        }

        let area = clipper_path.signed_area().abs();
        if area <= f64::EPSILON {
            return BoundingBox::new(Vec::new());
        }

        let coords: Vec<(f64, f64)> = clipper_path.iter().map(|p| (p.x(), p.y())).collect();
        let mut perimeter = 0.0f64;
        for i in 0..coords.len() {
            let (x1, y1) = coords[i];
            let (x2, y2) = coords[(i + 1) % coords.len()];
            let dx = x2 - x1;
            let dy = y2 - y1;
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        if perimeter <= f64::EPSILON {
            return BoundingBox::new(Vec::new());
        }

        let delta = area * unclip_ratio as f64 / perimeter;
        if delta.abs() <= f64::EPSILON {
            return BoundingBox::new(Vec::new());
        }

        let offset_paths = clipper_path.inflate(delta, JoinType::Round, EndType::Polygon, 2.0);

        if offset_paths.len() != 1 {
            return BoundingBox::new(Vec::new());
        }

        let Some(path) = offset_paths.into_iter().next() else {
            return BoundingBox::new(Vec::new());
        };

        let mut points: Vec<Point> = path
            .iter()
            .map(|pt| Point::new(pt.x() as f32, pt.y() as f32))
            .collect();

        // Remove duplicate closing point if present
        if points.len() > 1 {
            // Safe: we just verified len() > 1
            if let (Some(first), Some(last)) = (points.first(), points.last())
                && (first.x - last.x).abs() < f32::EPSILON
                && (first.y - last.y).abs() < f32::EPSILON
            {
                points.pop();
            }
        }

        if points.len() < 3 {
            return BoundingBox::new(Vec::new());
        }

        BoundingBox::new(points)
    }
}
