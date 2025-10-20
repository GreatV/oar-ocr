use crate::processors::geometry::{BoundingBox, Point};
use crate::processors::types::ScoreMode;
use image::GrayImage;
use imageproc::contours::find_contours;

use super::DBPostProcess;

impl DBPostProcess {
    pub(super) fn polygons_from_bitmap(
        &self,
        pred: &ndarray::Array2<f32>,
        bitmap: &[Vec<bool>],
        dest_width: u32,
        dest_height: u32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let height = bitmap.len();
        let width = if height > 0 { bitmap[0].len() } else { 0 };
        let width_scale = dest_width as f32 / width as f32;
        let height_scale = dest_height as f32 / height as f32;

        let mut gray_img = GrayImage::new(width as u32, height as u32);
        for (y, row) in bitmap.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let pixel_value = if value { 255 } else { 0 };
                gray_img.put_pixel(x as u32, y as u32, image::Luma([pixel_value]));
            }
        }

        let contours = find_contours::<u32>(&gray_img);
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
                    Point::new(
                        (point.x * width_scale).max(0.0).min(dest_width as f32),
                        (point.y * height_scale).max(0.0).min(dest_height as f32),
                    )
                })
                .collect();

            boxes.push(BoundingBox::new(scaled_points));
            scores.push(score);
        }

        (boxes, scores)
    }

    pub(super) fn boxes_from_bitmap(
        &self,
        pred: &ndarray::Array2<f32>,
        bitmap: &[Vec<bool>],
        dest_width: u32,
        dest_height: u32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let height = bitmap.len();
        let width = if height > 0 { bitmap[0].len() } else { 0 };
        let width_scale = dest_width as f32 / width as f32;
        let height_scale = dest_height as f32 / height as f32;

        let mut gray_img = GrayImage::new(width as u32, height as u32);
        for (y, row) in bitmap.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let pixel_value = if value { 255 } else { 0 };
                gray_img.put_pixel(x as u32, y as u32, image::Luma([pixel_value]));
            }
        }

        let contours = find_contours::<u32>(&gray_img);
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
                    Point::new(
                        (point.x * width_scale).max(0.0).min(dest_width as f32),
                        (point.y * height_scale).max(0.0).min(dest_height as f32),
                    )
                })
                .collect();

            boxes.push(BoundingBox::new(scaled_points));
            scores.push(score);
        }

        (boxes, scores)
    }

    fn unclip(&self, bbox: &BoundingBox, unclip_ratio: f32) -> BoundingBox {
        let area = bbox.area();
        let length = bbox.perimeter();

        if length <= f32::EPSILON {
            return bbox.clone();
        }

        let distance = area * unclip_ratio / length;

        let n = bbox.points.len() as f32;
        let center_x = bbox.points.iter().map(|p| p.x).sum::<f32>() / n;
        let center_y = bbox.points.iter().map(|p| p.y).sum::<f32>() / n;

        let expanded_points: Vec<Point> = bbox
            .points
            .iter()
            .map(|point| {
                let dx = point.x - center_x;
                let dy = point.y - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > f32::EPSILON {
                    let expansion = distance / dist;
                    Point::new(point.x + dx * expansion, point.y + dy * expansion)
                } else {
                    *point
                }
            })
            .collect();

        BoundingBox::new(expanded_points)
    }
}
