use crate::processors::geometry::{BoundingBox, MinAreaRect, Point};
use crate::processors::types::ScoreMode;
use clipper2::{EndType, JoinType, Path as ClipperPath};
use image::GrayImage;
use imageproc::contours::{Contour, find_contours};
use std::cmp::Ordering;
use std::f32::consts::PI;

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

            let unclipped = if approx.points.len() > 2 {
                self.unclip(&approx, unclip_ratio)
            } else {
                continue;
            };
            if unclipped.points.is_empty() {
                continue;
            }

            let Some((_, sside)) = self.get_mini_boxes_from_points(&unclipped.points) else {
                continue;
            };
            if sside < self.min_size + 2.0 {
                continue;
            }

            let scaled_points: Vec<Point> = unclipped
                .points
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
            let Some((mini_box_points, min_side)) = self.get_mini_boxes_from_contour(&contour)
            else {
                continue;
            };
            if min_side < self.min_size {
                continue;
            }
            let mini_box = BoundingBox::new(mini_box_points);

            let score = match self.score_mode {
                ScoreMode::Fast => self.box_score_fast(pred, &mini_box),
                ScoreMode::Slow => self.box_score_slow(pred, &contour),
            };

            if score < box_thresh {
                continue;
            }

            let unclipped = self.unclip(&mini_box, unclip_ratio);
            if unclipped.points.is_empty() {
                continue;
            }

            let Some((box_points, sside)) = self.get_mini_boxes_from_points(&unclipped.points)
            else {
                continue;
            };
            if sside < self.min_size + 2.0 {
                continue;
            }

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

    /// PaddleX `get_mini_boxes(contour)` equivalent.
    fn get_mini_boxes_from_contour(&self, contour: &Contour<u32>) -> Option<(Vec<Point>, f32)> {
        let points = contour
            .points
            .iter()
            .map(|p| Point::new(p.x as f32, p.y as f32))
            .collect::<Vec<_>>();
        let simplified = Self::simplify_chain_points(&points);
        if simplified.len() >= 3 {
            self.get_mini_boxes_from_points(&simplified)
        } else {
            self.get_mini_boxes_from_points(&points)
        }
    }

    /// PaddleX `get_mini_boxes` equivalent from polygon points.
    fn get_mini_boxes_from_points(&self, points: &[Point]) -> Option<(Vec<Point>, f32)> {
        if points.len() < 3 {
            return None;
        }

        let min_rect = BoundingBox::new(points.to_vec()).get_min_area_rect();
        let min_side = min_rect.min_side();
        if !min_side.is_finite() || min_side <= 0.0 {
            return None;
        }

        let raw_points = Self::box_points_without_reorder(&min_rect);
        if raw_points.len() != 4 {
            return None;
        }

        Some((Self::paddlex_order_mini_box_points(raw_points), min_side))
    }

    fn box_points_without_reorder(rect: &MinAreaRect) -> Vec<Point> {
        let cos_a = (rect.angle * PI / 180.0).cos();
        let sin_a = (rect.angle * PI / 180.0).sin();
        let w_2 = rect.width / 2.0;
        let h_2 = rect.height / 2.0;
        let corners = [(-w_2, -h_2), (w_2, -h_2), (w_2, h_2), (-w_2, h_2)];

        corners
            .iter()
            .map(|(x, y)| {
                let rotated_x = x * cos_a - y * sin_a + rect.center.x;
                let rotated_y = x * sin_a + y * cos_a + rect.center.y;
                Point::new(rotated_x, rotated_y)
            })
            .collect()
    }

    /// Compress contour chain points similarly to OpenCV CHAIN_APPROX_SIMPLE.
    ///
    /// This keeps turning points and removes interior points on straight segments.
    fn simplify_chain_points(points: &[Point]) -> Vec<Point> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        let mut simplified = Vec::with_capacity(points.len());
        let n = points.len();

        for i in 0..n {
            let prev = points[(i + n - 1) % n];
            let curr = points[i];
            let next = points[(i + 1) % n];

            let dir_prev = (
                Self::sign_step(curr.x - prev.x),
                Self::sign_step(curr.y - prev.y),
            );
            let dir_next = (
                Self::sign_step(next.x - curr.x),
                Self::sign_step(next.y - curr.y),
            );

            if dir_prev != dir_next {
                simplified.push(curr);
            }
        }

        if simplified.len() < 3 {
            points.to_vec()
        } else {
            simplified
        }
    }

    fn sign_step(v: f32) -> i8 {
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    }

    /// PaddleX `get_mini_boxes` point ordering:
    /// sort by x, then select [top-left, top-right, bottom-right, bottom-left].
    fn paddlex_order_mini_box_points(mut points: Vec<Point>) -> Vec<Point> {
        if points.len() != 4 {
            return points;
        }

        points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(Ordering::Equal));

        let (index_1, index_4) = if points[1].y > points[0].y {
            (0usize, 1usize)
        } else {
            (1usize, 0usize)
        };
        let (index_2, index_3) = if points[3].y > points[2].y {
            (2usize, 3usize)
        } else {
            (3usize, 2usize)
        };

        vec![
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4],
        ]
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

        // Safe: we just verified len() == 1
        let path = offset_paths.into_iter().next().unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paddlex_order_mini_box_points() {
        let input = vec![
            Point::new(20.0, 20.0),
            Point::new(10.0, 10.0),
            Point::new(20.0, 10.0),
            Point::new(10.0, 20.0),
        ];

        let ordered = DBPostProcess::paddlex_order_mini_box_points(input);
        assert_eq!(ordered.len(), 4);
        assert!((ordered[0].x - 10.0).abs() < 1e-6 && (ordered[0].y - 10.0).abs() < 1e-6);
        assert!((ordered[1].x - 20.0).abs() < 1e-6 && (ordered[1].y - 10.0).abs() < 1e-6);
        assert!((ordered[2].x - 20.0).abs() < 1e-6 && (ordered[2].y - 20.0).abs() < 1e-6);
        assert!((ordered[3].x - 10.0).abs() < 1e-6 && (ordered[3].y - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_mini_boxes_from_points_returns_min_side() {
        let post = DBPostProcess::new(None, None, None, None, None, None, None);
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(0.0, 5.0),
        ];

        let (_, min_side) = post
            .get_mini_boxes_from_points(&points)
            .expect("expected mini box");
        assert!((min_side - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_simplify_chain_points_removes_straight_segment_points() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(2.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(1.0, 2.0),
            Point::new(0.0, 2.0),
            Point::new(0.0, 1.0),
        ];

        let simplified = DBPostProcess::simplify_chain_points(&points);
        assert_eq!(simplified.len(), 4);
    }
}
