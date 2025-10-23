//! Layout Detection Post-processing
//!
//! This module implements post-processing for layout detection models including
//! PicoDet, RT-DETR, and PP-DocLayout series models.

use crate::core::Tensor4D;
use crate::processors::{BoundingBox, Point};
use ndarray::{ArrayView3, Axis};
use std::borrow::Cow;

/// Layout detection post-processor for models like PicoDet and RT-DETR.
///
/// This processor converts model predictions into bounding boxes with class labels
/// and confidence scores for document layout elements.
#[derive(Debug, Clone)]
pub struct LayoutPostProcess {
    /// Number of classes the model predicts
    num_classes: usize,
    /// Score threshold for filtering predictions
    score_threshold: f32,
    /// Non-maximum suppression threshold
    nms_threshold: f32,
    /// Maximum number of detections to return
    max_detections: usize,
    /// Model type (e.g., "picodet", "rtdetr", "pp-doclayout")
    model_type: String,
}

impl LayoutPostProcess {
    /// Creates a new layout detection post-processor.
    pub fn new(
        num_classes: usize,
        score_threshold: f32,
        nms_threshold: f32,
        max_detections: usize,
        model_type: String,
    ) -> Self {
        Self {
            num_classes,
            score_threshold,
            nms_threshold,
            max_detections,
            model_type,
        }
    }

    /// Applies post-processing to layout detection model predictions.
    ///
    /// # Arguments
    /// * `predictions` - Model output tensor [batch, num_boxes, 4 + num_classes]
    /// * `img_shapes` - Original image dimensions for each image in batch
    ///
    /// # Returns
    /// Tuple of (bounding_boxes, class_ids, scores) for each image in batch
    pub fn apply(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<[f32; 4]>,
    ) -> (Vec<Vec<BoundingBox>>, Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let batch_size = predictions.shape()[0];
        let mut all_boxes = Vec::with_capacity(batch_size);
        let mut all_classes = Vec::with_capacity(batch_size);
        let mut all_scores = Vec::with_capacity(batch_size);

        // Process each image in batch
        for batch_idx in 0..batch_size {
            let pred = predictions.index_axis(Axis(0), batch_idx);
            let img_shape = img_shapes[batch_idx];

            let (boxes, classes, scores) = match self.model_type.as_str() {
                "picodet" => self.process_picodet(pred, img_shape),
                "rtdetr" => self.process_rtdetr(pred, img_shape),
                "pp-doclayout" => self.process_pp_doclayout(pred, img_shape),
                _ => self.process_standard(pred, img_shape),
            };

            all_boxes.push(boxes);
            all_classes.push(classes);
            all_scores.push(scores);
        }

        (all_boxes, all_classes, all_scores)
    }

    /// Process PicoDet model output.
    fn process_picodet(
        &self,
        predictions: ArrayView3<f32>,
        img_shape: [f32; 4],
    ) -> (Vec<BoundingBox>, Vec<usize>, Vec<f32>) {
        let mut boxes = Vec::new();
        let mut classes = Vec::new();
        let mut scores = Vec::new();

        let orig_width = img_shape[1];
        let orig_height = img_shape[0];
        let shape = predictions.shape();
        if shape.len() != 3 || shape[2] == 0 {
            return (boxes, classes, scores);
        }

        let total_boxes = shape[0] * shape[1];
        if total_boxes == 0 {
            return (boxes, classes, scores);
        }

        let feature_dim = shape[2];
        let data: Cow<'_, [f32]> = match predictions.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => {
                let (mut vec, offset) = predictions.to_owned().into_raw_vec_and_offset();
                if let Some(offset) = offset {
                    if offset != 0 {
                        vec.drain(0..offset);
                    }
                }
                Cow::Owned(vec)
            }
        };

        for box_idx in 0..total_boxes {
            let start = box_idx * feature_dim;
            let end = start + feature_dim;

            if end > data.len() {
                break;
            }

            let row = &data[start..end];
            if feature_dim == 4 + self.num_classes {
                // Format: [x1, y1, x2, y2, scores...]
                let (max_class, max_score) = row[4..].iter().enumerate().fold(
                    (0usize, 0.0f32),
                    |(best_cls, best_score), (cls_idx, &score)| {
                        if score > best_score {
                            (cls_idx, score)
                        } else {
                            (best_cls, best_score)
                        }
                    },
                );

                if max_score < self.score_threshold {
                    continue;
                }

                let (sx1, sy1, sx2, sy2) = self.convert_bbox_coords(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    orig_width,
                    orig_height,
                );

                if !Self::is_valid_box(sx1, sy1, sx2, sy2) {
                    continue;
                }

                let bbox = BoundingBox::new(vec![
                    Point::new(sx1, sy1),
                    Point::new(sx2, sy1),
                    Point::new(sx2, sy2),
                    Point::new(sx1, sy2),
                ]);

                boxes.push(bbox);
                classes.push(max_class);
                scores.push(max_score);
            } else if feature_dim >= 6 {
                if let Some((class_id, score, x1, y1, x2, y2)) = self.parse_compact_prediction(row)
                {
                    if score < self.score_threshold || class_id >= self.num_classes {
                        continue;
                    }

                    let (sx1, sy1, sx2, sy2) =
                        self.convert_bbox_coords(x1, y1, x2, y2, orig_width, orig_height);

                    if !Self::is_valid_box(sx1, sy1, sx2, sy2) {
                        continue;
                    }

                    let bbox = BoundingBox::new(vec![
                        Point::new(sx1, sy1),
                        Point::new(sx2, sy1),
                        Point::new(sx2, sy2),
                        Point::new(sx1, sy2),
                    ]);

                    boxes.push(bbox);
                    classes.push(class_id);
                    scores.push(score);
                }
            }
        }

        self.apply_nms(boxes, classes, scores)
    }

    /// Process RT-DETR model output.
    fn process_rtdetr(
        &self,
        predictions: ArrayView3<f32>,
        img_shape: [f32; 4],
    ) -> (Vec<BoundingBox>, Vec<usize>, Vec<f32>) {
        // RT-DETR has similar output format to PicoDet
        self.process_picodet(predictions, img_shape)
    }

    /// Process PP-DocLayout model output.
    fn process_pp_doclayout(
        &self,
        predictions: ArrayView3<f32>,
        img_shape: [f32; 4],
    ) -> (Vec<BoundingBox>, Vec<usize>, Vec<f32>) {
        // PP-DocLayout outputs in [num_boxes, 1, 6] format
        // where each box is [class_id, score, x1, y1, x2, y2]
        let shape = predictions.shape();
        let num_boxes = shape[0];

        let mut boxes = Vec::new();
        let mut classes = Vec::new();
        let mut scores = Vec::new();

        let _orig_width = img_shape[1];
        let _orig_height = img_shape[0];

        // Extract predictions
        for box_idx in 0..num_boxes {
            // predictions is [num_boxes, 1, 6], so we use 3D indexing [box_idx, 0, i]
            let class_id = predictions[[box_idx, 0, 0]] as i32;
            let score = predictions[[box_idx, 0, 1]];
            let x1 = predictions[[box_idx, 0, 2]];
            let y1 = predictions[[box_idx, 0, 3]];
            let x2 = predictions[[box_idx, 0, 4]];
            let y2 = predictions[[box_idx, 0, 5]];

            // Filter by threshold and valid class
            if score >= self.score_threshold
                && class_id >= 0
                && (class_id as usize) < self.num_classes
            {
                // Note: PP-DocLayout already outputs scaled coordinates
                // No need to scale to original image size
                let bbox = BoundingBox::new(vec![
                    Point::new(x1, y1),
                    Point::new(x2, y1),
                    Point::new(x2, y2),
                    Point::new(x1, y2),
                ]);

                boxes.push(bbox);
                classes.push(class_id as usize);
                scores.push(score);
            }
        }

        // Apply NMS
        let (filtered_boxes, filtered_classes, filtered_scores) =
            self.apply_nms(boxes, classes, scores);

        (filtered_boxes, filtered_classes, filtered_scores)
    }

    /// Process standard detection model output.
    fn process_standard(
        &self,
        predictions: ArrayView3<f32>,
        img_shape: [f32; 4],
    ) -> (Vec<BoundingBox>, Vec<usize>, Vec<f32>) {
        self.process_picodet(predictions, img_shape)
    }

    fn parse_compact_prediction(&self, row: &[f32]) -> Option<(usize, f32, f32, f32, f32, f32)> {
        if row.len() < 6 {
            return None;
        }

        // Format: [class_id, score, x1, y1, x2, y2]
        let score_is_valid = if self.model_type == "rtdetr" {
            row[1].is_finite()
        } else {
            Self::is_valid_score(row[1])
        };

        if score_is_valid && Self::is_valid_class(row[0], self.num_classes) {
            let class_id = row[0].round() as i32;
            if class_id >= 0 {
                let score = self.adjust_score(row[1]);
                return Some((class_id as usize, score, row[2], row[3], row[4], row[5]));
            }
        }

        // Alternate format: [x1, y1, x2, y2, score, class_id]
        let score_is_valid = if self.model_type == "rtdetr" {
            row[4].is_finite()
        } else {
            Self::is_valid_score(row[4])
        };
        if score_is_valid && Self::is_valid_class(row[5], self.num_classes) {
            let class_id = row[5].round() as i32;
            if class_id >= 0 {
                let score = self.adjust_score(row[4]);
                return Some((class_id as usize, score, row[0], row[1], row[2], row[3]));
            }
        }

        // Alternate format: [score, class_id, x1, y1, x2, y2]
        let score_is_valid = if self.model_type == "rtdetr" {
            row[0].is_finite()
        } else {
            Self::is_valid_score(row[0])
        };
        if score_is_valid && Self::is_valid_class(row[1], self.num_classes) {
            let class_id = row[1].round() as i32;
            if class_id >= 0 {
                let score = self.adjust_score(row[0]);
                return Some((class_id as usize, score, row[2], row[3], row[4], row[5]));
            }
        }

        None
    }

    fn convert_bbox_coords(
        &self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        orig_width: f32,
        orig_height: f32,
    ) -> (f32, f32, f32, f32) {
        let normalized = x2 <= 1.05
            && y2 <= 1.05
            && x1 >= -0.05
            && y1 >= -0.05
            && orig_width > 0.0
            && orig_height > 0.0;

        if normalized {
            (
                x1.clamp(0.0, 1.0) * orig_width,
                y1.clamp(0.0, 1.0) * orig_height,
                x2.clamp(0.0, 1.0) * orig_width,
                y2.clamp(0.0, 1.0) * orig_height,
            )
        } else {
            (
                x1.clamp(0.0, orig_width),
                y1.clamp(0.0, orig_height),
                x2.clamp(0.0, orig_width),
                y2.clamp(0.0, orig_height),
            )
        }
    }

    fn is_valid_box(x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
        x2 > x1 && y2 > y1 && x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite()
    }

    fn is_valid_score(score: f32) -> bool {
        score.is_finite() && score >= 0.0 && score <= 1.0 + f32::EPSILON
    }

    fn is_valid_class(raw: f32, num_classes: usize) -> bool {
        if !raw.is_finite() {
            return false;
        }
        let class_id = raw.round() as i32;
        class_id >= 0 && (class_id as usize) < num_classes + 5
    }

    fn adjust_score(&self, raw_score: f32) -> f32 {
        if self.model_type == "rtdetr" {
            raw_score.clamp(0.0, 1.0)
        } else {
            raw_score
        }
    }

    /// Apply Non-Maximum Suppression to filter overlapping boxes.
    fn apply_nms(
        &self,
        boxes: Vec<BoundingBox>,
        classes: Vec<usize>,
        scores: Vec<f32>,
    ) -> (Vec<BoundingBox>, Vec<usize>, Vec<f32>) {
        if boxes.is_empty() {
            return (boxes, classes, scores);
        }

        // Sort by score in descending order
        let mut indices: Vec<usize> = (0..boxes.len()).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        let mut keep = Vec::new();
        let mut suppressed = vec![false; boxes.len()];

        for &i in &indices {
            if suppressed[i] {
                continue;
            }

            keep.push(i);
            if keep.len() >= self.max_detections {
                break;
            }

            // Suppress boxes with high IoU
            for &j in &indices {
                if i != j && !suppressed[j] && classes[i] == classes[j] {
                    let iou = self.calculate_iou(&boxes[i], &boxes[j]);
                    if iou > self.nms_threshold {
                        suppressed[j] = true;
                    }
                }
            }
        }

        let filtered_boxes: Vec<BoundingBox> = keep.iter().map(|&i| boxes[i].clone()).collect();
        let filtered_classes: Vec<usize> = keep.iter().map(|&i| classes[i]).collect();
        let filtered_scores: Vec<f32> = keep.iter().map(|&i| scores[i]).collect();

        (filtered_boxes, filtered_classes, filtered_scores)
    }

    /// Calculate Intersection over Union between two bounding boxes.
    fn calculate_iou(&self, box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        // Get bounding rectangle for box1
        let (x1_min, y1_min, x1_max, y1_max) = self.get_bbox_bounds(box1);

        // Get bounding rectangle for box2
        let (x2_min, y2_min, x2_max, y2_max) = self.get_bbox_bounds(box2);

        // Calculate intersection
        let x_min = x1_min.max(x2_min);
        let y_min = y1_min.max(y2_min);
        let x_max = x1_max.min(x2_max);
        let y_max = y1_max.min(y2_max);

        if x_max <= x_min || y_max <= y_min {
            return 0.0;
        }

        let intersection = (x_max - x_min) * (y_max - y_min);
        let area1 = (x1_max - x1_min) * (y1_max - y1_min);
        let area2 = (x2_max - x2_min) * (y2_max - y2_min);
        let union = area1 + area2 - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Get the minimum and maximum coordinates from a bounding box.
    fn get_bbox_bounds(&self, bbox: &BoundingBox) -> (f32, f32, f32, f32) {
        if bbox.points.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mut x_min = f32::INFINITY;
        let mut y_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_max = f32::NEG_INFINITY;

        for point in &bbox.points {
            x_min = x_min.min(point.x);
            y_min = y_min.min(point.y);
            x_max = x_max.max(point.x);
            y_max = y_max.max(point.y);
        }

        (x_min, y_min, x_max, y_max)
    }
}

impl Default for LayoutPostProcess {
    fn default() -> Self {
        Self {
            num_classes: 5, // Default for basic layout detection
            score_threshold: 0.5,
            nms_threshold: 0.5,
            max_detections: 100,
            model_type: "picodet".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_postprocess_creation() {
        let processor = LayoutPostProcess::default();
        assert_eq!(processor.num_classes, 5);
        assert_eq!(processor.score_threshold, 0.5);
    }

    #[test]
    fn test_iou_calculation() {
        let processor = LayoutPostProcess::default();

        // Two identical boxes should have IoU = 1.0
        let box1 = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 100.0),
            Point::new(0.0, 100.0),
        ]);
        let box2 = box1.clone();

        assert_eq!(processor.calculate_iou(&box1, &box2), 1.0);

        // Non-overlapping boxes should have IoU = 0.0
        let box3 = BoundingBox::new(vec![
            Point::new(200.0, 200.0),
            Point::new(300.0, 200.0),
            Point::new(300.0, 300.0),
            Point::new(200.0, 300.0),
        ]);

        assert_eq!(processor.calculate_iou(&box1, &box3), 0.0);
    }
}
