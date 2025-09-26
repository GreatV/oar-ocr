//! Post-processing for DB (Differentiable Binarization) text detection models.
//!
//! The [`DBPostProcess`] struct converts raw detection heatmaps into geometric
//! bounding boxes by thresholding, contour extraction, scoring, and optional
//! polygonal post-processing. Supporting functionality (bitmap extraction,
//! scoring, mask morphology) is split across helper modules within this
//! directory.

#[path = "db_bitmap.rs"]
mod db_bitmap;
#[path = "db_mask.rs"]
mod db_mask;
#[path = "db_score.rs"]
mod db_score;

use crate::core::Tensor4D;
use crate::processors::geometry::BoundingBox;
use crate::processors::types::{BoxType, ScoreMode};
use ndarray::Axis;

/// Post-processor for DB (Differentiable Binarization) text detection models.
#[derive(Debug)]
pub struct DBPostProcess {
    /// Threshold for binarizing the prediction map (default: 0.3).
    pub thresh: f32,
    /// Threshold for filtering bounding boxes based on their score (default: 0.7).
    pub box_thresh: f32,
    /// Maximum number of candidate bounding boxes to consider (default: 1000).
    pub max_candidates: usize,
    /// Ratio for unclipping (expanding) bounding boxes (default: 2.0).
    pub unclip_ratio: f32,
    /// Minimum side length for detected bounding boxes.
    pub min_size: f32,
    /// Method for calculating the score of a bounding box.
    pub score_mode: ScoreMode,
    /// Type of bounding box to generate (quadrilateral or polygon).
    pub box_type: BoxType,
    /// Whether to apply dilation to the segmentation mask before contour detection.
    pub use_dilation: bool,
}

impl DBPostProcess {
    /// Creates a new `DBPostProcess` instance with optional overrides.
    pub fn new(
        thresh: Option<f32>,
        box_thresh: Option<f32>,
        max_candidates: Option<usize>,
        unclip_ratio: Option<f32>,
        use_dilation: Option<bool>,
        score_mode: Option<ScoreMode>,
        box_type: Option<BoxType>,
    ) -> Self {
        Self {
            thresh: thresh.unwrap_or(0.3),
            box_thresh: box_thresh.unwrap_or(0.7),
            max_candidates: max_candidates.unwrap_or(1000),
            unclip_ratio: unclip_ratio.unwrap_or(2.0),
            min_size: 3.0,
            score_mode: score_mode.unwrap_or(ScoreMode::Fast),
            box_type: box_type.unwrap_or(BoxType::Quad),
            use_dilation: use_dilation.unwrap_or(false),
        }
    }

    /// Applies post-processing to a batch of prediction maps.
    pub fn apply(
        &self,
        preds: &Tensor4D,
        img_shapes: Vec<[f32; 4]>,
        thresh: Option<f32>,
        box_thresh: Option<f32>,
        unclip_ratio: Option<f32>,
    ) -> (Vec<Vec<BoundingBox>>, Vec<Vec<f32>>) {
        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();

        for (batch_idx, shape_batch) in img_shapes.iter().enumerate() {
            let pred_slice = preds.index_axis(Axis(0), batch_idx);
            let pred_channel = pred_slice.index_axis(Axis(0), 0);

            let (boxes, scores) = self.process(
                &pred_channel.to_owned(),
                *shape_batch,
                thresh.unwrap_or(self.thresh),
                box_thresh.unwrap_or(self.box_thresh),
                unclip_ratio.unwrap_or(self.unclip_ratio),
            );
            all_boxes.push(boxes);
            all_scores.push(scores);
        }

        (all_boxes, all_scores)
    }

    fn process(
        &self,
        pred: &ndarray::Array2<f32>,
        img_shape: [f32; 4],
        thresh: f32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let src_h = img_shape[0] as u32;
        let src_w = img_shape[1] as u32;

        let height = pred.shape()[0] as u32;
        let width = pred.shape()[1] as u32;

        let mut segmentation = vec![vec![false; width as usize]; height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                segmentation[y][x] = pred[[y, x]] > thresh;
            }
        }

        let mask = if self.use_dilation {
            self.dilate_mask(&segmentation)
        } else {
            segmentation
        };

        match self.box_type {
            BoxType::Poly => {
                self.polygons_from_bitmap(pred, &mask, src_w, src_h, box_thresh, unclip_ratio)
            }
            BoxType::Quad => {
                self.boxes_from_bitmap(pred, &mask, src_w, src_h, box_thresh, unclip_ratio)
            }
        }
    }
}
