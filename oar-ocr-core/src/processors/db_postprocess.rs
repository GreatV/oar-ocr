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

use crate::processors::geometry::BoundingBox;
use crate::processors::types::{BoxType, ImageScaleInfo, ScoreMode};
use ndarray::Axis;
use rayon::prelude::*;

/// Runtime configuration for DB post-processing.
///
/// This struct contains parameters that may vary per inference call,
/// such as detection thresholds and expansion ratios.
#[derive(Debug, Clone)]
pub struct DBPostProcessConfig {
    /// Threshold for binarizing the prediction map.
    pub thresh: f32,
    /// Threshold for filtering bounding boxes based on their score.
    pub box_thresh: f32,
    /// Ratio for unclipping (expanding) bounding boxes.
    pub unclip_ratio: f32,
}

impl DBPostProcessConfig {
    /// Creates a new runtime config with specified values.
    pub fn new(thresh: f32, box_thresh: f32, unclip_ratio: f32) -> Self {
        Self {
            thresh,
            box_thresh,
            unclip_ratio,
        }
    }
}

/// Post-processor for DB (Differentiable Binarization) text detection models.
#[derive(Debug)]
pub struct DBPostProcess {
    /// Default threshold for binarizing the prediction map (default: 0.3).
    pub thresh: f32,
    /// Default threshold for filtering bounding boxes based on their score (default: 0.6).
    pub box_thresh: f32,
    /// Maximum number of candidate bounding boxes to consider (default: 1000).
    pub max_candidates: usize,
    /// Default ratio for unclipping (expanding) bounding boxes (default: 1.5).
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
            box_thresh: box_thresh.unwrap_or(0.6),
            max_candidates: max_candidates.unwrap_or(1000),
            unclip_ratio: unclip_ratio.unwrap_or(1.5),
            min_size: 3.0,
            score_mode: score_mode.unwrap_or(ScoreMode::Fast),
            box_type: box_type.unwrap_or(BoxType::Quad),
            use_dilation: use_dilation.unwrap_or(false),
        }
    }

    /// Applies post-processing to a batch of prediction maps.
    ///
    /// # Arguments
    /// * `preds` - Model predictions (batch of heatmaps)
    /// * `img_shapes` - Original image dimensions for each image in batch
    /// * `config` - Runtime configuration for thresholds and ratios.
    ///   If `None`, uses the default values stored in this processor.
    ///
    /// # Returns
    /// Tuple of (bounding_boxes, scores) for each image in batch
    pub fn apply(
        &self,
        preds: &ndarray::Array4<f32>,
        img_shapes: Vec<ImageScaleInfo>,
        config: Option<&DBPostProcessConfig>,
    ) -> (Vec<Vec<BoundingBox>>, Vec<Vec<f32>>) {
        // Use provided config or fall back to stored defaults
        let thresh = config.map(|c| c.thresh).unwrap_or(self.thresh);
        let box_thresh = config.map(|c| c.box_thresh).unwrap_or(self.box_thresh);
        let unclip_ratio = config.map(|c| c.unclip_ratio).unwrap_or(self.unclip_ratio);

        // Process per-image: each batch entry is independent, so we can
        // process them in parallel via rayon. The body of `process` itself
        // parallelises the thresholding step below, so we keep this loop
        // serial here — fanning out the contour/postprocess work for a
        // small batch (<= a few images) rarely pays back the rayon setup
        // cost, but the dominant cost is the thresholded bitmap build
        // which is now multi-threaded.
        let mut all_boxes = Vec::with_capacity(img_shapes.len());
        let mut all_scores = Vec::with_capacity(img_shapes.len());

        for (batch_idx, shape_batch) in img_shapes.iter().enumerate() {
            let pred_slice = preds.index_axis(Axis(0), batch_idx);
            let pred_channel = pred_slice.index_axis(Axis(0), 0);

            let (boxes, scores) =
                self.process(&pred_channel, shape_batch, thresh, box_thresh, unclip_ratio);
            all_boxes.push(boxes);
            all_scores.push(scores);
        }

        (all_boxes, all_scores)
    }

    fn process(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        img_shape: &ImageScaleInfo,
        thresh: f32,
        box_thresh: f32,
        unclip_ratio: f32,
    ) -> (Vec<BoundingBox>, Vec<f32>) {
        let src_h = img_shape.src_h as u32;
        let src_w = img_shape.src_w as u32;

        let height = pred.shape()[0] as u32;
        let width = pred.shape()[1] as u32;

        tracing::debug!(
            "DBPostProcess: pred {}x{}, src {}x{} (dest dimensions)",
            height,
            width,
            src_h,
            src_w
        );

        // Build binary mask directly as a `GrayImage` buffer. The previous
        // implementation called `put_pixel` per element, which involves a
        // bounds check and a function call for every pixel. Writing into
        // the buffer with a single pass over the prediction map (chunked
        // across rows) is several times faster and trivially parallelises
        // over rows via rayon.
        let mask_img = self.threshold_to_mask(pred, thresh);

        // Apply dilation if needed
        let mask_img = if self.use_dilation {
            self.dilate_mask_img(&mask_img)
        } else {
            mask_img
        };

        match self.box_type {
            BoxType::Poly => {
                self.polygons_from_bitmap(pred, &mask_img, src_w, src_h, box_thresh, unclip_ratio)
            }
            BoxType::Quad => {
                self.boxes_from_bitmap(pred, &mask_img, src_w, src_h, box_thresh, unclip_ratio)
            }
        }
    }

    /// Builds a binary segmentation mask from a prediction heatmap.
    ///
    /// Writes into the `GrayImage` row buffer directly (no per-pixel
    /// `put_pixel` overhead) and parallelises the work over rows.
    fn threshold_to_mask(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        thresh: f32,
    ) -> image::GrayImage {
        let height = pred.shape()[0] as u32;
        let width = pred.shape()[1] as u32;
        let mut mask_img = image::GrayImage::new(width, height);
        let buf: &mut [u8] = mask_img.as_mut();
        let row_bytes = width as usize;

        // Fill the mask row by row. Pick between serial and parallel branches
        // based on the number of rows: for small masks the rayon per-row
        // overhead dominates; for typical 800x800 DB outputs (640x640
        // predictions), the parallel branch wins by a wide margin.
        let fill_row = |y: usize, row: &mut [u8]| {
            let pred_row = pred.row(y);
            for (x, dst) in row.iter_mut().enumerate() {
                *dst = if pred_row[x] > thresh { 255 } else { 0 };
            }
        };
        if height >= 64 {
            buf.par_chunks_mut(row_bytes)
                .enumerate()
                .for_each(|(y, row)| fill_row(y, row));
        } else {
            buf.chunks_mut(row_bytes)
                .enumerate()
                .for_each(|(y, row)| fill_row(y, row));
        }

        mask_img
    }
}
