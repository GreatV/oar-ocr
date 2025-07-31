//! Post-processing module for the OCR pipeline.
//!
//! This module contains implementations for post-processing steps in the OCR pipeline,
//! particularly for processing the output of text detection models like DB (Differentiable Binarization).
//! The main struct [`DBPostProcess`] handles the conversion of model predictions into bounding boxes
//! that represent detected text regions.

use crate::processors::geometry::{BoundingBox, Point, ScanlineBuffer};
use crate::processors::types::{BoxType, ScoreMode};
use image::GrayImage;
use imageproc::contours::{Contour, find_contours};
use imageproc::distance_transform::Norm;
use imageproc::morphology;
use itertools::Itertools;
use rayon::prelude::*;

/// Post-processor for DB (Differentiable Binarization) text detection models.
///
/// This struct contains parameters and methods for converting the output of a DB model
/// into bounding boxes that represent detected text regions. The process involves:
/// 1. Binarizing the prediction map using a threshold
/// 2. Finding connected components (contours) in the binary map
/// 3. Filtering contours based on size and score thresholds
/// 4. Expanding contours to compensate for the shrinkage effect during training
/// 5. Converting contours to bounding boxes (either quadrilateral or polygon)
#[derive(Debug)]
pub struct DBPostProcess {
    /// Threshold for binarizing the prediction map.
    ///
    /// Values in the prediction map above this threshold are considered text pixels.
    /// Default: 0.3
    pub thresh: f32,

    /// Threshold for filtering bounding boxes based on their score.
    ///
    /// Bounding boxes with a score below this threshold are discarded.
    /// Default: 0.7
    pub box_thresh: f32,

    /// Maximum number of candidate bounding boxes to consider.
    ///
    /// This limits the number of contours processed to prevent excessive computation.
    /// Default: 1000
    pub max_candidates: usize,

    /// Ratio for unclipping (expanding) bounding boxes.
    ///
    /// This value controls how much to expand the detected contours to compensate
    /// for the shrinkage effect during training. Higher values result in larger boxes.
    /// Default: 2.0
    pub unclip_ratio: f32,

    /// Minimum size for detected bounding boxes.
    ///
    /// Bounding boxes with a side length smaller than this value are discarded.
    /// Default: 3.0
    pub min_size: f32,

    /// Method for calculating the score of a bounding box.
    ///
    /// Can be either `Fast` (uses a simplified calculation) or `Slow` (more accurate but slower).
    /// Default: ScoreMode::Fast
    pub score_mode: ScoreMode,

    /// Type of bounding box to generate.
    ///
    /// Can be either `Quad` (quadrilateral) or `Poly` (polygon with more than 4 points).
    /// Default: BoxType::Quad
    pub box_type: BoxType,

    /// Whether to apply dilation to the segmentation mask.
    ///
    /// Dilation can help connect nearby text components but may also merge separate words.
    /// Default: false
    pub use_dilation: bool,
}

impl DBPostProcess {
    /// Creates a new DBPostProcess instance with optional parameters.
    ///
    /// If any parameter is None, a default value will be used.
    ///
    /// # Parameters
    /// * `thresh` - Threshold for binarizing the prediction map (default: 0.3)
    /// * `box_thresh` - Threshold for filtering bounding boxes based on their score (default: 0.7)
    /// * `max_candidates` - Maximum number of candidate bounding boxes to consider (default: 1000)
    /// * `unclip_ratio` - Ratio for unclipping (expanding) bounding boxes (default: 2.0)
    /// * `use_dilation` - Whether to apply dilation to the segmentation mask (default: false)
    /// * `score_mode` - Method for calculating the score of a bounding box (default: ScoreMode::Fast)
    /// * `box_type` - Type of bounding box to generate (default: BoxType::Quad)
    ///
    /// # Returns
    /// A new DBPostProcess instance with the specified or default parameters.
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

    /// Applies post-processing to the model predictions to generate bounding boxes.
    ///
    /// This function processes a batch of predictions from the DB model and converts them
    /// into bounding boxes representing detected text regions.
    ///
    /// # Parameters
    /// * `preds` - The model predictions as a 4D tensor
    /// * `img_shapes` - The shapes of the original images in the batch
    /// * `thresh` - Optional threshold for binarizing the prediction map (uses instance default if None)
    /// * `box_thresh` - Optional threshold for filtering bounding boxes (uses instance default if None)
    /// * `unclip_ratio` - Optional ratio for unclipping bounding boxes (uses instance default if None)
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of vectors of BoundingBox objects, one vector for each image in the batch
    /// * A vector of vectors of scores, one vector for each image in the batch
    pub fn apply(
        &self,
        preds: &crate::core::Tensor4D,
        img_shapes: Vec<[f32; 4]>,
        thresh: Option<f32>,
        box_thresh: Option<f32>,
        unclip_ratio: Option<f32>,
    ) -> (Vec<Vec<BoundingBox>>, Vec<Vec<f32>>) {
        let mut all_boxes = Vec::new();
        let mut all_scores = Vec::new();

        for (batch_idx, shape_batch) in img_shapes.iter().enumerate() {
            let pred_slice = preds.index_axis(ndarray::Axis(0), batch_idx);
            let pred_channel = pred_slice.index_axis(ndarray::Axis(0), 0);

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

    /// Processes a single prediction map to generate bounding boxes.
    ///
    /// This function takes a single 2D prediction map and converts it into bounding boxes
    /// representing detected text regions. It handles the core post-processing logic:
    /// binarization, contour detection, filtering, and box generation.
    ///
    /// # Parameters
    /// * `pred` - The 2D prediction map for a single image
    /// * `img_shape` - The shape of the original image
    /// * `thresh` - Threshold for binarizing the prediction map
    /// * `box_thresh` - Threshold for filtering bounding boxes based on their score
    /// * `unclip_ratio` - Ratio for unclipping (expanding) bounding boxes
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of BoundingBox objects representing detected text regions
    /// * A vector of scores for each bounding box
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

    /// Converts a bitmap to polygon bounding boxes.
    ///
    /// This function takes a binary bitmap and converts it into polygon bounding boxes
    /// with more than 4 points. It finds contours in the bitmap, filters them based on
    /// size and score, and then expands them using the unclip ratio.
    ///
    /// # Parameters
    /// * `pred` - The prediction map used for scoring contours
    /// * `bitmap` - The binary bitmap representing text regions
    /// * `dest_width` - The width of the destination image
    /// * `dest_height` - The height of the destination image
    /// * `box_thresh` - Threshold for filtering bounding boxes based on their score
    /// * `unclip_ratio` - Ratio for unclipping (expanding) bounding boxes
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of polygon BoundingBox objects
    /// * A vector of scores for each bounding box
    fn polygons_from_bitmap(
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

    /// Converts a bitmap to quadrilateral bounding boxes.
    ///
    /// This function takes a binary bitmap and converts it into quadrilateral bounding boxes.
    /// It finds contours in the bitmap, filters them based on size and score, and then
    /// expands them using the unclip ratio before converting to minimum area rectangles.
    ///
    /// # Parameters
    /// * `pred` - The prediction map used for scoring contours
    /// * `bitmap` - The binary bitmap representing text regions
    /// * `dest_width` - The width of the destination image
    /// * `dest_height` - The height of the destination image
    /// * `box_thresh` - Threshold for filtering bounding boxes based on their score
    /// * `unclip_ratio` - Ratio for unclipping (expanding) bounding boxes
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of quadrilateral BoundingBox objects
    /// * A vector of scores for each bounding box
    fn boxes_from_bitmap(
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

    /// Calculates the score of a bounding box using a fast approximation method.
    ///
    /// This function calculates the average prediction score within a bounding box
    /// by sampling points in the bounding box and averaging their scores. It uses
    /// a scanline algorithm for efficient computation.
    ///
    /// # Parameters
    /// * `pred` - The prediction map
    /// * `bbox` - The bounding box to score
    ///
    /// # Returns
    /// The average score of the bounding box
    pub fn box_score_fast(&self, pred: &ndarray::Array2<f32>, bbox: &BoundingBox) -> f32 {
        let height = pred.shape()[0];
        let width = pred.shape()[1];

        let (min_x, max_x) = bbox
            .points
            .iter()
            .map(|p| p.x)
            .minmax()
            .into_option()
            .unwrap_or((0.0, 0.0));
        let (min_y, max_y) = bbox
            .points
            .iter()
            .map(|p| p.y)
            .minmax()
            .into_option()
            .unwrap_or((0.0, 0.0));

        let min_x = min_x.max(0.0).min(width as f32 - 1.0);
        let max_x = max_x.max(0.0).min(width as f32 - 1.0);
        let min_y = min_y.max(0.0).min(height as f32 - 1.0);
        let max_y = max_y.max(0.0).min(height as f32 - 1.0);

        let start_y = min_y as usize;
        let end_y = max_y as usize + 1;
        let start_x = min_x as usize;
        let end_x = max_x as usize + 1;

        self.box_score_fast_contour(pred, bbox, start_y, end_y, start_x, end_x)
    }

    /// Calculates the score of a bounding box using a fast approximation method with scanline algorithm.
    ///
    /// This function calculates the average prediction score within a bounding box by using
    /// a scanline algorithm to efficiently sample points in the bounding box. For small regions,
    /// it uses a single thread, but for larger regions it uses parallel processing.
    ///
    /// # Parameters
    /// * `pred` - The prediction map
    /// * `bbox` - The bounding box to score
    /// * `start_y` - The starting y-coordinate of the region to process
    /// * `end_y` - The ending y-coordinate of the region to process
    /// * `start_x` - The starting x-coordinate of the region to process
    /// * `end_x` - The ending x-coordinate of the region to process
    ///
    /// # Returns
    /// The average score of the bounding box
    fn box_score_fast_contour(
        &self,
        pred: &ndarray::Array2<f32>,
        bbox: &BoundingBox,
        start_y: usize,
        end_y: usize,
        start_x: usize,
        end_x: usize,
    ) -> f32 {
        let region_height = end_y - start_y;
        let region_width = end_x - start_x;

        let max_polygon_points = bbox.points.len();
        let mut scanline_buffer = ScanlineBuffer::new(max_polygon_points);

        if region_height * region_width < 8_000 {
            let mut total_score = 0.0;
            let mut total_pixels = 0;

            for y in start_y..end_y {
                let scanline_y = y as f32 + 0.5;
                let (line_score, line_pixels) =
                    scanline_buffer.process_scanline(scanline_y, bbox, start_x, end_x, pred);
                total_score += line_score;
                total_pixels += line_pixels;
            }

            if total_pixels > 0 {
                total_score / total_pixels as f32
            } else {
                0.0
            }
        } else {
            let scanline_results: Vec<(f32, usize)> = (start_y..end_y)
                .into_par_iter()
                .map(|y| {
                    let scanline_y = y as f32 + 0.5;

                    let mut thread_buffer = ScanlineBuffer::new(max_polygon_points);
                    thread_buffer.process_scanline(scanline_y, bbox, start_x, end_x, pred)
                })
                .collect();

            let total_score: f32 = scanline_results.iter().map(|(score, _)| score).sum();
            let total_pixels: usize = scanline_results.iter().map(|(_, pixels)| pixels).sum();

            if total_pixels > 0 {
                total_score / total_pixels as f32
            } else {
                0.0
            }
        }
    }

    /// Calculates the score of a contour using a slower but more accurate method.
    ///
    /// This function calculates the average prediction score of points along a contour.
    /// It's more accurate than the fast method but slower because it processes each
    /// point in the contour individually.
    ///
    /// # Parameters
    /// * `pred` - The prediction map
    /// * `contour` - The contour to score
    ///
    /// # Returns
    /// The average score of the contour
    fn box_score_slow(&self, pred: &ndarray::Array2<f32>, contour: &Contour<u32>) -> f32 {
        let mut total_score = 0.0;
        let mut pixel_count = 0;

        for point in &contour.points {
            let x = point.x as usize;
            let y = point.y as usize;

            if y < pred.shape()[0] && x < pred.shape()[1] {
                total_score += pred[[y, x]];
                pixel_count += 1;
            }
        }

        if pixel_count > 0 {
            total_score / pixel_count as f32
        } else {
            0.0
        }
    }

    /// Expands a bounding box using the unclip ratio.
    ///
    /// This function expands a bounding box to compensate for the shrinkage effect
    /// that occurs during training. It calculates the area and perimeter of the
    /// bounding box and uses these values along with the unclip ratio to determine
    /// how much to expand the box.
    ///
    /// # Parameters
    /// * `bbox` - The bounding box to expand
    /// * `unclip_ratio` - The ratio for expanding the bounding box
    ///
    /// # Returns
    /// The expanded bounding box
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

    /// Applies dilation to a binary mask.
    ///
    /// This function applies morphological dilation to a binary mask using
    /// the LInf norm (Chebyshev distance) with a radius of 1. This can help
    /// connect nearby text components but may also merge separate words.
    ///
    /// # Parameters
    /// * `mask` - The binary mask to dilate
    ///
    /// # Returns
    /// The dilated binary mask
    fn dilate_mask(&self, mask: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let height = mask.len();
        let width = if height > 0 { mask[0].len() } else { 0 };

        if height == 0 || width == 0 {
            return vec![vec![false; width]; height];
        }

        let mut gray_img = GrayImage::new(width as u32, height as u32);
        for (y, row) in mask.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let pixel_value = if value { 255 } else { 0 };
                gray_img.put_pixel(x as u32, y as u32, image::Luma([pixel_value]));
            }
        }

        let dilated_img = morphology::dilate(&gray_img, Norm::LInf, 1);

        let mut dilated = vec![vec![false; width]; height];
        for (y, dilated_row) in dilated.iter_mut().enumerate() {
            for (x, dilated_pixel) in dilated_row.iter_mut().enumerate() {
                let pixel = dilated_img.get_pixel(x as u32, y as u32);
                *dilated_pixel = pixel[0] > 0;
            }
        }

        dilated
    }
}
