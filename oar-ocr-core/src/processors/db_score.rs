use super::DBPostProcess;
use crate::processors::geometry::{BoundingBox, ScanlineBuffer};
use itertools::Itertools;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct Region {
    start_y: usize,
    end_y: usize,
    start_x: usize,
    end_x: usize,
}

impl Region {
    fn new(start_y: usize, end_y: usize, start_x: usize, end_x: usize) -> Self {
        Self {
            start_y,
            end_y,
            start_x,
            end_x,
        }
    }

    fn height(&self) -> usize {
        self.end_y - self.start_y
    }

    fn width(&self) -> usize {
        self.end_x - self.start_x
    }
}

impl DBPostProcess {
    /// Calculates the score of a bounding box using a fast approximation method.
    pub fn box_score_fast(&self, pred: &ndarray::ArrayView2<f32>, bbox: &BoundingBox) -> f32 {
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

    fn box_score_fast_contour(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        bbox: &BoundingBox,
        start_y: usize,
        end_y: usize,
        start_x: usize,
        end_x: usize,
    ) -> f32 {
        let region = Region::new(start_y, end_y, start_x, end_x);
        self.box_score_fast_contour_with_policy(pred, bbox, region, None)
    }

    fn box_score_fast_contour_with_policy(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        bbox: &BoundingBox,
        region: Region,
        policy: Option<&crate::core::config::ParallelPolicy>,
    ) -> f32 {
        let region_height = region.height();
        let region_width = region.width();

        let max_polygon_points = bbox.points.len();
        let mut scanline_buffer = ScanlineBuffer::new(max_polygon_points);

        let pixel_threshold = policy
            .map(|p| p.postprocess_pixel_threshold)
            .unwrap_or(8_000);

        if region_height * region_width < pixel_threshold {
            let mut total_score = 0.0;
            let mut total_pixels = 0;

            for y in region.start_y..region.end_y {
                let scanline_y = y as f32 + 0.5;
                let (line_score, line_pixels) = scanline_buffer.process_scanline(
                    scanline_y,
                    bbox,
                    region.start_x,
                    region.end_x,
                    pred,
                );
                total_score += line_score;
                total_pixels += line_pixels;
            }

            if total_pixels > 0 {
                total_score / total_pixels as f32
            } else {
                0.0
            }
        } else {
            let scanline_results: Vec<(f32, usize)> = (region.start_y..region.end_y)
                .into_par_iter()
                .map(|y| {
                    let scanline_y = y as f32 + 0.5;

                    let mut thread_buffer = ScanlineBuffer::new(max_polygon_points);
                    thread_buffer.process_scanline(
                        scanline_y,
                        bbox,
                        region.start_x,
                        region.end_x,
                        pred,
                    )
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

    pub(super) fn box_score_slow(
        &self,
        pred: &ndarray::ArrayView2<f32>,
        contour: &imageproc::contours::Contour<u32>,
    ) -> f32 {
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
}
