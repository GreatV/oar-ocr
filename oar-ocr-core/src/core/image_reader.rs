//! Default implementation of the `ImageReader` trait used by inference builders.

use crate::core::{errors::OCRError, traits::ImageReader};
use image::RgbImage;
use std::path::Path;

/// Reads RGB images from disk with optional parallel batching.
#[derive(Debug)]
pub struct DefaultImageReader {
    /// Optional threshold that switches the loader into parallel mode.
    parallel_threshold: Option<usize>,
}

impl DefaultImageReader {
    /// Creates a reader with no parallel threshold.
    pub fn new() -> Self {
        Self {
            parallel_threshold: None,
        }
    }

    /// Creates a reader that switches to parallel loading after the given threshold.
    pub fn with_parallel_threshold(parallel_threshold: usize) -> Self {
        Self {
            parallel_threshold: Some(parallel_threshold),
        }
    }
}

impl Default for DefaultImageReader {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageReader for DefaultImageReader {
    type Error = OCRError;

    fn apply<P: AsRef<Path> + Send + Sync>(
        &self,
        imgs: impl IntoIterator<Item = P>,
    ) -> Result<Vec<RgbImage>, Self::Error> {
        use crate::utils::load_images_batch_with_threshold;

        let img_paths: Vec<_> = imgs.into_iter().collect();
        load_images_batch_with_threshold(&img_paths, self.parallel_threshold)
    }
}
