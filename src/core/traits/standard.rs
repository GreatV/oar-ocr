//! Core traits for sampling and image reading.
//!
//! This module provides traits used throughout the OCR pipeline for batch
//! sampling and image I/O operations.

use crate::core::errors::OCRError;
use image::RgbImage;
use std::path::Path;

/// Trait for sampling data into batches.
///
/// This trait defines the interface for sampling data into batches for processing.
pub trait Sampler<T> {
    /// The type of batch data produced by this sampler.
    type BatchData;

    /// Samples the given data into batches.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to sample.
    ///
    /// # Returns
    ///
    /// A vector of batch data.
    fn sample(&self, data: Vec<T>) -> Vec<Self::BatchData>;

    /// Samples the given slice of data into batches.
    ///
    /// # Arguments
    ///
    /// * `data` - The slice of data to sample.
    ///
    /// # Returns
    ///
    /// A vector of batch data.
    ///
    /// # Constraints
    ///
    /// * `T` must implement Clone.
    fn sample_slice(&self, data: &[T]) -> Vec<Self::BatchData>
    where
        T: Clone,
    {
        self.sample(data.to_vec())
    }

    /// Samples the given iterator of data into batches.
    ///
    /// # Arguments
    ///
    /// * `data` - The iterator of data to sample.
    ///
    /// # Returns
    ///
    /// A vector of batch data.
    ///
    /// # Constraints
    ///
    /// * `I` must implement IntoIterator<Item = T>.
    fn sample_iter<I>(&self, data: I) -> Vec<Self::BatchData>
    where
        I: IntoIterator<Item = T>,
    {
        self.sample(data.into_iter().collect())
    }
}

/// Trait for reading images.
///
/// This trait defines the interface for reading images from paths.
pub trait ImageReader {
    /// The error type of this image reader.
    type Error;

    /// Applies the image reader to the given paths.
    ///
    /// # Arguments
    ///
    /// * `imgs` - An iterator of paths to the images to read.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of RGB images or an error.
    ///
    /// # Constraints
    ///
    /// * `P` must implement `AsRef<Path>` + Send + Sync.
    fn apply<P: AsRef<Path> + Send + Sync>(
        &self,
        imgs: impl IntoIterator<Item = P>,
    ) -> Result<Vec<RgbImage>, Self::Error>;

    /// Reads a single image from the given path.
    ///
    /// # Arguments
    ///
    /// * `img_path` - The path to the image to read.
    ///
    /// # Returns
    ///
    /// A Result containing the RGB image or an error.
    ///
    /// # Constraints
    ///
    /// * `P` must implement `AsRef<Path>` + Send + Sync.
    fn read_single<P: AsRef<Path> + Send + Sync>(
        &self,
        img_path: P,
    ) -> Result<RgbImage, Self::Error>
    where
        Self::Error: From<OCRError>,
    {
        let mut results = self.apply(std::iter::once(img_path))?;
        results.pop().ok_or_else(|| {
            // Create a proper error instead of panicking
            OCRError::invalid_input("ImageReader::apply returned empty result for single image")
                .into()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OCRError;
    use image::RgbImage;
    use std::path::Path;

    /// Mock ImageReader that always returns empty results to test error handling
    struct MockEmptyImageReader;

    impl ImageReader for MockEmptyImageReader {
        type Error = OCRError;

        fn apply<P: AsRef<Path> + Send + Sync>(
            &self,
            _imgs: impl IntoIterator<Item = P>,
        ) -> Result<Vec<RgbImage>, Self::Error> {
            // Always return empty vector to trigger the error condition
            Ok(Vec::new())
        }
    }

    #[test]
    fn test_read_single_handles_empty_result_properly() {
        let reader = MockEmptyImageReader;
        let result = reader.read_single("test_path.jpg");

        // Should return an error instead of panicking
        assert!(result.is_err());

        // Check that it's the correct error type
        let err = result.unwrap_err();
        if let OCRError::InvalidInput { message } = err {
            assert!(message.contains("ImageReader::apply returned empty result for single image"));
        } else {
            panic!("Expected InvalidInput error, got {:?}", err);
        }
    }
}
