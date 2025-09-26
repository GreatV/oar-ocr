//! Traits for the OCR pipeline.
//!
//! This module defines various traits that are used throughout the OCR pipeline
//! to provide a consistent interface for different components.

use crate::core::{PredictionResult, batch::BatchData, batch::BatchSampler, errors::OCRError};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;

/// Trait for building predictors.
///
/// This trait defines the interface for building predictors with specific configurations.
pub trait PredictorBuilder: Sized {
    /// The configuration type for this builder.
    type Config;

    /// The predictor type that this builder creates.
    type Predictor;

    /// Builds a typed predictor.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// A Result containing the built predictor or an error.
    fn build_typed(self, model_path: &Path) -> crate::core::OcrResult<Self::Predictor>;

    /// Gets the type of predictor that this builder creates.
    ///
    /// # Returns
    ///
    /// The predictor type as a string.
    fn predictor_type(&self) -> &str;

    /// Configures the builder with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to use.
    ///
    /// # Returns
    ///
    /// The configured builder.
    fn with_config(self, config: Self::Config) -> Self;

    /// Builds a predictor (alias for build_typed).
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// A Result containing the built predictor or an error.
    fn build(self, model_path: &Path) -> crate::core::OcrResult<Self::Predictor> {
        self.build_typed(model_path)
    }
}

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

/// Trait for base predictors in the OCR pipeline.
///
/// This trait defines the interface for base predictors that process batch data.
pub trait BasePredictor {
    /// The result type of this predictor.
    type Result;

    /// The error type of this predictor.
    type Error;

    /// Processes the given batch data.
    ///
    /// # Arguments
    ///
    /// * `batch_data` - The batch data to process.
    ///
    /// # Returns
    ///
    /// A Result containing the processing result or an error.
    fn process(&self, batch_data: BatchData) -> Result<Self::Result, Self::Error>;

    /// Converts the processing result to a prediction result.
    ///
    /// # Arguments
    ///
    /// * `result` - The processing result to convert.
    ///
    /// # Returns
    ///
    /// The converted prediction result.
    fn convert_to_prediction_result(&self, result: Self::Result) -> PredictionResult<'static>;

    /// Gets the batch sampler used by this predictor.
    ///
    /// # Returns
    ///
    /// A reference to the batch sampler.
    fn batch_sampler(&self) -> &BatchSampler;

    /// Gets the name of the model used by this predictor.
    ///
    /// # Returns
    ///
    /// The name of the model.
    fn model_name(&self) -> &str;

    /// Gets the type name of this predictor.
    ///
    /// # Returns
    ///
    /// The type name of the predictor.
    fn predictor_type_name(&self) -> &str;
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
        Self::Error: From<crate::core::OCRError>,
    {
        let mut results = self.apply(std::iter::once(img_path))?;
        results.pop().ok_or_else(|| {
            // Create a proper error instead of panicking
            crate::core::OCRError::invalid_input(
                "ImageReader::apply returned empty result for single image",
            )
            .into()
        })
    }
}

/// Trait for predictor configurations.
///
/// This trait defines the interface for predictor configurations.
pub trait PredictorConfig {
    /// Gets the name of the model.
    ///
    /// # Returns
    ///
    /// The name of the model.
    fn model_name(&self) -> &str;

    /// Gets the batch size.
    ///
    /// # Returns
    ///
    /// The batch size.
    fn batch_size(&self) -> usize;

    /// Validates the configuration.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error.
    fn validate(&self) -> crate::core::OcrResult<()>;

    /// Validates the batch size.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error.
    ///
    /// # Validation Rules
    ///
    /// * Batch size must be greater than 0.
    /// * Batch size should not exceed 1000 for memory efficiency.
    fn validate_batch_size(&self) -> crate::core::OcrResult<()> {
        let batch_size = self.batch_size();
        if batch_size == 0 {
            return Err(OCRError::ConfigError {
                message: "Batch size must be greater than 0".to_string(),
            });
        }
        if batch_size > 1000 {
            return Err(OCRError::ConfigError {
                message: "Batch size should not exceed 1000 for memory efficiency".to_string(),
            });
        }
        Ok(())
    }

    /// Validates the model name.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an error.
    ///
    /// # Validation Rules
    ///
    /// * Model name cannot be empty.
    fn validate_model_name(&self) -> crate::core::OcrResult<()> {
        let name = self.model_name();
        if name.is_empty() {
            return Err(OCRError::ConfigError {
                message: "Model name cannot be empty".to_string(),
            });
        }
        Ok(())
    }
}

/// Trait for standard predictors.
///
/// This trait defines the interface for standard predictors that follow
/// a standard pipeline of reading images, preprocessing, inference, and postprocessing.
pub trait StandardPredictor {
    /// The configuration type for this predictor.
    type Config;

    /// The result type of this predictor.
    type Result;

    /// The output type of the preprocessing step.
    type PreprocessOutput;

    /// The output type of the inference step.
    type InferenceOutput;

    /// Reads images from the given paths.
    ///
    /// # Arguments
    ///
    /// * `paths` - An iterator of paths to the images to read.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of RGB images or an error.
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError>;

    /// Preprocesses the given images.
    ///
    /// # Arguments
    ///
    /// * `images` - The images to preprocess.
    /// * `config` - Optional configuration for preprocessing.
    ///
    /// # Returns
    ///
    /// A Result containing the preprocessed output or an error.
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::PreprocessOutput>;

    /// Performs inference on the preprocessed input.
    ///
    /// # Arguments
    ///
    /// * `input` - The preprocessed input.
    ///
    /// # Returns
    ///
    /// A Result containing the inference output or an error.
    fn infer(
        &self,
        input: &Self::PreprocessOutput,
    ) -> crate::core::OcrResult<Self::InferenceOutput>;

    /// Postprocesses the inference output.
    ///
    /// # Arguments
    ///
    /// * `output` - The inference output to postprocess.
    /// * `preprocessed` - The preprocessed input.
    /// * `batch_data` - The batch data.
    /// * `raw_images` - The raw images.
    /// * `config` - Optional configuration for postprocessing.
    ///
    /// # Returns
    ///
    /// A Result containing the final result or an error.
    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        preprocessed: &Self::PreprocessOutput,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result>;

    /// Performs prediction directly from in-memory images.
    ///
    /// This method bypasses file I/O by working directly with RgbImage instances,
    /// providing better performance when images are already in memory. This is
    /// the primary prediction method for most use cases.
    ///
    /// # Arguments
    ///
    /// * `images` - Vector of images to process
    /// * `config` - Optional configuration for the prediction
    ///
    /// # Returns
    ///
    /// A Result containing the prediction result or an OCRError
    fn predict(
        &self,
        images: Vec<RgbImage>,
        config: Option<Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        if images.is_empty() {
            return self.empty_result();
        }

        let batch_data = self.create_dummy_batch_data(images.len());
        let preprocessed = self.preprocess(images.clone(), config.as_ref())?;
        let inference_output = self.infer(&preprocessed)?;
        self.postprocess(
            inference_output,
            &preprocessed,
            &batch_data,
            images,
            config.as_ref(),
        )
    }

    /// Creates dummy batch data for in-memory processing.
    ///
    /// This method creates BatchData with dummy paths for in-memory processing,
    /// allowing the postprocessing step to work correctly without actual file paths.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of images to create batch data for
    ///
    /// # Returns
    ///
    /// BatchData with dummy paths and sequential indexes
    fn create_dummy_batch_data(&self, count: usize) -> BatchData {
        let dummy_paths: Vec<Arc<str>> = (0..count)
            .map(|i| Arc::from(format!("in_memory_{i}")))
            .collect();
        BatchData {
            instances: dummy_paths.clone(),
            input_paths: dummy_paths,
            indexes: (0..count).collect(),
        }
    }

    /// Returns an empty result for the predictor type.
    ///
    /// This method should return an empty instance of the result type,
    /// typically used when processing an empty list of images.
    ///
    /// # Returns
    ///
    /// A Result containing an empty result instance
    fn empty_result(&self) -> crate::core::OcrResult<Self::Result>;
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
        match result.unwrap_err() {
            OCRError::InvalidInput { message } => {
                assert!(
                    message.contains("ImageReader::apply returned empty result for single image")
                );
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }
}
