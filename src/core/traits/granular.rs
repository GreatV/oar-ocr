//! Granular traits for composable predictor components.
//!
//! This module provides granular traits that separate the concerns of the StandardPredictor
//! trait, making it easier to compose, test, and extend individual pipeline components.
//!
//! The design focuses on practical composability and clean separation of concerns:
//! - **ImageReader**: Handles I/O operations (loading images from files/memory)
//! - **Preprocessor**: Handles image preprocessing (resize, normalize, tensor conversion)
//! - **InferenceEngine**: Handles model inference (ONNX, TensorRT, etc.)
//! - **Postprocessor**: Handles result processing (decoding, formatting, filtering)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
//! │ImageReader  │───▶│Preprocessor │───▶│InferenceEng │───▶│Postprocessor │
//! │             │    │             │    │             │    │              │
//! │• read_images│    │• preprocess │    │• infer      │    │• postprocess │
//! │• validate   │    │• validate   │    │• engine_info│    │• empty_result│
//! └─────────────┘    └─────────────┘    └─────────────┘    └──────────────┘
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! use oar_ocr::core::traits::granular::Preprocessor;
//! use image::RgbImage;
//!
//! // Example of how you would implement and use a custom preprocessor
//! // (This is a conceptual example - actual implementations would be in separate modules)
//!
//! # #[derive(Debug)]
//! # struct MyPreprocessor;
//! # #[derive(Debug)]
//! # struct MyConfig { brightness_factor: f32 }
//! # impl Preprocessor for MyPreprocessor {
//! #     type Config = MyConfig;
//! #     type Output = Vec<RgbImage>;
//! #     fn preprocess(&self, images: Vec<RgbImage>, config: Option<&Self::Config>) -> Result<Self::Output, oar_ocr::core::OCRError> {
//! #         Ok(images)
//! #     }
//! #     fn preprocessing_info(&self) -> String { "MyPreprocessor".to_string() }
//! # }
//!
//! let preprocessor = MyPreprocessor;
//! let test_image = RgbImage::new(32, 32);
//! let images = vec![test_image];
//! let config = MyConfig { brightness_factor: 1.5 };
//!
//! // Apply preprocessing
//! let result = preprocessor.preprocess(images, Some(&config));
//! assert!(result.is_ok());
//! ```

use crate::core::{BatchData, OCRError};
use image::RgbImage;
use std::fmt::Debug;

/// Trait for image reading and I/O operations.
///
/// This trait handles loading images from various sources (file paths, URLs, memory)
/// and converting them to a standard RGB format for processing.
pub trait ImageReader: Send + Sync + Debug {
    /// Read images from file paths.
    ///
    /// # Arguments
    ///
    /// * `paths` - Iterator over file paths to read
    ///
    /// # Returns
    ///
    /// Vector of loaded RGB images or an error
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError>;

    /// Validate that images can be read from the given paths.
    ///
    /// # Arguments
    ///
    /// * `paths` - Iterator over file paths to validate
    ///
    /// # Returns
    ///
    /// Result indicating success or validation error
    fn validate_paths<'a>(&self, paths: impl Iterator<Item = &'a str>) -> Result<(), OCRError> {
        // Default implementation: try to read first path
        if let Some(path) = paths.into_iter().next() {
            let images = self.read_images(std::iter::once(path))?;
            if images.is_empty() {
                return Err(OCRError::InvalidInput {
                    message: "Failed to read any images".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Trait for image preprocessing operations.
///
/// This trait handles transforming raw images into the format required by
/// the inference engine (resizing, normalization, tensor conversion, etc.).
pub trait Preprocessor: Send + Sync + Debug {
    /// Configuration type for preprocessing
    type Config: Send + Sync + Debug;

    /// Output type after preprocessing
    type Output: Send + Sync + Debug;

    /// Preprocess input images into inference-ready format.
    ///
    /// # Arguments
    ///
    /// * `images` - Input images to preprocess
    /// * `config` - Optional configuration for preprocessing
    ///
    /// # Returns
    ///
    /// Preprocessed output ready for inference or an error
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> Result<Self::Output, OCRError>;

    /// Get information about the preprocessing requirements.
    ///
    /// # Returns
    ///
    /// String describing preprocessing requirements (input size, format, etc.)
    fn preprocessing_info(&self) -> String {
        "Generic preprocessing".to_string()
    }

    /// Validate that the input images are suitable for preprocessing.
    ///
    /// # Arguments
    ///
    /// * `images` - Input images to validate
    ///
    /// # Returns
    ///
    /// Result indicating success or validation error
    fn validate_input(&self, images: &[RgbImage]) -> Result<(), OCRError> {
        if images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided for preprocessing".to_string(),
            });
        }
        Ok(())
    }
}

/// Trait for inference engine operations.
///
/// This trait handles running the actual model inference, whether through
/// ONNX Runtime, TensorRT, PyTorch, or other backends.
pub trait InferenceEngine: Send + Sync + Debug {
    /// Input type for inference (typically a tensor)
    type Input: Send + Sync + Debug;

    /// Output type from inference (typically a tensor)
    type Output: Send + Sync + Debug;

    /// Perform inference on preprocessed input.
    ///
    /// # Arguments
    ///
    /// * `input` - Preprocessed input ready for inference
    ///
    /// # Returns
    ///
    /// Raw inference output or an error
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError>;

    /// Get information about the inference engine.
    ///
    /// # Returns
    ///
    /// String describing the inference engine (model type, backend, etc.)
    fn engine_info(&self) -> String;

    /// Validate that the input is suitable for inference.
    ///
    /// # Arguments
    ///
    /// * `input` - Input to validate
    ///
    /// # Returns
    ///
    /// Result indicating success or validation error
    fn validate_inference_input(&self, _input: &Self::Input) -> Result<(), OCRError> {
        // Default implementation - basic validation
        Ok(())
    }
}

/// Trait for postprocessing operations.
///
/// This trait handles converting raw inference outputs into meaningful results
/// (decoding, filtering, formatting, etc.).
pub trait Postprocessor: Send + Sync + Debug {
    /// Configuration type for postprocessing
    type Config: Send + Sync + Debug;

    /// Input type from inference engine
    type InferenceOutput: Send + Sync + Debug;

    /// Preprocessed data type for context
    type PreprocessOutput: Send + Sync + Debug;

    /// Final result type after postprocessing
    type Result: Send + Sync + Debug;

    /// Postprocess inference output into final results.
    ///
    /// # Arguments
    ///
    /// * `inference_output` - Raw output from inference engine
    /// * `preprocess_output` - Optional preprocessed data for context
    /// * `batch_data` - Batch metadata
    /// * `raw_images` - Original input images
    /// * `config` - Optional configuration for postprocessing
    ///
    /// # Returns
    ///
    /// Final processed result or an error
    fn postprocess(
        &self,
        inference_output: Self::InferenceOutput,
        preprocess_output: Option<&Self::PreprocessOutput>,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result>;

    /// Create an empty result for when no input is provided.
    ///
    /// # Returns
    ///
    /// Empty result instance
    fn empty_result(&self) -> Result<Self::Result, OCRError>;

    /// Get information about the postprocessing operations.
    ///
    /// # Returns
    ///
    /// String describing postprocessing operations
    fn postprocessing_info(&self) -> String {
        "Generic postprocessing".to_string()
    }
}
