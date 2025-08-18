//! Document Orientation Classifier
//!
//! This module provides functionality for classifying the orientation of documents in images.
//! It can detect if a document is rotated and by how much (0°, 90°, 180°, or 270°).
//!
//! The classifier uses a pre-trained model to analyze images and determine their orientation.
//! It supports batch processing for efficient handling of multiple images.

use crate::core::traits::ImageReader as CoreImageReader;
use crate::core::{
    BatchData, CommonBuilderConfig, DefaultImageReader, OCRError, OrtInfer, Tensor2D, Tensor4D,
    config::{ConfigValidator, ConfigValidatorExt},
};

use crate::impl_config_new_and_with_common;

use crate::core::get_document_orientation_labels;
use crate::processors::{NormalizeImage, Topk};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;

/// Results from document orientation classification
///
/// This struct contains the results of classifying document orientations in images.
/// For each image, it provides the predicted orientations along with confidence scores.
#[derive(Debug, Clone)]
pub struct DocOrientationResult {
    /// Paths to the input images
    pub input_path: Vec<Arc<str>>,
    /// Indexes of the images in the batch
    pub index: Vec<usize>,
    /// The input images
    pub input_img: Vec<Arc<RgbImage>>,
    /// Predicted class IDs for each image (sorted by confidence)
    pub class_ids: Vec<Vec<usize>>,
    /// Confidence scores for each prediction
    pub scores: Vec<Vec<f32>>,
    /// Label names for each prediction (e.g., "0", "90", "180", "270")
    pub label_names: Vec<Vec<Arc<str>>>,
}

/// Configuration for the document orientation classifier
///
/// This struct holds configuration parameters for the document orientation classifier.
/// It includes common configuration options as well as classifier-specific parameters.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DocOrientationClassifierConfig {
    /// Common configuration options shared across predictors
    pub common: CommonBuilderConfig,
    /// Number of top predictions to return for each image
    pub topk: Option<usize>,
    /// Input shape for the model (width, height)
    pub input_shape: Option<(u32, u32)>,
}

impl_config_new_and_with_common!(
    DocOrientationClassifierConfig,
    common_defaults: (Some("doc_orientation_classifier".to_string()), Some(1)),
    fields: {
        topk: Some(4),
        input_shape: Some((224, 224))
    }
);

impl DocOrientationClassifierConfig {
    /// Validates the document orientation classifier configuration
    ///
    /// Checks that all configuration parameters are valid and within acceptable ranges.
    ///
    /// # Returns
    ///
    /// Ok if the configuration is valid, or an error if validation fails
    pub fn validate(&self) -> Result<(), crate::core::ConfigError> {
        ConfigValidator::validate(self)
    }
}

impl ConfigValidator for DocOrientationClassifierConfig {
    /// Validates the document orientation classifier configuration
    ///
    /// Checks that all configuration parameters are valid and within acceptable ranges.
    /// This includes validating the common configuration, topk value, and input shape.
    ///
    /// # Returns
    ///
    /// Ok if the configuration is valid, or an error if validation fails
    fn validate(&self) -> Result<(), crate::core::ConfigError> {
        self.common.validate()?;

        if let Some(topk) = self.topk {
            self.validate_positive_usize(topk, "topk")?;
        }

        if let Some((width, height)) = self.input_shape {
            self.validate_image_dimensions(width, height)?;
        }

        Ok(())
    }

    /// Gets the default document orientation classifier configuration
    ///
    /// Returns a new instance of the document orientation classifier configuration
    /// with default values for all parameters.
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationClassifierConfig` with default settings
    fn get_defaults() -> Self {
        Self {
            common: CommonBuilderConfig::get_defaults(),
            topk: Some(4),
            input_shape: Some((224, 224)),
        }
    }
}

impl DocOrientationResult {
    /// Creates a new empty document orientation result
    ///
    /// Initializes a new instance of the document orientation result with empty vectors
    /// for all fields.
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationResult` with empty vectors
    pub fn new() -> Self {
        Self {
            input_path: Vec::new(),
            index: Vec::new(),
            input_img: Vec::new(),
            class_ids: Vec::new(),
            scores: Vec::new(),
            label_names: Vec::new(),
        }
    }
}

impl Default for DocOrientationResult {
    /// Creates a new empty document orientation result
    ///
    /// This is equivalent to calling `DocOrientationResult::new()`.
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationResult` with empty vectors
    fn default() -> Self {
        Self::new()
    }
}

/// Document orientation classifier built from modular components
///
/// This is a type alias over `ModularPredictor` with concrete, composable components
/// to eliminate duplicated StandardPredictor implementations across predictors.
pub type DocOrientationClassifier =
    ModularPredictor<DocOrImageReader, DocOrPreprocessor, OrtInfer2D, DocOrPostprocessor>;

// Granular trait adapters for the document orientation classifier
use crate::core::{
    GranularImageReader as GIReader, ModularPredictor, OrtInfer2D, Postprocessor as GPostprocessor,
    Preprocessor as GPreprocessor,
};
use image::DynamicImage;

#[derive(Debug, Clone)]
pub struct DocOrientationConfig;

#[derive(Debug)]
pub struct DocOrImageReader {
    inner: DefaultImageReader,
}

impl DocOrImageReader {
    pub fn new() -> Self {
        Self {
            inner: DefaultImageReader::new(),
        }
    }
}

impl Default for DocOrImageReader {
    fn default() -> Self {
        Self::new()
    }
}

impl GIReader for DocOrImageReader {
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.inner.apply(paths)
    }
}

#[derive(Debug)]
pub struct DocOrPreprocessor {
    pub input_shape: (u32, u32),
    pub normalize: NormalizeImage,
}

impl GPreprocessor for DocOrPreprocessor {
    type Config = DocOrientationConfig;
    type Output = Tensor4D;

    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Output, OCRError> {
        use crate::utils::resize_images_batch_to_dynamic;
        let dynamic_images: Vec<DynamicImage> =
            resize_images_batch_to_dynamic(&images, self.input_shape.0, self.input_shape.1, None);
        self.normalize.normalize_batch_to(dynamic_images)
    }

    fn preprocessing_info(&self) -> String {
        format!(
            "resize_to=({},{}) + normalize",
            self.input_shape.0, self.input_shape.1
        )
    }
}

#[derive(Debug)]
pub struct DocOrPostprocessor {
    pub topk: usize,
    pub topk_op: Topk,
}

impl GPostprocessor for DocOrPostprocessor {
    type Config = DocOrientationConfig;
    type InferenceOutput = Tensor2D;
    type PreprocessOutput = Tensor4D;
    type Result = DocOrientationResult;

    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _preprocess_output: Option<&Self::PreprocessOutput>,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        // Convert ndarray output to Vec<Vec<f32>> format expected by Topk
        let predictions: Vec<Vec<f32>> = output.outer_iter().map(|row| row.to_vec()).collect();
        let topk_result = self
            .topk_op
            .process(&predictions, self.topk)
            .map_err(|e| OCRError::ConfigError { message: e })?;

        Ok(DocOrientationResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            class_ids: topk_result.indexes,
            scores: topk_result.scores,
            label_names: topk_result
                .class_names
                .unwrap_or_default()
                .into_iter()
                .map(|names| names.into_iter().map(Arc::from).collect())
                .collect(),
        })
    }

    fn empty_result(&self) -> Result<Self::Result, OCRError> {
        Ok(DocOrientationResult::new())
    }
}

/// Builder for document orientation classifier
///
/// This struct provides a builder pattern for creating a document orientation classifier
/// with custom configuration options.
pub struct DocOrientationClassifierBuilder {
    /// Common configuration options shared across predictors
    common: CommonBuilderConfig,

    /// Number of top predictions to return for each image
    topk: Option<usize>,
    /// Input shape for the model (width, height)
    input_shape: Option<(u32, u32)>,
}

impl DocOrientationClassifierBuilder {
    /// Creates a new document orientation classifier builder
    ///
    /// Initializes a new instance of the document orientation classifier builder
    /// with default configuration options.
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationClassifierBuilder`
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::new(),
            topk: None,
            input_shape: None,
        }
    }

    /// Sets the model path for the classifier
    ///
    /// Specifies the path to the ONNX model file that will be used for inference.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn model_path(mut self, model_path: impl Into<std::path::PathBuf>) -> Self {
        self.common = self.common.model_path(model_path);
        self
    }

    /// Sets the model name for the classifier
    ///
    /// Specifies the name of the model being used.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Name of the model
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.common = self.common.model_name(model_name);
        self
    }

    /// Sets the batch size for the classifier
    ///
    /// Specifies the number of images to process in each batch.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of images to process in each batch
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.common = self.common.batch_size(batch_size);
        self
    }

    /// Enables or disables logging for the classifier
    ///
    /// Controls whether logging is enabled during classification.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable logging
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn enable_logging(mut self, enable: bool) -> Self {
        self.common = self.common.enable_logging(enable);
        self
    }

    /// Sets the ONNX Runtime session configuration
    ///
    /// This function sets the ONNX Runtime session configuration for the predictor.
    pub fn ort_session(mut self, config: crate::core::config::onnx::OrtSessionConfig) -> Self {
        self.common = self.common.ort_session(config);
        self
    }

    /// Sets the session pool size for concurrent predictions
    ///
    /// This function sets the size of the session pool used for concurrent predictions.
    /// The pool size must be >= 1.
    ///
    /// # Arguments
    ///
    /// * `size` - The session pool size (minimum 1)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.common = self.common.session_pool_size(size);
        self
    }

    /// Sets the number of top predictions to return
    ///
    /// Specifies how many of the top predictions to return for each image.
    ///
    /// # Arguments
    ///
    /// * `topk` - Number of top predictions to return
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn topk(mut self, topk: usize) -> Self {
        self.topk = Some(topk);
        self
    }

    /// Sets the input shape for the model
    ///
    /// Specifies the input shape (width, height) that the model expects.
    ///
    /// # Arguments
    ///
    /// * `input_shape` - Input shape as (width, height)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    /// Builds the document orientation classifier
    ///
    /// Creates a new instance of the document orientation classifier with the
    /// configured options.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationClassifier` or an error if building fails
    pub fn build(self, model_path: &Path) -> Result<DocOrientationClassifier, OCRError> {
        self.build_internal(model_path)
    }

    /// Internal method to build the document orientation classifier
    ///
    /// Creates a new instance of the document orientation classifier with the
    /// configured options. This method handles validation of the configuration
    /// and initialization of the classifier.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationClassifier` or an error if building fails
    fn build_internal(mut self, model_path: &Path) -> Result<DocOrientationClassifier, OCRError> {
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        let config = DocOrientationClassifierConfig {
            common: self.common,
            topk: self.topk,
            input_shape: self.input_shape,
        };

        let config = config.validate_and_wrap_ocr_error()?;

        // Build modular components
        let input_shape = config.input_shape.unwrap_or((224, 224));
        let image_reader = DocOrImageReader::new();
        let normalize = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.485, 0.456, 0.406]),
            Some(vec![0.229, 0.224, 0.225]),
            None,
        )?;
        let preprocessor = DocOrPreprocessor {
            input_shape,
            normalize,
        };
        let infer_inner = OrtInfer::from_common(&config.common, model_path, None)?;
        let inference_engine = OrtInfer2D(infer_inner);
        let postprocessor = DocOrPostprocessor {
            topk: config.topk.unwrap_or(4),
            topk_op: Topk::from_class_names(get_document_orientation_labels()),
        };

        Ok(ModularPredictor::new(
            image_reader,
            preprocessor,
            inference_engine,
            postprocessor,
        ))
    }
}

impl Default for DocOrientationClassifierBuilder {
    /// Creates a new document orientation classifier builder with default settings
    ///
    /// This is equivalent to calling `DocOrientationClassifierBuilder::new()`.
    ///
    /// # Returns
    ///
    /// A new instance of `DocOrientationClassifierBuilder` with default settings
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_orientation_config_defaults_and_validate() {
        let config = DocOrientationClassifierConfig::new();
        assert_eq!(config.topk, Some(4));
        assert_eq!(config.input_shape, Some((224, 224)));
        assert_eq!(
            config.common.model_name.as_deref(),
            Some("doc_orientation_classifier")
        );
        assert_eq!(config.common.batch_size, Some(1));
        assert!(config.validate().is_ok());
    }
}
