//! Text Line Classifier
//!
//! This module provides functionality for classifying the orientation of text lines in images.
//! It can detect if a text line is rotated and by how much (0° or 180°).
//!
//! The classifier uses a pre-trained model to analyze images and determine their orientation.
//! It supports batch processing for efficient handling of multiple images.

use crate::core::traits::StandardPredictor;
use crate::core::{
    BatchData, BatchSampler, CommonBuilderConfig, DefaultImageReader, ImageReader, OCRError,
    OrtInfer, Tensor2D, Tensor4D, ToBatch,
    config::{ConfigValidator, ConfigValidatorExt},
    get_text_line_orientation_labels,
};
use crate::processors::{Crop, NormalizeImage, Topk};
use image::{DynamicImage, RgbImage};
use std::path::Path;
use std::sync::Arc;

/// Results from text line classification
///
/// This struct contains the results of classifying text line orientations in images.
/// For each image, it provides the predicted orientations along with confidence scores.
#[derive(Debug, Clone)]
pub struct TextLineClasResult {
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
    /// Label names for each prediction (e.g., "0", "180")
    pub label_names: Vec<Vec<Arc<str>>>,
}

/// Configuration for the text line classifier
///
/// This struct holds configuration parameters for the text line classifier.
/// It includes common configuration options as well as classifier-specific parameters.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TextLineClasPredictorConfig {
    /// Common configuration options shared across predictors
    pub common: CommonBuilderConfig,
    /// Number of top predictions to return for each image
    pub topk: Option<usize>,
    /// Input shape for the model (width, height)
    pub input_shape: Option<(u32, u32)>,
}

impl TextLineClasPredictorConfig {
    /// Creates a new text line classifier configuration with default settings
    ///
    /// Initializes a new instance of the text line classifier configuration
    /// with default values for all parameters.
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictorConfig` with default settings
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::with_defaults(Some("PP-LCNet_x0_25".to_string()), Some(1)),
            topk: None,
            input_shape: Some((224, 224)),
        }
    }

    /// Creates a new text line classifier configuration with custom common settings
    ///
    /// Initializes a new instance of the text line classifier configuration
    /// with the provided common configuration and default values for other parameters.
    ///
    /// # Arguments
    ///
    /// * `common` - Common configuration options
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictorConfig` with custom common settings
    pub fn with_common(common: CommonBuilderConfig) -> Self {
        Self {
            common,
            topk: None,
            input_shape: Some((224, 224)),
        }
    }

    /// Validates the text line classifier configuration
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

impl ConfigValidator for TextLineClasPredictorConfig {
    /// Validates the text line classifier configuration
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

    /// Gets the default text line classifier configuration
    ///
    /// Returns a new instance of the text line classifier configuration
    /// with default values for all parameters.
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictorConfig` with default settings
    fn get_defaults() -> Self {
        Self {
            common: CommonBuilderConfig::get_defaults(),
            topk: Some(2),
            input_shape: Some((224, 224)),
        }
    }
}

impl TextLineClasResult {
    /// Creates a new empty text line classification result
    ///
    /// Initializes a new instance of the text line classification result with empty vectors
    /// for all fields.
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasResult` with empty vectors
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

impl Default for TextLineClasResult {
    /// Creates a new empty text line classification result
    ///
    /// This is equivalent to calling `TextLineClasResult::new()`.
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasResult` with empty vectors
    fn default() -> Self {
        Self::new()
    }
}

/// Text line classifier
///
/// This struct implements a classifier for determining the orientation of text lines in images.
/// It uses a pre-trained model to predict whether a text line is rotated by 0° or 180°.
#[derive(Debug)]
pub struct TextLineClasPredictor {
    /// Number of top predictions to return for each image
    pub topk: Option<usize>,
    /// Name of the model being used
    pub model_name: String,
    /// Input shape for the model (width, height)
    pub input_shape: (u32, u32),

    /// Batch sampler for processing images in batches
    pub batch_sampler: BatchSampler,
    /// Image reader for loading images from file paths
    pub read_image: DefaultImageReader,
    /// Image cropper for preprocessing images before inference
    pub crop: Option<Crop>,
    /// Image normalizer for preprocessing images before inference
    pub normalize: NormalizeImage,
    /// Batch converter for converting images to tensors
    pub to_batch: ToBatch,
    /// ONNX Runtime inference engine
    pub infer: OrtInfer,
    /// Top-k operator for selecting top predictions
    pub post_op: Topk,
}

impl TextLineClasPredictor {
    /// Creates a new text line classifier
    ///
    /// Initializes a new instance of the text line classifier with the provided
    /// configuration and model path.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the classifier
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictor` or an error if initialization fails
    pub fn new(config: TextLineClasPredictorConfig, model_path: &Path) -> Result<Self, OCRError> {
        let input_shape = config.input_shape.unwrap_or((224, 224));
        let (width, height) = input_shape;
        let topk = config.topk;
        // Use provided model name or default to "PP-LCNet_x0_25"
        let model_name = config
            .common
            .model_name
            .as_deref()
            .unwrap_or("PP-LCNet_x0_25")
            .to_string();
        let batch_size = config.common.batch_size.unwrap_or(1);

        // Create crop operation for preprocessing
        let crop = Some(
            Crop::new([width, height], crate::processors::CropMode::Center).map_err(|e| {
                OCRError::ConfigError {
                    message: format!("Failed to create crop operation: {e}"),
                }
            })?,
        );

        Ok(Self {
            topk,
            model_name,
            input_shape,
            batch_sampler: BatchSampler::new(batch_size),
            read_image: DefaultImageReader::new(),
            crop,
            // Normalize images using ImageNet normalization values
            normalize: NormalizeImage::new(
                Some(1.0 / 255.0),
                Some(vec![0.485, 0.456, 0.406]),
                Some(vec![0.229, 0.224, 0.225]),
                None,
            )?,
            to_batch: ToBatch::new(),
            infer: OrtInfer::from_common(
                &TextLineClasPredictorConfig {
                    common: config.common.clone(),
                    ..config.clone()
                }
                .common,
                model_path,
                None,
            )?,
            // Create Topk operator with label names for 0° and 180° rotations
            post_op: Topk::from_class_names(get_text_line_orientation_labels()),
        })
    }

    /// Processes a batch of images internally
    ///
    /// This method takes a batch of images and processes them through the full
    /// prediction pipeline, from preprocessing to postprocessing.
    ///
    /// # Arguments
    ///
    /// * `batch_data` - Batch of images to process
    ///
    /// Sets the number of top predictions to return
    ///
    /// Updates the classifier to return the specified number of top predictions
    /// for each image.
    ///
    /// # Arguments
    ///
    /// * `topk` - Number of top predictions to return
    pub fn set_topk(&mut self, topk: usize) {
        self.topk = Some(topk);
    }

    /// Gets the name of the model being used
    ///
    /// Returns a reference to the name of the model being used by the classifier.
    ///
    /// # Returns
    ///
    /// A reference to the model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[derive(Debug, Clone)]
pub struct TextLineClasConfig;

impl StandardPredictor for TextLineClasPredictor {
    type Config = TextLineClasConfig;
    type Result = TextLineClasResult;
    type PreprocessOutput = Tensor4D;
    type InferenceOutput = Tensor2D;

    /// Reads images from file paths
    ///
    /// Loads images from the provided file paths into memory.
    ///
    /// # Arguments
    ///
    /// * `paths` - Iterator over file paths to read
    ///
    /// # Returns
    ///
    /// Vector of loaded images or an error if reading fails
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.read_image.apply(paths)
    }

    /// Preprocesses images for inference
    ///
    /// Resizes images to the required input shape, crops them, and normalizes them for the model.
    ///
    /// # Arguments
    ///
    /// * `images` - Vector of images to preprocess
    /// * `_config` - Configuration (unused in this implementation)
    ///
    /// # Returns
    ///
    /// Preprocessed images as a 4D tensor or an error if preprocessing fails
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::PreprocessOutput, OCRError> {
        use crate::utils::resize_images_batch;

        let (width, height) = self.input_shape;

        // Resize images to model input size using shared utility
        let mut batch_imgs = resize_images_batch(&images, width, height, None);

        // Apply crop operation if available
        if let Some(crop_op) = &self.crop {
            batch_imgs = crop_op.process_batch(&batch_imgs).map_err(|e| {
                OCRError::post_processing("Crop operation failed during text classification", e)
            })?;
        }

        // Convert to DynamicImage for normalization
        let imgs_dynamic: Vec<DynamicImage> = batch_imgs
            .iter()
            .map(|img| DynamicImage::ImageRgb8(img.clone()))
            .collect();

        self.normalize.normalize_batch_to(imgs_dynamic)
    }

    /// Runs inference on preprocessed images
    ///
    /// Performs inference using the ONNX model on the preprocessed input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - Preprocessed input tensor
    ///
    /// # Returns
    ///
    /// Inference output as a 2D tensor or an error if inference fails
    fn infer(&self, input: &Self::PreprocessOutput) -> Result<Self::InferenceOutput, OCRError> {
        self.infer.infer_2d(input.clone())
    }

    /// Postprocesses inference output
    ///
    /// Converts the raw inference output into classification results, including
    /// class IDs, scores, and label names.
    ///
    /// # Arguments
    ///
    /// * `output` - Raw inference output
    /// * `_preprocessed` - Preprocessed input (unused in this implementation)
    /// * `batch_data` - Batch data containing input paths and indexes
    /// * `raw_images` - Original images
    /// * `_config` - Configuration (unused in this implementation)
    ///
    /// # Returns
    ///
    /// Classification results or an error if postprocessing fails
    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _preprocessed: &Self::PreprocessOutput,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Result, OCRError> {
        // Apply Top-k operation to get the most confident predictions
        // Convert ndarray output to Vec<Vec<f32>> format expected by Topk
        let predictions: Vec<Vec<f32>> = output.outer_iter().map(|row| row.to_vec()).collect();

        let topk_result = self
            .post_op
            .process(&predictions, self.topk.unwrap_or(2))
            .map_err(|e| OCRError::ConfigError { message: e })?;

        Ok(TextLineClasResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            // Wrap images in Arc for efficient sharing
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            class_ids: topk_result.indexes,
            scores: topk_result.scores,
            // Convert label names to Arc<str> for efficient sharing
            label_names: topk_result
                .class_names
                .unwrap_or_default()
                .into_iter()
                .map(|names| names.into_iter().map(Arc::from).collect())
                .collect(),
        })
    }

    fn empty_result(&self) -> Result<Self::Result, OCRError> {
        Ok(TextLineClasResult::new())
    }
}

/// Builder for text line classifier
///
/// This struct provides a builder pattern for creating a text line classifier
/// with custom configuration options.
pub struct TextLineClasPredictorBuilder {
    /// Common configuration options shared across predictors
    common: CommonBuilderConfig,

    /// Number of top predictions to return for each image
    topk: Option<usize>,
    /// Input shape for the model (width, height)
    input_shape: Option<(u32, u32)>,
}

impl TextLineClasPredictorBuilder {
    /// Creates a new text line classifier builder
    ///
    /// Initializes a new instance of the text line classifier builder
    /// with default configuration options.
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictorBuilder`
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

    /// Builds the text line classifier
    ///
    /// Creates a new instance of the text line classifier with the
    /// configured options.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictor` or an error if building fails
    pub fn build(self, model_path: &Path) -> Result<TextLineClasPredictor, OCRError> {
        self.build_internal(model_path)
    }

    /// Internal method to build the text line classifier
    ///
    /// Creates a new instance of the text line classifier with the
    /// configured options. This method handles validation of the configuration
    /// and initialization of the classifier.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictor` or an error if building fails
    fn build_internal(mut self, model_path: &Path) -> Result<TextLineClasPredictor, OCRError> {
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        let config = TextLineClasPredictorConfig {
            common: self.common,
            topk: self.topk,
            input_shape: self.input_shape,
        };

        let config = config.validate_and_wrap_ocr_error()?;

        TextLineClasPredictor::new(config, model_path)
    }
}

impl Default for TextLineClasPredictorBuilder {
    /// Creates a new text line classifier builder with default settings
    ///
    /// This is equivalent to calling `TextLineClasPredictorBuilder::new()`.
    ///
    /// # Returns
    ///
    /// A new instance of `TextLineClasPredictorBuilder` with default settings
    fn default() -> Self {
        Self::new()
    }
}
