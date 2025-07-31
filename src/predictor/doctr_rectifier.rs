//! Document Rectifier
//!
//! This module provides functionality for rectifying (correcting distortions in) document images.
//! It uses a pre-trained model to transform distorted document images into properly aligned versions.
//!
//! The rectifier supports batch processing for efficient handling of multiple images.

use crate::core::traits::StandardPredictor;
use crate::core::{
    BasePredictor, BatchData, BatchSampler, CommonBuilderConfig, DefaultImageReader, ImageReader,
    IntoPrediction, OCRError, OrtInfer, PredictionResult, Sampler, Tensor4D, ToBatch,
    config::ConfigValidator,
};

use crate::impl_predictor_from_generic;
use crate::impl_standard_predictor;
use crate::impl_standard_predictor_builder;
use crate::processors::{DocTrPostProcess, NormalizeImage};
use image::{DynamicImage, RgbImage};
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

/// Results from document rectification
///
/// This struct contains the results of rectifying document images.
/// For each image, it provides both the original and rectified versions.
#[derive(Debug, Clone)]
pub struct DoctrRectifierResult {
    /// Paths to the input images
    pub input_path: Vec<Arc<str>>,
    /// Indexes of the images in the batch
    pub index: Vec<usize>,
    /// The input images
    pub input_img: Vec<Arc<RgbImage>>,
    /// The rectified images
    pub rectified_img: Vec<Arc<RgbImage>>,
}

/// Configuration for the document rectifier
///
/// This struct holds configuration parameters for the document rectifier.
/// It includes common configuration options as well as rectifier-specific parameters.
#[derive(Debug, Clone, Default)]
pub struct DoctrRectifierPredictorConfig {
    /// Common configuration options shared across predictors
    pub common: CommonBuilderConfig,
    /// Input shape for the recognition model [channels, height, width]
    pub rec_image_shape: Option<[usize; 3]>,
}

impl DoctrRectifierPredictorConfig {
    /// Creates a new document rectifier configuration with default settings
    ///
    /// Initializes a new instance of the document rectifier configuration
    /// with default values for all parameters.
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictorConfig` with default settings
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::with_defaults(
                Some("doctr_rectifier".to_string()),
                Some(32),
            ),
            rec_image_shape: Some([3, 512, 512]),
        }
    }

    /// Creates a new document rectifier configuration with custom common settings
    ///
    /// Initializes a new instance of the document rectifier configuration
    /// with the provided common configuration and default values for other parameters.
    ///
    /// # Arguments
    ///
    /// * `common` - Common configuration options
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictorConfig` with custom common settings
    pub fn with_common(common: CommonBuilderConfig) -> Self {
        Self {
            common,
            rec_image_shape: Some([3, 512, 512]),
        }
    }

    /// Validates the document rectifier configuration
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

impl ConfigValidator for DoctrRectifierPredictorConfig {
    fn validate(&self) -> Result<(), crate::core::ConfigError> {
        self.common.validate()?;

        if let Some(rec_shape) = self.rec_image_shape {
            if rec_shape[0] == 0 || rec_shape[1] == 0 || rec_shape[2] == 0 {
                return Err(crate::core::ConfigError::InvalidConfig {
                    message: format!(
                        "Recognition image shape dimensions must be greater than 0, got [{}, {}, {}]",
                        rec_shape[0], rec_shape[1], rec_shape[2]
                    ),
                });
            }

            const MAX_SHAPE_SIZE: usize = 2048;
            for (i, &dim) in rec_shape.iter().enumerate() {
                if dim > MAX_SHAPE_SIZE {
                    return Err(crate::core::ConfigError::ResourceLimitExceeded {
                        message: format!(
                            "Recognition image shape dimension {i} ({dim}) exceeds maximum allowed size {MAX_SHAPE_SIZE}"
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn get_defaults() -> Self {
        Self {
            common: CommonBuilderConfig::get_defaults(),
            rec_image_shape: Some([3, 512, 512]),
        }
    }
}

impl DoctrRectifierResult {
    /// Creates a new empty document rectifier result
    ///
    /// Initializes a new instance of the document rectifier result with empty vectors
    /// for all fields.
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierResult` with empty vectors
    pub fn new() -> Self {
        Self {
            input_path: Vec::new(),
            index: Vec::new(),
            input_img: Vec::new(),
            rectified_img: Vec::new(),
        }
    }
}

impl Default for DoctrRectifierResult {
    /// Creates a new empty document rectifier result
    ///
    /// This is equivalent to calling `DoctrRectifierResult::new()`.
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierResult` with empty vectors
    fn default() -> Self {
        Self::new()
    }
}

impl IntoPrediction for DoctrRectifierResult {
    type Out = PredictionResult<'static>;

    fn into_prediction(self) -> Self::Out {
        PredictionResult::Rectification {
            input_path: self
                .input_path
                .into_iter()
                .map(|arc| Cow::Owned(arc.to_string()))
                .collect(),
            index: self.index,
            input_img: self.input_img,
            rectified_img: self.rectified_img,
        }
    }
}

/// Document rectifier
///
/// This struct implements a rectifier for correcting distortions in document images.
/// It uses a pre-trained model to transform distorted document images into properly aligned versions.
#[derive(Debug)]
pub struct DoctrRectifierPredictor {
    /// Input shape for the recognition model [channels, height, width]
    pub rec_image_shape: [usize; 3],
    /// Name of the model being used
    pub model_name: String,

    /// Batch sampler for processing images in batches
    pub batch_sampler: BatchSampler,
    /// Image reader for loading images from file paths
    pub read_image: DefaultImageReader,
    /// Image normalizer for preprocessing images before inference
    pub normalize: NormalizeImage,
    /// Batch converter for converting images to tensors
    pub to_batch: ToBatch,
    /// ONNX Runtime inference engine
    pub infer: OrtInfer,
    /// Post-processing operator for rectifying images
    pub post_op: DocTrPostProcess,
}

impl DoctrRectifierPredictor {
    /// Creates a new document rectifier
    ///
    /// Initializes a new instance of the document rectifier with the provided
    /// configuration and model path.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the rectifier
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictor` or an error if initialization fails
    pub fn new(config: DoctrRectifierPredictorConfig, model_path: &Path) -> Result<Self, OCRError> {
        let rec_image_shape = config.rec_image_shape.unwrap_or([3, 512, 512]);
        let model_name = config
            .common
            .model_name
            .unwrap_or_else(|| "doctr_rectifier".to_string());
        let batch_size = config.common.batch_size.unwrap_or(32);

        let batch_sampler = BatchSampler::new(batch_size);
        let read_image = DefaultImageReader::new();
        let normalize = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            None,
        )?;
        let to_batch = ToBatch::new();
        let infer = OrtInfer::with_auto_input_name(model_path)?;
        let post_op = DocTrPostProcess::new(None);

        Ok(Self {
            rec_image_shape,
            model_name,
            batch_sampler,
            read_image,
            normalize,
            to_batch,
            infer,
            post_op,
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
    /// # Returns
    ///
    /// Rectification results for the batch or an error if processing fails
    fn process_internal(
        &mut self,
        batch_data: BatchData,
    ) -> Result<DoctrRectifierResult, OCRError> {
        self.predict(batch_data, None)
    }

    /// Gets the model name
    ///
    /// Returns the name of the model being used by this rectifier.
    ///
    /// # Returns
    ///
    /// The model name as a string slice
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Configuration for document rectification
///
/// This struct is used as a placeholder for configuration options specific to
/// document rectification. Currently, it doesn't have any fields
/// as the configuration is handled by `DoctrRectifierPredictorConfig`.
#[derive(Debug, Clone)]
pub struct DoctrRectifierConfig;

impl StandardPredictor for DoctrRectifierPredictor {
    type Config = DoctrRectifierConfig;
    type Result = DoctrRectifierResult;
    type PreprocessOutput = Tensor4D;
    type InferenceOutput = Tensor4D;

    fn read_images<'a>(
        &mut self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.read_image.apply(paths)
    }

    fn preprocess(
        &mut self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::PreprocessOutput, OCRError> {
        let batch_imgs: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();

        self.normalize.normalize_batch_to(batch_imgs)
    }

    fn infer(&mut self, input: &Self::PreprocessOutput) -> Result<Self::InferenceOutput, OCRError> {
        self.infer.infer_4d(input.clone())
    }

    fn postprocess(
        &mut self,
        output: Self::InferenceOutput,
        _preprocessed: &Self::PreprocessOutput,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Result, OCRError> {
        let rectified_imgs = self
            .post_op
            .apply_batch(&output)
            .map_err(|e| OCRError::post_processing("DocTr post-processing failed", e))?;

        Ok(DoctrRectifierResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            rectified_img: rectified_imgs.into_iter().map(Arc::new).collect(),
        })
    }
}

/// Builder for document rectifier
///
/// This struct provides a builder pattern for creating a document rectifier
/// with custom configuration options.
pub struct DoctrRectifierPredictorBuilder {
    /// Common configuration options shared across predictors
    common: CommonBuilderConfig,

    /// Input shape for the recognition model [channels, height, width]
    rec_image_shape: Option<[usize; 3]>,
}

impl DoctrRectifierPredictorBuilder {
    /// Creates a new document rectifier builder
    ///
    /// Initializes a new instance of the document rectifier builder
    /// with default configuration options.
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictorBuilder`
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::new(),
            rec_image_shape: None,
        }
    }

    /// Sets the model path for the rectifier
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

    /// Sets the model name for the rectifier
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

    /// Sets the batch size for the rectifier
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

    /// Enables or disables logging for the rectifier
    ///
    /// Controls whether logging is enabled during rectification.
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

    /// Sets the input shape for the recognition model
    ///
    /// Specifies the input shape [channels, height, width] that the model expects.
    ///
    /// # Arguments
    ///
    /// * `rec_image_shape` - Input shape as [channels, height, width]
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn rec_image_shape(mut self, rec_image_shape: [usize; 3]) -> Self {
        self.rec_image_shape = Some(rec_image_shape);
        self
    }

    /// Builds the document rectifier
    ///
    /// Creates a new instance of the document rectifier with the
    /// configured options.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictor` or an error if building fails
    pub fn build(self, model_path: &Path) -> Result<DoctrRectifierPredictor, OCRError> {
        self.build_internal(model_path)
    }

    /// Internal method to build the document rectifier
    ///
    /// Creates a new instance of the document rectifier with the
    /// configured options. This method handles validation of the configuration
    /// and initialization of the rectifier.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictor` or an error if building fails
    fn build_internal(mut self, model_path: &Path) -> Result<DoctrRectifierPredictor, OCRError> {
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        let config = DoctrRectifierPredictorConfig {
            common: self.common,
            rec_image_shape: self.rec_image_shape,
        };

        config.validate().map_err(|e| OCRError::ConfigError {
            message: e.to_string(),
        })?;

        DoctrRectifierPredictor::new(config, model_path)
    }
}

impl Default for DoctrRectifierPredictorBuilder {
    /// Creates a new document rectifier builder with default settings
    ///
    /// This is equivalent to calling `DoctrRectifierPredictorBuilder::new()`.
    ///
    /// # Returns
    ///
    /// A new instance of `DoctrRectifierPredictorBuilder` with default settings
    fn default() -> Self {
        Self::new()
    }
}

impl_standard_predictor!(
    DoctrRectifierPredictor,
    DoctrRectifierResult,
    OCRError,
    "DoctrRectifier"
);

impl_predictor_from_generic!(DoctrRectifierPredictor);

impl_standard_predictor_builder!(
    DoctrRectifierPredictorBuilder,
    DoctrRectifierPredictor,
    "DoctrRectifier"
);
