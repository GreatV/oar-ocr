//! Document Rectifier
//!
//! This module provides functionality for rectifying (correcting distortions in) document images.
//! It uses a pre-trained model to transform distorted document images into properly aligned versions.
//!
//! The rectifier supports batch processing for efficient handling of multiple images.

use crate::core::ImageReader as CoreImageReader;
use crate::core::StandardPredictor;
use crate::core::{
    BatchData, CommonBuilderConfig, DefaultImageReader, OCRError, OrtInfer, Tensor4D,
    config::{ConfigValidator, ConfigValidatorExt},
};
use crate::core::{
    GranularImageReader as GIReader, InferenceEngine as GInferenceEngine, ModularPredictor,
    OrtInfer4D, Postprocessor as GPostprocessor, Preprocessor as GPreprocessor,
};
use crate::processors::{DocTrPostProcess, NormalizeImage};

use image::{DynamicImage, RgbImage};
use std::path::Path;
use std::sync::Arc;

use crate::impl_config_new_and_with_common;

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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DoctrRectifierPredictorConfig {
    /// Common configuration options shared across predictors
    pub common: CommonBuilderConfig,
    /// Input shape for the recognition model [channels, height, width]
    pub rec_image_shape: Option<[usize; 3]>,
}

impl_config_new_and_with_common!(
    DoctrRectifierPredictorConfig,
    common_defaults: (Some("doctr_rectifier".to_string()), Some(32)),
    fields: {
        rec_image_shape: Some([3, 512, 512])
    }
);

impl DoctrRectifierPredictorConfig {
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

/// Document rectifier
///
/// This struct implements a rectifier for correcting distortions in document images.
/// It uses a pre-trained model to transform distorted document images into properly aligned versions.
#[derive(Debug)]
pub struct DoctrRectifierPredictor {
    pub rec_image_shape: [usize; 3],
    pub model_name: String,
    inner: ModularPredictor<DRImageReader, DRPreprocessor, OrtInfer4D, DRPostprocessor>,
}

#[derive(Debug)]
pub struct DRImageReader {
    inner: DefaultImageReader,
}
impl DRImageReader {
    pub fn new() -> Self {
        Self {
            inner: DefaultImageReader::new(),
        }
    }
}
impl Default for DRImageReader {
    fn default() -> Self {
        Self::new()
    }
}
impl GIReader for DRImageReader {
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.inner.apply(paths)
    }
}

#[derive(Debug)]
pub struct DRPreprocessor {
    pub normalize: NormalizeImage,
}
impl GPreprocessor for DRPreprocessor {
    type Config = DoctrRectifierConfig;
    type Output = Tensor4D;
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Output, OCRError> {
        let batch_imgs: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();
        self.normalize.normalize_batch_to(batch_imgs)
    }
}

#[derive(Debug)]
pub struct DRPostprocessor {
    pub op: DocTrPostProcess,
}
impl GPostprocessor for DRPostprocessor {
    type Config = DoctrRectifierConfig;
    type InferenceOutput = Tensor4D;
    type PreprocessOutput = Tensor4D;
    type Result = DoctrRectifierResult;
    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _pre: Option<&Self::PreprocessOutput>,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        let rectified_imgs = self
            .op
            .apply_batch(&output)
            .map_err(|e| OCRError::ConfigError {
                message: format!("DocTr post-processing failed: {}", e),
            })?;
        Ok(DoctrRectifierResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            rectified_img: rectified_imgs.into_iter().map(Arc::new).collect(),
        })
    }
    fn empty_result(&self) -> crate::core::OcrResult<Self::Result> {
        Ok(DoctrRectifierResult::new())
    }
}

impl DoctrRectifierPredictor {
    pub fn new(config: DoctrRectifierPredictorConfig, model_path: &Path) -> Result<Self, OCRError> {
        let rec_image_shape = config.rec_image_shape.unwrap_or([3, 512, 512]);
        let model_name = config
            .common
            .model_name
            .as_deref()
            .unwrap_or("doctr_rectifier")
            .to_string();

        let image_reader = DRImageReader::new();
        let normalize = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            None,
        )?;
        let preprocessor = DRPreprocessor { normalize };
        let infer = OrtInfer::from_common_with_auto_input(&config.common, model_path)?;
        let inference_engine = OrtInfer4D(infer);
        let postprocessor = DRPostprocessor {
            op: DocTrPostProcess::new(1.0),
        };
        let inner =
            ModularPredictor::new(image_reader, preprocessor, inference_engine, postprocessor);

        Ok(Self {
            rec_image_shape,
            model_name,
            inner,
        })
    }

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
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.inner.image_reader.read_images(paths)
    }

    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> Result<Self::PreprocessOutput, OCRError> {
        self.inner.preprocessor.preprocess(images, config)
    }

    fn infer(&self, input: &Self::PreprocessOutput) -> Result<Self::InferenceOutput, OCRError> {
        self.inner.inference_engine.infer(input)
    }

    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _preprocessed: &Self::PreprocessOutput,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        self.inner
            .postprocessor
            .postprocess(output, None, batch_data, raw_images, config)
    }

    fn empty_result(&self) -> crate::core::OcrResult<Self::Result> {
        self.inner.postprocessor.empty_result()
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

crate::impl_common_builder_methods!(DoctrRectifierPredictorBuilder, common);

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

        let config = config.validate_and_wrap_ocr_error()?;

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
