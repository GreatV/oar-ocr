//! Text Line Classifier
//!
//! This module provides functionality for classifying the orientation of text lines in images.
//! It can detect if a text line is rotated and by how much (0° or 180°).
//!
//! The classifier uses a pre-trained model to analyze images and determine their orientation.
//! It supports batch processing for efficient handling of multiple images.

use crate::common_builder_methods;
use crate::core::ImageReader as CoreImageReader;
use crate::core::{
    BatchData, CommonBuilderConfig, DefaultImageReader, OCRError, OrtInfer, Tensor2D, Tensor4D,
    config::{ConfigValidator, ConfigValidatorExt},
    get_text_line_orientation_labels,
};
use crate::core::{
    GranularImageReader as GIReader, ModularPredictor, OrtInfer2D, Postprocessor as GPostprocessor,
    Preprocessor as GPreprocessor,
};

use crate::processors::{Crop, NormalizeImage, Topk};
use image::{DynamicImage, RgbImage};
use std::path::Path;
use std::sync::Arc;

use crate::impl_config_new_and_with_common;

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

impl_config_new_and_with_common!(
    TextLineClasPredictorConfig,
    common_defaults: (Some("PP-LCNet_x0_25".to_string()), Some(1)),
    fields: {
        topk: None,
        input_shape: Some((224, 224))
    }
);

impl TextLineClasPredictorConfig {
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

/// Text line classifier built from modular components
///
/// This is a type alias over `ModularPredictor` with concrete, composable components
/// to eliminate duplicated StandardPredictor implementations.
pub type TextLineClasPredictor =
    ModularPredictor<TLImageReader, TLPreprocessor, OrtInfer2D, TLPostprocessor>;

#[derive(Debug, Clone)]
pub struct TextLineClasConfig;

#[derive(Debug)]
pub struct TLImageReader {
    inner: DefaultImageReader,
}
impl TLImageReader {
    pub fn new() -> Self {
        Self {
            inner: DefaultImageReader::new(),
        }
    }
}
impl Default for TLImageReader {
    fn default() -> Self {
        Self::new()
    }
}
impl GIReader for TLImageReader {
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.inner.apply(paths)
    }
}

#[derive(Debug)]
pub struct TLPreprocessor {
    pub input_shape: (u32, u32),
    pub crop: Option<Crop>,
    pub normalize: NormalizeImage,
}
impl GPreprocessor for TLPreprocessor {
    type Config = TextLineClasConfig;
    type Output = Tensor4D;
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Output, OCRError> {
        use crate::utils::resize_images_batch;
        let (width, height) = self.input_shape;
        let mut batch_imgs = resize_images_batch(&images, width, height, None);
        if let Some(crop_op) = &self.crop {
            batch_imgs = crop_op.process_batch(&batch_imgs).map_err(|e| {
                OCRError::post_processing("Crop operation failed during text classification", e)
            })?;
        }
        let imgs_dynamic: Vec<DynamicImage> = batch_imgs
            .iter()
            .map(|img| DynamicImage::ImageRgb8(img.clone()))
            .collect();
        self.normalize.normalize_batch_to(imgs_dynamic)
    }
    fn preprocessing_info(&self) -> String {
        format!(
            "resize_to=({},{}) + crop? + normalize",
            self.input_shape.0, self.input_shape.1
        )
    }
}

#[derive(Debug)]
pub struct TLPostprocessor {
    pub topk: usize,
    pub topk_op: Topk,
}
impl GPostprocessor for TLPostprocessor {
    type Config = TextLineClasConfig;
    type InferenceOutput = Tensor2D;
    type PreprocessOutput = Tensor4D;
    type Result = TextLineClasResult;
    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _pre: Option<&Self::PreprocessOutput>,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        let predictions: Vec<Vec<f32>> = output.outer_iter().map(|row| row.to_vec()).collect();
        let topk_result = self
            .topk_op
            .process(&predictions, self.topk)
            .map_err(|e| OCRError::ConfigError { message: e })?;
        Ok(TextLineClasResult {
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
    fn empty_result(&self) -> crate::core::OcrResult<Self::Result> {
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

    // Inject common builder methods
    common_builder_methods!(common);

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
    pub fn build(self, model_path: &Path) -> crate::core::OcrResult<TextLineClasPredictor> {
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
    fn build_internal(
        mut self,
        model_path: &Path,
    ) -> crate::core::OcrResult<TextLineClasPredictor> {
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        let config = TextLineClasPredictorConfig {
            common: self.common,
            topk: self.topk,
            input_shape: self.input_shape,
        };

        let config = config.validate_and_wrap_ocr_error()?;

        let input_shape = config.input_shape.unwrap_or((224, 224));
        let (width, height) = input_shape;
        let crop = Some(
            Crop::new([width, height], crate::processors::CropMode::Center).map_err(|e| {
                OCRError::ConfigError {
                    message: format!("Failed to create crop operation: {e}"),
                }
            })?,
        );
        let normalize = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.485, 0.456, 0.406]),
            Some(vec![0.229, 0.224, 0.225]),
            None,
        )?;
        let preprocessor = TLPreprocessor {
            input_shape,
            crop,
            normalize,
        };
        let infer_inner = OrtInfer::from_common(&config.common, model_path, None)?;
        let inference_engine = OrtInfer2D::new(infer_inner);
        let postprocessor = TLPostprocessor {
            topk: config.topk.unwrap_or(2),
            topk_op: Topk::from_class_names(get_text_line_orientation_labels()),
        };
        let image_reader = TLImageReader::new();
        Ok(ModularPredictor::new(
            image_reader,
            preprocessor,
            inference_engine,
            postprocessor,
        ))
    }
}

impl Default for TextLineClasPredictorBuilder {
    /// Creates a new text line classifier builder with default settings
    ///
    /// This is equivalent to calling `TextLineClasPredictorBuilder::new()`.
    ///
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_line_clas_config_defaults_and_validate() {
        let config = TextLineClasPredictorConfig::new();
        // Defaults from impl_config_new_and_with_common! invocation
        assert_eq!(config.topk, None); // overridden by get_defaults when used
        assert_eq!(config.input_shape, Some((224, 224)));
        assert_eq!(config.common.model_name.as_deref(), Some("PP-LCNet_x0_25"));
        assert_eq!(config.common.batch_size, Some(1));
        // Validate should pass
        assert!(config.validate().is_ok());
    }
}
