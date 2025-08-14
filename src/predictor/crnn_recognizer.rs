//! CRNN (Convolutional Recurrent Neural Network) Text Recognizer
//!
//! This module implements a text recognition predictor using the CRNN model,
//! which combines convolutional layers for feature extraction and recurrent layers
//! for sequence modeling. It's commonly used for recognizing text in images.
//!
//! The main components are:
//! - `TextRecPredictor`: The main predictor that performs text recognition
//! - `TextRecPredictorConfig`: Configuration for the predictor
//! - `TextRecResult`: Results from text recognition
//! - `TextRecPredictorBuilder`: Builder for creating predictor instances

use crate::core::traits::StandardPredictor;
use crate::core::{
    BatchData, BatchSampler, CommonBuilderConfig, ConfigValidator, ConfigValidatorExt,
    DefaultImageReader, ImageReader, OCRError, OrtInfer, Tensor3D, Tensor4D, ToBatch,
};
use crate::processors::{CTCLabelDecode, NormalizeImage, OCRResize};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;

/// Results from text recognition
///
/// This struct holds the results of text recognition operations,
/// including the recognized text, confidence scores, and associated metadata.
#[derive(Debug, Clone)]
pub struct TextRecResult {
    /// Paths to the input images
    pub input_path: Vec<Arc<str>>,
    /// Indexes of the input images
    pub index: Vec<usize>,
    /// Input images
    pub input_img: Vec<Arc<RgbImage>>,
    /// Recognized text
    pub rec_text: Vec<Arc<str>>,
    /// Confidence scores for the recognized text
    pub rec_score: Vec<f32>,
}

/// Configuration for the text recognition predictor
///
/// This struct holds the configuration parameters for the text recognition predictor.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TextRecPredictorConfig {
    /// Common configuration parameters
    pub common: CommonBuilderConfig,
    /// Model input shape for image resizing [channels, height, width]
    /// When specified, images are resized to fit this shape while maintaining aspect ratio.
    pub model_input_shape: Option<[usize; 3]>,
    /// Character dictionary for recognition
    pub character_dict: Option<Vec<String>>,
    /// Score threshold for filtering recognition results
    pub score_thresh: Option<f32>,
}

impl TextRecPredictorConfig {
    /// Creates a new `TextRecPredictorConfig` with default values
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::with_defaults(Some("crnn".to_string()), Some(32)),
            model_input_shape: Some([3, 48, 320]),
            character_dict: None,
            score_thresh: None,
        }
    }

    /// Creates a new `TextRecPredictorConfig` with the provided common configuration
    pub fn with_common(common: CommonBuilderConfig) -> Self {
        Self {
            common,
            model_input_shape: Some([3, 32, 320]),
            character_dict: None,
            score_thresh: None,
        }
    }
}

impl ConfigValidator for TextRecPredictorConfig {
    fn validate(&self) -> Result<(), crate::core::ConfigError> {
        self.common.validate()?;

        if let Some(shape) = self.model_input_shape
            && (shape[0] == 0 || shape[1] == 0 || shape[2] == 0)
        {
            return Err(crate::core::ConfigError::InvalidConfig {
                message: "Model input shape dimensions must be greater than 0".to_string(),
            });
        }

        Ok(())
    }

    fn get_defaults() -> Self {
        Self::new()
    }
}

impl TextRecResult {
    /// Creates a new, empty `TextRecResult`
    pub fn new() -> Self {
        Self {
            input_path: Vec::new(),
            index: Vec::new(),
            input_img: Vec::new(),
            rec_text: Vec::new(),
            rec_score: Vec::new(),
        }
    }
}

impl Default for TextRecResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Text recognition predictor
///
/// This struct holds the components needed for text recognition.
#[derive(Debug)]
pub struct TextRecPredictor {
    /// Model input shape for image resizing [channels, height, width]
    pub model_input_shape: [usize; 3],
    /// Character dictionary for recognition
    pub character_dict: Option<Vec<String>>,
    /// Name of the model
    pub model_name: String,

    /// Batch sampler
    pub batch_sampler: BatchSampler,
    /// Image reader
    pub read_image: DefaultImageReader,
    /// Image resizer
    pub resize: OCRResize,
    /// Image normalizer
    pub normalize: NormalizeImage,
    /// Batch converter
    pub to_batch: ToBatch,
    /// Model inference engine
    pub infer: OrtInfer,
    /// Post-processing operation
    pub post_op: CTCLabelDecode,
}

impl TextRecPredictor {
    /// Creates a new `TextRecPredictor`
    ///
    /// This function initializes a new text recognition predictor with the provided configuration
    /// and model path.
    pub fn new(config: TextRecPredictorConfig, model_path: &Path) -> Result<Self, OCRError> {
        let model_input_shape = config.model_input_shape.unwrap_or([3, 32, 320]);
        let character_dict = config.character_dict.clone();
        let model_name = config
            .common
            .model_name
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "crnn".to_string());
        let batch_size = config.common.batch_size.unwrap_or(32);

        let batch_sampler = BatchSampler::new(batch_size);
        let read_image = DefaultImageReader::new();
        // Use dynamic resizing (old rec_image_shape behavior) - maintains aspect ratio
        let resize = OCRResize::new(Some(model_input_shape), None);
        let normalize = NormalizeImage::for_ocr_recognition()?;
        let to_batch = ToBatch::new();
        let infer = OrtInfer::from_common(
            &TextRecPredictorConfig {
                common: config.common.clone(),
                ..config.clone()
            }
            .common,
            model_path,
            None,
        )?;
        let post_op = CTCLabelDecode::from_string_list(character_dict.as_deref(), true, false);

        Ok(Self {
            model_input_shape,
            character_dict,
            model_name,
            batch_sampler,
            read_image,
            resize,
            normalize,
            to_batch,
            infer,
            post_op,
        })
    }

    /// Sets the model input shape
    ///
    /// This function updates the model input shape and the resize component.
    pub fn set_model_input_shape(&mut self, shape: [usize; 3]) {
        self.model_input_shape = shape;
        self.resize = OCRResize::new(Some(shape), None);
    }

    /// Returns the model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Configuration for text recognition
///
/// This struct is used as a placeholder for text recognition configuration.
#[derive(Debug, Clone)]
pub struct TextRecConfig;

impl StandardPredictor for TextRecPredictor {
    type Config = TextRecConfig;
    type Result = TextRecResult;
    type PreprocessOutput = Tensor4D;
    type InferenceOutput = Tensor3D;

    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.read_image.apply(paths)
    }

    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::PreprocessOutput, OCRError> {
        let resized_imgs = self.resize.apply_to_images(&images)?;

        let dynamic_imgs: Vec<image::DynamicImage> = resized_imgs
            .into_iter()
            .map(image::DynamicImage::ImageRgb8)
            .collect();

        let batch_size = dynamic_imgs.len();
        self.normalize
            .normalize_batch_to(dynamic_imgs)
            .map_err(|e| {
                OCRError::model_inference_error(
                    &self.model_name,
                    "preprocessing_normalization",
                    0,
                    &[batch_size], // batch size as shape info
                    &format!(
                        "Text recognition normalization failed for {} images with input shape {:?}",
                        batch_size, self.model_input_shape
                    ),
                    e,
                )
            })
    }

    fn infer(&self, input: &Self::PreprocessOutput) -> Result<Self::InferenceOutput, OCRError> {
        let input_shape = input.shape().to_vec();
        self.infer.infer_3d(input.clone()).map_err(|e| {
            OCRError::model_inference_error(
                &self.model_name,
                "inference_3d",
                0,
                &input_shape,
                &format!("Text recognition inference failed with input shape {:?}, model input shape {:?}",
                    input_shape, self.model_input_shape),
                e,
            )
        })
    }

    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _preprocessed: &Self::PreprocessOutput,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Result, OCRError> {
        let (texts, scores) = self.post_op.apply(&output);

        Ok(TextRecResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            rec_text: texts.into_iter().map(Arc::from).collect(),
            rec_score: scores,
        })
    }

    fn empty_result(&self) -> Result<Self::Result, OCRError> {
        Ok(TextRecResult::new())
    }
}

/// Builder for `TextRecPredictor`
///
/// This struct is used to build a `TextRecPredictor` with the desired configuration.
pub struct TextRecPredictorBuilder {
    /// Common configuration parameters
    common: CommonBuilderConfig,

    /// Model input shape for image resizing [channels, height, width]
    model_input_shape: Option<[usize; 3]>,
    /// Character dictionary for recognition
    character_dict: Option<Vec<String>>,
    /// Score threshold for filtering recognition results
    score_thresh: Option<f32>,
}

impl TextRecPredictorBuilder {
    /// Creates a new `TextRecPredictorBuilder`
    ///
    /// This function initializes a new builder with default values.
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::new(),
            model_input_shape: None,
            character_dict: None,
            score_thresh: None,
        }
    }

    /// Sets the model path
    ///
    /// This function sets the path to the model file.
    pub fn model_path(mut self, model_path: impl Into<std::path::PathBuf>) -> Self {
        self.common = self.common.model_path(model_path);
        self
    }

    /// Sets the model name
    ///
    /// This function sets the name of the model.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.common = self.common.model_name(model_name);
        self
    }

    /// Sets the batch size
    ///
    /// This function sets the batch size for processing.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.common = self.common.batch_size(batch_size);
        self
    }

    /// Enables or disables logging
    ///
    /// This function enables or disables logging for the predictor.
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

    /// Sets the model input shape
    ///
    /// This function sets the model input shape for image resizing.
    /// Images will be resized to fit this shape while maintaining aspect ratio.
    pub fn model_input_shape(mut self, shape: [usize; 3]) -> Self {
        self.model_input_shape = Some(shape);
        self
    }

    /// Sets the character dictionary
    ///
    /// This function sets the character dictionary for recognition.
    pub fn character_dict(mut self, character_dict: Vec<String>) -> Self {
        self.character_dict = Some(character_dict);
        self
    }

    /// Sets the score threshold for filtering recognition results
    ///
    /// This function sets the minimum score threshold for recognition results.
    /// Results with scores below this threshold will be filtered out.
    pub fn score_thresh(mut self, score_thresh: f32) -> Self {
        self.score_thresh = Some(score_thresh);
        self
    }

    /// Builds the `TextRecPredictor`
    ///
    /// This function builds the `TextRecPredictor` with the provided configuration.
    pub fn build(self, model_path: &Path) -> Result<TextRecPredictor, OCRError> {
        self.build_internal(model_path)
    }

    /// Builds the `TextRecPredictor` internally
    ///
    /// This function builds the `TextRecPredictor` with the provided configuration.
    /// It also validates the configuration and handles the model path.
    fn build_internal(mut self, model_path: &Path) -> Result<TextRecPredictor, OCRError> {
        // Ensure model path is set first
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        // Build the configuration
        let config = TextRecPredictorConfig {
            common: self.common,
            model_input_shape: self.model_input_shape,
            character_dict: self.character_dict,
            score_thresh: self.score_thresh,
        };

        // Validate the configuration
        let config = config.validate_and_wrap_ocr_error()?;

        // Create the predictor
        TextRecPredictor::new(config, model_path)
    }
}

impl Default for TextRecPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_rec_predictor_config_score_thresh() {
        // Test default configuration
        let config = TextRecPredictorConfig::new();
        assert_eq!(config.score_thresh, None);

        // Test configuration with score threshold
        let mut config = TextRecPredictorConfig::new();
        config.score_thresh = Some(0.5);
        assert_eq!(config.score_thresh, Some(0.5));
    }

    #[test]
    fn test_text_rec_predictor_builder_score_thresh() {
        // Test builder with score threshold
        let builder = TextRecPredictorBuilder::new().score_thresh(0.7);

        assert_eq!(builder.score_thresh, Some(0.7));
    }
}
