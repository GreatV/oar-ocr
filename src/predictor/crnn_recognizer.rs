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

use crate::core::ImageReader as CoreImageReader;
use crate::core::StandardPredictor;
use crate::core::{
    BatchData, CommonBuilderConfig, ConfigValidator, ConfigValidatorExt, DefaultImageReader,
    OCRError, OrtInfer, Tensor3D, Tensor4D,
};
use crate::core::{
    GranularImageReader as GIReader, InferenceEngine as GInferenceEngine, ModularPredictor,
    OrtInfer3D, Postprocessor as GPostprocessor, Preprocessor as GPreprocessor,
};
use crate::impl_common_builder_methods;
use crate::impl_config_new_and_with_common;
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
    /// If None, the predictor defaults to [3, 48, 320] (DEFAULT_REC_IMAGE_SHAPE).
    pub model_input_shape: Option<[usize; 3]>,
    /// Character dictionary for recognition
    pub character_dict: Option<Vec<String>>,
    /// Score threshold for filtering recognition results
    pub score_thresh: Option<f32>,
}

impl_config_new_and_with_common!(
    TextRecPredictorConfig,
    common_defaults: (Some("crnn".to_string()), Some(32)),
    fields: {
        model_input_shape: Some([3, 48, 320]),
        character_dict: None,
        score_thresh: None
    }
);

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

/// Text recognition predictor built from modular components
#[derive(Debug)]
pub struct TextRecPredictor {
    pub model_input_shape: [usize; 3],
    pub character_dict: Option<Vec<String>>, // preserved for API completeness
    pub model_name: String,
    inner: ModularPredictor<TRImageReader, TRPreprocessor, OrtInfer3D, TRPostprocessor>,
}

#[derive(Debug)]
pub struct TRImageReader {
    inner: DefaultImageReader,
}
impl TRImageReader {
    pub fn new() -> Self {
        Self {
            inner: DefaultImageReader::new(),
        }
    }
}
impl Default for TRImageReader {
    fn default() -> Self {
        Self::new()
    }
}
impl GIReader for TRImageReader {
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.inner.apply(paths)
    }
}

#[derive(Debug)]
pub struct TRPreprocessor {
    pub resize: OCRResize,
    pub normalize: NormalizeImage,
}
impl GPreprocessor for TRPreprocessor {
    type Config = TextRecConfig;
    type Output = Tensor4D;
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> Result<Self::Output, OCRError> {
        let resized_imgs = self.resize.apply_to_images(&images)?;
        let dynamic_imgs: Vec<image::DynamicImage> = resized_imgs
            .into_iter()
            .map(image::DynamicImage::ImageRgb8)
            .collect();
        self.normalize.normalize_batch_to(dynamic_imgs)
    }
}

#[derive(Debug)]
pub struct TRPostprocessor {
    pub decoder: CTCLabelDecode,
}
impl GPostprocessor for TRPostprocessor {
    type Config = TextRecConfig;
    type InferenceOutput = Tensor3D;
    type PreprocessOutput = Tensor4D;
    type Result = TextRecResult;
    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        _pre: Option<&Self::PreprocessOutput>,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        _config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        let (texts, scores) = self.decoder.apply(&output);
        Ok(TextRecResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            rec_text: texts.into_iter().map(Arc::from).collect(),
            rec_score: scores,
        })
    }
    fn empty_result(&self) -> crate::core::OcrResult<Self::Result> {
        Ok(TextRecResult::new())
    }
}

impl TextRecPredictor {
    pub fn new(config: TextRecPredictorConfig, model_path: &Path) -> crate::core::OcrResult<Self> {
        let model_input_shape = config.model_input_shape.unwrap_or([3, 48, 320]);
        let character_dict = config.character_dict.clone();
        let model_name = config
            .common
            .model_name
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "crnn".to_string());

        let image_reader = TRImageReader::new();
        let resize = OCRResize::new(Some(model_input_shape), None);
        let normalize = NormalizeImage::for_ocr_recognition()?;
        let preprocessor = TRPreprocessor { resize, normalize };
        let infer = OrtInfer::from_common(&config.common, model_path, None)?;
        let inference_engine = OrtInfer3D(infer);
        let decoder = CTCLabelDecode::from_string_list(character_dict.as_deref(), true, false);
        let postprocessor = TRPostprocessor { decoder };
        let inner =
            ModularPredictor::new(image_reader, preprocessor, inference_engine, postprocessor);

        Ok(Self {
            model_input_shape,
            character_dict,
            model_name,
            inner,
        })
    }

    pub fn set_model_input_shape(&mut self, shape: [usize; 3]) {
        self.model_input_shape = shape;
        self.inner.preprocessor.resize = OCRResize::new(Some(shape), None);
    }

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
    ) -> crate::core::OcrResult<Vec<RgbImage>> {
        self.inner.image_reader.read_images(paths)
    }

    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::PreprocessOutput> {
        self.inner.preprocessor.preprocess(images, config)
    }

    fn infer(
        &self,
        input: &Self::PreprocessOutput,
    ) -> crate::core::OcrResult<Self::InferenceOutput> {
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

impl_common_builder_methods!(TextRecPredictorBuilder, common);

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
    pub fn build(self, model_path: &Path) -> crate::core::OcrResult<TextRecPredictor> {
        self.build_internal(model_path)
    }

    /// Builds the `TextRecPredictor` internally
    ///
    /// This function builds the `TextRecPredictor` with the provided configuration.
    /// It also validates the configuration and handles the model path.
    fn build_internal(mut self, model_path: &Path) -> crate::core::OcrResult<TextRecPredictor> {
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

    #[cfg(test)]
    mod tests_local {
        use super::*;

        #[test]
        fn test_text_rec_config_defaults_and_validate() {
            let config = TextRecPredictorConfig::new();
            assert_eq!(config.model_input_shape, Some([3, 48, 320]));
            assert_eq!(config.common.model_name.as_deref(), Some("crnn"));
            assert_eq!(config.common.batch_size, Some(32));
            assert!(config.validate().is_ok());
        }
    }

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
