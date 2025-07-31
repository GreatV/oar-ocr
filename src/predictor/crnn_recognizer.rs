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
    BasePredictor, BatchData, BatchSampler, CommonBuilderConfig, DefaultImageReader, ImageReader,
    IntoPrediction, OCRError, OrtInfer, PredictionResult, Sampler, Tensor3D, Tensor4D, ToBatch,
};

use crate::impl_predictor_from_generic;
use crate::impl_standard_predictor;
use crate::impl_standard_predictor_builder;
use crate::processors::{CTCLabelDecode, NormalizeImage, OCRResize};
use image::RgbImage;
use std::borrow::Cow;
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
#[derive(Debug, Clone, Default)]
pub struct TextRecPredictorConfig {
    /// Common configuration parameters
    pub common: CommonBuilderConfig,
    /// Input shape for the model
    pub input_shape: Option<[usize; 3]>,
    /// Recognition image shape
    pub rec_image_shape: Option<[usize; 3]>,
    /// Character dictionary for recognition
    pub character_dict: Option<Vec<String>>,
}

impl TextRecPredictorConfig {
    /// Creates a new `TextRecPredictorConfig` with default values
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::with_defaults(Some("crnn".to_string()), Some(32)),
            input_shape: None,
            rec_image_shape: Some([3, 32, 320]),
            character_dict: None,
        }
    }

    /// Creates a new `TextRecPredictorConfig` with the provided common configuration
    pub fn with_common(common: CommonBuilderConfig) -> Self {
        Self {
            common,
            input_shape: None,
            rec_image_shape: Some([3, 32, 320]),
            character_dict: None,
        }
    }

    /// Validates the configuration
    ///
    /// This function checks that the configuration parameters are valid.
    /// It returns an error if any of the parameters are invalid.
    pub fn validate(&self) -> Result<(), crate::core::ConfigError> {
        self.common.validate()?;

        if let Some(rec_shape) = self.rec_image_shape {
            if rec_shape[0] == 0 || rec_shape[1] == 0 || rec_shape[2] == 0 {
                return Err(crate::core::ConfigError::InvalidConfig {
                    message: "Recognition image shape dimensions must be greater than 0"
                        .to_string(),
                });
            }
        }

        if let Some(input_shape) = self.input_shape {
            if input_shape[0] == 0 || input_shape[1] == 0 || input_shape[2] == 0 {
                return Err(crate::core::ConfigError::InvalidConfig {
                    message: "Input shape dimensions must be greater than 0".to_string(),
                });
            }
        }

        Ok(())
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

impl IntoPrediction for TextRecResult {
    type Out = PredictionResult<'static>;

    fn into_prediction(self) -> Self::Out {
        PredictionResult::Recognition {
            input_path: self
                .input_path
                .into_iter()
                .map(|arc| Cow::Owned(arc.to_string()))
                .collect(),
            index: self.index,
            input_img: self.input_img,
            rec_text: self
                .rec_text
                .into_iter()
                .map(|arc| Cow::Owned(arc.to_string()))
                .collect(),
            rec_score: self.rec_score,
        }
    }
}

/// Text recognition predictor
///
/// This struct holds the components needed for text recognition.
#[derive(Debug)]
pub struct TextRecPredictor {
    /// Input shape for the model
    pub input_shape: Option<[usize; 3]>,
    /// Recognition image shape
    pub rec_image_shape: [usize; 3],
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
        let rec_image_shape = config.rec_image_shape.unwrap_or([3, 32, 320]);
        let input_shape = config.input_shape;
        let character_dict = config.character_dict;
        let model_name = config
            .common
            .model_name
            .unwrap_or_else(|| "crnn".to_string());
        let batch_size = config.common.batch_size.unwrap_or(32);

        let batch_sampler = BatchSampler::new(batch_size);
        let read_image = DefaultImageReader::new();
        let resize = OCRResize::new(Some(rec_image_shape), input_shape);
        let normalize = NormalizeImage::for_ocr_recognition()?;
        let to_batch = ToBatch::new();
        let infer = OrtInfer::new(model_path, None)?;
        let post_op = CTCLabelDecode::from_string_list(character_dict.as_deref(), true, false);

        Ok(Self {
            input_shape,
            rec_image_shape,
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

    /// Processes a batch of data internally
    ///
    /// This function processes a batch of data and returns the recognition results.
    fn process_internal(&mut self, batch_data: BatchData) -> Result<TextRecResult, OCRError> {
        self.predict(batch_data, None)
    }

    /// Sets the recognition image shape
    ///
    /// This function updates the recognition image shape and the resize component.
    pub fn set_rec_image_shape(&mut self, shape: [usize; 3]) {
        self.rec_image_shape = shape;
        self.resize = OCRResize::new(Some(shape), self.input_shape);
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
        let resized_imgs = self.resize.apply_to_images(&images)?;

        let dynamic_imgs: Vec<image::DynamicImage> = resized_imgs
            .into_iter()
            .map(image::DynamicImage::ImageRgb8)
            .collect();

        self.normalize.normalize_batch_to(dynamic_imgs)
    }

    fn infer(&mut self, input: &Self::PreprocessOutput) -> Result<Self::InferenceOutput, OCRError> {
        self.infer.infer_3d(input.clone())
    }

    fn postprocess(
        &mut self,
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
}

/// Builder for `TextRecPredictor`
///
/// This struct is used to build a `TextRecPredictor` with the desired configuration.
pub struct TextRecPredictorBuilder {
    /// Common configuration parameters
    common: CommonBuilderConfig,

    /// Input shape for the model
    input_shape: Option<[usize; 3]>,
    /// Recognition image shape
    rec_image_shape: Option<[usize; 3]>,
    /// Character dictionary for recognition
    character_dict: Option<Vec<String>>,
}

impl TextRecPredictorBuilder {
    /// Creates a new `TextRecPredictorBuilder`
    ///
    /// This function initializes a new builder with default values.
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::new(),
            input_shape: None,
            rec_image_shape: None,
            character_dict: None,
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

    /// Sets the input shape
    ///
    /// This function sets the input shape for the model.
    pub fn input_shape(mut self, input_shape: [usize; 3]) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    /// Sets the recognition image shape
    ///
    /// This function sets the recognition image shape for the model.
    pub fn rec_image_shape(mut self, rec_image_shape: [usize; 3]) -> Self {
        self.rec_image_shape = Some(rec_image_shape);
        self
    }

    /// Sets the character dictionary
    ///
    /// This function sets the character dictionary for recognition.
    pub fn character_dict(mut self, character_dict: Vec<String>) -> Self {
        self.character_dict = Some(character_dict);
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
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        let config = TextRecPredictorConfig {
            common: self.common,
            input_shape: self.input_shape,
            rec_image_shape: self.rec_image_shape,
            character_dict: self.character_dict,
        };

        config.validate().map_err(|e| OCRError::ConfigError {
            message: e.to_string(),
        })?;

        TextRecPredictor::new(config, model_path)
    }
}

impl Default for TextRecPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl_standard_predictor!(TextRecPredictor, TextRecResult, OCRError, "TextRecognition");

impl_predictor_from_generic!(TextRecPredictor);

impl_standard_predictor_builder!(TextRecPredictorBuilder, TextRecPredictor, "TextRecognition");
