//! DB (Differentiable Binarization) Text Detector
//!
//! This module implements a text detection predictor using the DB model,
//! which is designed for detecting text regions in images. The DB model
//! uses a differentiable binarization technique to improve the accuracy
//! of text detection.
//!
//! The main components are:
//! - `TextDetPredictor`: The main predictor that performs text detection
//! - `TextDetPredictorConfig`: Configuration for the predictor
//! - `TextDetResult`: Results from text detection
//! - `TextDetPredictorBuilder`: Builder for creating predictor instances

use crate::core::batch::ToBatch;
use crate::processors::{BoundingBox, DBPostProcess, DetResizeForTest, LimitType, NormalizeImage};
use image::{DynamicImage, RgbImage};
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use crate::core::traits::StandardPredictor;
use crate::core::{
    BasePredictor, BatchData, CommonBuilderConfig, IntoPrediction, OCRError, PredictionResult,
    Tensor4D,
    config::ConfigValidator,
    constants::{DEFAULT_BATCH_SIZE, DEFAULT_MAX_SIDE_LIMIT},
};
use crate::impl_predictor_from_generic;
use crate::impl_standard_predictor;

const DEFAULT_THRESH: f32 = 0.3;

const DEFAULT_BOX_THRESH: f32 = 0.6;

const DEFAULT_UNCLIP_RATIO: f32 = 1.5;

/// Configuration for text detection
///
/// This struct holds configuration parameters for text detection.
#[derive(Debug, Clone, Default)]
pub struct TextDetConfig {
    /// Limit for the side length of the image
    pub limit_side_len: Option<u32>,
    /// Type of limit to apply (Max or Min)
    pub limit_type: Option<LimitType>,
    /// Threshold for binarization
    pub thresh: Option<f32>,
    /// Threshold for filtering text boxes
    pub box_thresh: Option<f32>,
    /// Ratio for unclipping text boxes
    pub unclip_ratio: Option<f32>,
    /// Maximum side limit for the image
    pub max_side_limit: Option<u32>,
}

/// Configuration for the text detection predictor
///
/// This struct holds configuration parameters for the text detection predictor.
#[derive(Debug, Clone, Default)]
pub struct TextDetPredictorConfig {
    /// Common configuration parameters
    pub common: CommonBuilderConfig,
    /// Limit for the side length of the image
    pub limit_side_len: Option<u32>,
    /// Type of limit to apply (Max or Min)
    pub limit_type: Option<LimitType>,
    /// Threshold for binarization
    pub thresh: Option<f32>,
    /// Threshold for filtering text boxes
    pub box_thresh: Option<f32>,
    /// Ratio for unclipping text boxes
    pub unclip_ratio: Option<f32>,
    /// Input shape for the model (channels, height, width)
    pub input_shape: Option<(u32, u32, u32)>,
    /// Maximum side limit for the image
    pub max_side_limit: Option<u32>,
}

impl TextDetPredictorConfig {
    /// Creates a new `TextDetPredictorConfig` with default values
    ///
    /// This function initializes a new text detection predictor configuration
    /// with default values for all parameters.
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::with_defaults(None, Some(DEFAULT_BATCH_SIZE)),
            limit_side_len: None,
            limit_type: None,
            thresh: None,
            box_thresh: None,
            unclip_ratio: None,
            input_shape: None,
            max_side_limit: Some(DEFAULT_MAX_SIDE_LIMIT),
        }
    }

    /// Creates a new `TextDetPredictorConfig` with the provided common configuration
    ///
    /// This function initializes a new text detection predictor configuration
    /// with the provided common configuration and default values for other parameters.
    pub fn with_common(common: CommonBuilderConfig) -> Self {
        Self {
            common,
            limit_side_len: None,
            limit_type: None,
            thresh: None,
            box_thresh: None,
            unclip_ratio: None,
            input_shape: None,
            max_side_limit: Some(DEFAULT_MAX_SIDE_LIMIT),
        }
    }

    /// Validates the configuration
    ///
    /// This function validates the configuration parameters to ensure they are within
    /// acceptable ranges and formats.
    pub fn validate(&self) -> Result<(), crate::core::ConfigError> {
        ConfigValidator::validate(self)
    }
}

impl ConfigValidator for TextDetPredictorConfig {
    fn validate(&self) -> Result<(), crate::core::ConfigError> {
        self.common.validate()?;

        if let Some(thresh) = self.thresh {
            self.validate_f32_range(thresh, 0.0, 1.0, "threshold")?;
        }

        if let Some(box_thresh) = self.box_thresh {
            self.validate_f32_range(box_thresh, 0.0, 1.0, "box threshold")?;
        }

        if let Some(unclip_ratio) = self.unclip_ratio {
            self.validate_positive_f32(unclip_ratio, "unclip ratio")?;
        }

        if let Some(max_side_limit) = self.max_side_limit {
            self.validate_positive_usize(max_side_limit as usize, "max side limit")?;
        }

        if let Some(limit_side_len) = self.limit_side_len {
            self.validate_positive_usize(limit_side_len as usize, "limit side length")?;
        }

        if let Some((c, h, w)) = self.input_shape {
            if c == 0 || h == 0 || w == 0 {
                return Err(crate::core::ConfigError::InvalidConfig {
                    message: format!(
                        "Input shape dimensions must be greater than 0, got ({}, {}, {})",
                        c, h, w
                    ),
                });
            }
        }

        Ok(())
    }

    fn get_defaults() -> Self {
        Self {
            common: CommonBuilderConfig::get_defaults(),
            limit_side_len: Some(960),
            limit_type: Some(LimitType::Max),
            thresh: Some(DEFAULT_THRESH),
            box_thresh: Some(DEFAULT_BOX_THRESH),
            unclip_ratio: Some(DEFAULT_UNCLIP_RATIO),
            input_shape: Some((3, 640, 640)),
            max_side_limit: Some(DEFAULT_MAX_SIDE_LIMIT),
        }
    }
}

/// Results from text detection
///
/// This struct holds the results of text detection operations.
#[derive(Debug, Clone)]
pub struct TextDetResult {
    /// Paths to the input images
    pub input_path: Vec<Arc<str>>,
    /// Indexes of the input images
    pub index: Vec<usize>,
    /// Input images
    pub input_img: Vec<Arc<RgbImage>>,
    /// Detected polygons
    pub dt_polys: Vec<Vec<BoundingBox>>,
    /// Detection scores
    pub dt_scores: Vec<Vec<f32>>,
}

impl TextDetResult {
    /// Creates a new, empty `TextDetResult`
    ///
    /// This function initializes a new text detection result with empty vectors
    /// for all fields.
    pub fn new() -> Self {
        Self {
            input_path: Vec::new(),
            index: Vec::new(),
            input_img: Vec::new(),
            dt_polys: Vec::new(),
            dt_scores: Vec::new(),
        }
    }
}

impl Default for TextDetResult {
    fn default() -> Self {
        Self::new()
    }
}

use crate::core::{BatchSampler, DefaultImageReader, ImageReader, OrtInfer, Sampler};

/// Text detection predictor
///
/// This struct holds the components needed for text detection.
pub struct TextDetPredictor {
    /// Limit for the side length of the image
    pub limit_side_len: Option<u32>,
    /// Type of limit to apply (Max or Min)
    pub limit_type: Option<LimitType>,
    /// Threshold for binarization
    pub thresh: Option<f32>,
    /// Threshold for filtering text boxes
    pub box_thresh: Option<f32>,
    /// Ratio for unclipping text boxes
    pub unclip_ratio: Option<f32>,
    /// Input shape for the model (channels, height, width)
    pub input_shape: Option<(u32, u32, u32)>,
    /// Maximum side limit for the image
    pub max_side_limit: u32,
    /// Name of the model
    pub model_name: String,

    /// Batch sampler
    pub batch_sampler: BatchSampler,
    /// Image reader
    pub read_image: DefaultImageReader,
    /// Image resizer
    pub resize: DetResizeForTest,
    /// Image normalizer
    pub normalize: NormalizeImage,
    /// Batch converter
    pub to_batch: ToBatch,
    /// Model inference engine
    pub infer: OrtInfer,
    /// Post-processing operation
    pub post_op: DBPostProcess,
}

impl TextDetPredictor {
    /// Creates a new `TextDetPredictor`
    ///
    /// This function initializes a new text detection predictor with the provided
    /// configuration and model path.
    pub fn new(config: TextDetPredictorConfig, model_path: &Path) -> Result<Self, OCRError> {
        let batch_size = config.common.batch_size.unwrap_or(6);
        let batch_sampler = BatchSampler::new(batch_size);
        let read_image = DefaultImageReader::new();

        let (default_limit_side_len, default_limit_type) =
            if let Some(model_name) = &config.common.model_name {
                match model_name.as_str() {
                    "PP-OCRv5_server_det"
                    | "PP-OCRv5_mobile_det"
                    | "PP-OCRv4_server_det"
                    | "PP-OCRv4_mobile_det"
                    | "PP-OCRv3_server_det"
                    | "PP-OCRv3_mobile_det" => (960, LimitType::Max),
                    _ => (736, LimitType::Min),
                }
            } else {
                (736, LimitType::Min)
            };

        let limit_side_len = config.limit_side_len.unwrap_or(default_limit_side_len);
        let limit_type = config.limit_type.clone().unwrap_or(default_limit_type);
        let max_side_limit = config.max_side_limit.unwrap_or(DEFAULT_MAX_SIDE_LIMIT);

        let resize = DetResizeForTest::new(
            config.input_shape,
            None,
            None,
            Some(limit_side_len),
            Some(limit_type),
            None,
            Some(max_side_limit),
        );
        let normalize = NormalizeImage::new(None, None, None, None)?;
        let to_batch = ToBatch::new();
        let infer = OrtInfer::new(model_path, None)?;
        let post_op = DBPostProcess::new(None, None, None, None, None, None, None);

        Ok(TextDetPredictor {
            limit_side_len: config.limit_side_len,
            limit_type: config.limit_type,
            thresh: config.thresh,
            box_thresh: config.box_thresh,
            unclip_ratio: config.unclip_ratio,
            input_shape: config.input_shape,
            max_side_limit,
            model_name: config
                .common
                .model_name
                .unwrap_or_else(|| "TextDetection".to_string()),
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
    /// This function processes a batch of data and returns the detection results.
    fn process_internal(
        &mut self,
        batch_data: BatchData,
        config: Option<TextDetConfig>,
    ) -> Result<TextDetResult, OCRError> {
        self.predict(batch_data, config)
    }

    /// Sets the threshold for binarization
    ///
    /// This function sets the threshold value used for binarization in text detection.
    pub fn set_thresh(&mut self, thresh: f32) {
        self.thresh = Some(thresh);
    }

    /// Sets the threshold for filtering text boxes
    ///
    /// This function sets the threshold value used for filtering text boxes in text detection.
    pub fn set_box_thresh(&mut self, box_thresh: f32) {
        self.box_thresh = Some(box_thresh);
    }

    /// Sets the ratio for unclipping text boxes
    ///
    /// This function sets the ratio used for unclipping text boxes in text detection.
    pub fn set_unclip_ratio(&mut self, unclip_ratio: f32) {
        self.unclip_ratio = Some(unclip_ratio);
    }

    /// Sets the limit for the side length of the image
    ///
    /// This function sets the limit for the side length of the image used in text detection.
    pub fn set_limit_side_len(&mut self, limit_side_len: u32) {
        self.limit_side_len = Some(limit_side_len);
    }

    /// Sets the type of limit to apply
    ///
    /// This function sets the type of limit (Max or Min) to apply to the image side length
    /// in text detection.
    pub fn set_limit_type(&mut self, limit_type: LimitType) {
        self.limit_type = Some(limit_type);
    }

    /// Returns the name of the model
    ///
    /// This function returns the name of the model used for text detection.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Processes a batch of data with a specific configuration
    ///
    /// This function processes a batch of data with the provided configuration
    /// and returns the detection results.
    pub fn process_with_config(
        &mut self,
        batch_data: BatchData,
        config: Option<TextDetConfig>,
    ) -> Result<TextDetResult, OCRError> {
        self.process_internal(batch_data, config)
    }
}

/// Preprocessing output for text detection
///
/// This struct holds the output of the preprocessing step for text detection.
#[derive(Debug)]
pub struct TextDetPreprocessOutput {
    /// Tensor output from preprocessing
    pub tensor: Tensor4D,
    /// Shapes of the preprocessed images
    pub shapes: Vec<[f32; 4]>,
}

impl StandardPredictor for TextDetPredictor {
    type Config = TextDetConfig;
    type Result = TextDetResult;
    type PreprocessOutput = TextDetPreprocessOutput;
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
        config: Option<&Self::Config>,
    ) -> Result<Self::PreprocessOutput, OCRError> {
        let config = config.cloned().unwrap_or_default();

        let limit_side_len = config.limit_side_len.or(self.limit_side_len).unwrap_or(960);
        let limit_type = config
            .limit_type
            .or(self.limit_type.clone())
            .unwrap_or(LimitType::Min);
        let max_side_limit = config.max_side_limit.unwrap_or(self.max_side_limit);

        let batch_imgs: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();

        let (resized_imgs, shapes) = self.resize.apply(
            batch_imgs,
            Some(limit_side_len),
            Some(limit_type),
            Some(max_side_limit),
        );

        let tensor = self.normalize.normalize_batch_to(resized_imgs)?;

        Ok(TextDetPreprocessOutput { tensor, shapes })
    }

    fn infer(&mut self, input: &Self::PreprocessOutput) -> Result<Self::InferenceOutput, OCRError> {
        self.infer.infer_4d(input.tensor.clone())
    }

    fn postprocess(
        &mut self,
        output: Self::InferenceOutput,
        preprocessed: &Self::PreprocessOutput,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> Result<Self::Result, OCRError> {
        let config = config.cloned().unwrap_or_default();

        let thresh = config.thresh.or(self.thresh).unwrap_or(DEFAULT_THRESH);
        let box_thresh = config
            .box_thresh
            .or(self.box_thresh)
            .unwrap_or(DEFAULT_BOX_THRESH);
        let unclip_ratio = config
            .unclip_ratio
            .or(self.unclip_ratio)
            .unwrap_or(DEFAULT_UNCLIP_RATIO);

        let (polys, scores) = self.post_op.apply(
            &output,
            preprocessed.shapes.clone(),
            Some(thresh),
            Some(box_thresh),
            Some(unclip_ratio),
        );

        Ok(TextDetResult {
            input_path: batch_data.input_paths.clone(),
            index: batch_data.indexes.clone(),
            input_img: raw_images.into_iter().map(Arc::new).collect(),
            dt_polys: polys,
            dt_scores: scores,
        })
    }
}

/// Builder for `TextDetPredictor`
///
/// This struct is used to build a `TextDetPredictor` with the desired configuration.
pub struct TextDetPredictorBuilder {
    /// Common configuration parameters
    common: CommonBuilderConfig,

    /// Limit for the side length of the image
    limit_side_len: Option<u32>,
    /// Type of limit to apply (Max or Min)
    limit_type: Option<LimitType>,
    /// Threshold for binarization
    thresh: Option<f32>,
    /// Threshold for filtering text boxes
    box_thresh: Option<f32>,
    /// Ratio for unclipping text boxes
    unclip_ratio: Option<f32>,
    /// Input shape for the model (channels, height, width)
    input_shape: Option<(u32, u32, u32)>,
    /// Maximum side limit for the image
    max_side_limit: Option<u32>,
}

impl TextDetPredictorBuilder {
    /// Creates a new `TextDetPredictorBuilder`
    ///
    /// This function initializes a new builder with default values.
    pub fn new() -> Self {
        Self {
            common: CommonBuilderConfig::new(),
            limit_side_len: None,
            limit_type: None,
            thresh: None,
            box_thresh: None,
            unclip_ratio: None,
            input_shape: None,
            max_side_limit: None,
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

    /// Sets the limit for the side length of the image
    ///
    /// This function sets the limit for the side length of the image used in text detection.
    pub fn limit_side_len(mut self, limit_side_len: u32) -> Self {
        self.limit_side_len = Some(limit_side_len);
        self
    }

    /// Sets the type of limit to apply
    ///
    /// This function sets the type of limit (Max or Min) to apply to the image side length
    /// in text detection.
    pub fn limit_type(mut self, limit_type: LimitType) -> Self {
        self.limit_type = Some(limit_type);
        self
    }

    /// Sets the threshold for binarization
    ///
    /// This function sets the threshold value used for binarization in text detection.
    pub fn thresh(mut self, thresh: f32) -> Self {
        self.thresh = Some(thresh);
        self
    }

    /// Sets the threshold for filtering text boxes
    ///
    /// This function sets the threshold value used for filtering text boxes in text detection.
    pub fn box_thresh(mut self, box_thresh: f32) -> Self {
        self.box_thresh = Some(box_thresh);
        self
    }

    /// Sets the ratio for unclipping text boxes
    ///
    /// This function sets the ratio used for unclipping text boxes in text detection.
    pub fn unclip_ratio(mut self, unclip_ratio: f32) -> Self {
        self.unclip_ratio = Some(unclip_ratio);
        self
    }

    /// Sets the input shape for the model
    ///
    /// This function sets the input shape (channels, height, width) for the model.
    pub fn input_shape(mut self, input_shape: (u32, u32, u32)) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    /// Sets the maximum side limit for the image
    ///
    /// This function sets the maximum side limit for the image used in text detection.
    pub fn max_side_limit(mut self, max_side_limit: u32) -> Self {
        self.max_side_limit = Some(max_side_limit);
        self
    }

    /// Builds the `TextDetPredictor`
    ///
    /// This function builds the `TextDetPredictor` with the provided configuration.
    pub fn build(self, model_path: &Path) -> Result<TextDetPredictor, OCRError> {
        self.build_internal(model_path)
    }

    /// Builds the `TextDetPredictor` internally
    ///
    /// This function builds the `TextDetPredictor` with the provided configuration.
    /// It also validates the configuration and handles the model path.
    fn build_internal(mut self, model_path: &Path) -> Result<TextDetPredictor, OCRError> {
        if self.common.model_path.is_none() {
            self.common = self.common.model_path(model_path.to_path_buf());
        }

        let config = TextDetPredictorConfig {
            common: self.common,
            limit_side_len: self.limit_side_len,
            limit_type: self.limit_type,
            thresh: self.thresh,
            box_thresh: self.box_thresh,
            unclip_ratio: self.unclip_ratio,
            input_shape: self.input_shape,
            max_side_limit: self.max_side_limit,
        };
        config.validate().map_err(|e| OCRError::ConfigError {
            message: e.to_string(),
        })?;
        TextDetPredictor::new(config, model_path)
    }
}

impl Default for TextDetPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl_standard_predictor!(
    TextDetPredictor,
    TextDetResult,
    OCRError,
    "TextDetection",
    process_internal(None)
);

impl IntoPrediction for TextDetResult {
    type Out = PredictionResult<'static>;

    fn into_prediction(self) -> Self::Out {
        PredictionResult::Detection {
            input_path: self
                .input_path
                .into_iter()
                .map(|arc| Cow::Owned(arc.to_string()))
                .collect(),
            index: self.index,
            input_img: self.input_img,
            dt_polys: self.dt_polys,
            dt_scores: self.dt_scores,
        }
    }
}

impl_predictor_from_generic!(TextDetPredictor);
