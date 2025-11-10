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

use crate::processors::{BoundingBox, DBPostProcess, DetResizeForTest, LimitType, NormalizeImage};
use image::{DynamicImage, RgbImage};
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use crate::impl_config_new_and_with_common;

use crate::impl_common_builder_methods;

use crate::core::ImageReader as CoreImageReader;
use crate::core::{
    BatchData, CommonBuilderConfig, OCRError, Tensor4D,
    config::{ConfigValidator, ConfigValidatorExt},
    constants::{DEFAULT_BATCH_SIZE, DEFAULT_MAX_SIDE_LIMIT},
};
use crate::core::{DefaultImageReader, OrtInfer};
use crate::core::{
    GranularImageReader as GIReader, InferenceEngine as GInferenceEngine, ModularPredictor,
    Postprocessor as GPostprocessor, Preprocessor as GPreprocessor,
};

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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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

impl_config_new_and_with_common!(
    TextDetPredictorConfig,
    common_defaults: (None, Some(DEFAULT_BATCH_SIZE)),
    fields: {
        limit_side_len: None,
        limit_type: None,
        thresh: None,
        box_thresh: None,
        unclip_ratio: None,
        input_shape: None,
        max_side_limit: Some(DEFAULT_MAX_SIDE_LIMIT)
    }
);

impl TextDetPredictorConfig {
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

        if let Some((c, h, w)) = self.input_shape
            && (c == 0 || h == 0 || w == 0)
        {
            return Err(crate::core::ConfigError::InvalidConfig {
                message: format!(
                    "Input shape dimensions must be greater than 0, got ({c}, {h}, {w})"
                ),
            });
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

impl fmt::Display for TextDetResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ((path, polys), scores)) in self
            .input_path
            .iter()
            .zip(self.dt_polys.iter())
            .zip(self.dt_scores.iter())
            .enumerate()
        {
            writeln!(f, "Image {} of {}: {}", i + 1, self.input_path.len(), path)?;
            writeln!(f, "  Total regions: {}", polys.len())?;

            if !polys.is_empty() {
                writeln!(f, "  Detection polygons:")?;
                for (j, (bbox, &score)) in polys.iter().zip(scores.iter()).enumerate() {
                    if bbox.points.is_empty() {
                        writeln!(f, "    Region {j}: [] (empty, score: {score:.3})")?;
                        continue;
                    }

                    write!(f, "    Region {j}: [")?;
                    for (k, point) in bbox.points.iter().enumerate() {
                        if k == 0 {
                            write!(f, "[{:.0}, {:.0}]", point.x, point.y)?;
                        } else {
                            write!(f, ", [{:.0}, {:.0}]", point.x, point.y)?;
                        }
                    }
                    writeln!(f, "] (score: {score:.3})")?;
                }
            }

            if i < self.input_path.len() - 1 {
                writeln!(f)?;
            }
        }

        Ok(())
    }
}

impl Default for TextDetResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Text detection predictor built from modular components
///
/// This is a type alias over `ModularPredictor` with concrete, composable components
/// to eliminate duplicated StandardPredictor implementations across predictors.
pub type TextDetPredictor =
    ModularPredictor<TDImageReader, TDPreprocessor, TDOrtInfer, TDPostprocessor>;

#[derive(Debug)]
pub struct TDImageReader {
    inner: DefaultImageReader,
}
impl TDImageReader {
    pub fn new() -> Self {
        Self {
            inner: DefaultImageReader::new(),
        }
    }
}
impl Default for TDImageReader {
    fn default() -> Self {
        Self::new()
    }
}
impl GIReader for TDImageReader {
    fn read_images<'a>(
        &self,
        paths: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<RgbImage>, OCRError> {
        self.inner.apply(paths)
    }
}

#[derive(Debug)]
pub struct TDPreprocessor {
    pub resize: DetResizeForTest,
    pub normalize: NormalizeImage,
    // Store default configuration values for merging with runtime config
    pub default_config: TextDetConfig,
}
#[derive(Debug)]
pub struct TextDetPreprocessOutput {
    pub tensor: Tensor4D,
    pub shapes: Vec<[f32; 4]>,
}
impl GPreprocessor for TDPreprocessor {
    type Config = TextDetConfig;
    type Output = TextDetPreprocessOutput;
    fn preprocess(
        &self,
        images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Output> {
        // Merge runtime config with stored defaults
        let merged = match config {
            Some(runtime_config) => TextDetConfig {
                limit_side_len: runtime_config
                    .limit_side_len
                    .or(self.default_config.limit_side_len),
                limit_type: runtime_config
                    .limit_type
                    .clone()
                    .or(self.default_config.limit_type.clone()),
                thresh: runtime_config.thresh.or(self.default_config.thresh),
                box_thresh: runtime_config.box_thresh.or(self.default_config.box_thresh),
                unclip_ratio: runtime_config
                    .unclip_ratio
                    .or(self.default_config.unclip_ratio),
                max_side_limit: runtime_config
                    .max_side_limit
                    .or(self.default_config.max_side_limit),
            },
            None => self.default_config.clone(),
        };

        let limit_side_len = merged
            .limit_side_len
            .unwrap_or(self.resize.limit_side_len.unwrap_or(960));
        let limit_type = merged
            .limit_type
            .unwrap_or(self.resize.limit_type.clone().unwrap_or(LimitType::Min));
        let max_side_limit = merged.max_side_limit.unwrap_or(self.resize.max_side_limit);
        let batch_imgs: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();
        let (resized_imgs, shapes) = self.resize.apply(
            batch_imgs,
            Some(limit_side_len),
            Some(limit_type.clone()),
            Some(max_side_limit),
        );
        let tensor = self
            .normalize
            .normalize_batch_to(resized_imgs)
            .map_err(|e| {
                OCRError::model_inference_error(
                    "TextDetection",
                    "preprocessing_normalization",
                    0,
                    &[shapes.len()],
                    "Normalization failed in TDPreprocessor",
                    e,
                )
            })?;
        Ok(TextDetPreprocessOutput { tensor, shapes })
    }
}

#[derive(Debug)]
pub struct TDOrtInfer(pub OrtInfer);
impl GInferenceEngine for TDOrtInfer {
    type Input = TextDetPreprocessOutput;
    type Output = Tensor4D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        // Performance improvement: Pass reference instead of cloning the tensor
        self.0.infer_4d(&input.tensor)
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-4D".to_string()
    }
}

#[derive(Debug)]
pub struct TDPostprocessor {
    pub op: DBPostProcess,
    // Store default configuration values for merging with runtime config
    pub default_config: TextDetConfig,
}
impl GPostprocessor for TDPostprocessor {
    type Config = TextDetConfig;
    type InferenceOutput = Tensor4D;
    type PreprocessOutput = TextDetPreprocessOutput;
    type Result = TextDetResult;
    fn postprocess(
        &self,
        output: Self::InferenceOutput,
        pre: Option<&Self::PreprocessOutput>,
        batch_data: &BatchData,
        raw_images: Vec<RgbImage>,
        config: Option<&Self::Config>,
    ) -> crate::core::OcrResult<Self::Result> {
        // Merge runtime config with stored defaults
        let merged = match config {
            Some(runtime_config) => TextDetConfig {
                limit_side_len: runtime_config
                    .limit_side_len
                    .or(self.default_config.limit_side_len),
                limit_type: runtime_config
                    .limit_type
                    .clone()
                    .or(self.default_config.limit_type.clone()),
                thresh: runtime_config.thresh.or(self.default_config.thresh),
                box_thresh: runtime_config.box_thresh.or(self.default_config.box_thresh),
                unclip_ratio: runtime_config
                    .unclip_ratio
                    .or(self.default_config.unclip_ratio),
                max_side_limit: runtime_config
                    .max_side_limit
                    .or(self.default_config.max_side_limit),
            },
            None => self.default_config.clone(),
        };

        let thresh = merged.thresh.unwrap_or(DEFAULT_THRESH);
        let box_thresh = merged.box_thresh.unwrap_or(DEFAULT_BOX_THRESH);
        let unclip_ratio = merged.unclip_ratio.unwrap_or(DEFAULT_UNCLIP_RATIO);
        let shapes = pre.map(|p| p.shapes.clone()).unwrap_or_default();
        let (polys, scores) = self.op.apply(
            &output,
            shapes,
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
    fn empty_result(&self) -> crate::core::OcrResult<Self::Result> {
        Ok(TextDetResult::new())
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

impl_common_builder_methods!(TextDetPredictorBuilder, common);

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
        let config = config.validate_and_wrap_ocr_error()?;

        // Determine default values based on model name
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

        // Create default configuration for components
        let default_config = TextDetConfig {
            limit_side_len: Some(limit_side_len),
            limit_type: Some(limit_type.clone()),
            thresh: config.thresh,
            box_thresh: config.box_thresh,
            unclip_ratio: config.unclip_ratio,
            max_side_limit: Some(max_side_limit),
        };

        // Build modular components
        let image_reader = TDImageReader::new();
        let resize = DetResizeForTest::new(
            config.input_shape,
            None,
            None,
            Some(limit_side_len),
            Some(limit_type.clone()),
            None,
            Some(max_side_limit),
        );
        let normalize = NormalizeImage::new(None, None, None, None, None)?;
        let preprocessor = TDPreprocessor {
            resize,
            normalize,
            default_config: default_config.clone(),
        };
        let infer = OrtInfer::from_common(&config.common, model_path, None)?;
        let inference_engine = TDOrtInfer(infer);
        let post_op = DBPostProcess::new(None, None, None, None, None, None, None);
        let postprocessor = TDPostprocessor {
            op: post_op,
            default_config,
        };

        Ok(ModularPredictor::new(
            image_reader,
            preprocessor,
            inference_engine,
            postprocessor,
        ))
    }
}

impl Default for TextDetPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests_local {
    use super::*;

    #[test]
    fn test_text_det_config_defaults_and_validate() {
        let config = TextDetPredictorConfig::new();
        // Defaults via get_defaults
        assert_eq!(config.max_side_limit, Some(DEFAULT_MAX_SIDE_LIMIT));
        assert!(config.validate().is_ok());
    }
}
