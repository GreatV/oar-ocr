//! The main OCR pipeline implementation.
//!
//! This module provides the complete OCR pipeline that combines multiple
//! components to perform document orientation classification, text detection,
//! text recognition, and text line classification.
//!
//! The pipeline can be configured to use different models for each step
//! and can be customized with various parameters.

use crate::core::traits::StandardPredictor;
use crate::core::{OCRError, PipelineStats};
use crate::predictor::{
    DocOrientationClassifier, DocOrientationClassifierBuilder, DocOrientationClassifierConfig,
    DoctrRectifierPredictor, DoctrRectifierPredictorBuilder, DoctrRectifierPredictorConfig,
    TextDetPredictor, TextDetPredictorBuilder, TextDetPredictorConfig, TextLineClasPredictor,
    TextLineClasPredictorBuilder, TextLineClasPredictorConfig, TextRecPredictor,
    TextRecPredictorBuilder, TextRecPredictorConfig,
};
use crate::processors::{BoundingBox, LimitType};
use crate::utils::transform::{Point2f, get_rotate_crop_image};
use image::{RgbImage, imageops};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use tracing::{debug, info, warn};

/// Configuration for the OAROCR pipeline.
///
/// This struct holds all the configuration parameters needed to initialize
/// and run the OCR pipeline. It uses a hierarchical structure with nested
/// configuration structs for each component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAROCRConfig {
    /// Configuration for text detection.
    #[serde(default)]
    pub detection: TextDetPredictorConfig,

    /// Configuration for text recognition.
    #[serde(default)]
    pub recognition: TextRecPredictorConfig,

    /// Configuration for document orientation classification (optional).
    #[serde(default)]
    pub orientation: Option<DocOrientationClassifierConfig>,

    /// Configuration for document rectification/unwarping (optional).
    #[serde(default)]
    pub rectification: Option<DoctrRectifierPredictorConfig>,

    /// Configuration for text line orientation classification (optional).
    #[serde(default)]
    pub text_line_orientation: Option<TextLineClasPredictorConfig>,

    /// Path to the character dictionary file for text recognition.
    pub character_dict_path: PathBuf,

    /// Whether to use document orientation classification.
    #[serde(default)]
    pub use_doc_orientation_classify: bool,

    /// Whether to use document unwarping.
    #[serde(default)]
    pub use_doc_unwarping: bool,

    /// Whether to use text line orientation classification.
    #[serde(default)]
    pub use_textline_orientation: bool,

    /// Maximum number of threads to use for parallel processing.
    /// If None, rayon will use the default thread pool size (typically number of CPU cores).
    /// Default: None (use rayon's default)
    #[serde(default)]
    pub max_parallel_threads: Option<usize>,
}

impl OAROCRConfig {
    /// Creates a new OAROCRConfig with the required parameters.
    ///
    /// This constructor initializes the configuration with default values
    /// for optional parameters while requiring the essential model paths.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model_path` - Path to the text detection model file
    /// * `text_recognition_model_path` - Path to the text recognition model file
    /// * `character_dict_path` - Path to the character dictionary file
    ///
    /// # Returns
    ///
    /// A new OAROCRConfig instance with default values
    pub fn new(
        text_detection_model_path: impl Into<PathBuf>,
        text_recognition_model_path: impl Into<PathBuf>,
        character_dict_path: impl Into<PathBuf>,
    ) -> Self {
        let mut detection_config = TextDetPredictorConfig::new();
        detection_config.common.model_path = Some(text_detection_model_path.into());
        detection_config.common.batch_size = Some(1);
        detection_config.limit_side_len = Some(736);
        detection_config.limit_type = Some(LimitType::Max);

        let mut recognition_config = TextRecPredictorConfig::new();
        recognition_config.common.model_path = Some(text_recognition_model_path.into());
        recognition_config.common.batch_size = Some(1);

        Self {
            detection: detection_config,
            recognition: recognition_config,
            orientation: None,
            rectification: None,
            text_line_orientation: None,
            character_dict_path: character_dict_path.into(),
            use_doc_orientation_classify: false,
            use_doc_unwarping: false,
            use_textline_orientation: false,
            max_parallel_threads: None,
        }
    }
}

/// Implementation of Default for OAROCRConfig.
///
/// This provides a default configuration that can be used for testing.
/// Note: This default configuration will not work for actual OCR processing
/// as it doesn't specify valid model paths.
impl Default for OAROCRConfig {
    fn default() -> Self {
        Self::new(
            "default_detection_model.onnx",
            "default_recognition_model.onnx",
            "default_char_dict.txt",
        )
    }
}

/// Result of the OAROCR pipeline execution.
///
/// This struct contains all the results from processing an image through
/// the OCR pipeline, including detected text boxes, recognized text, and
/// any intermediate processing results.
#[derive(Debug, Clone)]
pub struct OAROCRResult {
    /// Path to the input image file.
    pub input_path: Arc<str>,
    /// Index of the image in a batch (0 for single image processing).
    pub index: usize,
    /// The input image.
    pub input_img: Arc<RgbImage>,
    /// Detected text bounding boxes.
    pub text_boxes: Vec<BoundingBox>,
    /// Recognized text for each bounding box.
    pub rec_texts: Vec<Arc<str>>,
    /// Confidence scores for each recognized text.
    pub rec_scores: Vec<f32>,
    /// Document orientation angle (if orientation classification was used).
    pub orientation_angle: Option<f32>,
    /// Text line orientation angles for each text box (if text line orientation classification was used).
    pub text_line_orientation_angles: Vec<Option<f32>>,
    /// Rectified image (if document unwarping was used).
    pub rectified_img: Option<Arc<RgbImage>>,
}

impl fmt::Display for OAROCRResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Input path: {}", self.input_path)?;
        writeln!(f, "Page index: {}", self.index)?;
        writeln!(
            f,
            "Image dimensions: [{}, {}]",
            self.input_img.width(),
            self.input_img.height()
        )?;

        if let Some(angle) = self.orientation_angle {
            writeln!(f, "Orientation angle: {angle:.1}°")?;
        } else {
            writeln!(f, "Orientation angle: not detected")?;
        }

        writeln!(f, "Total text regions: {}", self.text_boxes.len())?;
        writeln!(f, "Recognized texts: {}", self.rec_texts.len())?;

        if !self.text_boxes.is_empty() {
            writeln!(f, "Text regions (detection + recognition):")?;

            // Create a mapping from detection index to recognition data
            // Since recognition results may be filtered, we need to track which
            // detection regions have corresponding recognition results
            let mut rec_index = 0;

            for (det_index, bbox) in self.text_boxes.iter().enumerate() {
                write!(f, "  Region {}: ", det_index + 1)?;

                // Display bounding box
                if bbox.points.is_empty() {
                    write!(f, "[] (empty)")?;
                } else {
                    write!(f, "[")?;
                    for (j, point) in bbox.points.iter().enumerate() {
                        if j == 0 {
                            write!(f, "[{:.0}, {:.0}]", point.x, point.y)?;
                        } else {
                            write!(f, ", [{:.0}, {:.0}]", point.x, point.y)?;
                        }
                    }
                    write!(f, "]")?;
                }

                // Display recognition result if available
                if rec_index < self.rec_texts.len() {
                    let text = &self.rec_texts[rec_index];
                    let score = self.rec_scores[rec_index];
                    let orientation = self
                        .text_line_orientation_angles
                        .get(rec_index)
                        .unwrap_or(&None);

                    let orientation_str = match orientation {
                        Some(angle) => format!(" (orientation: {angle:.1}°)"),
                        None => String::new(),
                    };

                    writeln!(f, " -> '{text}' (confidence: {score:.3}){orientation_str}")?;
                    rec_index += 1;
                } else {
                    writeln!(f, " -> [no text recognized]")?;
                }
            }
        }

        if let Some(rectified_img) = &self.rectified_img {
            writeln!(
                f,
                "Rectified image: available [{} x {}]",
                rectified_img.width(),
                rectified_img.height()
            )?;
        } else {
            writeln!(
                f,
                "Rectified image: not available (document unwarping not enabled)"
            )?;
        }

        Ok(())
    }
}

/// Builder for creating OAROCR instances.
///
/// This struct provides a fluent API for configuring and building
/// OAROCR pipeline instances with various options.
#[derive(Debug)]
pub struct OAROCRBuilder {
    config: OAROCRConfig,
}

impl OAROCRBuilder {
    /// Creates a new OAROCRBuilder with the required parameters.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model_path` - Path to the text detection model file
    /// * `text_recognition_model_path` - Path to the text recognition model file
    /// * `text_rec_character_dict_path` - Path to the character dictionary file
    ///
    /// # Returns
    ///
    /// A new OAROCRBuilder instance
    pub fn new(
        text_detection_model_path: String,
        text_recognition_model_path: String,
        text_rec_character_dict_path: String,
    ) -> Self {
        Self {
            config: OAROCRConfig::new(
                text_detection_model_path,
                text_recognition_model_path,
                text_rec_character_dict_path,
            ),
        }
    }

    /// Creates a new OAROCRBuilder from an existing configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The OAROCRConfig to use
    ///
    /// # Returns
    ///
    /// A new OAROCRBuilder instance
    pub fn from_config(config: OAROCRConfig) -> Self {
        Self { config }
    }

    /// Sets the document orientation classification model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_orientation_classify_model_name(mut self, name: String) -> Self {
        if self.config.orientation.is_none() {
            self.config.orientation = Some(DocOrientationClassifierConfig::new());
        }
        if let Some(ref mut config) = self.config.orientation {
            config.common.model_name = Some(name);
        }
        self
    }

    /// Sets the document orientation classification model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_orientation_classify_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        if self.config.orientation.is_none() {
            self.config.orientation = Some(DocOrientationClassifierConfig::new());
        }
        if let Some(ref mut config) = self.config.orientation {
            config.common.model_path = Some(path.into());
        }
        self
    }

    /// Sets the document unwarping model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_unwarping_model_name(mut self, name: String) -> Self {
        if self.config.rectification.is_none() {
            self.config.rectification = Some(DoctrRectifierPredictorConfig::new());
        }
        if let Some(ref mut config) = self.config.rectification {
            config.common.model_name = Some(name);
        }
        self
    }

    /// Sets the document unwarping model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_unwarping_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        if self.config.rectification.is_none() {
            self.config.rectification = Some(DoctrRectifierPredictorConfig::new());
        }
        if let Some(ref mut config) = self.config.rectification {
            config.common.model_path = Some(path.into());
        }
        self
    }

    /// Sets the text detection model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_model_name(mut self, name: String) -> Self {
        self.config.detection.common.model_name = Some(name);
        self
    }

    /// Sets the text detection model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.detection.common.model_path = Some(path.into());
        self
    }

    /// Sets the text detection batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for inference
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_batch_size(mut self, batch_size: usize) -> Self {
        self.config.detection.common.batch_size = Some(batch_size);
        self
    }

    /// Sets the text recognition model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_model_name(mut self, name: String) -> Self {
        self.config.recognition.common.model_name = Some(name);
        self
    }

    /// Sets the text recognition model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.recognition.common.model_path = Some(path.into());
        self
    }

    /// Sets the text recognition batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for inference
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_batch_size(mut self, batch_size: usize) -> Self {
        self.config.recognition.common.batch_size = Some(batch_size);
        self
    }

    /// Sets the text line orientation classification model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_model_name(mut self, name: String) -> Self {
        if self.config.text_line_orientation.is_none() {
            self.config.text_line_orientation = Some(TextLineClasPredictorConfig::new());
        }
        if let Some(ref mut config) = self.config.text_line_orientation {
            config.common.model_name = Some(name);
        }
        self
    }

    /// Sets the text line orientation classification model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        if self.config.text_line_orientation.is_none() {
            self.config.text_line_orientation = Some(TextLineClasPredictorConfig::new());
        }
        if let Some(ref mut config) = self.config.text_line_orientation {
            config.common.model_path = Some(path.into());
        }
        self
    }

    /// Sets the text line orientation classification batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for inference
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_batch_size(mut self, batch_size: usize) -> Self {
        if self.config.text_line_orientation.is_none() {
            self.config.text_line_orientation = Some(TextLineClasPredictorConfig::new());
        }
        if let Some(ref mut config) = self.config.text_line_orientation {
            config.common.batch_size = Some(batch_size);
        }
        self
    }

    /// Sets the text line orientation classification input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The input shape as (width, height)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_input_shape(mut self, shape: (u32, u32)) -> Self {
        if self.config.text_line_orientation.is_none() {
            self.config.text_line_orientation = Some(TextLineClasPredictorConfig::new());
        }
        if let Some(ref mut config) = self.config.text_line_orientation {
            config.input_shape = Some(shape);
        }
        self
    }

    /// Sets whether to use document orientation classification.
    ///
    /// # Arguments
    ///
    /// * `use_it` - Whether to use document orientation classification
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn use_doc_orientation_classify(mut self, use_it: bool) -> Self {
        self.config.use_doc_orientation_classify = use_it;
        self
    }

    /// Sets whether to use document unwarping.
    ///
    /// # Arguments
    ///
    /// * `use_it` - Whether to use document unwarping
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn use_doc_unwarping(mut self, use_it: bool) -> Self {
        self.config.use_doc_unwarping = use_it;
        self
    }

    /// Sets whether to use text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `use_it` - Whether to use text line orientation classification
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn use_textline_orientation(mut self, use_it: bool) -> Self {
        self.config.use_textline_orientation = use_it;
        self
    }

    /// Sets the maximum number of threads to use for parallel processing.
    ///
    /// If not set or set to None, rayon will use the default thread pool size
    /// (typically the number of CPU cores). This setting only applies when
    /// parallel processing is enabled.
    ///
    /// # Arguments
    ///
    /// * `max_threads` - Maximum number of threads to use
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn max_parallel_threads(mut self, max_threads: Option<usize>) -> Self {
        self.config.max_parallel_threads = max_threads;
        self
    }

    /// Sets the text detection limit side length.
    ///
    /// # Arguments
    ///
    /// * `limit` - The maximum side length for resizing
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_limit_side_len(mut self, limit: u32) -> Self {
        self.config.detection.limit_side_len = Some(limit);
        self
    }

    /// Sets the text detection limit type.
    ///
    /// # Arguments
    ///
    /// * `limit_type` - The type of limit for resizing
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_limit_type(mut self, limit_type: LimitType) -> Self {
        self.config.detection.limit_type = Some(limit_type);
        self
    }

    /// Sets the text detection input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The input shape as (channels, height, width)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_input_shape(mut self, shape: (u32, u32, u32)) -> Self {
        self.config.detection.input_shape = Some(shape);
        self
    }

    /// Sets the text recognition score threshold.
    ///
    /// # Arguments
    ///
    /// * `thresh` - The minimum score threshold for recognition results
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_score_thresh(mut self, thresh: f32) -> Self {
        self.config.recognition.score_thresh = Some(thresh);
        self
    }

    /// Sets the text recognition model input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The model input shape as (channels, height, width)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_model_input_shape(mut self, shape: (u32, u32, u32)) -> Self {
        self.config.recognition.model_input_shape =
            Some([shape.0 as usize, shape.1 as usize, shape.2 as usize]);
        self
    }

    /// Sets the text recognition character dictionary path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the character dictionary file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_character_dict_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.character_dict_path = path.into();
        self
    }

    /// Builds the OAROCR instance with the configured parameters.
    ///
    /// # Returns
    ///
    /// A Result containing the OAROCR instance or an OCRError
    pub fn build(self) -> Result<OAROCR, OCRError> {
        OAROCR::new(self.config)
    }
}

/// The main OCR pipeline that combines multiple components to perform
/// document processing and text recognition.
///
/// This struct manages the complete OCR pipeline, including document
/// orientation classification, text detection, text recognition, and
/// text line classification. It initializes and coordinates all the
/// required components based on the provided configuration.
pub struct OAROCR {
    /// Configuration for the OCR pipeline.
    config: OAROCRConfig,
    /// Document orientation classifier (optional).
    doc_orientation_classifier: Option<DocOrientationClassifier>,
    /// Document rectifier for unwarping (optional).
    doc_rectifier: Option<DoctrRectifierPredictor>,
    /// Text detector for finding text regions.
    text_detector: Option<TextDetPredictor>,
    /// Text line classifier for orientation (optional).
    text_line_classifier: Option<TextLineClasPredictor>,
    /// Text recognizer for reading text content.
    text_recognizer: Option<TextRecPredictor>,
    /// Statistics for the pipeline execution (thread-safe).
    stats: Mutex<PipelineStats>,
}

impl OAROCR {
    /// Creates a new OAROCR instance with the provided configuration.
    ///
    /// This method initializes all the required components based on the
    /// configuration and builds the complete OCR pipeline.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the OCR pipeline
    ///
    /// # Returns
    ///
    /// A Result containing the OAROCR instance or an OCRError
    pub fn new(config: OAROCRConfig) -> Result<Self, OCRError> {
        info!("Initializing OAROCR pipeline with config: {:?}", config);

        let mut pipeline = Self {
            config: config.clone(),
            doc_orientation_classifier: None,
            doc_rectifier: None,
            text_detector: None,
            text_line_classifier: None,
            text_recognizer: None,
            stats: Mutex::new(PipelineStats::default()),
        };

        pipeline.initialize_components()?;

        info!("OAROCR pipeline initialized successfully");
        Ok(pipeline)
    }

    /// Initializes all the components required for the OCR pipeline.
    ///
    /// This method creates and configures all the predictors based on
    /// the pipeline configuration. Components are only initialized if
    /// they are enabled in the configuration.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError
    fn initialize_components(&mut self) -> Result<(), OCRError> {
        // Initialize document orientation classifier if enabled
        if self.config.use_doc_orientation_classify {
            info!("Initializing document orientation classifier");
            self.doc_orientation_classifier = Some(self.build_doc_orientation_classifier()?);
        }

        // Initialize document rectifier if enabled
        if self.config.use_doc_unwarping {
            info!("Initializing document rectifier");
            self.doc_rectifier = Some(self.build_doc_rectifier()?);
        }

        // Text detector is always required
        info!("Initializing text detector");
        self.text_detector = Some(self.build_text_detector()?);

        // Initialize text line classifier if enabled
        if self.config.use_textline_orientation {
            info!("Initializing text line classifier");
            self.text_line_classifier = Some(self.build_text_line_classifier()?);
        }

        // Text recognizer is always required
        info!("Initializing text recognizer");
        self.text_recognizer = Some(self.build_text_recognizer()?);

        Ok(())
    }

    /// Builds the document orientation classifier.
    ///
    /// This method creates a document orientation classifier using the
    /// configured model path and batch size. It uses the builder pattern
    /// to construct the classifier with the appropriate settings.
    ///
    /// # Returns
    ///
    /// A Result containing the DocOrientationClassifier or an OCRError
    fn build_doc_orientation_classifier(&self) -> Result<DocOrientationClassifier, OCRError> {
        let config = self
            .config
            .orientation
            .as_ref()
            .ok_or_else(|| OCRError::ConfigError {
                message: "Document orientation classification config not specified".to_string(),
            })?;

        // Get the model path from configuration, returning an error if not specified
        let model_path =
            config
                .common
                .model_path
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Document orientation classification model path not specified"
                        .to_string(),
                })?;

        // Create a new builder for the document orientation classifier
        let mut builder = DocOrientationClassifierBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = config.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Set model name if specified
        if let Some(ref name) = config.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Set input shape if specified
        if let Some(shape) = config.input_shape {
            builder = builder.input_shape(shape);
        }

        // Build and return the classifier
        builder.build(model_path)
    }

    /// Builds the document rectifier.
    ///
    /// # Returns
    ///
    /// A Result containing the DoctrRectifierPredictor or an OCRError
    fn build_doc_rectifier(&self) -> Result<DoctrRectifierPredictor, OCRError> {
        let config = self
            .config
            .rectification
            .as_ref()
            .ok_or_else(|| OCRError::ConfigError {
                message: "Document rectification config not specified".to_string(),
            })?;

        let model_path =
            config
                .common
                .model_path
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Document rectifier model path not specified".to_string(),
                })?;

        let mut builder = DoctrRectifierPredictorBuilder::new();

        if let Some(batch_size) = config.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        if let Some(ref name) = config.common.model_name {
            builder = builder.model_name(name.clone());
        }

        builder.build(model_path)
    }

    /// Builds the text detector.
    ///
    /// This method creates a text detector using the configured model path
    /// and various detection parameters. It uses the builder pattern to
    /// construct the detector with the appropriate settings.
    ///
    /// # Returns
    ///
    /// A Result containing the TextDetPredictor or an OCRError
    fn build_text_detector(&self) -> Result<TextDetPredictor, OCRError> {
        // Get the model path from configuration
        let model_path = self
            .config
            .detection
            .common
            .model_path
            .as_ref()
            .ok_or_else(|| OCRError::ConfigError {
                message: "Text detection model path not specified".to_string(),
            })?;

        // Create a new builder for the text detector
        let mut builder = TextDetPredictorBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = self.config.detection.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Set model name if specified
        if let Some(ref name) = self.config.detection.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Configure the limit side length if specified in the configuration
        if let Some(limit_side_len) = self.config.detection.limit_side_len {
            builder = builder.limit_side_len(limit_side_len);
        }

        // Configure the limit type if specified in the configuration
        if let Some(limit_type) = &self.config.detection.limit_type {
            builder = builder.limit_type(limit_type.clone());
        }

        // Configure input shape if specified
        if let Some(shape) = self.config.detection.input_shape {
            builder = builder.input_shape(shape);
        }

        // Build and return the text detector
        builder.build(model_path)
    }

    /// Builds the text line classifier.
    ///
    /// # Returns
    ///
    /// A Result containing the TextLineClasPredictor or an OCRError
    fn build_text_line_classifier(&self) -> Result<TextLineClasPredictor, OCRError> {
        let config =
            self.config
                .text_line_orientation
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Text line orientation config not specified".to_string(),
                })?;

        let model_path =
            config
                .common
                .model_path
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Text line classifier model path not specified".to_string(),
                })?;

        let mut builder = TextLineClasPredictorBuilder::new();

        if let Some(batch_size) = config.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        if let Some(ref model_name) = config.common.model_name {
            builder = builder.model_name(model_name.clone());
        }

        if let Some(input_shape) = config.input_shape {
            builder = builder.input_shape(input_shape);
        }

        builder.build(model_path)
    }

    /// Builds the text recognizer.
    ///
    /// This method creates a text recognizer using the configured model path,
    /// input shape, and character dictionary. It uses the builder pattern to
    /// construct the recognizer with the appropriate settings.
    ///
    /// # Returns
    ///
    /// A Result containing the TextRecPredictor or an OCRError
    fn build_text_recognizer(&self) -> Result<TextRecPredictor, OCRError> {
        // Get the model path from configuration
        let model_path = self
            .config
            .recognition
            .common
            .model_path
            .as_ref()
            .ok_or_else(|| OCRError::ConfigError {
                message: "Text recognition model path not specified".to_string(),
            })?;

        // Create a new builder for the text recognizer
        let mut builder = TextRecPredictorBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = self.config.recognition.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Set model name if specified
        if let Some(ref name) = self.config.recognition.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Configure the model input shape if specified in the configuration
        if let Some(shape) = self.config.recognition.model_input_shape {
            builder = builder.model_input_shape(shape);
        }

        // Load the character dictionary and configure it in the builder
        let character_dict =
            self.load_character_dict(self.config.character_dict_path.to_str().ok_or_else(
                || OCRError::ConfigError {
                    message: "Invalid character dictionary path".to_string(),
                },
            )?)?;
        builder = builder.character_dict(character_dict);

        // Configure the score threshold if specified in the configuration
        if let Some(score_thresh) = self.config.recognition.score_thresh {
            builder = builder.score_thresh(score_thresh);
        }

        // Build and return the text recognizer
        builder.build(model_path)
    }

    /// Loads the character dictionary from a file.
    ///
    /// This function reads a text file containing characters (one per line)
    /// and returns them as a vector of strings. This dictionary is used
    /// by the text recognizer to map model outputs to actual characters.
    ///
    /// # Arguments
    ///
    /// * `dict_path` - The path to the character dictionary file
    ///
    /// # Returns
    ///
    /// A Result containing the character dictionary as a Vec<String> or an OCRError
    fn load_character_dict(&self, dict_path: &str) -> Result<Vec<String>, OCRError> {
        // Read the entire dictionary file into a string
        let content = std::fs::read_to_string(dict_path).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to load character dictionary from {dict_path}: {e}"),
        })?;

        // Split the content into lines and collect them into a vector
        Ok(content.lines().map(|line| line.to_string()).collect())
    }

    /// Applies document orientation rotation to an image.
    ///
    /// This function rotates an image based on the detected orientation angle.
    /// It supports rotation by 0°, 90°, 180°, and 270° degrees.
    ///
    /// # Arguments
    ///
    /// * `image` - The input image to rotate
    /// * `angle` - The rotation angle in degrees (0, 90, 180, or 270)
    ///
    /// # Returns
    ///
    /// The rotated image
    fn apply_document_orientation(&self, image: RgbImage, angle: f32) -> RgbImage {
        match angle as i32 {
            0 => image,
            90 => imageops::rotate90(&image),
            180 => imageops::rotate180(&image),
            270 => imageops::rotate270(&image),
            _ => {
                warn!(
                    "Unsupported rotation angle: {:.1}°, using original image",
                    angle
                );
                image
            }
        }
    }

    /// Crops an image based on a bounding box.
    ///
    /// This function calculates the bounding rectangle of a polygonal bounding box
    /// and crops the image to that region. It handles edge cases like empty bounding
    /// boxes and ensures the crop region is within the image boundaries.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image
    /// * `bbox` - The bounding box defining the crop region
    ///
    /// # Returns
    ///
    /// A Result containing the cropped image or an OCRError
    fn crop_bounding_box(
        &self,
        image: &RgbImage,
        bbox: &BoundingBox,
    ) -> Result<RgbImage, OCRError> {
        // Check if the bounding box is empty
        if bbox.points.is_empty() {
            return Err(OCRError::ConfigError {
                message: "Empty bounding box".to_string(),
            });
        }

        // Calculate the bounding rectangle of the polygon
        let min_x = bbox
            .points
            .iter()
            .map(|p| p.x)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let max_x = bbox
            .points
            .iter()
            .map(|p| p.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = bbox
            .points
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let max_y = bbox
            .points
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);

        // Convert to integer coordinates, ensuring they're within image bounds
        let x1 = (min_x as u32).min(image.width().saturating_sub(1));
        let y1 = (min_y as u32).min(image.height().saturating_sub(1));
        let x2 = (max_x as u32).min(image.width());
        let y2 = (max_y as u32).min(image.height());

        // Validate the crop region
        if x2 <= x1 || y2 <= y1 {
            return Err(OCRError::ConfigError {
                message: format!("Invalid crop region: ({x1}, {y1}) to ({x2}, {y2})"),
            });
        }

        let coords = (x1, y1, x2, y2);
        Ok(self.slice_rgb_image(image, coords))
    }

    /// Slices an RGB image based on coordinates.
    ///
    /// This function creates a new image by copying pixels from a rectangular
    /// region of the source image. It performs bounds checking to ensure
    /// that only valid pixels are copied.
    ///
    /// # Arguments
    ///
    /// * `img` - The source image
    /// * `coords` - The coordinates as (x1, y1, x2, y2)
    ///
    /// # Returns
    ///
    /// The sliced image
    fn slice_rgb_image(&self, img: &RgbImage, coords: (u32, u32, u32, u32)) -> RgbImage {
        let (x1, y1, x2, y2) = coords;
        let width = x2 - x1;
        let height = y2 - y1;

        // Create a new image with the cropped dimensions
        let mut cropped = RgbImage::new(width, height);

        // Copy pixels from the source image to the cropped image
        for y in 0..height {
            for x in 0..width {
                let src_x = x1 + x;
                let src_y = y1 + y;
                // Ensure we don't read beyond the source image boundaries
                if src_x < img.width() && src_y < img.height() {
                    let pixel = img.get_pixel(src_x, src_y);
                    cropped.put_pixel(x, y, *pixel);
                }
            }
        }

        cropped
    }

    /// Crops and rectifies an image region using rotated crop with perspective transformation.
    ///
    /// This function implements the same functionality as OpenCV's GetRotateCropImage.
    /// It takes a bounding box (quadrilateral) and applies perspective transformation
    /// to rectify it into a rectangular image. This is particularly useful for text
    /// regions that may be rotated or have perspective distortion.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image
    /// * `bbox` - The bounding box defining the quadrilateral region
    ///
    /// # Returns
    ///
    /// A Result containing the rotated and cropped image or an OCRError
    fn crop_rotated_bounding_box(
        &self,
        image: &RgbImage,
        bbox: &BoundingBox,
    ) -> Result<RgbImage, OCRError> {
        // Check if the bounding box has exactly 4 points
        if bbox.points.len() != 4 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Bounding box must have exactly 4 points, got {}",
                    bbox.points.len()
                ),
            });
        }

        // Convert BoundingBox points to Point2f
        let box_points: Vec<Point2f> = bbox.points.iter().map(|p| Point2f::new(p.x, p.y)).collect();

        // Apply rotated crop transformation
        get_rotate_crop_image(image, &box_points)
    }

    /// Processes one or more images through the OCR pipeline.
    ///
    /// This method runs the complete OCR pipeline on either a single image or
    /// a batch of images, including document orientation classification, text detection,
    /// text recognition, and text line classification (if enabled).
    ///
    /// Multiple images are processed in parallel using rayon for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - A slice of paths to the image files
    ///
    /// # Returns
    ///
    /// A Result containing a vector of OAROCRResult or an OCRError
    pub fn predict(&self, image_paths: &[&Path]) -> Result<Vec<OAROCRResult>, OCRError> {
        let start_time = std::time::Instant::now();

        info!(
            "Starting OCR pipeline for {} image(s) using parallel processing",
            image_paths.len()
        );

        // Configure rayon thread pool if max_threads is specified
        if let Some(max_threads) = self.config.max_parallel_threads {
            debug!("Configuring rayon thread pool with {} threads", max_threads);
            rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build_global()
                .map_err(|e| OCRError::ConfigError {
                    message: format!("Failed to configure thread pool: {e}"),
                })?;
        }

        let result = self.procss_images(image_paths);

        // Update statistics based on the result
        let processing_time = start_time.elapsed();
        let total_time_ms = processing_time.as_millis() as f64;

        match &result {
            Ok(results) => {
                // All images processed successfully
                self.update_stats(image_paths.len(), results.len(), 0, total_time_ms);
            }
            Err(_) => {
                // Processing failed - count as all failed
                self.update_stats(image_paths.len(), 0, image_paths.len(), total_time_ms);
            }
        }

        result
    }

    fn process_single_image(
        &self,
        image_path: &Path,
        index: usize,
    ) -> Result<OAROCRResult, OCRError> {
        let input_img = crate::utils::load_image(image_path)?;
        let input_img_arc = Arc::new(input_img.clone());

        // Stage 1: Document orientation classification
        // Note: We process orientation first since text detection may benefit from corrected orientation
        let (orientation_angle, mut current_img) =
            if let Some(ref classifier) = self.doc_orientation_classifier {
                let result = classifier.predict(vec![input_img.clone()], None)?;
                let angle = if let (Some(labels), Some(scores)) =
                    (result.label_names.first(), result.scores.first())
                {
                    if let (Some(label), Some(&_score)) = (labels.first(), scores.first()) {
                        match label.as_ref() {
                            "0" => 0.0,
                            "90" => 90.0,
                            "180" => 180.0,
                            "270" => 270.0,
                            _ => {
                                warn!("Unknown orientation label: {}", label);
                                0.0
                            }
                        }
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let corrected_img = if angle != 0.0 {
                    self.apply_document_orientation(input_img.clone(), angle)
                } else {
                    input_img.clone()
                };

                (Some(angle), corrected_img)
            } else {
                (None, input_img.clone())
            };

        // Stage 2: Document rectification (depends on orientation correction)
        let rectified_img = if let Some(ref rectifier) = self.doc_rectifier {
            let result = rectifier.predict(vec![current_img.clone()], None)?;
            if let Some(rectified) = result.rectified_img.first() {
                current_img = (**rectified).clone();
                Some(rectified.clone())
            } else {
                Some(Arc::new(current_img.clone()))
            }
        } else {
            None
        };

        // Stage 3: Text detection (on the processed image)
        let text_boxes: Vec<crate::processors::BoundingBox> =
            if let Some(ref detector) = self.text_detector {
                let result = detector.predict(vec![current_img.clone()], None)?;
                result.dt_polys.into_iter().flatten().collect()
            } else {
                return Err(OCRError::ConfigError {
                    message: "Text detector not initialized".to_string(),
                });
            };

        // Stage 4: Text box cropping (can be parallelized)
        let cropped_images: Vec<Option<RgbImage>> = text_boxes
            .par_iter()
            .map(|bbox| {
                let crop_result = if bbox.points.len() == 4 {
                    self.crop_rotated_bounding_box(&current_img, bbox)
                } else {
                    self.crop_bounding_box(&current_img, bbox)
                };
                crop_result.ok()
            })
            .collect();

        // Stage 5: Text line orientation classification
        let mut text_line_orientations: Vec<Option<f32>> = Vec::new();
        if self.config.use_textline_orientation && !text_boxes.is_empty() {
            if let Some(ref classifier) = self.text_line_classifier {
                let valid_images: Vec<RgbImage> = cropped_images
                    .iter()
                    .filter_map(|o| o.as_ref().cloned())
                    .collect();
                if !valid_images.is_empty() {
                    match classifier.predict(valid_images, None) {
                        Ok(result) => {
                            let mut result_idx = 0usize;
                            for cropped_img_opt in &cropped_images {
                                if cropped_img_opt.is_some() {
                                    if let (Some(labels), Some(score_list)) = (
                                        result.label_names.get(result_idx),
                                        result.scores.get(result_idx),
                                    ) {
                                        if let (Some(label), Some(_score)) =
                                            (labels.first(), score_list.first())
                                        {
                                            let angle = match label.as_ref() {
                                                "0" => Some(0.0),
                                                "180" => Some(180.0),
                                                _ => None,
                                            };
                                            text_line_orientations.push(angle);
                                        } else {
                                            text_line_orientations.push(None);
                                        }
                                    } else {
                                        text_line_orientations.push(None);
                                    }
                                    result_idx += 1;
                                } else {
                                    text_line_orientations.push(None);
                                }
                            }
                        }
                        Err(_) => {
                            text_line_orientations.resize(text_boxes.len(), None);
                        }
                    }
                } else {
                    text_line_orientations.resize(text_boxes.len(), None);
                }
            } else {
                text_line_orientations.resize(text_boxes.len(), None);
            }
        } else {
            text_line_orientations.resize(text_boxes.len(), None);
        }

        // Stage 6: Text recognition (with dimension-based batching)
        let (rec_texts, rec_scores) = if text_boxes.is_empty() {
            (Vec::new(), Vec::new())
        } else if let Some(ref recognizer) = self.text_recognizer {
            let mut dimension_groups: std::collections::HashMap<
                (u32, u32),
                Vec<(usize, RgbImage)>,
            > = std::collections::HashMap::new();
            for (i, cropped_img_opt) in cropped_images.iter().enumerate() {
                if let Some(cropped_img) = cropped_img_opt {
                    let dims = (cropped_img.height(), cropped_img.width());
                    dimension_groups
                        .entry(dims)
                        .or_default()
                        .push((i, cropped_img.clone()));
                }
            }
            let mut recognition_results: Vec<(usize, Arc<str>, f32)> = Vec::new();
            for ((_h, _w), group) in dimension_groups {
                let (indices, images): (Vec<usize>, Vec<RgbImage>) = group.into_iter().unzip();
                match recognizer.predict(images, None) {
                    Ok(result) => {
                        for (original_idx, (text, score)) in indices
                            .into_iter()
                            .zip(result.rec_text.iter().zip(result.rec_score.iter()))
                        {
                            recognition_results.push((original_idx, text.clone(), *score));
                        }
                    }
                    Err(_) => {
                        for original_idx in indices {
                            recognition_results.push((original_idx, Arc::from(""), 0.0));
                        }
                    }
                }
            }
            recognition_results.sort_by_key(|(idx, _, _)| *idx);
            let mut texts = Vec::new();
            let mut scores = Vec::new();
            for (_, text, score) in recognition_results {
                texts.push(text);
                scores.push(score);
            }
            (texts, scores)
        } else {
            return Err(OCRError::ConfigError {
                message: "Text recognizer not initialized".to_string(),
            });
        };

        // Stage 7: Final filtering and result assembly
        let score_thresh = self.config.recognition.score_thresh.unwrap_or(0.0);
        let mut final_texts: Vec<Arc<str>> = Vec::new();
        let mut final_scores: Vec<f32> = Vec::new();
        let mut final_orientations: Vec<Option<f32>> = Vec::new();
        for ((text, score), orientation) in rec_texts
            .into_iter()
            .zip(rec_scores)
            .zip(text_line_orientations.iter().cloned())
        {
            if score >= score_thresh {
                final_texts.push(text);
                final_scores.push(score);
                final_orientations.push(orientation);
            }
        }

        Ok(OAROCRResult {
            input_path: Arc::from(image_path.to_string_lossy().as_ref()),
            index,
            input_img: input_img_arc,
            text_boxes,
            rec_texts: final_texts,
            rec_scores: final_scores,
            orientation_angle,
            text_line_orientation_angles: final_orientations,
            rectified_img,
        })
    }

    /// Processes images in parallel through the OCR pipeline (internal implementation).
    ///
    /// This method uses rayon to process multiple images concurrently while
    /// maintaining the original order of results.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - A slice of paths to the image files
    ///
    /// # Returns
    ///
    /// A Result containing a vector of OAROCRResult or an OCRError
    fn procss_images(&self, image_paths: &[&Path]) -> Result<Vec<OAROCRResult>, OCRError> {
        debug!("Processing {} images in parallel", image_paths.len());

        // Process images in parallel using rayon, maintaining order with enumerate
        let results: Result<Vec<_>, OCRError> = image_paths
            .par_iter()
            .enumerate()
            .map(|(index, &image_path)| {
                debug!(
                    "Processing image {} of {}: {:?}",
                    index + 1,
                    image_paths.len(),
                    image_path
                );

                // Process single image using helper
                let mut result = self.process_single_image(image_path, index)?;
                // Ensure index is set correctly
                result.index = index;
                Ok((index, result))
            })
            .collect();

        // Extract results and sort by index to maintain original order
        let mut indexed_results = results?;
        indexed_results.sort_by_key(|(index, _)| *index);
        let final_results: Vec<OAROCRResult> = indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect();

        info!(
            "OCR pipeline completed for {} images in parallel",
            final_results.len()
        );
        Ok(final_results)
    }

    /// Gets the pipeline statistics.
    ///
    /// # Returns
    ///
    /// A copy of the current PipelineStats
    pub fn get_stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }

    /// Updates the pipeline statistics after processing images.
    ///
    /// # Arguments
    ///
    /// * `processed_count` - Number of images processed
    /// * `successful_count` - Number of successful predictions
    /// * `failed_count` - Number of failed predictions
    /// * `inference_time_ms` - Total inference time in milliseconds
    fn update_stats(
        &self,
        processed_count: usize,
        successful_count: usize,
        failed_count: usize,
        inference_time_ms: f64,
    ) {
        let mut stats = self.stats.lock().unwrap();

        // Update counters
        stats.total_processed += processed_count;
        stats.successful_predictions += successful_count;
        stats.failed_predictions += failed_count;

        // Update average inference time using incremental average formula
        if processed_count > 0 {
            let old_total = stats.total_processed - processed_count;
            if old_total == 0 {
                stats.average_inference_time_ms = inference_time_ms / processed_count as f64;
            } else {
                let old_avg = stats.average_inference_time_ms;
                let new_avg_contribution = inference_time_ms / processed_count as f64;
                stats.average_inference_time_ms = (old_avg * old_total as f64
                    + new_avg_contribution * processed_count as f64)
                    / stats.total_processed as f64;
            }
        }
    }

    /// Resets the pipeline statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = PipelineStats::default();
    }

    /// Gets the pipeline configuration.
    ///
    /// # Returns
    ///
    /// A reference to the OAROCRConfig
    pub fn get_config(&self) -> &OAROCRConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oarocr_builder_text_rec_score_thresh() {
        // Test that the text_rec_score_thresh method properly sets the score threshold
        let builder = OAROCRBuilder::new(
            "dummy_det_path".to_string(),
            "dummy_rec_path".to_string(),
            "dummy_dict_path".to_string(),
        )
        .text_rec_score_thresh(0.8);

        assert_eq!(builder.config.recognition.score_thresh, Some(0.8));
    }

    #[test]
    fn test_oarocr_config_default_score_thresh() {
        // Test that the default configuration has no score threshold set
        let config = OAROCRConfig::default();
        assert_eq!(config.recognition.score_thresh, None);
    }
}
