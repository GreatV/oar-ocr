//! The main OCR pipeline implementation.
//!
//! This module provides the complete OCR pipeline that combines multiple
//! components to perform document orientation classification, text detection,
//! text recognition, and text line classification.
//!
//! The pipeline can be configured to use different models for each step
//! and can be customized with various parameters.

use crate::core::{OCRError, PipelineExecutor, PipelineStats, PredictionResult, Predictor};
use crate::predictor::{
    DocOrientationClassifier, DocOrientationClassifierBuilder, DoctrRectifierPredictor,
    DoctrRectifierPredictorBuilder, TextDetPredictor, TextDetPredictorBuilder,
    TextLineClasPredictor, TextLineClasPredictorBuilder, TextRecPredictor, TextRecPredictorBuilder,
};
use crate::processors::{BoundingBox, LimitType};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Configuration for the OAROCR pipeline.
///
/// This struct holds all the configuration parameters needed to initialize
/// and run the OCR pipeline. It includes paths to model files, batch sizes,
/// and various processing parameters.
#[derive(Debug, Clone)]
pub struct OAROCRConfig {
    /// Model name for document orientation classification.
    pub doc_orientation_classify_model_name: Option<String>,
    /// Path to the document orientation classification model file.
    pub doc_orientation_classify_model_path: Option<String>,

    /// Model name for document unwarping/rectification.
    pub doc_unwarping_model_name: Option<String>,
    /// Path to the document unwarping model file.
    pub doc_unwarping_model_path: Option<String>,

    /// Model name for text detection.
    pub text_detection_model_name: Option<String>,
    /// Path to the text detection model file.
    pub text_detection_model_path: String,
    /// Batch size for text detection inference.
    pub text_detection_batch_size: Option<usize>,

    /// Model name for text recognition.
    pub text_recognition_model_name: Option<String>,
    /// Path to the text recognition model file.
    pub text_recognition_model_path: String,
    /// Batch size for text recognition inference.
    pub text_recognition_batch_size: Option<usize>,

    /// Model name for text line orientation classification.
    pub textline_orientation_classify_model_name: Option<String>,
    /// Path to the text line orientation classification model file.
    pub textline_orientation_classify_model_path: Option<String>,
    /// Batch size for text line orientation classification inference.
    pub textline_orientation_classify_batch_size: Option<usize>,

    /// Whether to use document orientation classification.
    pub use_doc_orientation_classify: Option<bool>,
    /// Whether to use document unwarping.
    pub use_doc_unwarping: Option<bool>,
    /// Whether to use text line orientation classification.
    pub use_textline_orientation: Option<bool>,

    /// Maximum side length for text detection resizing.
    pub text_det_limit_side_len: Option<u32>,
    /// Type of limit for text detection resizing.
    pub text_det_limit_type: Option<LimitType>,
    /// Input shape for text detection model.
    pub text_det_input_shape: Option<(u32, u32, u32)>,

    /// Minimum score threshold for text recognition results.
    pub text_rec_score_thresh: Option<f32>,
    /// Input shape for text recognition model.
    pub text_rec_input_shape: Option<(u32, u32, u32)>,
    /// Path to the character dictionary file for text recognition.
    pub text_rec_character_dict_path: String,
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
    /// * `text_rec_character_dict_path` - Path to the character dictionary file
    ///
    /// # Returns
    ///
    /// A new OAROCRConfig instance with default values
    pub fn new(
        text_detection_model_path: String,
        text_recognition_model_path: String,
        text_rec_character_dict_path: String,
    ) -> Self {
        Self {
            doc_orientation_classify_model_name: None,
            doc_orientation_classify_model_path: None,
            doc_unwarping_model_name: None,
            doc_unwarping_model_path: None,
            text_detection_model_name: None,
            text_detection_model_path,
            text_detection_batch_size: Some(1),
            text_recognition_model_name: None,
            text_recognition_model_path,
            text_recognition_batch_size: Some(1),
            textline_orientation_classify_model_name: None,
            textline_orientation_classify_model_path: None,
            textline_orientation_classify_batch_size: Some(1),
            use_doc_orientation_classify: Some(false),
            use_doc_unwarping: Some(false),
            use_textline_orientation: Some(false),
            text_det_limit_side_len: Some(736),
            text_det_limit_type: Some(LimitType::Max),
            text_det_input_shape: None,
            text_rec_score_thresh: Some(0.0),
            text_rec_input_shape: None,
            text_rec_character_dict_path,
        }
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
    /// Rectified image (if document unwarping was used).
    pub rectified_img: Option<Arc<RgbImage>>,
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
        self.config.doc_orientation_classify_model_name = Some(name);
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
    pub fn doc_orientation_classify_model_path(mut self, path: String) -> Self {
        self.config.doc_orientation_classify_model_path = Some(path);
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
        self.config.doc_unwarping_model_name = Some(name);
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
    pub fn doc_unwarping_model_path(mut self, path: String) -> Self {
        self.config.doc_unwarping_model_path = Some(path);
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
        self.config.text_detection_model_name = Some(name);
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
    pub fn text_detection_model_path(mut self, path: String) -> Self {
        self.config.text_detection_model_path = path;
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
        self.config.text_detection_batch_size = Some(batch_size);
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
        self.config.text_recognition_model_name = Some(name);
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
    pub fn text_recognition_model_path(mut self, path: String) -> Self {
        self.config.text_recognition_model_path = path;
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
        self.config.text_recognition_batch_size = Some(batch_size);
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
        self.config.textline_orientation_classify_model_name = Some(name);
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
    pub fn textline_orientation_classify_model_path(mut self, path: String) -> Self {
        self.config.textline_orientation_classify_model_path = Some(path);
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
        self.config.textline_orientation_classify_batch_size = Some(batch_size);
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
        self.config.use_doc_orientation_classify = Some(use_it);
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
        self.config.use_doc_unwarping = Some(use_it);
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
        self.config.use_textline_orientation = Some(use_it);
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
        self.config.text_det_limit_side_len = Some(limit);
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
        self.config.text_det_limit_type = Some(limit_type);
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
        self.config.text_det_input_shape = Some(shape);
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
        self.config.text_rec_score_thresh = Some(thresh);
        self
    }

    /// Sets the text recognition input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The input shape as (channels, height, width)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_input_shape(mut self, shape: (u32, u32, u32)) -> Self {
        self.config.text_rec_input_shape = Some(shape);
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
    pub fn text_rec_character_dict_path(mut self, path: String) -> Self {
        self.config.text_rec_character_dict_path = path;
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
    /// Statistics for the pipeline execution.
    stats: PipelineStats,
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
            stats: PipelineStats::default(),
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
        if self.config.use_doc_orientation_classify.unwrap_or(false) {
            info!("Initializing document orientation classifier");
            self.doc_orientation_classifier = Some(self.build_doc_orientation_classifier()?);
        }

        // Initialize document rectifier if enabled
        if self.config.use_doc_unwarping.unwrap_or(false) {
            info!("Initializing document rectifier");
            self.doc_rectifier = Some(self.build_doc_rectifier()?);
        }

        // Text detector is always required
        info!("Initializing text detector");
        self.text_detector = Some(self.build_text_detector()?);

        // Initialize text line classifier if enabled
        if self.config.use_textline_orientation.unwrap_or(false) {
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
        // Get the model path from configuration, returning an error if not specified
        let model_path = if let Some(ref path) = self.config.doc_orientation_classify_model_path {
            self.get_model_path(path)
        } else {
            return Err(OCRError::ConfigError {
                message: "Document orientation classifier model path not specified".to_string(),
            });
        };

        // Create a new builder for the document orientation classifier
        let mut builder = DocOrientationClassifierBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = self.config.text_detection_batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Build and return the classifier
        builder.build(&model_path)
    }

    /// Builds the document rectifier.
    ///
    /// # Returns
    ///
    /// A Result containing the DoctrRectifierPredictor or an OCRError
    fn build_doc_rectifier(&self) -> Result<DoctrRectifierPredictor, OCRError> {
        let model_path = if let Some(ref path) = self.config.doc_unwarping_model_path {
            self.get_model_path(path)
        } else {
            return Err(OCRError::ConfigError {
                message: "Document rectifier model path not specified".to_string(),
            });
        };

        let mut builder = DoctrRectifierPredictorBuilder::new();

        if let Some(batch_size) = self.config.text_detection_batch_size {
            builder = builder.batch_size(batch_size);
        }

        builder.build(&model_path)
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
        let model_path = self.get_model_path(&self.config.text_detection_model_path);

        // Create a new builder for the text detector
        let mut builder = TextDetPredictorBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = self.config.text_detection_batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Configure the limit side length if specified in the configuration
        if let Some(limit_side_len) = self.config.text_det_limit_side_len {
            builder = builder.limit_side_len(limit_side_len);
        }

        // Configure the limit type if specified in the configuration
        if let Some(limit_type) = &self.config.text_det_limit_type {
            builder = builder.limit_type(limit_type.clone());
        }

        // Build and return the text detector
        builder.build(&model_path)
    }

    /// Builds the text line classifier.
    ///
    /// # Returns
    ///
    /// A Result containing the TextLineClasPredictor or an OCRError
    fn build_text_line_classifier(&self) -> Result<TextLineClasPredictor, OCRError> {
        let model_path =
            if let Some(ref path) = self.config.textline_orientation_classify_model_path {
                self.get_model_path(path)
            } else {
                return Err(OCRError::ConfigError {
                    message: "Text line classifier model path not specified".to_string(),
                });
            };

        let mut builder = TextLineClasPredictorBuilder::new();

        if let Some(batch_size) = self.config.textline_orientation_classify_batch_size {
            builder = builder.batch_size(batch_size);
        }

        if let Some(model_name) = &self.config.textline_orientation_classify_model_name {
            builder = builder.model_name(model_name.clone());
        }

        builder.build(&model_path)
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
        let model_path = self.get_model_path(&self.config.text_recognition_model_path);

        // Create a new builder for the text recognizer
        let mut builder = TextRecPredictorBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = self.config.text_recognition_batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Configure the input shape if specified in the configuration
        if let Some(shape) = self.config.text_rec_input_shape {
            builder =
                builder.rec_image_shape([shape.0 as usize, shape.1 as usize, shape.2 as usize]);
        }

        // Load the character dictionary and configure it in the builder
        let character_dict = self.load_character_dict(&self.config.text_rec_character_dict_path)?;
        builder = builder.character_dict(character_dict);

        // Build and return the text recognizer
        builder.build(&model_path)
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

    /// Gets the model path from a configuration path.
    ///
    /// # Arguments
    ///
    /// * `config_path` - The path from the configuration
    ///
    /// # Returns
    ///
    /// A PathBuf representing the model path
    fn get_model_path(&self, config_path: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(config_path)
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

    /// Processes a single image through the OCR pipeline.
    ///
    /// This method runs the complete OCR pipeline on a single image,
    /// including document orientation classification, text detection,
    /// text recognition, and text line classification (if enabled).
    ///
    /// # Arguments
    ///
    /// * `image_path` - The path to the image file
    ///
    /// # Returns
    ///
    /// A Result containing the OAROCRResult or an OCRError
    pub fn predict(&mut self, image_path: &Path) -> Result<OAROCRResult, OCRError> {
        info!("Starting OCR pipeline for image: {:?}", image_path);

        let input_img = crate::utils::load_image(image_path)?;
        let input_img_arc = Arc::new(input_img.clone());

        let mut _current_img = input_img;
        let mut orientation_angle = None;
        let mut rectified_img = None;

        if let Some(ref mut classifier) = self.doc_orientation_classifier {
            debug!("Running document orientation classification");
            let _result = classifier.predict_single(image_path)?;

            orientation_angle = Some(0.0);
        }

        if let Some(ref mut rectifier) = self.doc_rectifier {
            debug!("Running document rectification");
            let _result = rectifier.predict_single(image_path)?;

            rectified_img = Some(input_img_arc.clone());
        }

        let text_boxes = if let Some(ref mut detector) = self.text_detector {
            debug!("Running text detection");
            let result = detector.predict_single(image_path)?;

            match result {
                PredictionResult::Detection { dt_polys, .. } => {
                    dt_polys.into_iter().flatten().collect()
                }
                _ => {
                    warn!("Unexpected result type from text detector");
                    Vec::new()
                }
            }
        } else {
            return Err(OCRError::ConfigError {
                message: "Text detector not initialized".to_string(),
            });
        };

        let mut cropped_images = Vec::new();
        for (i, bbox) in text_boxes.iter().enumerate() {
            match self.crop_bounding_box(&_current_img, bbox) {
                Ok(cropped_img) => {
                    debug!(
                        "Successfully cropped region {}: {}x{}",
                        i,
                        cropped_img.width(),
                        cropped_img.height()
                    );
                    cropped_images.push(cropped_img);
                }
                Err(e) => {
                    warn!("Failed to crop bounding box {}: {}", i, e);

                    cropped_images.push(RgbImage::new(1, 1));
                }
            }
        }

        // Process text recognition if we have detected text boxes
        let (rec_texts, rec_scores) = if text_boxes.is_empty() {
            // No text boxes detected, return empty results
            (Vec::new(), Vec::new())
        } else if let Some(ref mut recognizer) = self.text_recognizer {
            debug!(
                "Running text recognition on {} detected regions",
                text_boxes.len()
            );

            let mut all_texts = Vec::new();
            let mut all_scores = Vec::new();

            // Process each cropped image through the text recognizer
            for (i, cropped_img) in cropped_images.iter().enumerate() {
                // Skip placeholder images (1x1 pixels)
                if cropped_img.width() == 1 && cropped_img.height() == 1 {
                    continue;
                }

                // Save cropped image to temporary file for recognition
                // This is needed because the recognizer expects file paths
                let temp_dir = std::env::temp_dir();
                let temp_path = temp_dir.join(format!("oar_temp_crop_{i}.jpg"));

                if let Err(e) = cropped_img.save(&temp_path) {
                    warn!("Failed to save temporary image {}: {}", i, e);
                    continue;
                }

                // Run text recognition on the cropped image
                match recognizer.predict_single(&temp_path) {
                    Ok(result) => match result {
                        PredictionResult::Recognition {
                            rec_text,
                            rec_score,
                            ..
                        } => {
                            // Extract the first recognized text and score
                            if let (Some(text), Some(score)) = (rec_text.first(), rec_score.first())
                            {
                                all_texts.push(Arc::from(text.as_ref()));
                                all_scores.push(*score);
                                debug!("Recognized text {}: '{}' (score: {:.3})", i, text, score);
                            }
                        }
                        _ => {
                            warn!(
                                "Unexpected result type from text recognizer for region {}",
                                i
                            );
                        }
                    },
                    Err(e) => {
                        warn!("Text recognition failed for region {}: {}", i, e);
                    }
                }

                // Clean up temporary file
                if let Err(e) = std::fs::remove_file(&temp_path) {
                    warn!("Failed to remove temporary file {:?}: {}", temp_path, e);
                }
            }

            debug!(
                "Text recognition completed: {} texts recognized from {} regions",
                all_texts.len(),
                text_boxes.len()
            );
            (all_texts, all_scores)
        } else {
            return Err(OCRError::ConfigError {
                message: "Text recognizer not initialized".to_string(),
            });
        };

        // Filter recognition results based on score threshold
        let score_thresh = self.config.text_rec_score_thresh.unwrap_or(0.0);
        let filtered_results: Vec<(Arc<str>, f32)> = rec_texts
            .into_iter()
            .zip(rec_scores)
            // Keep only results with scores above the threshold
            .filter(|(_, score)| *score >= score_thresh)
            .collect();

        // Separate the filtered texts and scores into separate vectors
        let (final_texts, final_scores): (Vec<Arc<str>>, Vec<f32>) =
            filtered_results.into_iter().unzip();

        info!(
            "OCR pipeline completed. Found {} text regions with {} recognized texts",
            text_boxes.len(),
            final_texts.len()
        );

        Ok(OAROCRResult {
            input_path: Arc::from(image_path.to_string_lossy().as_ref()),
            index: 0,
            input_img: input_img_arc,
            text_boxes,
            rec_texts: final_texts,
            rec_scores: final_scores,
            orientation_angle,
            rectified_img,
        })
    }

    /// Processes a batch of images through the OCR pipeline.
    ///
    /// This method runs the complete OCR pipeline on a batch of images.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - A slice of paths to the image files
    ///
    /// # Returns
    ///
    /// A Result containing a vector of OAROCRResult or an OCRError
    pub fn predict_batch(&mut self, image_paths: &[&Path]) -> Result<Vec<OAROCRResult>, OCRError> {
        info!(
            "Starting batch OCR pipeline for {} images",
            image_paths.len()
        );

        let mut results = Vec::with_capacity(image_paths.len());

        for (index, &path) in image_paths.iter().enumerate() {
            debug!(
                "Processing image {} of {}: {:?}",
                index + 1,
                image_paths.len(),
                path
            );
            let mut result = self.predict(path)?;
            result.index = index;
            results.push(result);
        }

        info!("Batch OCR pipeline completed for {} images", results.len());
        Ok(results)
    }

    /// Gets the pipeline statistics.
    ///
    /// # Returns
    ///
    /// A reference to the PipelineStats
    pub fn get_stats(&self) -> &PipelineStats {
        &self.stats
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

impl PipelineExecutor for OAROCR {
    /// Executes the OCR pipeline on a single image.
    ///
    /// This method runs the complete OCR pipeline on a single image and
    /// returns a PredictionResult with the combined text and average score.
    ///
    /// # Arguments
    ///
    /// * `image_path` - The path to the image file
    ///
    /// # Returns
    ///
    /// A Result containing the PredictionResult or an OCRError
    fn execute_pipeline(
        &mut self,
        image_path: &Path,
    ) -> Result<PredictionResult<'static>, OCRError> {
        let result = self.predict(image_path)?;

        // Combine all recognized texts into a single string
        let combined_text = result.rec_texts.join(" ");
        // Calculate the average score of all recognition results
        let avg_score = if result.rec_scores.is_empty() {
            0.0
        } else {
            result.rec_scores.iter().sum::<f32>() / result.rec_scores.len() as f32
        };

        Ok(PredictionResult::Recognition {
            input_path: vec![std::borrow::Cow::Owned(result.input_path.to_string())],
            index: vec![result.index],
            input_img: vec![result.input_img],
            rec_text: vec![std::borrow::Cow::Owned(combined_text)],
            rec_score: vec![avg_score],
        })
    }

    /// Executes the OCR pipeline on a batch of images.
    ///
    /// This method runs the complete OCR pipeline on a batch of images and
    /// returns a vector of PredictionResult with the combined text and average score
    /// for each image.
    ///
    /// # Arguments
    ///
    /// * `image_paths` - A slice of paths to the image files
    ///
    /// # Returns
    ///
    /// A Result containing a vector of PredictionResult or an OCRError
    fn execute_batch_pipeline(
        &mut self,
        image_paths: &[&Path],
    ) -> Result<Vec<PredictionResult<'static>>, OCRError> {
        let results = self.predict_batch(image_paths)?;

        let mut prediction_results = Vec::with_capacity(results.len());

        // Process each result in the batch
        for result in results {
            // Combine all recognized texts into a single string
            let combined_text = result.rec_texts.join(" ");
            // Calculate the average score of all recognition results
            let avg_score = if result.rec_scores.is_empty() {
                0.0
            } else {
                result.rec_scores.iter().sum::<f32>() / result.rec_scores.len() as f32
            };

            prediction_results.push(PredictionResult::Recognition {
                input_path: vec![std::borrow::Cow::Owned(result.input_path.to_string())],
                index: vec![result.index],
                input_img: vec![result.input_img],
                rec_text: vec![std::borrow::Cow::Owned(combined_text)],
                rec_score: vec![avg_score],
            });
        }

        Ok(prediction_results)
    }

    /// Gets the pipeline statistics.
    ///
    /// # Returns
    ///
    /// The pipeline statistics
    fn get_pipeline_stats(&self) -> PipelineStats {
        self.stats.clone()
    }

    /// Checks if the pipeline supports parallel execution.
    ///
    /// # Returns
    ///
    /// True, as the pipeline supports parallel execution
    fn supports_parallel_execution(&self) -> bool {
        true
    }

    /// Gets the recommended batch size for the pipeline.
    ///
    /// # Returns
    ///
    /// The recommended batch size based on the text recognition configuration
    fn recommended_batch_size(&self) -> usize {
        self.config.text_recognition_batch_size.unwrap_or(1)
    }
}
