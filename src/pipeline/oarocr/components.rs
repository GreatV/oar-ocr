//! Component initialization utilities for the OAROCR pipeline.

use crate::core::OCRError;
use crate::pipeline::oarocr::config::OAROCRConfig;
use crate::predictor::{
    DocOrientationClassifier, DocOrientationClassifierBuilder, DoctrRectifierPredictor,
    DoctrRectifierPredictorBuilder, TextDetPredictor, TextDetPredictorBuilder,
    TextLineClasPredictor, TextLineClasPredictorBuilder, TextRecPredictor, TextRecPredictorBuilder,
};

/// Component builder for initializing OAROCR pipeline components.
pub struct ComponentBuilder;

impl ComponentBuilder {
    /// Builds the document orientation classifier.
    ///
    /// This method creates a document orientation classifier using the
    /// configured model path and batch size. It uses the builder pattern
    /// to construct the classifier with the appropriate settings.
    ///
    /// # Arguments
    ///
    /// * `config` - The OAROCR configuration
    ///
    /// # Returns
    ///
    /// A Result containing the DocOrientationClassifier or an OCRError
    pub fn build_doc_orientation_classifier(
        config: &OAROCRConfig,
    ) -> Result<DocOrientationClassifier, OCRError> {
        let orientation_config =
            config
                .orientation
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Document orientation classification config not specified".to_string(),
                })?;

        // Get the model path from configuration, returning an error if not specified
        let model_path = orientation_config
            .common
            .model_path
            .as_ref()
            .ok_or_else(|| OCRError::ConfigError {
                message: "Document orientation classification model path not specified".to_string(),
            })?;

        // Create a new builder for the document orientation classifier
        let mut builder = DocOrientationClassifierBuilder::new();

        // Configure the batch size if specified in the configuration
        if let Some(batch_size) = orientation_config.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Set model name if specified
        if let Some(ref name) = orientation_config.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Set ORT session configuration if specified
        if let Some(ref ort_session) = orientation_config.common.ort_session {
            builder = builder.ort_session(ort_session.clone());
        }

        // Set session pool size if specified
        if let Some(session_pool_size) = orientation_config.common.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Set enable logging if specified
        if let Some(enable_logging) = orientation_config.common.enable_logging {
            builder = builder.enable_logging(enable_logging);
        }

        // Set input shape if specified
        if let Some(shape) = orientation_config.input_shape {
            builder = builder.input_shape(shape);
        }

        // Build and return the classifier
        builder.build(model_path)
    }

    /// Builds the document rectifier.
    ///
    /// # Arguments
    ///
    /// * `config` - The OAROCR configuration
    ///
    /// # Returns
    ///
    /// A Result containing the DoctrRectifierPredictor or an OCRError
    pub fn build_doc_rectifier(config: &OAROCRConfig) -> Result<DoctrRectifierPredictor, OCRError> {
        let rectification_config =
            config
                .rectification
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Document rectification config not specified".to_string(),
                })?;

        let model_path = rectification_config
            .common
            .model_path
            .as_ref()
            .ok_or_else(|| OCRError::ConfigError {
                message: "Document rectifier model path not specified".to_string(),
            })?;

        let mut builder = DoctrRectifierPredictorBuilder::new();

        if let Some(batch_size) = rectification_config.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        if let Some(ref name) = rectification_config.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Set ORT session configuration if specified
        if let Some(ref ort_session) = rectification_config.common.ort_session {
            builder = builder.ort_session(ort_session.clone());
        }

        // Set session pool size if specified
        if let Some(session_pool_size) = rectification_config.common.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Set enable logging if specified
        if let Some(enable_logging) = rectification_config.common.enable_logging {
            builder = builder.enable_logging(enable_logging);
        }

        builder.build(model_path)
    }

    /// Builds the text detector.
    ///
    /// This method creates a text detector using the configured model path
    /// and various detection parameters. It uses the builder pattern to
    /// construct the detector with the appropriate settings.
    ///
    /// # Arguments
    ///
    /// * `config` - The OAROCR configuration
    ///
    /// # Returns
    ///
    /// A Result containing the TextDetPredictor or an OCRError
    pub fn build_text_detector(config: &OAROCRConfig) -> Result<TextDetPredictor, OCRError> {
        // Get the model path from configuration
        let model_path =
            config
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
        if let Some(batch_size) = config.detection.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Set model name if specified
        if let Some(ref name) = config.detection.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Set ORT session configuration if specified
        if let Some(ref ort_session) = config.detection.common.ort_session {
            builder = builder.ort_session(ort_session.clone());
        }

        // Set session pool size if specified
        if let Some(session_pool_size) = config.detection.common.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Set enable logging if specified
        if let Some(enable_logging) = config.detection.common.enable_logging {
            builder = builder.enable_logging(enable_logging);
        }

        // Configure the limit side length if specified in the configuration
        if let Some(limit_side_len) = config.detection.limit_side_len {
            builder = builder.limit_side_len(limit_side_len);
        }

        // Configure the limit type if specified in the configuration
        if let Some(limit_type) = &config.detection.limit_type {
            builder = builder.limit_type(limit_type.clone());
        }

        // Configure input shape if specified
        if let Some(shape) = config.detection.input_shape {
            builder = builder.input_shape(shape);
        }

        // Configure max side limit if specified
        if let Some(max_side_limit) = config.detection.max_side_limit {
            builder = builder.max_side_limit(max_side_limit);
        }

        // Configure binarization threshold if specified
        if let Some(thresh) = config.detection.thresh {
            builder = builder.thresh(thresh);
        }

        // Configure box score threshold if specified
        if let Some(box_thresh) = config.detection.box_thresh {
            builder = builder.box_thresh(box_thresh);
        }

        // Configure unclip ratio if specified
        if let Some(unclip_ratio) = config.detection.unclip_ratio {
            builder = builder.unclip_ratio(unclip_ratio);
        }

        // Build and return the text detector
        builder.build(model_path)
    }

    /// Builds the text line classifier.
    ///
    /// # Arguments
    ///
    /// * `config` - The OAROCR configuration
    ///
    /// # Returns
    ///
    /// A Result containing the TextLineClasPredictor or an OCRError
    pub fn build_text_line_classifier(
        config: &OAROCRConfig,
    ) -> Result<TextLineClasPredictor, OCRError> {
        let text_line_config =
            config
                .text_line_orientation
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Text line orientation config not specified".to_string(),
                })?;

        let model_path =
            text_line_config
                .common
                .model_path
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Text line classifier model path not specified".to_string(),
                })?;

        let mut builder = TextLineClasPredictorBuilder::new();

        if let Some(batch_size) = text_line_config.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        if let Some(ref model_name) = text_line_config.common.model_name {
            builder = builder.model_name(model_name.clone());
        }

        // Set ORT session configuration if specified
        if let Some(ref ort_session) = text_line_config.common.ort_session {
            builder = builder.ort_session(ort_session.clone());
        }

        // Set session pool size if specified
        if let Some(session_pool_size) = text_line_config.common.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Set enable logging if specified
        if let Some(enable_logging) = text_line_config.common.enable_logging {
            builder = builder.enable_logging(enable_logging);
        }

        if let Some(input_shape) = text_line_config.input_shape {
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
    /// # Arguments
    ///
    /// * `config` - The OAROCR configuration
    ///
    /// # Returns
    ///
    /// A Result containing the TextRecPredictor or an OCRError
    pub fn build_text_recognizer(config: &OAROCRConfig) -> Result<TextRecPredictor, OCRError> {
        // Get the model path from configuration
        let model_path = config
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
        if let Some(batch_size) = config.recognition.common.batch_size {
            builder = builder.batch_size(batch_size);
        }

        // Set model name if specified
        if let Some(ref name) = config.recognition.common.model_name {
            builder = builder.model_name(name.clone());
        }

        // Set ORT session configuration if specified
        if let Some(ref ort_session) = config.recognition.common.ort_session {
            builder = builder.ort_session(ort_session.clone());
        }

        // Set session pool size if specified
        if let Some(session_pool_size) = config.recognition.common.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Set enable logging if specified
        if let Some(enable_logging) = config.recognition.common.enable_logging {
            builder = builder.enable_logging(enable_logging);
        }

        // Configure the model input shape if specified in the configuration
        if let Some(shape) = config.recognition.model_input_shape {
            builder = builder.model_input_shape(shape);
        }

        // Load the character dictionary and configure it in the builder
        let character_dict =
            Self::load_character_dict(config.character_dict_path.to_str().ok_or_else(|| {
                OCRError::ConfigError {
                    message: "Invalid character dictionary path".to_string(),
                }
            })?)?;
        builder = builder.character_dict(character_dict);

        // Configure the score threshold if specified in the configuration
        if let Some(score_thresh) = config.recognition.score_thresh {
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
    fn load_character_dict(dict_path: &str) -> Result<Vec<String>, OCRError> {
        // Read the entire dictionary file into a string
        let content = std::fs::read_to_string(dict_path).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to load character dictionary from {dict_path}: {e}"),
        })?;

        // Split the content into lines and collect them into a vector
        Ok(content.lines().map(|line| line.to_string()).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::builder::CommonBuilderConfig;
    use crate::predictor::{TextDetPredictorConfig, TextRecPredictorConfig};
    use crate::processors::LimitType;
    use std::path::PathBuf;

    #[test]
    fn test_common_builder_config_propagation() {
        // Test that all CommonBuilderConfig fields are properly propagated
        let mut common_config = CommonBuilderConfig::new()
            .model_name("test_model")
            .batch_size(16)
            .enable_logging(false)
            .session_pool_size(4);

        // Set a dummy model path
        common_config.model_path = Some(PathBuf::from("test_model.onnx"));

        // Test TextDetPredictorConfig::with_common
        let det_config = TextDetPredictorConfig::with_common(common_config.clone());
        assert_eq!(det_config.common.model_name, Some("test_model".to_string()));
        assert_eq!(det_config.common.batch_size, Some(16));
        assert_eq!(det_config.common.enable_logging, Some(false));
        assert_eq!(det_config.common.session_pool_size, Some(4));

        // Test TextRecPredictorConfig::with_common
        let rec_config = TextRecPredictorConfig::with_common(common_config.clone());
        assert_eq!(rec_config.common.model_name, Some("test_model".to_string()));
        assert_eq!(rec_config.common.batch_size, Some(16));
        assert_eq!(rec_config.common.enable_logging, Some(false));
        assert_eq!(rec_config.common.session_pool_size, Some(4));
    }

    #[test]
    fn test_text_detection_config_with_thresholds() {
        // Test that TextDetPredictorConfig properly holds threshold values
        let mut common_config = CommonBuilderConfig::new();
        common_config.model_path = Some(PathBuf::from("test_model.onnx"));

        let mut det_config = TextDetPredictorConfig::with_common(common_config);
        det_config.thresh = Some(0.4);
        det_config.box_thresh = Some(0.7);
        det_config.unclip_ratio = Some(2.0);
        det_config.limit_side_len = Some(960);
        det_config.limit_type = Some(LimitType::Max);
        det_config.max_side_limit = Some(5000);

        // Verify all threshold values are properly set
        assert_eq!(det_config.thresh, Some(0.4));
        assert_eq!(det_config.box_thresh, Some(0.7));
        assert_eq!(det_config.unclip_ratio, Some(2.0));
        assert_eq!(det_config.limit_side_len, Some(960));
        assert_eq!(det_config.limit_type, Some(LimitType::Max));
        assert_eq!(det_config.max_side_limit, Some(5000));
    }
}
