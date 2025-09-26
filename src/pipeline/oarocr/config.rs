//! Configuration types for the OAROCR pipeline.

use crate::core::DynamicBatchConfig;
use crate::pipeline::stages::{OrientationConfig, TextLineOrientationConfig};
use crate::predictor::{
    DocOrientationClassifierConfig, DoctrRectifierPredictorConfig, TextDetPredictorConfig,
    TextLineClasPredictorConfig, TextRecPredictorConfig,
};
use crate::processors::{AspectRatioBucketingConfig, LimitType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use crate::core::config::{OnnxThreadingConfig, ParallelPolicy};

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

    /// Configuration for document orientation stage processing.
    #[serde(default)]
    pub orientation_stage: Option<OrientationConfig>,

    /// Configuration for text line orientation stage processing.
    #[serde(default)]
    pub text_line_orientation_stage: Option<TextLineOrientationConfig>,

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

    /// Configuration for aspect ratio bucketing in text recognition.
    /// If None, falls back to exact dimension grouping.
    #[serde(default)]
    pub aspect_ratio_bucketing: Option<AspectRatioBucketingConfig>,

    /// Configuration for dynamic batching across multiple images.
    /// If None, uses default dynamic batching configuration.
    #[serde(default)]
    pub dynamic_batching: Option<DynamicBatchConfig>,

    /// Centralized parallel processing policy configuration
    #[serde(default)]
    pub parallel_policy: ParallelPolicy,
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
            orientation_stage: None,
            text_line_orientation_stage: None,
            character_dict_path: character_dict_path.into(),
            use_doc_orientation_classify: false,
            use_doc_unwarping: false,
            use_textline_orientation: false,
            aspect_ratio_bucketing: None,
            dynamic_batching: None,
            parallel_policy: ParallelPolicy::default(),
        }
    }

    /// Get the effective parallel policy
    pub fn effective_parallel_policy(&self) -> ParallelPolicy {
        self.parallel_policy.clone()
    }

    /// Get the maximum number of threads for parallel processing
    pub fn max_threads(&self) -> Option<usize> {
        self.effective_parallel_policy().max_threads
    }

    /// Get the image processing threshold
    pub fn image_threshold(&self) -> usize {
        self.effective_parallel_policy().image_threshold
    }

    /// Get the text box processing threshold
    pub fn text_box_threshold(&self) -> usize {
        self.effective_parallel_policy().text_box_threshold
    }

    /// Get the batch processing threshold
    pub fn batch_threshold(&self) -> usize {
        self.effective_parallel_policy().batch_threshold
    }

    /// Get the utility operations threshold
    pub fn utility_threshold(&self) -> usize {
        self.effective_parallel_policy().utility_threshold
    }

    /// Get the postprocessing pixel threshold
    pub fn postprocess_pixel_threshold(&self) -> usize {
        self.effective_parallel_policy().postprocess_pixel_threshold
    }

    /// Get the ONNX threading configuration
    pub fn onnx_threading(&self) -> OnnxThreadingConfig {
        self.effective_parallel_policy().onnx_threading
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_policy_builder() {
        let onnx_config = OnnxThreadingConfig {
            intra_threads: Some(4),
            inter_threads: Some(2),
            parallel_execution: Some(true),
        };

        let policy = ParallelPolicy::new()
            .with_max_threads(Some(8))
            .with_image_threshold(2)
            .with_text_box_threshold(5)
            .with_batch_threshold(20)
            .with_utility_threshold(8)
            .with_postprocess_pixel_threshold(16000)
            .with_onnx_threading(onnx_config.clone());

        assert_eq!(policy.max_threads, Some(8));
        assert_eq!(policy.image_threshold, 2);
        assert_eq!(policy.text_box_threshold, 5);
        assert_eq!(policy.batch_threshold, 20);
        assert_eq!(policy.utility_threshold, 8);
        assert_eq!(policy.postprocess_pixel_threshold, 16000);
        assert_eq!(policy.onnx_threading.intra_threads, Some(4));
        assert_eq!(policy.onnx_threading.inter_threads, Some(2));
        assert_eq!(policy.onnx_threading.parallel_execution, Some(true));
    }

    #[test]
    fn test_parallel_policy_serialization() {
        let policy = ParallelPolicy::new()
            .with_max_threads(Some(4))
            .with_image_threshold(3);

        let serialized = serde_json::to_string(&policy).unwrap();
        let deserialized: ParallelPolicy = serde_json::from_str(&serialized).unwrap();

        assert_eq!(policy.max_threads, deserialized.max_threads);
        assert_eq!(policy.image_threshold, deserialized.image_threshold);
        assert_eq!(policy.text_box_threshold, deserialized.text_box_threshold);
        assert_eq!(policy.batch_threshold, deserialized.batch_threshold);
        assert_eq!(policy.utility_threshold, deserialized.utility_threshold);
    }

    #[test]
    fn test_oarocr_config_effective_parallel_policy() {
        let mut config = OAROCRConfig::default();

        // Test with default policy
        let policy = config.effective_parallel_policy();
        assert_eq!(policy.max_threads, None);
        assert_eq!(policy.image_threshold, 1);
        assert_eq!(policy.text_box_threshold, 1);

        // Test with custom parallel policy
        config.parallel_policy = ParallelPolicy::new()
            .with_max_threads(Some(6))
            .with_image_threshold(3);

        let policy = config.effective_parallel_policy();
        assert_eq!(policy.max_threads, Some(6));
        assert_eq!(policy.image_threshold, 3);
        assert_eq!(policy.text_box_threshold, 1);
    }

    #[test]
    fn test_oarocr_config_parallel_policy() {
        let config = OAROCRConfig {
            parallel_policy: ParallelPolicy::new()
                .with_max_threads(Some(4))
                .with_image_threshold(2),
            ..Default::default()
        };

        let policy = config.effective_parallel_policy();
        assert_eq!(policy.max_threads, Some(4));
        assert_eq!(policy.image_threshold, 2);
        assert_eq!(policy.text_box_threshold, 1); // Default

        // Test convenience methods
        assert_eq!(config.max_threads(), Some(4));
        assert_eq!(config.image_threshold(), 2);
        assert_eq!(config.text_box_threshold(), 1);
        assert_eq!(config.batch_threshold(), 10); // Default
        assert_eq!(config.utility_threshold(), 4); // Default
    }
}
