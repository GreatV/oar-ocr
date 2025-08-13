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

/// Centralized configuration for parallel processing behavior across the pipeline.
///
/// This struct consolidates all parallel processing configuration that was previously
/// scattered across different components, providing a unified way to tune parallelism
/// behavior throughout the OCR pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPolicy {
    /// Maximum number of threads to use for parallel processing.
    /// If None, rayon will use the default thread pool size (typically number of CPU cores).
    /// Default: None (use rayon's default)
    #[serde(default)]
    pub max_threads: Option<usize>,

    /// Threshold for number of images to process sequentially (<= this uses sequential)
    /// Default: 1 (process single images sequentially, use parallel for multiple images)
    #[serde(default = "ParallelPolicy::default_image_threshold")]
    pub image_threshold: usize,

    /// Threshold for number of text boxes to crop sequentially (<= this uses sequential)
    /// Default: 1 (process single text boxes sequentially, use parallel for multiple boxes)
    #[serde(default = "ParallelPolicy::default_text_box_threshold")]
    pub text_box_threshold: usize,

    /// Threshold for batch processing operations (<= this uses sequential)
    /// Default: 10 (use sequential for small batches, parallel for larger ones)
    #[serde(default = "ParallelPolicy::default_batch_threshold")]
    pub batch_threshold: usize,

    /// Threshold for general utility operations like image loading (<= this uses sequential)
    /// Default: 4 (matches DEFAULT_PARALLEL_THRESHOLD constant)
    #[serde(default = "ParallelPolicy::default_utility_threshold")]
    pub utility_threshold: usize,

    /// Threshold for postprocessing operations based on pixel area (<= this uses sequential)
    /// Default: 8000 (use sequential for regions with <= 8000 pixels, parallel for larger)
    #[serde(default = "ParallelPolicy::default_postprocess_pixel_threshold")]
    pub postprocess_pixel_threshold: usize,

    /// ONNX Runtime threading configuration
    #[serde(default)]
    pub onnx_threading: OnnxThreadingConfig,
}

/// ONNX Runtime threading configuration that's part of the centralized parallel policy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnnxThreadingConfig {
    /// Number of threads used to parallelize execution within nodes
    /// If None, uses ONNX Runtime default
    #[serde(default)]
    pub intra_threads: Option<usize>,

    /// Number of threads used to parallelize execution across nodes
    /// If None, uses ONNX Runtime default
    #[serde(default)]
    pub inter_threads: Option<usize>,

    /// Enable parallel execution mode
    /// If None, uses ONNX Runtime default
    #[serde(default)]
    pub parallel_execution: Option<bool>,
}

impl OnnxThreadingConfig {
    /// Create a new OnnxThreadingConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the intra-op threads
    pub fn with_intra_threads(mut self, threads: Option<usize>) -> Self {
        self.intra_threads = threads;
        self
    }

    /// Set the inter-op threads
    pub fn with_inter_threads(mut self, threads: Option<usize>) -> Self {
        self.inter_threads = threads;
        self
    }

    /// Set parallel execution mode
    pub fn with_parallel_execution(mut self, enabled: Option<bool>) -> Self {
        self.parallel_execution = enabled;
        self
    }

    /// Convert to OrtSessionConfig for use with ONNX Runtime
    pub fn to_ort_session_config(&self) -> crate::core::config::OrtSessionConfig {
        let mut config = crate::core::config::OrtSessionConfig::new();

        if let Some(intra) = self.intra_threads {
            config = config.with_intra_threads(intra);
        }

        if let Some(inter) = self.inter_threads {
            config = config.with_inter_threads(inter);
        }

        if let Some(parallel) = self.parallel_execution {
            config = config.with_parallel_execution(parallel);
        }

        config
    }
}

impl ParallelPolicy {
    /// Create a new ParallelPolicy with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of threads
    pub fn with_max_threads(mut self, max_threads: Option<usize>) -> Self {
        self.max_threads = max_threads;
        self
    }

    /// Set the image processing threshold
    pub fn with_image_threshold(mut self, threshold: usize) -> Self {
        self.image_threshold = threshold;
        self
    }

    /// Set the text box processing threshold
    pub fn with_text_box_threshold(mut self, threshold: usize) -> Self {
        self.text_box_threshold = threshold;
        self
    }

    /// Set the batch processing threshold
    pub fn with_batch_threshold(mut self, threshold: usize) -> Self {
        self.batch_threshold = threshold;
        self
    }

    /// Set the postprocessing pixel threshold
    pub fn with_postprocess_pixel_threshold(mut self, threshold: usize) -> Self {
        self.postprocess_pixel_threshold = threshold;
        self
    }

    /// Set the ONNX threading configuration
    pub fn with_onnx_threading(mut self, config: OnnxThreadingConfig) -> Self {
        self.onnx_threading = config;
        self
    }

    /// Set the utility operations threshold
    pub fn with_utility_threshold(mut self, threshold: usize) -> Self {
        self.utility_threshold = threshold;
        self
    }

    /// Default value for image threshold
    fn default_image_threshold() -> usize {
        1
    }

    /// Default value for text box threshold
    fn default_text_box_threshold() -> usize {
        1
    }

    /// Default value for batch threshold
    fn default_batch_threshold() -> usize {
        10
    }

    /// Default value for utility threshold
    fn default_utility_threshold() -> usize {
        4 // Matches DEFAULT_PARALLEL_THRESHOLD from constants
    }

    /// Default postprocessing pixel threshold
    fn default_postprocess_pixel_threshold() -> usize {
        8000
    }
}

impl Default for ParallelPolicy {
    fn default() -> Self {
        Self {
            max_threads: None,
            image_threshold: Self::default_image_threshold(),
            text_box_threshold: Self::default_text_box_threshold(),
            batch_threshold: Self::default_batch_threshold(),
            utility_threshold: Self::default_utility_threshold(),
            postprocess_pixel_threshold: Self::default_postprocess_pixel_threshold(),
            onnx_threading: OnnxThreadingConfig::default(),
        }
    }
}

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
    fn test_parallel_policy_default() {
        let policy = ParallelPolicy::default();
        assert_eq!(policy.max_threads, None);
        assert_eq!(policy.image_threshold, 1);
        assert_eq!(policy.text_box_threshold, 1);
        assert_eq!(policy.batch_threshold, 10);
        assert_eq!(policy.utility_threshold, 4);
        assert_eq!(policy.postprocess_pixel_threshold, 8000);
        assert_eq!(policy.onnx_threading.intra_threads, None);
        assert_eq!(policy.onnx_threading.inter_threads, None);
        assert_eq!(policy.onnx_threading.parallel_execution, None);
    }

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
