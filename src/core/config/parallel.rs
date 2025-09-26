//! Shared parallel processing configuration types.

use serde::{Deserialize, Serialize};

/// Centralized configuration for parallel processing behavior across the OCR pipeline.
///
/// This struct consolidates parallel processing configuration that was previously
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

/// ONNX Runtime threading configuration that's part of the centralized parallel policy.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OnnxThreadingConfig {
    /// Number of threads used to parallelize execution within nodes.
    /// If None, uses ONNX Runtime default.
    #[serde(default)]
    pub intra_threads: Option<usize>,

    /// Number of threads used to parallelize execution across nodes.
    /// If None, uses ONNX Runtime default.
    #[serde(default)]
    pub inter_threads: Option<usize>,

    /// Enable parallel execution mode.
    /// If None, uses ONNX Runtime default.
    #[serde(default)]
    pub parallel_execution: Option<bool>,
}

impl OnnxThreadingConfig {
    /// Create a new OnnxThreadingConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the intra-op threads.
    pub fn with_intra_threads(mut self, threads: Option<usize>) -> Self {
        self.intra_threads = threads;
        self
    }

    /// Set the inter-op threads.
    pub fn with_inter_threads(mut self, threads: Option<usize>) -> Self {
        self.inter_threads = threads;
        self
    }

    /// Set parallel execution mode.
    pub fn with_parallel_execution(mut self, enabled: Option<bool>) -> Self {
        self.parallel_execution = enabled;
        self
    }

    /// Convert to OrtSessionConfig for use with ONNX Runtime.
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
    /// Create a new ParallelPolicy with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of threads.
    pub fn with_max_threads(mut self, max_threads: Option<usize>) -> Self {
        self.max_threads = max_threads;
        self
    }

    /// Set the image processing threshold.
    pub fn with_image_threshold(mut self, threshold: usize) -> Self {
        self.image_threshold = threshold;
        self
    }

    /// Set the text box processing threshold.
    pub fn with_text_box_threshold(mut self, threshold: usize) -> Self {
        self.text_box_threshold = threshold;
        self
    }

    /// Set the batch processing threshold.
    pub fn with_batch_threshold(mut self, threshold: usize) -> Self {
        self.batch_threshold = threshold;
        self
    }

    /// Set the postprocessing pixel threshold.
    pub fn with_postprocess_pixel_threshold(mut self, threshold: usize) -> Self {
        self.postprocess_pixel_threshold = threshold;
        self
    }

    /// Set the ONNX threading configuration.
    pub fn with_onnx_threading(mut self, config: OnnxThreadingConfig) -> Self {
        self.onnx_threading = config;
        self
    }

    /// Set the utility operations threshold.
    pub fn with_utility_threshold(mut self, threshold: usize) -> Self {
        self.utility_threshold = threshold;
        self
    }

    /// Default value for image threshold.
    fn default_image_threshold() -> usize {
        1
    }

    /// Default value for text box threshold.
    fn default_text_box_threshold() -> usize {
        1
    }

    /// Default value for batch threshold.
    fn default_batch_threshold() -> usize {
        10
    }

    /// Default value for utility threshold.
    fn default_utility_threshold() -> usize {
        4 // Matches DEFAULT_PARALLEL_THRESHOLD from constants.
    }

    /// Default postprocessing pixel threshold.
    fn default_postprocess_pixel_threshold() -> usize {
        8_000
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
