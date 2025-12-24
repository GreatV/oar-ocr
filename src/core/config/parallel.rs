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

    /// Threshold for general utility operations like image loading (<= this uses sequential)
    /// Default: 4 (matches DEFAULT_PARALLEL_THRESHOLD constant)
    #[serde(default = "ParallelPolicy::default_utility_threshold")]
    pub utility_threshold: usize,

    /// Threshold for postprocessing operations based on pixel area (<= this uses sequential)
    /// Default: 8000 (use sequential for regions with <= 8000 pixels, parallel for larger)
    #[serde(default = "ParallelPolicy::default_postprocess_pixel_threshold")]
    pub postprocess_pixel_threshold: usize,
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

    /// Set the postprocessing pixel threshold.
    pub fn with_postprocess_pixel_threshold(mut self, threshold: usize) -> Self {
        self.postprocess_pixel_threshold = threshold;
        self
    }

    /// Set the utility operations threshold.
    pub fn with_utility_threshold(mut self, threshold: usize) -> Self {
        self.utility_threshold = threshold;
        self
    }

    /// Install the global rayon thread pool with the configured number of threads.
    ///
    /// This method should be called once at application startup before any parallel
    /// processing occurs. If `max_threads` is None, this method does nothing and
    /// rayon will use its default thread pool size.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if the thread pool was successfully configured
    /// - `Ok(false)` if `max_threads` is None (no configuration needed)
    /// - `Err` if the thread pool has already been initialized
    ///
    /// # Example
    ///
    /// ```ignore
    /// let policy = ParallelPolicy::new().with_max_threads(Some(4));
    /// policy.install_global_thread_pool().expect("Failed to configure thread pool");
    /// ```
    pub fn install_global_thread_pool(&self) -> Result<bool, rayon::ThreadPoolBuildError> {
        if let Some(num_threads) = self.max_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()?;
            Ok(true)
        } else {
            Ok(false)
        }
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
            utility_threshold: Self::default_utility_threshold(),
            postprocess_pixel_threshold: Self::default_postprocess_pixel_threshold(),
        }
    }
}
