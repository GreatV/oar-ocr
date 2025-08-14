//! Helper utilities for stage processors to reduce code duplication.
//!
//! This module provides concrete implementations of common patterns found
//! across stage processors, including timing, metrics collection, and
//! batch processing coordination.
//!
//! ## Problem Statement
//!
//! The original stage processors (orientation, cropping, recognition) exhibited
//! significant code repetition following a common lifecycle pattern:
//!
//! 1. **Start timer** - `let start_time = Instant::now();`
//! 2. **Branch for empty input** - Check if input collections are empty and return early
//! 3. **Process items** - Handle items individually or in parallel based on thresholds
//! 4. **Accumulate metrics** - Track success/failure counts and processing time
//! 5. **Wrap results** - Create `StageResult` with data and metrics
//!
//! This repetition accounted for approximately 25-35% of the code in each processor.
//!
//! ## Solution
//!
//! This module provides three main abstractions:
//!
//! - [`process_items`] - Generic utility function for the most common batch processing patterns
//! - [`SingleItemProcessor`] - For processing individual items with automatic timing and metrics
//! - [`BatchProcessor`] - For processing collections with automatic parallel/sequential decisions
//!
//! ## Benefits
//!
//! - **Reduced code duplication** - Common patterns are centralized
//! - **Consistent metrics** - All processors use the same metrics format
//! - **Automatic parallelization** - Batch processing decisions are handled automatically
//! - **Error handling** - Standardized error handling with fallback support
//! - **Maintainability** - Changes to common patterns only need to be made in one place
//!
//! ## Usage Examples
//!
//! ### Single Item Processing
//!
//! ```rust
//! use oar_ocr::pipeline::stages::{SingleItemProcessor, StageResult};
//! use oar_ocr::processors::{BoundingBox, Point};
//!
//! fn process_text_detection() -> StageResult<Vec<BoundingBox>> {
//!     let processor = SingleItemProcessor::new("text_detection");
//!
//!     // Simulate text detection work...
//!     let detected_boxes = vec![
//!         BoundingBox::new(vec![
//!             Point::new(10.0, 10.0), Point::new(100.0, 10.0),
//!             Point::new(100.0, 30.0), Point::new(10.0, 30.0)
//!         ]),
//!         BoundingBox::new(vec![
//!             Point::new(10.0, 40.0), Point::new(150.0, 40.0),
//!             Point::new(150.0, 60.0), Point::new(10.0, 60.0)
//!         ]),
//!     ];
//!
//!     let additional_info = vec![("boxes_detected", detected_boxes.len().to_string())];
//!     processor.complete_with_info(detected_boxes, true, additional_info)
//! }
//! ```
//!
//! ### Batch Processing
//!
//! ```rust
//! use oar_ocr::pipeline::stages::{BatchConfig, BatchProcessor, StageResult};
//! use oar_ocr::core::OCRError;
//! use image::RgbImage;
//! use std::sync::Arc;
//!
//! fn process_image_batch(images: Vec<RgbImage>) -> Result<StageResult<Vec<Arc<str>>>, OCRError> {
//!     let config = BatchConfig::new("text_recognition")
//!         .with_fallback_results(true);
//!
//!     let processor = BatchProcessor::new(&config);
//!
//!     processor.process_items_with_policy(
//!         images,
//!         |image| {
//!             // Simulate text recognition on a single image
//!             let recognized_text = format!("Text from image {}x{}", image.width(), image.height());
//!             Ok(Arc::from(recognized_text))
//!         },
//!         |_error, index| Some(Arc::from(format!("Failed to process image {}", index))),
//!         Some(5), // Parallel threshold
//!     )
//! }
//! ```

use rayon::prelude::*;
use std::time::Instant;
use tracing::{debug, warn};

use super::types::{StageMetrics, StageResult};
use crate::core::OCRError;
use crate::metrics;

/// Trait for stage algorithms that can be run with automatic timing and metrics.
///
/// This trait provides a clean interface for implementing stage processing logic
/// while delegating the common concerns (timing, metrics, error handling) to
/// the generic stage harness.
///
/// # Example
///
/// ```rust,ignore
/// struct TextRecognitionAlgorithm {
///     predictor: TextRecPredictor,
/// }
///
/// impl StageAlgorithm<RgbImage, String> for TextRecognitionAlgorithm {
///     fn run(&self, input: RgbImage) -> Result<String, OCRError> {
///         // Implement the core recognition logic
///         self.predictor.predict(vec![input], None)
///             .map(|results| results.into_iter().next().unwrap_or_default())
///     }
/// }
///
/// // Use with the generic harness
/// let algorithm = TextRecognitionAlgorithm { predictor };
/// let result = run_with_metrics("text_recognition", images, &algorithm, None)?;
/// ```
pub trait StageAlgorithm<Input, Output> {
    /// Run the algorithm on a single input item.
    ///
    /// This method should contain the core processing logic without worrying
    /// about timing, metrics, or error handling patterns.
    fn run(&self, input: Input) -> Result<Output, OCRError>;
}

/// Generic stage harness that runs an algorithm with automatic timing and metrics.
///
/// This function provides a standardized way to run stage algorithms with:
/// - Automatic timing
/// - Metrics collection
/// - Error handling with optional fallback
/// - Parallel/sequential processing decisions
///
/// # Arguments
///
/// * `stage_name` - Name of the processing stage for metrics
/// * `inputs` - Collection of inputs to process
/// * `algorithm` - The algorithm implementation
/// * `parallel_threshold` - Threshold for switching to parallel processing
///
/// # Returns
///
/// A `StageResult` containing the processed outputs and metrics
pub fn run_with_metrics<I, O, A>(
    stage_name: &str,
    inputs: Vec<I>,
    algorithm: &A,
    parallel_threshold: Option<usize>,
) -> Result<StageResult<Vec<O>>, OCRError>
where
    I: Send,
    O: Send,
    A: StageAlgorithm<I, O> + Sync,
{
    process_items(
        stage_name,
        inputs,
        |input| algorithm.run(input),
        |_error, _index| None, // Default: skip failed items
        parallel_threshold,
    )
}

/// Generic stage harness with custom error handling.
///
/// This variant allows custom error handling strategies while still providing
/// the automatic timing and metrics collection.
///
/// # Arguments
///
/// * `stage_name` - Name of the processing stage for metrics
/// * `inputs` - Collection of inputs to process
/// * `algorithm` - The algorithm implementation
/// * `error_handler` - Custom error handling function
/// * `parallel_threshold` - Threshold for switching to parallel processing
///
/// # Returns
///
/// A `StageResult` containing the processed outputs and metrics
pub fn run_with_metrics_and_fallback<I, O, A, E>(
    stage_name: &str,
    inputs: Vec<I>,
    algorithm: &A,
    error_handler: E,
    parallel_threshold: Option<usize>,
) -> Result<StageResult<Vec<O>>, OCRError>
where
    I: Send,
    O: Send,
    A: StageAlgorithm<I, O> + Sync,
    E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
{
    process_items(
        stage_name,
        inputs,
        |input| algorithm.run(input),
        error_handler,
        parallel_threshold,
    )
}

// Shared helper to aggregate outputs and success/failure counts from (Option<O>, bool) pairs.
// The bool indicates success (true) or failure (false).
fn collect_outputs_and_counts<O>(results: Vec<(Option<O>, bool)>) -> (Vec<O>, usize, usize)
where
    O: Send,
{
    let mut outputs = Vec::new();
    let mut success_count = 0;
    let mut failure_count = 0;

    for (maybe_output, success) in results {
        if let Some(output) = maybe_output {
            outputs.push(output);
        }
        if success {
            success_count += 1;
        } else {
            failure_count += 1;
        }
    }

    (outputs, success_count, failure_count)
}

/// Generic utility function for processing collections with common patterns.
///
/// This function encapsulates the most common batch processing pattern found across
/// stage processors:
/// 1. Empty input fast path
/// 2. Sequential vs parallel processing decision
/// 3. Success/failure accumulation
/// 4. Automatic metrics collection
///
/// # Arguments
///
/// * `stage_name` - Name of the processing stage for metrics
/// * `items` - Collection of items to process
/// * `processor` - Function to process each item
/// * `error_handler` - Function to handle errors (returns None to skip, Some(value) to include fallback)
/// * `parallel_threshold` - Threshold for switching to parallel processing (default: 10)
///
/// # Returns
///
/// A `StageResult` containing the processed items and metrics
///
/// # Example
///
/// ```rust,ignore
/// let result = process_items(
///     "text_recognition",
///     images,
///     |image| recognize_text(image),
///     |_error, _index| None, // Skip failed items
///     Some(5), // Use parallel processing for 5+ items
/// )?;
/// ```
pub fn process_items<I, O, F, E>(
    stage_name: &str,
    items: Vec<I>,
    processor: F,
    error_handler: E,
    parallel_threshold: Option<usize>,
) -> Result<StageResult<Vec<O>>, OCRError>
where
    I: Send,
    O: Send,
    F: Fn(I) -> Result<O, OCRError> + Send + Sync,
    E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
{
    let start_time = Instant::now();

    // Fast path for empty input
    if items.is_empty() {
        let metrics = metrics!(0, 0, start_time;
            stage = stage_name,
            total_items = 0
        );
        return Ok(StageResult::new(Vec::new(), metrics));
    }

    let total_items = items.len();
    let threshold = parallel_threshold.unwrap_or(10);
    let use_parallel = total_items > threshold;

    debug!(
        "Processing {} items for stage '{}' (parallel: {})",
        total_items, stage_name, use_parallel
    );

    let mapped_results: Vec<(Option<O>, bool)> = if use_parallel {
        items
            .into_par_iter()
            .enumerate()
            .map(|(index, item)| match processor(item) {
                Ok(output) => (Some(output), true),
                Err(error) => (error_handler(error, index), false),
            })
            .collect()
    } else {
        items
            .into_iter()
            .enumerate()
            .map(|(index, item)| match processor(item) {
                Ok(output) => (Some(output), true),
                Err(error) => (error_handler(error, index), false),
            })
            .collect()
    };

    let (results, success_count, failure_count) = collect_outputs_and_counts(mapped_results);

    let metrics = metrics!(success_count, failure_count, start_time;
        stage = stage_name,
        total_items = total_items,
        parallel_processing = use_parallel
    );

    Ok(StageResult::new(results, metrics))
}

/// Configuration for batch processing behavior
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Whether to include failed items as fallback results
    pub include_fallback_results: bool,
    /// Stage name for metrics and logging
    pub stage_name: String,
}

impl BatchConfig {
    /// Create a new batch configuration
    pub fn new(stage_name: impl Into<String>) -> Self {
        Self {
            include_fallback_results: false,
            stage_name: stage_name.into(),
        }
    }

    /// Enable fallback results for failed items
    pub fn with_fallback_results(mut self, include: bool) -> Self {
        self.include_fallback_results = include;
        self
    }

    /// Get the effective parallel threshold from the provided policy
    pub fn effective_threshold(&self, policy_threshold: Option<usize>) -> usize {
        policy_threshold.unwrap_or(10) // Default threshold
    }
}

/// Helper for processing collections with common lifecycle patterns
pub struct BatchProcessor<'a> {
    config: &'a BatchConfig,
    start_time: Instant,
}

impl<'a> BatchProcessor<'a> {
    /// Create a new batch processor and start timing
    pub fn new(config: &'a BatchConfig) -> Self {
        debug!("Starting batch processing for stage: {}", config.stage_name);
        Self {
            config,
            start_time: Instant::now(),
        }
    }

    /// Process a collection of items with automatic parallel/sequential decision
    pub fn process_items<I, O, F, E>(
        self,
        items: Vec<I>,
        processor: F,
        error_handler: E,
    ) -> Result<StageResult<Vec<O>>, OCRError>
    where
        I: Send,
        O: Send,
        F: Fn(I) -> Result<O, OCRError> + Send + Sync,
        E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
    {
        self.process_items_with_policy(items, processor, error_handler, None)
    }

    /// Process a collection of items with automatic parallel/sequential decision using policy threshold
    pub fn process_items_with_policy<I, O, F, E>(
        self,
        items: Vec<I>,
        processor: F,
        error_handler: E,
        policy_threshold: Option<usize>,
    ) -> Result<StageResult<Vec<O>>, OCRError>
    where
        I: Send,
        O: Send,
        F: Fn(I) -> Result<O, OCRError> + Send + Sync,
        E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
    {
        if items.is_empty() {
            return Ok(self.empty_result());
        }

        let total_items = items.len();
        let effective_threshold = self.config.effective_threshold(policy_threshold);
        let use_parallel = total_items > effective_threshold;

        debug!(
            "Processing {} items for stage '{}' (parallel: {})",
            total_items, self.config.stage_name, use_parallel
        );

        let (results, success_count, failure_count) = if use_parallel {
            self.process_parallel(items, processor, error_handler)
        } else {
            self.process_sequential(items, processor, error_handler)
        };

        let metrics = metrics!(success_count, failure_count, self.start_time;
            stage = &self.config.stage_name,
            total_items = total_items,
            parallel_processing = use_parallel
        );

        Ok(StageResult::new(results, metrics))
    }

    /// Process items in parallel
    fn process_parallel<I, O, F, E>(
        &self,
        items: Vec<I>,
        processor: F,
        error_handler: E,
    ) -> (Vec<O>, usize, usize)
    where
        I: Send,
        O: Send,
        F: Fn(I) -> Result<O, OCRError> + Send + Sync,
        E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
    {
        let processed_results: Vec<(usize, Result<O, OCRError>)> = items
            .into_par_iter()
            .enumerate()
            .map(|(index, item)| (index, processor(item)))
            .collect();

        self.collect_results(processed_results, error_handler)
    }

    /// Process items sequentially
    fn process_sequential<I, O, F, E>(
        &self,
        items: Vec<I>,
        processor: F,
        error_handler: E,
    ) -> (Vec<O>, usize, usize)
    where
        I: Send,
        O: Send,
        F: Fn(I) -> Result<O, OCRError> + Send + Sync,
        E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
    {
        let processed_results: Vec<(usize, Result<O, OCRError>)> = items
            .into_iter()
            .enumerate()
            .map(|(index, item)| (index, processor(item)))
            .collect();

        self.collect_results(processed_results, error_handler)
    }

    /// Collect results and handle errors
    fn collect_results<O, E>(
        &self,
        processed_results: Vec<(usize, Result<O, OCRError>)>,
        error_handler: E,
    ) -> (Vec<O>, usize, usize)
    where
        O: Send,
        E: Fn(OCRError, usize) -> Option<O> + Send + Sync,
    {
        let mut results = Vec::new();
        let mut success_count = 0;
        let mut failure_count = 0;

        for (index, result) in processed_results {
            match result {
                Ok(output) => {
                    results.push(output);
                    success_count += 1;
                }
                Err(e) => {
                    warn!(
                        "Processing failed for item {} in stage '{}': {}",
                        index, self.config.stage_name, e
                    );

                    if let Some(fallback) = error_handler(e, index) {
                        results.push(fallback);
                    }
                    failure_count += 1;
                }
            }
        }

        (results, success_count, failure_count)
    }

    /// Create an empty result for when no items are provided
    fn empty_result<O>(&self) -> StageResult<Vec<O>> {
        let metrics = metrics!(0, 0, self.start_time;
            stage = &self.config.stage_name,
            total_items = 0
        );

        StageResult::new(Vec::new(), metrics)
    }
}

/// Helper for single item processing with timing and metrics
pub struct SingleItemProcessor {
    stage_name: String,
    start_time: Instant,
}

impl SingleItemProcessor {
    /// Create a new single item processor and start timing
    pub fn new(stage_name: impl Into<String>) -> Self {
        Self {
            stage_name: stage_name.into(),
            start_time: Instant::now(),
        }
    }

    /// Complete processing and create a result with metrics
    pub fn complete<T>(self, data: T, success: bool) -> StageResult<T> {
        let (success_count, failure_count) = if success { (1, 0) } else { (0, 1) };

        let metrics = metrics!(success_count, failure_count, self.start_time;
            stage = &self.stage_name
        );

        StageResult::new(data, metrics)
    }

    /// Complete processing with additional metrics information
    pub fn complete_with_info<T>(
        self,
        data: T,
        success: bool,
        additional_info: Vec<(&str, String)>,
    ) -> StageResult<T> {
        let (success_count, failure_count) = if success { (1, 0) } else { (0, 1) };

        let mut metrics = StageMetrics::new(success_count, failure_count)
            .with_processing_time(self.start_time.elapsed())
            .with_info("stage", &self.stage_name);

        for (key, value) in additional_info {
            metrics = metrics.with_info(key, value);
        }

        StageResult::new(data, metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new("test_stage").with_fallback_results(true);

        assert_eq!(config.effective_threshold(None), 10);
        assert!(config.include_fallback_results);
        assert_eq!(config.stage_name, "test_stage");
    }

    #[test]
    fn test_batch_config_effective_threshold() {
        let config = BatchConfig::new("test_stage");

        // Test with policy threshold
        assert_eq!(config.effective_threshold(Some(8)), 8);

        // Test without policy threshold (uses default)
        assert_eq!(config.effective_threshold(None), 10);
    }

    #[test]
    fn test_batch_processor_empty_items() {
        let config = BatchConfig::new("test_stage");
        let processor = BatchProcessor::new(&config);

        let result = processor
            .process_items(Vec::<i32>::new(), |x| Ok(x * 2), |_err, _idx| Some(0))
            .unwrap();

        assert!(result.data.is_empty());
        assert_eq!(result.metrics.success_count, 0);
        assert_eq!(result.metrics.failure_count, 0);
    }

    #[test]
    fn test_batch_processor_with_policy() {
        let config = BatchConfig::new("test_stage");
        let processor = BatchProcessor::new(&config);

        let items = vec![1, 2, 3];
        let result = processor
            .process_items_with_policy(
                items,
                |x| Ok(x * 2),
                |_err, _idx| Some(0),
                Some(2), // Policy threshold
            )
            .unwrap();

        assert_eq!(result.data, vec![2, 4, 6]);
        assert_eq!(result.metrics.success_count, 3);
        assert_eq!(result.metrics.failure_count, 0);
    }

    #[test]
    fn test_generic_process_items() {
        let items = vec![1, 2, 3, 4, 5];

        // Test successful processing
        let result = process_items(
            "test_stage",
            items,
            |x| Ok(x * 2),
            |_err, _idx| None, // Skip failed items
            Some(3),           // Use parallel for 3+ items
        )
        .unwrap();

        assert_eq!(result.data, vec![2, 4, 6, 8, 10]);
        assert_eq!(result.metrics.success_count, 5);
        assert_eq!(result.metrics.failure_count, 0);
        assert_eq!(
            result.metrics.additional_info.get("stage"),
            Some(&"test_stage".to_string())
        );
        assert_eq!(
            result.metrics.additional_info.get("parallel_processing"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn test_generic_process_items_with_errors() {
        let items = vec![1, 2, 3, 4, 5];

        // Test processing with some failures
        let result = process_items(
            "test_stage",
            items,
            |x| {
                if x % 2 == 0 {
                    Err(OCRError::invalid_input("even number"))
                } else {
                    Ok(x * 2)
                }
            },
            |_err, _idx| Some(0), // Use fallback for failed items
            Some(10),             // Use sequential processing
        )
        .unwrap();

        assert_eq!(result.data, vec![2, 0, 6, 0, 10]); // Odd numbers doubled, even numbers become 0
        assert_eq!(result.metrics.success_count, 3); // 1, 3, 5 succeeded
        assert_eq!(result.metrics.failure_count, 2); // 2, 4 failed
        assert_eq!(
            result.metrics.additional_info.get("parallel_processing"),
            Some(&"false".to_string())
        );
    }

    #[test]
    fn test_generic_process_items_empty() {
        let items: Vec<i32> = vec![];

        let result =
            process_items("test_stage", items, |x| Ok(x * 2), |_err, _idx| None, None).unwrap();

        assert!(result.data.is_empty());
        assert_eq!(result.metrics.success_count, 0);
        assert_eq!(result.metrics.failure_count, 0);
        assert_eq!(
            result.metrics.additional_info.get("total_items"),
            Some(&"0".to_string())
        );
    }

    // Test implementation of StageAlgorithm for testing
    struct TestAlgorithm {
        multiplier: i32,
    }

    impl StageAlgorithm<i32, i32> for TestAlgorithm {
        fn run(&self, input: i32) -> Result<i32, OCRError> {
            if input < 0 {
                Err(OCRError::invalid_input("negative numbers not allowed"))
            } else {
                Ok(input * self.multiplier)
            }
        }
    }

    #[test]
    fn test_stage_algorithm_harness() {
        let algorithm = TestAlgorithm { multiplier: 3 };
        let inputs = vec![1, 2, 3, 4, 5];

        let result = run_with_metrics(
            "test_algorithm",
            inputs,
            &algorithm,
            Some(3), // Use parallel for 3+ items
        )
        .unwrap();

        assert_eq!(result.data, vec![3, 6, 9, 12, 15]);
        assert_eq!(result.metrics.success_count, 5);
        assert_eq!(result.metrics.failure_count, 0);
        assert_eq!(
            result.metrics.additional_info.get("stage"),
            Some(&"test_algorithm".to_string())
        );
    }

    #[test]
    fn test_stage_algorithm_harness_with_errors() {
        let algorithm = TestAlgorithm { multiplier: 2 };
        let inputs = vec![-1, 2, -3, 4, 5]; // Negative numbers will fail

        let result = run_with_metrics(
            "test_algorithm",
            inputs,
            &algorithm,
            None, // Use sequential processing
        )
        .unwrap();

        assert_eq!(result.data, vec![4, 8, 10]); // Only positive numbers processed (2*2=4, 4*2=8, 5*2=10)
        assert_eq!(result.metrics.success_count, 3); // 2, 4, 5 succeeded
        assert_eq!(result.metrics.failure_count, 2); // -1, -3 failed
    }

    #[test]
    fn test_stage_algorithm_harness_with_fallback() {
        let algorithm = TestAlgorithm { multiplier: 2 };
        let inputs = vec![-1, 2, -3, 4, 5];

        let result = run_with_metrics_and_fallback(
            "test_algorithm",
            inputs,
            &algorithm,
            |_error, _index| Some(0), // Use 0 as fallback for failed items
            None,
        )
        .unwrap();

        assert_eq!(result.data, vec![0, 4, 0, 8, 10]); // Failed items become 0
        assert_eq!(result.metrics.success_count, 3);
        assert_eq!(result.metrics.failure_count, 2);
    }
}
