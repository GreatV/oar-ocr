//! Shared types and traits for pipeline stage processors.

use crate::core::OCRError;
use std::time::{Duration, Instant};

/// Result wrapper for stage processing operations.
///
/// This provides a consistent interface for all stage processors,
/// including the processed data and associated metadata.
#[derive(Debug, Clone)]
pub struct StageResult<T> {
    /// The processed data from the stage
    pub data: T,
    /// Performance and error metrics for the stage
    pub metrics: StageMetrics,
}

impl<T> StageResult<T> {
    /// Create a new stage result with the given data and metrics
    pub fn new(data: T, metrics: StageMetrics) -> Self {
        Self { data, metrics }
    }

    /// Create a stage result with default metrics
    pub fn with_data(data: T) -> Self {
        Self {
            data,
            metrics: StageMetrics::default(),
        }
    }
}

/// Metrics collected during stage processing.
///
/// This provides consistent performance and error tracking
/// across all pipeline stages.
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    /// Time taken to process the stage
    pub processing_time: Option<Duration>,
    /// Number of items successfully processed
    pub success_count: usize,
    /// Number of items that failed processing
    pub failure_count: usize,
    /// Additional stage-specific metrics
    pub additional_info: std::collections::HashMap<String, String>,
}

impl StageMetrics {
    /// Create new metrics with the given counts
    pub fn new(success_count: usize, failure_count: usize) -> Self {
        Self {
            processing_time: None,
            success_count,
            failure_count,
            additional_info: std::collections::HashMap::new(),
        }
    }

    /// Set the processing time
    pub fn with_processing_time(mut self, duration: Duration) -> Self {
        self.processing_time = Some(duration);
        self
    }

    /// Add additional information to the metrics
    pub fn with_info<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }

    /// Get the total number of items processed
    pub fn total_count(&self) -> usize {
        self.success_count + self.failure_count
    }

    /// Get the success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        let total = self.total_count();
        if total == 0 {
            0.0
        } else {
            (self.success_count as f64 / total as f64) * 100.0
        }
    }
}

/// Trait for stage processors that follow the common lifecycle pattern.
///
/// This trait provides a unified interface for stage processors that need to:
/// - Start timing operations
/// - Handle empty input collections
/// - Process items (potentially in parallel)
/// - Accumulate metrics
/// - Wrap results
///
/// ## Design Note
///
/// This trait was designed to capture the common patterns found across
/// orientation, cropping, and recognition stage processors. However, in practice,
/// the concrete helper implementations ([`crate::pipeline::stages::processor_helper`])
/// proved more flexible and easier to use than this trait-based approach.
///
/// The trait remains for potential future use cases where a more formal
/// interface is needed, but the helper utilities are recommended for most
/// stage processor implementations.
pub trait StageProcessor<Input, Output: Default, Config = ()>
where
    Config: Sync,
{
    /// The name of the stage for metrics and logging
    fn stage_name(&self) -> &'static str;

    /// Process a single item
    fn process_item(&self, input: Input, config: Option<&Config>) -> Result<Output, OCRError>;

    /// Check if the input collection is empty and should be handled specially
    fn is_empty_input<I>(&self, input: &[I]) -> bool {
        input.is_empty()
    }

    /// Create a result for empty input
    fn empty_result(&self, start_time: Instant) -> StageResult<Vec<Output>>
    where
        Output: Default,
    {
        let metrics = StageMetrics::new(0, 0)
            .with_processing_time(start_time.elapsed())
            .with_info("stage", self.stage_name())
            .with_info("items", "0");

        StageResult::new(Vec::new(), metrics)
    }

    /// Determine if parallel processing should be used
    fn should_use_parallel(&self, item_count: usize, config: Option<&Config>) -> bool {
        let _ = config;
        item_count > 10 // Default threshold
    }

    /// Process a collection of items following the common lifecycle pattern
    fn process_collection<I, F>(
        &self,
        items: Vec<I>,
        config: Option<&Config>,
        processor: F,
    ) -> Result<StageResult<Vec<Output>>, OCRError>
    where
        I: Send,
        Output: Send,
        F: Fn(I, Option<&Config>) -> Result<Output, OCRError> + Send + Sync,
    {
        let start_time = Instant::now();

        if self.is_empty_input(&items) {
            return Ok(self.empty_result(start_time));
        }

        let total_items = items.len();
        let mut results = Vec::with_capacity(total_items);
        let mut success_count = 0;
        let mut failure_count = 0;

        if self.should_use_parallel(total_items, config) {
            // Parallel processing
            use rayon::prelude::*;
            let processed_results: Vec<Result<Output, OCRError>> = items
                .into_par_iter()
                .map(|item| processor(item, config))
                .collect();

            for result in processed_results {
                match result {
                    Ok(output) => {
                        results.push(output);
                        success_count += 1;
                    }
                    Err(_e) => {
                        // Create fallback result - this should be customizable per processor
                        failure_count += 1;
                        // For now, skip failed items - processors can override this behavior
                    }
                }
            }
        } else {
            // Sequential processing
            for item in items {
                match processor(item, config) {
                    Ok(output) => {
                        results.push(output);
                        success_count += 1;
                    }
                    Err(_e) => {
                        failure_count += 1;
                        // For now, skip failed items - processors can override this behavior
                    }
                }
            }
        }

        let processing_time = start_time.elapsed();
        let metrics = StageMetrics::new(success_count, failure_count)
            .with_processing_time(processing_time)
            .with_info("stage", self.stage_name())
            .with_info("total_items", total_items.to_string())
            .with_info(
                "parallel_processing",
                self.should_use_parallel(total_items, config).to_string(),
            );

        Ok(StageResult::new(results, metrics))
    }
}

/// Helper utility for common stage processing patterns.
///
/// This struct provides reusable methods for the common lifecycle patterns
/// found across stage processors, reducing code duplication.
pub struct StageProcessorHelper {
    stage_name: &'static str,
}

impl StageProcessorHelper {
    /// Create a new stage processor helper
    pub fn new(stage_name: &'static str) -> Self {
        Self { stage_name }
    }

    /// Start timing an operation
    pub fn start_timer(&self) -> Instant {
        Instant::now()
    }

    /// Create metrics for a single item operation
    pub fn single_item_metrics(&self, start_time: Instant, success: bool) -> StageMetrics {
        let (success_count, failure_count) = if success { (1, 0) } else { (0, 1) };
        StageMetrics::new(success_count, failure_count)
            .with_processing_time(start_time.elapsed())
            .with_info("stage", self.stage_name)
    }

    /// Create metrics for batch operations
    pub fn batch_metrics(
        &self,
        start_time: Instant,
        success_count: usize,
        failure_count: usize,
        total_items: usize,
        parallel: bool,
    ) -> StageMetrics {
        StageMetrics::new(success_count, failure_count)
            .with_processing_time(start_time.elapsed())
            .with_info("stage", format!("{}_batch", self.stage_name))
            .with_info("total_items", total_items.to_string())
            .with_info("parallel_processing", parallel.to_string())
    }

    /// Wrap a result with metrics
    pub fn wrap_result<T>(&self, data: T, metrics: StageMetrics) -> StageResult<T> {
        StageResult::new(data, metrics)
    }
}
