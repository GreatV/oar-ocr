//! Pipeline-wide statistics helpers.
//!
//! This module defines the `PipelineStats` structure used to track execution metrics
//! for OCR pipeline runs and the `StatsManager` helper that coordinates thread-safe
//! updates to these metrics.

use std::fmt;
use std::sync::Mutex;

/// Statistics for the OCR pipeline.
///
/// Tracks how many images were processed and performance metrics such as average
/// inference time and success ratios.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// The total number of images processed.
    pub total_processed: usize,
    /// The number of successful predictions.
    pub successful_predictions: usize,
    /// The number of failed predictions.
    pub failed_predictions: usize,
    /// The average inference time in milliseconds.
    pub average_inference_time_ms: f64,
}

impl PipelineStats {
    /// Creates a new PipelineStats instance with default values.
    pub fn new() -> Self {
        Self {
            total_processed: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            average_inference_time_ms: 0.0,
        }
    }

    /// Returns the success rate as a percentage (0.0 to 100.0).
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            0.0
        } else {
            (self.successful_predictions as f64 / self.total_processed as f64) * 100.0
        }
    }

    /// Returns the failure rate as a percentage (0.0 to 100.0).
    pub fn failure_rate(&self) -> f64 {
        if self.total_processed == 0 {
            0.0
        } else {
            (self.failed_predictions as f64 / self.total_processed as f64) * 100.0
        }
    }

    /// Returns the average processing speed in images per second.
    pub fn images_per_second(&self) -> f64 {
        if self.average_inference_time_ms == 0.0 {
            0.0
        } else {
            1000.0 / self.average_inference_time_ms
        }
    }
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Statistics:")?;
        writeln!(f, "  Total processed: {}", self.total_processed)?;
        writeln!(
            f,
            "  Successful: {} ({:.1}%)",
            self.successful_predictions,
            self.success_rate()
        )?;
        writeln!(
            f,
            "  Failed: {} ({:.1}%)",
            self.failed_predictions,
            self.failure_rate()
        )?;
        writeln!(
            f,
            "  Average inference time: {:.2} ms",
            self.average_inference_time_ms
        )?;
        writeln!(
            f,
            "  Processing speed: {:.2} images/sec",
            self.images_per_second()
        )?;
        Ok(())
    }
}

/// Thread-safe manager for updating pipeline statistics during OCR execution.
#[derive(Debug, Default)]
pub struct StatsManager {
    /// Shared statistics state guarded by a mutex.
    stats: Mutex<PipelineStats>,
}

impl StatsManager {
    /// Creates a new `StatsManager` instance with zeroed metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of the current statistics snapshot.
    pub fn get_stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }

    /// Updates the tracked metrics using the results from a batch run.
    pub fn update_stats(
        &self,
        processed_count: usize,
        successful_count: usize,
        failed_count: usize,
        inference_time_ms: f64,
    ) {
        let mut stats = self.stats.lock().unwrap();

        let previous_total = stats.total_processed;
        let previous_average = stats.average_inference_time_ms;
        let new_total = previous_total + processed_count;

        stats.total_processed = new_total;
        stats.successful_predictions += successful_count;
        stats.failed_predictions += failed_count;

        if new_total > 0 {
            let accumulated_time = previous_average * previous_total as f64;
            let new_total_time = accumulated_time + inference_time_ms;
            stats.average_inference_time_ms = new_total_time / new_total as f64;
        } else {
            stats.average_inference_time_ms = 0.0;
        }
    }

    /// Resets the tracked statistics to their default state.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = PipelineStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::{PipelineStats, StatsManager};

    #[test]
    fn success_rate_handles_zero_processed() {
        let stats = PipelineStats::default();
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn success_rate_computes_percentage() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 7,
            failed_predictions: 3,
            average_inference_time_ms: 50.0,
        };
        assert_eq!(stats.success_rate(), 70.0);
    }

    #[test]
    fn failure_rate_handles_zero_processed() {
        let stats = PipelineStats::default();
        assert_eq!(stats.failure_rate(), 0.0);
    }

    #[test]
    fn failure_rate_computes_percentage() {
        let stats = PipelineStats {
            total_processed: 8,
            successful_predictions: 6,
            failed_predictions: 2,
            average_inference_time_ms: 75.0,
        };
        assert_eq!(stats.failure_rate(), 25.0);
    }

    #[test]
    fn images_per_second_handles_zero_time() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 10,
            failed_predictions: 0,
            average_inference_time_ms: 0.0,
        };
        assert_eq!(stats.images_per_second(), 0.0);
    }

    #[test]
    fn images_per_second_computes_rate() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 10,
            failed_predictions: 0,
            average_inference_time_ms: 100.0,
        };
        assert_eq!(stats.images_per_second(), 10.0);
    }

    #[test]
    fn display_formats_metrics() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 8,
            failed_predictions: 2,
            average_inference_time_ms: 125.0,
        };

        let display = stats.to_string();
        assert!(display.contains("Pipeline Statistics:"));
        assert!(display.contains("Total processed: 10"));
        assert!(display.contains("Successful: 8 (80.0%)"));
        assert!(display.contains("Failed: 2 (20.0%)"));
        assert!(display.contains("Average inference time: 125.00 ms"));
        assert!(display.contains("Processing speed: 8.00 images/sec"));
    }

    #[test]
    fn stats_manager_updates_counters_and_average() {
        let manager = StatsManager::new();

        manager.update_stats(1, 1, 0, 100.0);
        let stats = manager.get_stats();
        assert_eq!(stats.total_processed, 1);
        assert_eq!(stats.successful_predictions, 1);
        assert_eq!(stats.failed_predictions, 0);
        assert_eq!(stats.average_inference_time_ms, 100.0);

        manager.update_stats(1, 0, 1, 200.0);
        let stats = manager.get_stats();
        assert_eq!(stats.total_processed, 2);
        assert_eq!(stats.successful_predictions, 1);
        assert_eq!(stats.failed_predictions, 1);
        assert!((stats.average_inference_time_ms - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_manager_resets_metrics() {
        let manager = StatsManager::new();
        manager.update_stats(5, 4, 1, 500.0);
        manager.reset_stats();

        let stats = manager.get_stats();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.successful_predictions, 0);
        assert_eq!(stats.failed_predictions, 0);
        assert_eq!(stats.average_inference_time_ms, 0.0);
    }
}
