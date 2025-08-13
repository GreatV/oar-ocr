//! Statistics management for the OAROCR pipeline.

use crate::core::PipelineStats;
use std::sync::Mutex;

/// Statistics management functionality for the OAROCR pipeline.
pub struct StatsManager {
    /// Statistics for the pipeline execution (thread-safe).
    stats: Mutex<PipelineStats>,
}

impl StatsManager {
    /// Creates a new StatsManager instance.
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(PipelineStats::default()),
        }
    }

    /// Gets a copy of the current pipeline statistics.
    ///
    /// # Returns
    ///
    /// A copy of the current PipelineStats
    pub fn get_stats(&self) -> PipelineStats {
        self.stats.lock().unwrap().clone()
    }

    /// Updates the pipeline statistics with new processing results.
    ///
    /// # Arguments
    ///
    /// * `processed_count` - Number of images processed in this batch
    /// * `successful_count` - Number of images successfully processed
    /// * `failed_count` - Number of images that failed processing
    /// * `inference_time_ms` - Total inference time in milliseconds
    pub fn update_stats(
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

        // Update average inference time using incremental formula
        // For batch processing, we need to account for multiple images
        let new_count = stats.total_processed;
        if new_count > 0 {
            let old_count = new_count - processed_count;
            let old_total_time = stats.average_inference_time_ms * old_count as f64;
            let new_total_time = old_total_time + inference_time_ms;
            stats.average_inference_time_ms = new_total_time / new_count as f64;
        }
    }

    /// Resets the pipeline statistics.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = PipelineStats::default();
    }
}

impl Default for StatsManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_update_incremental_average() {
        // Test that the stats update formula correctly calculates incremental averages
        let stats_manager = StatsManager::new();

        // First update: 1 image, 100ms
        stats_manager.update_stats(1, 1, 0, 100.0);
        let stats = stats_manager.get_stats();
        assert_eq!(stats.total_processed, 1);
        assert_eq!(stats.successful_predictions, 1);
        assert_eq!(stats.failed_predictions, 0);
        assert_eq!(stats.average_inference_time_ms, 100.0);

        // Second update: 1 more image, 200ms
        stats_manager.update_stats(1, 1, 0, 200.0);
        let stats = stats_manager.get_stats();
        assert_eq!(stats.total_processed, 2);
        assert_eq!(stats.successful_predictions, 2);
        assert_eq!(stats.failed_predictions, 0);
        assert_eq!(stats.average_inference_time_ms, 150.0); // (100 + 200) / 2

        // Third update: 2 more images, 50ms total (25ms average)
        stats_manager.update_stats(2, 1, 1, 50.0);
        let stats = stats_manager.get_stats();
        assert_eq!(stats.total_processed, 4);
        assert_eq!(stats.successful_predictions, 3);
        assert_eq!(stats.failed_predictions, 1);
        // Average should be (100 + 200 + 50) / 4 = 87.5
        assert_eq!(stats.average_inference_time_ms, 87.5);
    }

    #[test]
    fn test_stats_reset() {
        let stats_manager = StatsManager::new();

        // Add some stats
        stats_manager.update_stats(5, 4, 1, 500.0);
        let stats = stats_manager.get_stats();
        assert_eq!(stats.total_processed, 5);

        // Reset stats
        stats_manager.reset_stats();
        let stats = stats_manager.get_stats();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.successful_predictions, 0);
        assert_eq!(stats.failed_predictions, 0);
        assert_eq!(stats.average_inference_time_ms, 0.0);
    }
}
