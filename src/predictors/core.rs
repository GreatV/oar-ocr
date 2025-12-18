//! Core predictor functionality
//!
//! This module provides a generic predictor implementation that can be reused
//! across all task-specific predictors, eliminating boilerplate code.

use crate::core::traits::adapter::ModelAdapter;
use crate::core::traits::task::Task;

/// Generic task predictor core.
///
/// This struct encapsulates the common pattern used across all predictors:
/// holding an adapter and configuration, and executing predictions through
/// the adapter with proper validation.
///
/// # Type Parameters
///
/// * `T` - The task type that implements the `Task` trait
pub struct TaskPredictorCore<T: Task> {
    /// The model adapter for this task
    pub(crate) adapter: Box<dyn ModelAdapter<Task = T>>,
    /// The task configuration
    pub(crate) config: T::Config,
}

impl<T: Task> TaskPredictorCore<T> {
    /// Creates a new task predictor core.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The model adapter to use for predictions
    /// * `config` - The task configuration
    pub fn new(adapter: Box<dyn ModelAdapter<Task = T>>, config: T::Config) -> Self {
        Self { adapter, config }
    }

    /// Executes prediction on the given input.
    ///
    /// This method delegates to the adapter's execute method, which handles
    /// validation internally. The configuration is passed to the adapter to
    /// ensure task-specific parameters are applied.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for prediction
    ///
    /// # Returns
    ///
    /// The task output on success, or an error if validation or execution fails.
    pub fn predict(&self, input: T::Input) -> Result<T::Output, Box<dyn std::error::Error>> {
        // Execute prediction through the adapter
        // The adapter handles validation and processing
        let output = self.adapter.execute(input, Some(&self.config))?;

        Ok(output)
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &T::Config {
        &self.config
    }

    /// Returns a mutable reference to the configuration.
    ///
    /// This allows modifying the configuration after the predictor is created,
    /// though creating a new predictor with a different configuration is generally
    /// preferred for clarity.
    pub fn config_mut(&mut self) -> &mut T::Config {
        &mut self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::tasks::text_detection::{TextDetectionConfig, TextDetectionTask};

    #[test]
    fn test_task_predictor_core_creation() {
        // This test just verifies the type compiles
        // We can't actually create an adapter without model files
        let _check = || -> Option<TaskPredictorCore<TextDetectionTask>> { None };
    }

    #[test]
    fn test_config_accessors() {
        // Verify config() and config_mut() compile with correct types
        let _check = || {
            let mut core: Option<TaskPredictorCore<TextDetectionTask>> = None;
            if let Some(c) = core.as_ref() {
                let _cfg: &TextDetectionConfig = c.config();
            }
            if let Some(c) = core.as_mut() {
                let _cfg: &mut TextDetectionConfig = c.config_mut();
            }
        };
    }
}
