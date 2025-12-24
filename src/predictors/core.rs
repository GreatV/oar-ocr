//! Core predictor functionality
//!
//! This module provides a generic predictor implementation that can be reused
//! across all task-specific predictors, eliminating boilerplate code.

use crate::core::traits::adapter::ModelAdapter;
use crate::core::traits::task::Task;

/// Generic task predictor core.
///
/// This struct encapsulates the common pattern used across all predictors:
/// holding an adapter, task instance, and configuration, and executing
/// predictions through the adapter with proper validation.
///
/// # Type Parameters
///
/// * `T` - The task type that implements the `Task` trait
pub struct TaskPredictorCore<T: Task> {
    /// The model adapter for this task
    pub(crate) adapter: Box<dyn ModelAdapter<Task = T>>,
    /// The task instance for validation
    pub(crate) task: T,
    /// The task configuration
    pub(crate) config: T::Config,
}

impl<T: Task> TaskPredictorCore<T> {
    /// Creates a new task predictor core.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The model adapter to use for predictions
    /// * `task` - The task instance for validation
    /// * `config` - The task configuration
    pub fn new(adapter: Box<dyn ModelAdapter<Task = T>>, task: T, config: T::Config) -> Self {
        Self {
            adapter,
            task,
            config,
        }
    }

    /// Executes prediction on the given input.
    ///
    /// This method performs the complete validation and execution pipeline:
    /// 1. Validate input using the task's validation logic
    /// 2. Execute inference through the adapter
    /// 3. Validate output using the task's validation logic
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for prediction
    ///
    /// # Returns
    ///
    /// The task output on success, or an error if validation or execution fails.
    pub fn predict(&self, input: T::Input) -> Result<T::Output, Box<dyn std::error::Error>> {
        // 1. Validate input
        self.task.validate_input(&input)?;

        // 2. Execute prediction through the adapter
        let output = self.adapter.execute(input, Some(&self.config))?;

        // 3. Validate output
        self.task.validate_output(&output)?;

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

    /// Returns a reference to the task instance.
    pub fn task(&self) -> &T {
        &self.task
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
