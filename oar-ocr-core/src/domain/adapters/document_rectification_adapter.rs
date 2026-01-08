//! UVDoc rectifier adapter implementation.
//!
//! This adapter uses the UVDoc model and adapts its output to the DocumentRectification task format.

use crate::apply_ort_config;
use crate::core::OCRError;
use crate::core::traits::adapter::{AdapterInfo, ModelAdapter};
use crate::core::traits::task::Task;
use crate::domain::tasks::document_rectification::{
    DocumentRectificationConfig, DocumentRectificationOutput, DocumentRectificationTask,
};
use crate::impl_adapter_builder;
use crate::models::rectification::uvdoc::{UVDocModel, UVDocModelBuilder, UVDocPreprocessConfig};

/// UVDoc rectifier adapter that uses the UVDoc model.
#[derive(Debug)]
pub struct UVDocRectifierAdapter {
    /// The underlying UVDoc model
    model: UVDocModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration (stored for potential future use)
    _config: DocumentRectificationConfig,
}

impl ModelAdapter for UVDocRectifierAdapter {
    type Task = DocumentRectificationTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        _config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let batch_len = input.images.len();
        // Use the UVDoc model to rectify images
        let model_output = self.model.forward(input.images).map_err(|e| {
            OCRError::adapter_execution_error(
                "UVDocRectifierAdapter",
                format!("model forward (batch_size={})", batch_len),
                e,
            )
        })?;

        // Adapt model output to task output
        Ok(DocumentRectificationOutput {
            rectified_images: model_output.images,
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        // Document rectification is computationally intensive
        // Use smaller batch size for better memory management
        8
    }
}

impl_adapter_builder! {
    builder_name: UVDocRectifierAdapterBuilder,
    adapter_name: UVDocRectifierAdapter,
    config_type: DocumentRectificationConfig,
    adapter_type: "uvdoc_rectifier",
    adapter_desc: "Corrects geometric distortions in document images",
    task_type: DocumentRectification,

    fields: {
        preprocess_config: UVDocPreprocessConfig = UVDocPreprocessConfig::default(),
        model_name_override: Option<String> = None,
    },

    methods: {
        /// Sets a custom model name for registry registration.
        pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
            self.model_name_override = Some(model_name.into());
            self
        }
    }

    overrides: {
        with_config: |builder: UVDocRectifierAdapterBuilder, config: DocumentRectificationConfig| -> UVDocRectifierAdapterBuilder {
            // Create a new builder with updated preprocess_config
            let mut result = builder;
            result.preprocess_config.rec_image_shape = config.rec_image_shape;
            result.config = result.config.with_task_config(config);
            result
        },
    }

    build: |builder: UVDocRectifierAdapterBuilder, model_path: &std::path::Path| -> Result<UVDocRectifierAdapter, OCRError> {
        let (task_config, ort_config) = builder.config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the UVDoc model
        let model = apply_ort_config!(
            UVDocModelBuilder::new().preprocess_config(builder.preprocess_config),
            ort_config
        )
        .build(model_path)?;

        // Create adapter info using the helper
        let mut info = UVDocRectifierAdapterBuilder::base_adapter_info();
        if let Some(model_name) = builder.model_name_override {
            info.model_name = model_name;
        }

        Ok(UVDocRectifierAdapter {
            model,
            info,
            _config: task_config,
        })
    },
}

// Custom impl for input_shape that updates both preprocess_config and config
impl UVDocRectifierAdapterBuilder {
    /// Sets the input shape for the rectification model.
    ///
    /// # Arguments
    ///
    /// * `shape` - Input shape as [channels, height, width]
    pub fn input_shape(mut self, shape: [usize; 3]) -> Self {
        self.preprocess_config.rec_image_shape = shape;
        // Also update the config to match the test expectation
        let task_config = DocumentRectificationConfig {
            rec_image_shape: shape,
        };
        self.config = self.config.with_task_config(task_config);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::adapter::AdapterBuilder;

    #[test]
    fn test_builder_creation() {
        let builder = UVDocRectifierAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "uvdoc_rectifier");
    }

    #[test]
    fn test_builder_with_config() {
        let config = DocumentRectificationConfig {
            rec_image_shape: [3, 1024, 1024],
        };

        let builder = UVDocRectifierAdapterBuilder::new().with_config(config.clone());
        assert_eq!(
            builder.config.task_config().rec_image_shape,
            [3, 1024, 1024]
        );
        assert_eq!(builder.preprocess_config.rec_image_shape, [3, 1024, 1024]);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = UVDocRectifierAdapterBuilder::new().input_shape([3, 768, 768]);

        assert_eq!(builder.config.task_config().rec_image_shape, [3, 768, 768]);
        assert_eq!(builder.preprocess_config.rec_image_shape, [3, 768, 768]);
    }

    #[test]
    fn test_default_builder() {
        let builder = UVDocRectifierAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "uvdoc_rectifier");
        assert_eq!(builder.config.task_config().rec_image_shape, [3, 0, 0]);
    }
}
