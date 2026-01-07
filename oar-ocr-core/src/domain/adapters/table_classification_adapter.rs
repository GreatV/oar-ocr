//! Table Classification Adapter
//!
//! This adapter uses the PP-LCNet model to classify table images as wired or wireless.

use crate::apply_ort_config;
use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::domain::tasks::{
    Classification, TableClassificationConfig, TableClassificationOutput, TableClassificationTask,
};
use crate::impl_adapter_builder;
use crate::models::classification::{PPLCNetModel, PPLCNetModelBuilder, PPLCNetPostprocessConfig};
use std::path::Path;

/// Table classification adapter that uses the PP-LCNet model.
#[derive(Debug)]
pub struct TableClassificationAdapter {
    /// The underlying PP-LCNet model
    model: PPLCNetModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TableClassificationConfig,
    /// Postprocessing configuration
    postprocess_config: PPLCNetPostprocessConfig,
}

impl TableClassificationAdapter {
    /// Creates a new table classification adapter.
    pub fn new(
        model: PPLCNetModel,
        info: AdapterInfo,
        config: TableClassificationConfig,
        postprocess_config: PPLCNetPostprocessConfig,
    ) -> Self {
        Self {
            model,
            info,
            config,
            postprocess_config,
        }
    }

    /// Default input shape for table classification (224x224 as per model spec).
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (224, 224);

    /// Class labels for table classification.
    pub fn labels() -> Vec<String> {
        vec!["wired_table".to_string(), "wireless_table".to_string()]
    }
}

impl ModelAdapter for TableClassificationAdapter {
    type Task = TableClassificationTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // Update postprocess config with task-specific topk
        let mut postprocess_config = self.postprocess_config.clone();
        postprocess_config.topk = effective_config.topk;

        // Use model to get predictions with error context
        let model_output = self
            .model
            .forward(input.images, &postprocess_config)
            .map_err(|e| {
                OCRError::adapter_execution_error(
                    "TableClassificationAdapter",
                    format!(
                        "failed to classify table type (topk={})",
                        effective_config.topk
                    ),
                    e,
                )
            })?;

        // Convert model output to task-specific output with structured classifications
        let label_names = model_output.label_names.unwrap_or_else(|| {
            model_output
                .class_ids
                .iter()
                .map(|ids| {
                    ids.iter()
                        .map(|&id| {
                            if id == 0 {
                                "wired_table".to_string()
                            } else {
                                "wireless_table".to_string()
                            }
                        })
                        .collect()
                })
                .collect()
        });

        // Create structured classifications
        let classifications = model_output
            .class_ids
            .into_iter()
            .zip(model_output.scores)
            .zip(label_names)
            .map(|((class_ids, scores), labels)| {
                class_ids
                    .into_iter()
                    .zip(scores)
                    .zip(labels)
                    .map(|((class_id, score), label)| Classification::new(class_id, label, score))
                    .collect()
            })
            .collect();

        Ok(TableClassificationOutput { classifications })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        32
    }
}

// Builder macro invocation - generates the builder struct and all trait implementations
impl_adapter_builder! {
    builder_name: TableClassificationAdapterBuilder,
    adapter_name: TableClassificationAdapter,
    config_type: TableClassificationConfig,
    adapter_type: "table_classification",
    adapter_desc: "Classifies table images as wired or wireless",
    task_type: TableClassification,

    fields: {
        input_shape: (u32, u32) = TableClassificationAdapter::DEFAULT_INPUT_SHAPE,
        model_name_override: Option<String> = None,
    },

    methods: {
        pub fn input_shape(mut self, shape: (u32, u32)) -> Self {
            self.input_shape = shape;
            self
        }

        pub fn model_name(mut self, name: impl Into<String>) -> Self {
            self.model_name_override = Some(name.into());
            self
        }
    }

    build: |builder: TableClassificationAdapterBuilder, model_path: &Path| -> Result<TableClassificationAdapter, OCRError> {
        let (task_config, ort_config) = builder.config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the PP-LCNet model
        let preprocess_config = super::preprocessing::pp_lcnet_preprocess(builder.input_shape);

        let model = apply_ort_config!(
            PPLCNetModelBuilder::new().preprocess_config(preprocess_config),
            ort_config
        )
        .build(model_path)?;

        // Create postprocessing configuration
        let postprocess_config = PPLCNetPostprocessConfig {
            labels: TableClassificationAdapter::labels(),
            topk: 1, // Will be overridden by task config
        };

        // Create adapter info using the helper
        let mut info = TableClassificationAdapterBuilder::base_adapter_info();
        if let Some(model_name) = builder.model_name_override {
            info.model_name = model_name;
        }

        Ok(TableClassificationAdapter::new(
            model,
            info,
            task_config,
            postprocess_config,
        ))
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::adapter::AdapterBuilder;

    #[test]
    fn test_builder_creation() {
        let builder = TableClassificationAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "table_classification");
    }

    #[test]
    fn test_builder_with_config() {
        let config = TableClassificationConfig {
            score_threshold: 0.7,
            topk: 2,
        };

        let builder = TableClassificationAdapterBuilder::new().with_config(config.clone());
        // Verify adapter_type is correct
        assert_eq!(builder.adapter_type(), "table_classification");
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = TableClassificationAdapterBuilder::new().input_shape((256, 256));
        // Builder method chaining works - verified by compilation
        let _ = builder;
    }

    #[test]
    fn test_default_builder() {
        let builder = TableClassificationAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "table_classification");
    }

    #[test]
    fn test_labels() {
        let labels = TableClassificationAdapter::labels();
        assert_eq!(labels, vec!["wired_table", "wireless_table"]);
    }
}
