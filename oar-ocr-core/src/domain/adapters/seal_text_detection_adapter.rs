//! Seal Text Detection Task Adapter
//!
//! This adapter uses the DB model configured for seal text detection (curved text).

use crate::apply_ort_config;
use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{
    Detection, SealTextDetectionConfig, SealTextDetectionOutput, SealTextDetectionTask,
};
use crate::impl_adapter_builder;
use crate::models::detection::db::{DBModel, DBModelBuilder, DBPostprocessConfig};
use crate::processors::{BoxType, ScoreMode};

/// Seal text detection adapter that uses the DB model.
#[derive(Debug)]
pub struct SealTextDetectionAdapter {
    /// The underlying DB model
    model: DBModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: SealTextDetectionConfig,
}

impl SealTextDetectionAdapter {
    /// Creates a new seal text detection adapter.
    pub fn new(model: DBModel, info: AdapterInfo, config: SealTextDetectionConfig) -> Self {
        Self {
            model,
            info,
            config,
        }
    }
}

impl ModelAdapter for SealTextDetectionAdapter {
    type Task = SealTextDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // Use the DB model to detect seal text with error context
        let model_output = self.model
            .forward(
                input.images,
                effective_config.score_threshold,
                effective_config.box_threshold,
                effective_config.unclip_ratio,
            )
            .map_err(|e| {
                OCRError::adapter_execution_error(
                    "SealTextDetectionAdapter",
                    format!(
                        "failed to detect seal text (score_threshold={}, box_threshold={}, unclip_ratio={})",
                        effective_config.score_threshold,
                        effective_config.box_threshold,
                        effective_config.unclip_ratio
                    ),
                    e,
                )
            })?;

        // Convert model output to structured detections
        let detections = model_output
            .boxes
            .into_iter()
            .zip(model_output.scores)
            .map(|(boxes, scores)| {
                boxes
                    .into_iter()
                    .zip(scores)
                    .map(|(bbox, score)| Detection::new(bbox, score))
                    .collect()
            })
            .collect();

        Ok(SealTextDetectionOutput { detections })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}

impl_adapter_builder! {
    builder_name: SealTextDetectionAdapterBuilder,
    adapter_name: SealTextDetectionAdapter,
    config_type: SealTextDetectionConfig,
    adapter_type: "SealTextDetection",
    adapter_desc: "Detects curved seal text with polygon bounding boxes",
    task_type: SealTextDetection,

    fields: {},
    methods: {}

    build: |builder: SealTextDetectionAdapterBuilder, model_path: &std::path::Path| -> Result<SealTextDetectionAdapter, OCRError> {
        let (task_config, ort_config) = builder.config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Configure DB model for seal text detection
        // Use seal text preprocessing configuration (limit_side_len=736, limit_type=Min)
        let preprocess_config = super::preprocessing::db_preprocess_for_text_type(Some("seal"));

        let postprocess_config = DBPostprocessConfig {
            score_threshold: task_config.score_threshold,
            box_threshold: task_config.box_threshold,
            unclip_ratio: task_config.unclip_ratio,
            max_candidates: task_config.max_candidates,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
            box_type: BoxType::Poly, // Seal detection uses polygon boxes for curved text
        };

        // Build the DB model
        let model = apply_ort_config!(
            DBModelBuilder::new()
                .preprocess_config(preprocess_config)
                .postprocess_config(postprocess_config),
            ort_config
        )
        .build(model_path)?;

        // Create adapter info
        let info = AdapterInfo::new(
            "seal_text_detection",
            "1.0.0",
            TaskType::SealTextDetection,
            "Seal text detection using DB model with polygon output",
        );

        Ok(SealTextDetectionAdapter::new(
            model,
            info,
            task_config,
        ))
    },
}
