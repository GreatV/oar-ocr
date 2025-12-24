//! Layout Detection Predictor
//!
//! This module provides a high-level API for document layout detection.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::LayoutDetectionAdapterBuilder;
use crate::domain::tasks::layout_detection::{LayoutDetectionConfig, LayoutDetectionTask};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Layout detection prediction result
#[derive(Debug, Clone)]
pub struct LayoutDetectionResult {
    /// Detected layout elements for each input image
    pub elements: Vec<Vec<crate::domain::tasks::layout_detection::LayoutElement>>,
}

/// Layout detection predictor
pub struct LayoutDetectionPredictor {
    core: TaskPredictorCore<LayoutDetectionTask>,
}

impl LayoutDetectionPredictor {
    pub fn builder() -> LayoutDetectionPredictorBuilder {
        LayoutDetectionPredictorBuilder::new()
    }

    /// Predict layout elements in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<LayoutDetectionResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(LayoutDetectionResult {
            elements: output.elements,
        })
    }
}

pub struct LayoutDetectionPredictorBuilder {
    state: PredictorBuilderState<LayoutDetectionConfig>,
    model_name: Option<String>,
}

impl LayoutDetectionPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(LayoutDetectionConfig::default()),
            model_name: None,
        }
    }

    /// Creates a builder with PP-StructureV3 default class thresholds.
    pub fn with_pp_structurev3_thresholds() -> Self {
        Self {
            state: PredictorBuilderState::new(
                LayoutDetectionConfig::with_pp_structurev3_thresholds(),
            ),
            model_name: None,
        }
    }

    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<LayoutDetectionPredictor, Box<dyn std::error::Error>> {
        let (config, ort_config) = self.state.into_parts();
        let mut adapter_builder = LayoutDetectionAdapterBuilder::new().task_config(config.clone());

        // Set model configuration if model_name was provided
        if let Some(model_name) = self.model_name {
            let model_config = Self::get_model_config(&model_name)?;
            adapter_builder = adapter_builder.model_config(model_config);
        }

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = LayoutDetectionTask::new(config.clone());
        Ok(LayoutDetectionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }

    fn get_model_config(
        model_name: &str,
    ) -> Result<crate::domain::adapters::LayoutModelConfig, Box<dyn std::error::Error>> {
        use crate::domain::adapters::LayoutModelConfig;

        let normalized = model_name.to_lowercase().replace('-', "_");
        let config = match normalized.as_str() {
            "picodet_layout_1x" => LayoutModelConfig::picodet_layout_1x(),
            "picodet_layout_1x_table" => LayoutModelConfig::picodet_layout_1x_table(),
            "picodet_s_layout_3cls" => LayoutModelConfig::picodet_s_layout_3cls(),
            "picodet_l_layout_3cls" => LayoutModelConfig::picodet_l_layout_3cls(),
            "picodet_s_layout_17cls" => LayoutModelConfig::picodet_s_layout_17cls(),
            "picodet_l_layout_17cls" => LayoutModelConfig::picodet_l_layout_17cls(),
            "rtdetr_h_layout_3cls" | "rt_detr_h_layout_3cls" => {
                LayoutModelConfig::rtdetr_h_layout_3cls()
            }
            "rtdetr_h_layout_17cls" | "rt_detr_h_layout_17cls" => {
                LayoutModelConfig::rtdetr_h_layout_17cls()
            }
            "pp_docblocklayout" => LayoutModelConfig::pp_docblocklayout(),
            "pp_doclayout_s" => LayoutModelConfig::pp_doclayout_s(),
            "pp_doclayout_m" => LayoutModelConfig::pp_doclayout_m(),
            "pp_doclayout_l" => LayoutModelConfig::pp_doclayout_l(),
            "pp_doclayout_plus_l" => LayoutModelConfig::pp_doclayout_plus_l(),
            _ => {
                return Err(format!("Unknown model name: {}. Supported models: picodet_layout_1x, picodet_layout_1x_table, picodet_s_layout_3cls, picodet_l_layout_3cls, picodet_s_layout_17cls, picodet_l_layout_17cls, rtdetr_h_layout_3cls, rtdetr_h_layout_17cls, pp_docblocklayout, pp_doclayout_s, pp_doclayout_m, pp_doclayout_l, pp_doclayout_plus_l", model_name).into());
            }
        };

        Ok(config)
    }
}

impl_task_predictor_builder!(LayoutDetectionPredictorBuilder, LayoutDetectionConfig);

impl Default for LayoutDetectionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
