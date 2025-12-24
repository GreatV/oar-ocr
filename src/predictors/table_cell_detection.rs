//! Table Cell Detection Predictor
//!
//! This module provides a high-level API for table cell detection in images.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::{TableCellDetectionAdapterBuilder, TableCellModelConfig};
use crate::domain::tasks::table_cell_detection::{
    TableCellDetectionConfig, TableCellDetectionTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Table cell detection prediction result
#[derive(Debug, Clone)]
pub struct TableCellDetectionResult {
    /// Detected table cells for each input image
    pub cells: Vec<Vec<crate::domain::tasks::table_cell_detection::TableCell>>,
}

/// Table cell detection predictor
pub struct TableCellDetectionPredictor {
    core: TaskPredictorCore<TableCellDetectionTask>,
}

impl TableCellDetectionPredictor {
    pub fn builder() -> TableCellDetectionPredictorBuilder {
        TableCellDetectionPredictorBuilder::new()
    }

    /// Predict table cells in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<TableCellDetectionResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(TableCellDetectionResult {
            cells: output.cells,
        })
    }
}

pub struct TableCellDetectionPredictorBuilder {
    state: PredictorBuilderState<TableCellDetectionConfig>,
    model_variant: Option<TableCellModelVariant>,
}

/// Supported table cell model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableCellModelVariant {
    /// RT-DETR-L wired table cell detector.
    RTDetrLWired,
    /// RT-DETR-L wireless table cell detector.
    RTDetrLWireless,
}

impl TableCellModelVariant {
    /// Canonical string identifier for the variant.
    pub fn as_str(&self) -> &'static str {
        match self {
            TableCellModelVariant::RTDetrLWired => "rt-detr-l_wired_table_cell_det",
            TableCellModelVariant::RTDetrLWireless => "rt-detr-l_wireless_table_cell_det",
        }
    }

    /// Converts the variant into the adapter configuration.
    pub fn to_model_config(&self) -> TableCellModelConfig {
        match self {
            TableCellModelVariant::RTDetrLWired => {
                TableCellModelConfig::rtdetr_l_wired_table_cell_det()
            }
            TableCellModelVariant::RTDetrLWireless => {
                TableCellModelConfig::rtdetr_l_wireless_table_cell_det()
            }
        }
    }

    /// Detects the variant from an ONNX filename.
    pub fn detect_from_path(path: &Path) -> Option<Self> {
        let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
        if stem.contains("wired_table_cell") {
            Some(TableCellModelVariant::RTDetrLWired)
        } else if stem.contains("wireless_table_cell") {
            Some(TableCellModelVariant::RTDetrLWireless)
        } else {
            None
        }
    }

    /// Parses from a user-provided model type string.
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        let normalized = model_type.to_ascii_lowercase();
        let canonical = normalized.replace('-', "_");
        match canonical.as_str() {
            "rt-detr-l_wired_table_cell_det" => Some(TableCellModelVariant::RTDetrLWired),
            "rt-detr-l_wireless_table_cell_det" => Some(TableCellModelVariant::RTDetrLWireless),
            _ => None,
        }
    }
}

impl TableCellDetectionPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TableCellDetectionConfig {
                score_threshold: 0.3,
                max_cells: 300,
            }),
            model_variant: None,
        }
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    /// Sets the model variant to use (wired / wireless).
    pub fn model_variant(mut self, variant: TableCellModelVariant) -> Self {
        self.model_variant = Some(variant);
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<TableCellDetectionPredictor, Box<dyn std::error::Error>> {
        let (config, ort_config) = self.state.into_parts();
        let path_ref = model_path.as_ref();
        let variant = self
            .model_variant
            .or_else(|| TableCellModelVariant::detect_from_path(path_ref))
            .ok_or_else(|| {
                format!(
                    "Unable to determine table cell model variant from '{}'. \
                     Provide `model_variant(...)` on the builder or use a filename \
                     containing 'wired_table_cell' or 'wireless_table_cell'.",
                    path_ref.display()
                )
            })?;

        let mut adapter_builder = TableCellDetectionAdapterBuilder::new()
            .model_config(variant.to_model_config())
            .with_config(config.clone());

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(path_ref)?);
        let task = TableCellDetectionTask::new(config.clone());
        Ok(TableCellDetectionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(TableCellDetectionPredictorBuilder, TableCellDetectionConfig);

impl Default for TableCellDetectionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
