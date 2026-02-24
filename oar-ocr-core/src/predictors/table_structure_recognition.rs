//! Table Structure Recognition Predictor
//!
//! This module provides a high-level API for table structure recognition.

use super::builder::PredictorBuilderState;
use crate::TaskPredictorBuilder;
use crate::core::OcrResult;
use crate::core::errors::OCRError;
use crate::core::traits::OrtConfigurable;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::{SLANetWiredAdapterBuilder, SLANetWirelessAdapterBuilder};
use crate::domain::tasks::table_structure_recognition::{
    TableStructureRecognitionConfig, TableStructureRecognitionTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::{Path, PathBuf};

/// Table structure recognition prediction result
#[derive(Debug, Clone)]
pub struct TableStructureRecognitionResult {
    /// Recognized table structures in HTML format (one per image)
    pub structures: Vec<Vec<String>>,
    /// Bounding boxes for table cells as 8-point coordinates (one per image)
    pub bboxes: Vec<Vec<Vec<f32>>>,
}

/// Supported table structure model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableStructureModelFamily {
    Wired,
    Wireless,
}

impl TableStructureModelFamily {
    fn from_model_name(model_name: &str) -> Option<Self> {
        match model_name {
            "SLANet" | "SLANeXt_wired" | "SLANeXt_wireless" => Some(Self::Wired),
            "SLANet_plus" => Some(Self::Wireless),
            _ => None,
        }
    }

    fn detect_from_path(path: &Path) -> Option<Self> {
        let stem = path.file_stem()?.to_str()?.to_ascii_lowercase();
        if stem.contains("slanet_plus") {
            Some(Self::Wireless)
        } else if stem.contains("wired") || stem.contains("slanet") || stem.contains("slanext") {
            Some(Self::Wired)
        } else {
            None
        }
    }
}

/// Table structure recognition predictor
pub struct TableStructureRecognitionPredictor {
    core: TaskPredictorCore<TableStructureRecognitionTask>,
}

impl TableStructureRecognitionPredictor {
    pub fn builder() -> TableStructureRecognitionPredictorBuilder {
        TableStructureRecognitionPredictorBuilder::new()
    }

    /// Predict table structures in the given images.
    pub fn predict(&self, images: Vec<RgbImage>) -> OcrResult<TableStructureRecognitionResult> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(TableStructureRecognitionResult {
            structures: output.structures,
            bboxes: output.bboxes,
        })
    }
}

#[derive(TaskPredictorBuilder)]
#[builder(config = TableStructureRecognitionConfig)]
pub struct TableStructureRecognitionPredictorBuilder {
    state: PredictorBuilderState<TableStructureRecognitionConfig>,
    dict_path: Option<PathBuf>,
    model_name: Option<String>,
    /// Custom input shape (height, width). If None, uses adapter default.
    input_shape: Option<(u32, u32)>,
}

impl TableStructureRecognitionPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TableStructureRecognitionConfig {
                score_threshold: 0.5,
                max_structure_length: 500,
            }),
            dict_path: None,
            model_name: None,
            input_shape: None,
        }
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    pub fn dict_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.dict_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the table structure model preset name.
    ///
    /// Supported names:
    /// - `SLANeXt_wired`
    /// - `SLANeXt_wireless`
    /// - `SLANet`
    /// - `SLANet_plus`
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = Some(name.into());
        self
    }

    /// Sets the input shape for the model.
    ///
    /// If not set, the input shape will be auto-detected from the ONNX model.
    /// Example ONNX shapes: SLANeXt_wired=512×512, SLANet_plus=488×488.
    pub fn input_shape(mut self, height: u32, width: u32) -> Self {
        self.input_shape = Some((height, width));
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> OcrResult<TableStructureRecognitionPredictor> {
        let Self {
            state,
            dict_path,
            model_name,
            input_shape,
        } = self;
        let (config, ort_config) = state.into_parts();
        let dict_path = dict_path.ok_or_else(|| {
            OCRError::missing_field("dict_path", "TableStructureRecognitionPredictor")
        })?;
        let model_path = model_path.as_ref();

        let model_family = if let Some(name) = model_name.as_deref() {
            TableStructureModelFamily::from_model_name(name).ok_or_else(|| {
                OCRError::ConfigError {
                    message: format!(
                        "Unknown model name '{}'. Supported names: {}",
                        name,
                        Self::SUPPORTED_MODEL_NAMES.join(", ")
                    ),
                }
            })?
        } else {
            TableStructureModelFamily::detect_from_path(model_path)
                .unwrap_or(TableStructureModelFamily::Wired)
        };

        let adapter = match model_family {
            TableStructureModelFamily::Wired => {
                let mut adapter_builder = SLANetWiredAdapterBuilder::new()
                    .with_config(config.clone())
                    .dict_path(dict_path.clone());

                if let Some(name) = model_name.as_deref() {
                    adapter_builder = adapter_builder.model_name(name);
                }

                if let Some((h, w)) = input_shape {
                    adapter_builder = adapter_builder.input_shape((h, w));
                }

                if let Some(ort_cfg) = ort_config.clone() {
                    adapter_builder = adapter_builder.with_ort_config(ort_cfg);
                }

                Box::new(adapter_builder.build(model_path)?)
            }
            TableStructureModelFamily::Wireless => {
                let mut adapter_builder = SLANetWirelessAdapterBuilder::new()
                    .with_config(config.clone())
                    .dict_path(dict_path);

                if let Some(name) = model_name.as_deref() {
                    adapter_builder = adapter_builder.model_name(name);
                }

                if let Some((h, w)) = input_shape {
                    adapter_builder = adapter_builder.input_shape((h, w));
                }

                if let Some(ort_cfg) = ort_config {
                    adapter_builder = adapter_builder.with_ort_config(ort_cfg);
                }

                Box::new(adapter_builder.build(model_path)?)
            }
        };
        let task = TableStructureRecognitionTask::new(config.clone());
        Ok(TableStructureRecognitionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }

    /// Supported table structure model names.
    const SUPPORTED_MODEL_NAMES: &'static [&'static str] =
        &["SLANeXt_wired", "SLANeXt_wireless", "SLANet", "SLANet_plus"];
}

impl Default for TableStructureRecognitionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::TableStructureModelFamily;
    use std::path::Path;

    #[test]
    fn test_model_family_from_model_name() {
        assert_eq!(
            TableStructureModelFamily::from_model_name("SLANeXt_wired"),
            Some(TableStructureModelFamily::Wired)
        );
        assert_eq!(
            TableStructureModelFamily::from_model_name("SLANeXt_wireless"),
            Some(TableStructureModelFamily::Wired)
        );
        assert_eq!(
            TableStructureModelFamily::from_model_name("SLANet_plus"),
            Some(TableStructureModelFamily::Wireless)
        );
        assert_eq!(TableStructureModelFamily::from_model_name("unknown"), None);
    }

    #[test]
    fn test_model_family_detect_from_path() {
        assert_eq!(
            TableStructureModelFamily::detect_from_path(Path::new("models/slanet_plus.onnx")),
            Some(TableStructureModelFamily::Wireless)
        );
        assert_eq!(
            TableStructureModelFamily::detect_from_path(Path::new("models/slanext_wired.onnx")),
            Some(TableStructureModelFamily::Wired)
        );
        assert_eq!(
            TableStructureModelFamily::detect_from_path(Path::new("models/custom.onnx")),
            None
        );
    }
}
