//! Table Structure Recognition Adapter
//!
//! This adapter uses the SLANet model to recognize table structure as HTML tokens with bounding boxes.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{ImageTaskInput, Task, TaskType},
};
use crate::domain::tasks::{TableStructureRecognitionConfig, TableStructureRecognitionTask};
use crate::models::recognition::{SLANetModel, SLANetModelBuilder};
use crate::processors::TableStructureDecode;
use std::path::Path;

/// Table structure recognition adapter that uses the SLANet model.
#[derive(Debug)]
pub struct TableStructureRecognitionAdapter {
    /// The underlying SLANet model
    model: SLANetModel,
    /// Table structure decoder
    decoder: TableStructureDecode,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TableStructureRecognitionConfig,
}

impl TableStructureRecognitionAdapter {
    /// Creates a new table structure recognition adapter.
    pub fn new(
        model: SLANetModel,
        decoder: TableStructureDecode,
        info: AdapterInfo,
        config: TableStructureRecognitionConfig,
    ) -> Self {
        Self {
            model,
            decoder,
            info,
            config,
        }
    }

    /// Default input shape for table structure recognition (488x488 as per PaddleOCR config).
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (488, 488);
}

impl ModelAdapter for TableStructureRecognitionAdapter {
    type Task = TableStructureRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // For now, process only the first image (matching PaddleOCR behavior)
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided".to_string(),
            });
        }

        let single_image_input = ImageTaskInput::new(vec![input.images[0].clone()]);

        // Run model forward pass
        let model_output = self.model.forward(single_image_input.images)?;

        // Decode structure and bboxes
        let decode_output = self.decoder.decode(
            &model_output.structure_logits,
            &model_output.bbox_preds,
            &model_output.shape_info,
        )?;

        // Get the first (and only) result
        let structure_tokens =
            decode_output
                .structure_tokens
                .first()
                .ok_or_else(|| OCRError::InvalidInput {
                    message: "No structure tokens decoded".to_string(),
                })?;

        let bboxes = decode_output
            .bboxes
            .first()
            .ok_or_else(|| OCRError::InvalidInput {
                message: "No bboxes decoded".to_string(),
            })?;

        let structure_score = decode_output
            .structure_scores
            .first()
            .copied()
            .unwrap_or(0.0);

        if structure_score < effective_config.score_threshold {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Structure score {:.3} below threshold {:.3}",
                    structure_score, effective_config.score_threshold
                ),
            });
        }

        let trimmed_tokens: Vec<String> = structure_tokens
            .iter()
            .take(effective_config.max_structure_length)
            .cloned()
            .collect();

        let trimmed_len = trimmed_tokens.len();

        if trimmed_len < structure_tokens.len() {
            tracing::warn!(
                "Structure tokens {} exceed max {}, truncating output",
                structure_tokens.len(),
                effective_config.max_structure_length
            );
        }

        // Add HTML wrapping like PaddleX
        let mut structure = vec![
            "<html>".to_string(),
            "<body>".to_string(),
            "<table>".to_string(),
        ];
        structure.extend(trimmed_tokens);
        structure.extend(vec![
            "</table>".to_string(),
            "</body>".to_string(),
            "</html>".to_string(),
        ]);

        tracing::debug!("Final HTML structure output: {:?}", structure);

        // Convert bboxes from [f32; 8] to Vec<Vec<i32>> (round to integers like PaddleX)
        let bbox: Vec<Vec<i32>> = bboxes
            .iter()
            .take(trimmed_len)
            .map(|&bbox_coords| {
                let int_coords: Vec<i32> = bbox_coords
                    .iter()
                    .map(|&coord| coord.round() as i32)
                    .collect();
                tracing::debug!("Converted bbox {:?} to {:?}", bbox_coords, int_coords);
                int_coords
            })
            .collect();

        if bbox.len() < trimmed_len {
            tracing::warn!(
                "Only {} bounding boxes available for {} structure tokens",
                bbox.len(),
                trimmed_len
            );
        }

        tracing::debug!("Final bbox output: {:?}", bbox);

        Ok(crate::domain::tasks::TableStructureRecognitionOutput {
            structure,
            bbox,
            structure_score,
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}

/// Builder for table structure recognition adapter (wired tables).
pub struct SLANetWiredAdapterBuilder {
    /// Task configuration
    task_config: TableStructureRecognitionConfig,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Dictionary path
    dict_path: Option<std::path::PathBuf>,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl SLANetWiredAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            task_config: TableStructureRecognitionConfig::default(),
            input_shape: TableStructureRecognitionAdapter::DEFAULT_INPUT_SHAPE,
            session_pool_size: 1,
            dict_path: None,
            model_name_override: None,
            ort_config: None,
        }
    }

    /// Sets the input shape.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = input_shape;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the dictionary path.
    pub fn dict_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.dict_path = Some(path.into());
        self
    }

    /// Sets a custom model name for registry registration.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name_override = Some(model_name.into());
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }
}

impl Default for SLANetWiredAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for SLANetWiredAdapterBuilder {
    type Config = TableStructureRecognitionConfig;
    type Adapter = TableStructureRecognitionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Build the SLANet model
        let mut model_builder = SLANetModelBuilder::new()
            .session_pool_size(self.session_pool_size)
            .input_size(self.input_shape);

        if let Some(ort_config) = self.ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Dictionary path is required
        let dict_path = self.dict_path.ok_or_else(|| OCRError::ConfigError {
            message: "Dictionary path is required. Use .dict_path() to specify the path to table_structure_dict_ch.txt".to_string(),
        })?;

        // Create decoder
        let decoder = TableStructureDecode::from_dict_path(&dict_path)?;

        // Create adapter info
        let mut info = AdapterInfo::new(
            "table_structure_recognition_wired",
            "1.0.0",
            TaskType::TableStructureRecognition,
            "Table structure recognition (wired tables) using SLANeXt model",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(TableStructureRecognitionAdapter::new(
            model,
            decoder,
            info,
            self.task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "TableStructureRecognitionWired"
    }
}

/// Builder for table structure recognition adapter (wireless tables).
pub struct SLANetWirelessAdapterBuilder {
    /// Task configuration
    task_config: TableStructureRecognitionConfig,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Dictionary path
    dict_path: Option<std::path::PathBuf>,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl SLANetWirelessAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            task_config: TableStructureRecognitionConfig::default(),
            input_shape: TableStructureRecognitionAdapter::DEFAULT_INPUT_SHAPE,
            session_pool_size: 1,
            dict_path: None,
            model_name_override: None,
            ort_config: None,
        }
    }

    /// Sets the input shape.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = input_shape;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the dictionary path.
    pub fn dict_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.dict_path = Some(path.into());
        self
    }

    /// Sets a custom model name for registry registration.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name_override = Some(model_name.into());
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }
}

impl Default for SLANetWirelessAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for SLANetWirelessAdapterBuilder {
    type Config = TableStructureRecognitionConfig;
    type Adapter = TableStructureRecognitionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Build the SLANet model
        let mut model_builder = SLANetModelBuilder::new()
            .session_pool_size(self.session_pool_size)
            .input_size(self.input_shape);

        if let Some(ort_config) = self.ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Dictionary path is required
        let dict_path = self.dict_path.ok_or_else(|| OCRError::ConfigError {
            message: "Dictionary path is required. Use .dict_path() to specify the path to table_structure_dict_ch.txt".to_string(),
        })?;

        // Create decoder
        let decoder = TableStructureDecode::from_dict_path(&dict_path)?;

        // Create adapter info
        let mut info = AdapterInfo::new(
            "table_structure_recognition_wireless",
            "1.0.0",
            TaskType::TableStructureRecognition,
            "Table structure recognition (wireless tables) using SLANeXt model",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(TableStructureRecognitionAdapter::new(
            model,
            decoder,
            info,
            self.task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "TableStructureRecognitionWireless"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wired_builder_creation() {
        let builder = SLANetWiredAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "TableStructureRecognitionWired");
    }

    #[test]
    fn test_wireless_builder_creation() {
        let builder = SLANetWirelessAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "TableStructureRecognitionWireless");
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = SLANetWiredAdapterBuilder::new()
            .input_shape((640, 640))
            .session_pool_size(4)
            .dict_path("models/table_structure_dict_ch.txt");

        assert_eq!(builder.input_shape, (640, 640));
        assert_eq!(builder.session_pool_size, 4);
    }
}
