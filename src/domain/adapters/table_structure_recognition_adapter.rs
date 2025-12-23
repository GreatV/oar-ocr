//! Table Structure Recognition Adapter
//!
//! This adapter uses the SLANet model to recognize table structure as HTML tokens with bounding boxes.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
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

    /// Default input shape for wired table structure recognition (SLANeXt_wired).
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (512, 512);

    /// Default input shape for wireless table structure recognition (SLANet_plus).
    ///
    /// PP-StructureV3 uses SLANet_plus (not SLANeXt_wireless) for wireless tables,
    /// which requires 488×488 input size.
    pub const DEFAULT_WIRELESS_INPUT_SHAPE: (u32, u32) = (488, 488);
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

        // Validate input
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided".to_string(),
            });
        }

        let num_images = input.images.len();
        tracing::debug!("Processing {} table images", num_images);

        // Run model forward pass on all images
        let model_output = self.model.forward(input.images).map_err(|e| {
            OCRError::adapter_execution_error(
                "TableStructureRecognitionAdapter",
                format!("model forward (batch_size={})", num_images),
                e,
            )
        })?;

        // Decode structure and bboxes for all images
        let decode_output = self
            .decoder
            .decode(
                &model_output.structure_logits,
                &model_output.bbox_preds,
                &model_output.shape_info,
            )
            .map_err(|e| {
                OCRError::adapter_execution_error("TableStructureRecognitionAdapter", "decode", e)
            })?;

        // Process each image's results
        let mut structures = Vec::with_capacity(num_images);
        let mut bboxes = Vec::with_capacity(num_images);
        let mut structure_scores = Vec::with_capacity(num_images);

        for img_idx in 0..num_images {
            let structure_tokens =
                decode_output.structure_tokens.get(img_idx).ok_or_else(|| {
                    OCRError::InvalidInput {
                        message: format!("No structure tokens decoded for image {}", img_idx),
                    }
                })?;

            let image_bboxes =
                decode_output
                    .bboxes
                    .get(img_idx)
                    .ok_or_else(|| OCRError::InvalidInput {
                        message: format!("No bboxes decoded for image {}", img_idx),
                    })?;

            let structure_score = decode_output
                .structure_scores
                .get(img_idx)
                .copied()
                .unwrap_or(0.0);

            if structure_score < effective_config.score_threshold {
                // Do not drop results, just log for visibility.
                tracing::warn!(
                    "Image {}: Structure score {:.3} below threshold {:.3}, keeping result",
                    img_idx,
                    structure_score,
                    effective_config.score_threshold
                );
            }

            let trimmed_tokens: Vec<String> = structure_tokens
                .iter()
                .take(effective_config.max_structure_length)
                .cloned()
                .collect();

            let trimmed_len = trimmed_tokens.len();

            if trimmed_len < structure_tokens.len() {
                tracing::warn!(
                    "Image {}: Structure tokens {} exceed max {}, truncating output",
                    img_idx,
                    structure_tokens.len(),
                    effective_config.max_structure_length
                );
            }

            // Do NOT add HTML wrapping here - it will be done by wrap_table_html later
            // The adapter should just return the raw structure tokens
            // TableLabelDecode adds wrapping in postprocessor, but we handle it in wrap_table_html
            let structure = trimmed_tokens;

            tracing::debug!("Image {}: Final structure tokens: {:?}", img_idx, structure);

            // Return bboxes as [f32; 8] without rounding to preserve precision for IoA calculation
            let bbox: Vec<Vec<f32>> = image_bboxes
                .iter()
                .take(trimmed_len)
                .map(|&bbox_coords| {
                    let coords: Vec<f32> = bbox_coords.to_vec();
                    tracing::debug!("Image {}: BBox coords: {:?}", img_idx, coords);
                    coords
                })
                .collect();

            if bbox.len() < trimmed_len {
                // This is expected: structure tokens include all tags (<tr>, </tr>, etc.)
                // while bboxes are only for TD tokens (<td>, <td></td>)
                tracing::debug!(
                    "Image {}: {} bounding boxes for {} structure tokens (TD tokens only have bboxes)",
                    img_idx,
                    bbox.len(),
                    trimmed_len
                );
            }

            tracing::debug!("Image {}: Final bbox output: {:?}", img_idx, bbox);

            structures.push(structure);
            bboxes.push(bbox);
            structure_scores.push(structure_score);
        }

        Ok(crate::domain::tasks::TableStructureRecognitionOutput {
            structures,
            bboxes,
            structure_scores,
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
///
/// Uses SLANeXt_wired model which requires 512×512 input size (per PP-StructureV3 configuration).
/// If no input shape is specified, it will be auto-detected from the ONNX model.
pub struct SLANetWiredAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<TableStructureRecognitionConfig>,
    /// Input shape (height, width) - if None, will be auto-detected from ONNX
    input_shape: Option<(u32, u32)>,
    /// Dictionary path
    dict_path: Option<std::path::PathBuf>,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl SLANetWiredAdapterBuilder {
    /// Creates a new builder with default configuration.
    ///
    /// Input shape will be auto-detected from the ONNX model if not explicitly set.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            input_shape: Some((512, 512)), // SLANeXt_wired is trained/evaluated at 512
            dict_path: None,
            model_name_override: None,
        }
    }

    /// Sets the input shape explicitly.
    ///
    /// If not set, the input shape will be auto-detected from the ONNX model.
    /// For SLANeXt_wired, the expected shape is 512×512.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.config = self.config.with_session_pool_size(size);
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
        self.config = self.config.with_ort_config(config);
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
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the SLANet model - input shape will be auto-detected from ONNX if not set
        let mut model_builder = SLANetModelBuilder::new().session_pool_size(session_pool_size);

        // Only set input size if explicitly provided; otherwise let ONNX auto-detect
        if let Some(input_shape) = self.input_shape {
            model_builder = model_builder.input_size(input_shape);
        }

        if let Some(ort_config) = ort_config {
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
            task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "TableStructureRecognitionWired"
    }
}

/// Builder for table structure recognition adapter (wireless tables).
///
/// Uses SLANet_plus model which requires 488×488 input size (per PP-StructureV3 configuration).
/// If no input shape is specified, it will be auto-detected from the ONNX model.
pub struct SLANetWirelessAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<TableStructureRecognitionConfig>,
    /// Input shape (height, width) - if None, will be auto-detected from ONNX
    input_shape: Option<(u32, u32)>,
    /// Dictionary path
    dict_path: Option<std::path::PathBuf>,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl SLANetWirelessAdapterBuilder {
    /// Creates a new builder with default configuration.
    ///
    /// Input shape will be auto-detected from the ONNX model.
    /// For SLANet/SLANet_plus with dynamic input, this enables no-padding preprocessing.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            input_shape: Some((488, 488)), // SLANet_plus requires 488x488 with padding
            dict_path: None,
            model_name_override: None,
        }
    }

    /// Sets the input shape explicitly.
    ///
    /// If not set, the input shape will be auto-detected from the ONNX model.
    /// For SLANet_plus, the expected shape is 488×488.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.config = self.config.with_session_pool_size(size);
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
        self.config = self.config.with_ort_config(config);
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
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the SLANet model - input shape will be auto-detected from ONNX if not set
        let mut model_builder = SLANetModelBuilder::new().session_pool_size(session_pool_size);

        // Only set input size if explicitly provided; otherwise let ONNX auto-detect
        if let Some(input_shape) = self.input_shape {
            model_builder = model_builder.input_size(input_shape);
        }

        if let Some(ort_config) = ort_config {
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
            "Table structure recognition (wireless tables) using SLANet_plus model",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(TableStructureRecognitionAdapter::new(
            model,
            decoder,
            info,
            task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
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

        assert_eq!(builder.input_shape, Some((640, 640)));
        assert_eq!(builder.config.session_pool_size(), 4);
    }
}
