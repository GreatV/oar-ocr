//! High-level builder API for document structure analysis.
//!
//! This module provides a fluent builder interface for constructing document structure
//! analysis pipelines that can detect layout elements, recognize tables, extract formulas,
//! and optionally integrate OCR for text extraction.

use crate::core::OCRError;
use crate::core::config::OrtSessionConfig;
use crate::core::registry::{DynModelAdapter, TaskAdapter};
use crate::core::traits::adapter::AdapterBuilder;
use crate::domain::adapters::{
    LayoutDetectionAdapterBuilder, PPFormulaNetAdapterBuilder, SLANetWiredAdapterBuilder,
    SLANetWirelessAdapterBuilder, TableCellDetectionAdapterBuilder,
    TableClassificationAdapterBuilder, TextDetectionAdapterBuilder, TextRecognitionAdapterBuilder,
    UniMERNetFormulaAdapterBuilder,
};
use crate::domain::structure::{StructureResult, TableResult};
use crate::domain::tasks::{
    FormulaRecognitionConfig, LayoutDetectionConfig, TableCellDetectionConfig,
    TableClassificationConfig, TableStructureRecognitionConfig, TextDetectionConfig,
    TextRecognitionConfig,
};
use std::path::PathBuf;
use std::sync::Arc;

/// IoU threshold for removing overlapping layout elements (0.5 = 50% overlap).
const LAYOUT_OVERLAP_IOU_THRESHOLD: f32 = 0.5;

/// IoU threshold for determining if an OCR box overlaps with table cells.
const CELL_OVERLAP_IOU_THRESHOLD: f32 = 0.5;

/// IoA threshold for assigning layout elements to region blocks during reading order.
/// A low threshold (0.1 = 10%) allows elements near region boundaries to be included.
const REGION_MEMBERSHIP_IOA_THRESHOLD: f32 = 0.1;

/// IoA threshold for splitting text boxes that intersect with container elements.
/// A moderate threshold (0.3 = 30%) balances precision with avoiding over-splitting.
const TEXT_BOX_SPLIT_IOA_THRESHOLD: f32 = 0.3;

/// Internal structure holding the structure analysis pipeline adapters.
#[derive(Debug)]
struct StructurePipeline {
    // Document preprocessing (optional)
    document_orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
    rectification_adapter: Option<Arc<dyn DynModelAdapter>>,

    // Layout analysis (required)
    layout_detection_adapter: Arc<dyn DynModelAdapter>,

    // Region detection for hierarchical ordering (optional, PP-DocBlockLayout)
    region_detection_adapter: Option<Arc<dyn DynModelAdapter>>,

    // Table analysis (optional)
    table_classification_adapter: Option<Arc<dyn DynModelAdapter>>,
    table_orientation_adapter: Option<Arc<dyn DynModelAdapter>>, // Reuses doc orientation model
    table_cell_detection_adapter: Option<Arc<dyn DynModelAdapter>>,
    table_structure_recognition_adapter: Option<Arc<dyn DynModelAdapter>>,
    // PP-StructureV3 auto-switch: separate adapters for wired/wireless tables
    wired_table_structure_adapter: Option<Arc<dyn DynModelAdapter>>,
    wireless_table_structure_adapter: Option<Arc<dyn DynModelAdapter>>,
    wired_table_cell_adapter: Option<Arc<dyn DynModelAdapter>>,
    wireless_table_cell_adapter: Option<Arc<dyn DynModelAdapter>>,
    // E2E mode: when true, skip cell detection and use only structure model output
    use_e2e_wired_table_rec: bool,
    use_e2e_wireless_table_rec: bool,

    formula_recognition_adapter: Option<Arc<dyn DynModelAdapter>>,

    seal_text_detection_adapter: Option<Arc<dyn DynModelAdapter>>,

    // OCR integration (optional)
    text_detection_adapter: Option<Arc<dyn DynModelAdapter>>,
    text_line_orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
    text_recognition_adapter: Option<Arc<dyn DynModelAdapter>>,

    // Batch size for region-level processing (table cells, text recognition)
    region_batch_size: Option<usize>,
}

/// High-level builder for document structure analysis pipelines.
///
/// This builder provides a fluent API for constructing document structure analysis
/// pipelines with various components:
/// - Document preprocessing (optional): orientation detection and rectification
/// - Layout detection (required)
/// - Table classification (optional)
/// - Table cell detection (optional)
/// - Table structure recognition (optional)
/// - Formula recognition (optional)
/// - Seal text detection (optional)
/// - OCR integration (optional)
///
/// # Example
///
/// ```no_run
/// use oar_ocr::oarocr::structure::OARStructureBuilder;
///
/// let structure = OARStructureBuilder::new("models/layout.onnx")
///     .with_table_classification("models/table_cls.onnx")
///     .with_table_cell_detection("models/table_cell.onnx", "wired")
///     .with_table_structure_recognition("models/table_struct.onnx", "wired")
///     .with_formula_recognition(
///         "models/formula.onnx",
///         "models/tokenizer.json",
///         "pp_formulanet"
///     )
///     .build()
///     .expect("Failed to build structure analyzer");
/// ```
#[derive(Debug, Clone)]
pub struct OARStructureBuilder {
    // Required models
    layout_detection_model: PathBuf,
    layout_model_name: Option<String>,

    // Optional document preprocessing
    document_orientation_model: Option<PathBuf>,
    document_rectification_model: Option<PathBuf>,

    // Optional region detection for hierarchical ordering (PP-DocBlockLayout)
    region_detection_model: Option<PathBuf>,

    // Optional table analysis models
    table_classification_model: Option<PathBuf>,
    table_orientation_model: Option<PathBuf>, // Reuses doc orientation model for rotated tables
    table_cell_detection_model: Option<PathBuf>,
    table_cell_detection_type: Option<String>, // "wired" or "wireless"
    table_structure_recognition_model: Option<PathBuf>,
    table_structure_recognition_type: Option<String>, // "wired" or "wireless"
    table_structure_dict_path: Option<PathBuf>,

    wired_table_structure_model: Option<PathBuf>,
    wireless_table_structure_model: Option<PathBuf>,
    wired_table_cell_model: Option<PathBuf>,
    wireless_table_cell_model: Option<PathBuf>,
    // E2E mode: when true, skip cell detection and use only structure model output
    // Defaults: wired=false, wireless=true
    use_e2e_wired_table_rec: bool,
    use_e2e_wireless_table_rec: bool,

    // Optional formula recognition
    formula_recognition_model: Option<PathBuf>,
    formula_recognition_type: Option<String>, // "pp_formulanet" or "unimernet"
    formula_tokenizer_path: Option<PathBuf>,

    // Optional seal text detection
    seal_text_detection_model: Option<PathBuf>,

    // Optional OCR integration
    text_detection_model: Option<PathBuf>,
    text_line_orientation_model: Option<PathBuf>,
    text_recognition_model: Option<PathBuf>,
    character_dict_path: Option<PathBuf>,

    // Model name presets for loading correct pre/post processors
    region_model_name: Option<String>,
    wired_table_structure_model_name: Option<String>,
    wireless_table_structure_model_name: Option<String>,
    wired_table_cell_model_name: Option<String>,
    wireless_table_cell_model_name: Option<String>,
    text_detection_model_name: Option<String>,
    text_recognition_model_name: Option<String>,

    // Configuration
    ort_session_config: Option<OrtSessionConfig>,
    layout_detection_config: Option<LayoutDetectionConfig>,
    table_classification_config: Option<TableClassificationConfig>,
    table_cell_detection_config: Option<TableCellDetectionConfig>,
    table_structure_recognition_config: Option<TableStructureRecognitionConfig>,
    formula_recognition_config: Option<FormulaRecognitionConfig>,
    text_detection_config: Option<TextDetectionConfig>,
    text_recognition_config: Option<TextRecognitionConfig>,

    // Batch sizes
    image_batch_size: Option<usize>,
    region_batch_size: Option<usize>,
}

impl OARStructureBuilder {
    /// Creates a new structure builder with the required layout detection model.
    ///
    /// # Arguments
    ///
    /// * `layout_detection_model` - Path to the layout detection model file
    pub fn new(layout_detection_model: impl Into<PathBuf>) -> Self {
        Self {
            layout_detection_model: layout_detection_model.into(),
            layout_model_name: None,
            document_orientation_model: None,
            document_rectification_model: None,
            region_detection_model: None,
            table_classification_model: None,
            table_orientation_model: None,
            table_cell_detection_model: None,
            table_cell_detection_type: None,
            table_structure_recognition_model: None,
            table_structure_recognition_type: None,
            table_structure_dict_path: None,
            wired_table_structure_model: None,
            wireless_table_structure_model: None,
            wired_table_cell_model: None,
            wireless_table_cell_model: None,
            // Defaults: wired=false (use cell detection), wireless=true (E2E mode)
            use_e2e_wired_table_rec: false,
            use_e2e_wireless_table_rec: true,
            formula_recognition_model: None,
            formula_recognition_type: None,
            formula_tokenizer_path: None,
            seal_text_detection_model: None,
            text_detection_model: None,
            text_line_orientation_model: None,
            text_recognition_model: None,
            character_dict_path: None,
            region_model_name: None,
            wired_table_structure_model_name: None,
            wireless_table_structure_model_name: None,
            wired_table_cell_model_name: None,
            wireless_table_cell_model_name: None,
            text_detection_model_name: None,
            text_recognition_model_name: None,
            ort_session_config: None,
            layout_detection_config: None,
            table_classification_config: None,
            table_cell_detection_config: None,
            table_structure_recognition_config: None,
            formula_recognition_config: None,
            text_detection_config: None,
            text_recognition_config: None,
            image_batch_size: None,
            region_batch_size: None,
        }
    }

    /// Sets the ONNX Runtime session configuration.
    ///
    /// This configuration will be applied to all models in the pipeline.
    pub fn ort_session(mut self, config: OrtSessionConfig) -> Self {
        self.ort_session_config = Some(config);
        self
    }

    /// Sets the layout detection model configuration.
    pub fn layout_detection_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.layout_detection_config = Some(config);
        self
    }

    /// Overrides the built-in layout model preset used to configure preprocessing/postprocessing.
    ///
    /// This is useful when the ONNX file name alone is not enough to infer the correct
    /// model family. Supported presets include:
    /// - `pp-doclayout_plus-l` (default)
    /// - `pp-doclayout-s`, `pp-doclayout-m`, `pp-doclayout-l`
    /// - `pp-docblocklayout`
    /// - `picodet_layout_1x`, `picodet_layout_1x_table`
    /// - `rt-detr-h_layout_3cls`, `rt-detr-h_layout_17cls`
    pub fn layout_model_name(mut self, name: impl Into<String>) -> Self {
        self.layout_model_name = Some(name.into());
        self
    }

    /// Sets the region detection model name preset.
    ///
    /// This is used to load the correct preprocessing/postprocessing for the region
    /// detection model. Supported presets: `PP-DocBlockLayout`.
    pub fn region_model_name(mut self, name: impl Into<String>) -> Self {
        self.region_model_name = Some(name.into());
        self
    }

    /// Sets the wired table structure model name preset.
    ///
    /// Supported presets: `SLANet`, `SLANeXt_wired`.
    pub fn wired_table_structure_model_name(mut self, name: impl Into<String>) -> Self {
        self.wired_table_structure_model_name = Some(name.into());
        self
    }

    /// Sets the wireless table structure model name preset.
    ///
    /// Supported presets: `SLANet_plus`.
    pub fn wireless_table_structure_model_name(mut self, name: impl Into<String>) -> Self {
        self.wireless_table_structure_model_name = Some(name.into());
        self
    }

    /// Sets the wired table cell detection model name preset.
    ///
    /// Supported presets: `RT-DETR-L_wired_table_cell_det`.
    pub fn wired_table_cell_model_name(mut self, name: impl Into<String>) -> Self {
        self.wired_table_cell_model_name = Some(name.into());
        self
    }

    /// Sets the wireless table cell detection model name preset.
    ///
    /// Supported presets: `RT-DETR-L_wireless_table_cell_det`.
    pub fn wireless_table_cell_model_name(mut self, name: impl Into<String>) -> Self {
        self.wireless_table_cell_model_name = Some(name.into());
        self
    }

    /// Sets the text detection model name preset.
    ///
    /// Supported presets: `PP-OCRv5_mobile_det`, `PP-OCRv5_server_det`.
    pub fn text_detection_model_name(mut self, name: impl Into<String>) -> Self {
        self.text_detection_model_name = Some(name.into());
        self
    }

    /// Sets the text recognition model name preset.
    ///
    /// Supported presets: `PP-OCRv5_mobile_rec`, `PP-OCRv5_server_rec`.
    pub fn text_recognition_model_name(mut self, name: impl Into<String>) -> Self {
        self.text_recognition_model_name = Some(name.into());
        self
    }

    /// Sets the batch size for image-level processing.
    ///
    /// Note: Currently not used in structure analysis as each image is processed individually.
    pub fn image_batch_size(mut self, size: usize) -> Self {
        self.image_batch_size = Some(size);
        self
    }

    /// Sets the batch size for region-level processing (text recognition).
    ///
    /// Controls how many text regions are processed together during OCR recognition.
    /// Larger values improve throughput but use more memory.
    pub fn region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = Some(size);
        self
    }

    /// Adds document orientation detection to the pipeline.
    ///
    /// This component detects and corrects document rotation (0°, 90°, 180°, 270°).
    /// Should be run before other processing for best results.
    pub fn with_document_orientation(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.document_orientation_model = Some(model_path.into());
        self
    }

    /// Adds document rectification to the pipeline.
    ///
    /// This component corrects document distortion and perspective issues.
    /// Should be run after orientation detection if both are enabled.
    pub fn with_document_rectification(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.document_rectification_model = Some(model_path.into());
        self
    }

    /// Adds region detection to the pipeline (PP-DocBlockLayout).
    ///
    /// This component detects document regions (columns, blocks) for hierarchical
    /// layout ordering. Region blocks provide grouping information for improved
    /// reading order within multi-column or complex layouts.
    ///
    /// # PP-StructureV3 Integration
    ///
    /// When enabled, the pipeline uses region detection results to:
    /// 1. Group layout elements by their parent regions
    /// 2. Apply XY-cut ordering within each region
    /// 3. Order regions based on their relative positions
    pub fn with_region_detection(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.region_detection_model = Some(model_path.into());
        self
    }

    /// Adds seal text detection to the pipeline.
    ///
    /// This component detects circular/curved seal and stamp text regions.
    /// Seal regions will be included in the layout elements.
    pub fn with_seal_text_detection(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.seal_text_detection_model = Some(model_path.into());
        self
    }

    /// Adds table classification to the pipeline.
    ///
    /// This component classifies tables as wired or wireless.
    pub fn with_table_classification(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.table_classification_model = Some(model_path.into());
        self
    }

    /// Sets the table classification configuration.
    pub fn table_classification_config(mut self, config: TableClassificationConfig) -> Self {
        self.table_classification_config = Some(config);
        self
    }

    /// Adds table orientation detection to the pipeline.
    ///
    /// This component detects if tables are rotated (0°, 90°, 180°, 270°) and corrects them
    /// before structure recognition. Uses the same model as document orientation detection
    /// (PP-LCNet_x1_0_doc_ori).
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the orientation classification model (same as document orientation)
    pub fn with_table_orientation(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.table_orientation_model = Some(model_path.into());
        self
    }

    /// Sets whether to use end-to-end mode for wired table recognition.
    ///
    /// When enabled, cell detection model is skipped and only the table structure
    /// recognition model's cell output is used. When disabled, RT-DETR cell detection
    /// provides more precise cell bounding boxes.
    ///
    /// Default: `false` (use cell detection for wired tables)
    pub fn use_e2e_wired_table_rec(mut self, enabled: bool) -> Self {
        self.use_e2e_wired_table_rec = enabled;
        self
    }

    /// Sets whether to use end-to-end mode for wireless table recognition.
    ///
    /// When enabled, cell detection model is skipped and only the table structure
    /// recognition model's cell output is used. When disabled, RT-DETR cell detection
    /// provides more precise cell bounding boxes.
    ///
    /// Default: `true` (E2E mode for wireless tables)
    pub fn use_e2e_wireless_table_rec(mut self, enabled: bool) -> Self {
        self.use_e2e_wireless_table_rec = enabled;
        self
    }

    /// Adds table cell detection to the pipeline.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the table cell detection model
    /// * `cell_type` - Type of cells to detect: "wired" or "wireless"
    pub fn with_table_cell_detection(
        mut self,
        model_path: impl Into<PathBuf>,
        cell_type: impl Into<String>,
    ) -> Self {
        self.table_cell_detection_model = Some(model_path.into());
        self.table_cell_detection_type = Some(cell_type.into());
        self
    }

    /// Sets the table cell detection configuration.
    pub fn table_cell_detection_config(mut self, config: TableCellDetectionConfig) -> Self {
        self.table_cell_detection_config = Some(config);
        self
    }

    /// Adds table structure recognition to the pipeline.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the table structure recognition model
    /// * `table_type` - Type of table structure: "wired" or "wireless"
    ///
    /// This component recognizes the structure of tables and outputs HTML.
    pub fn with_table_structure_recognition(
        mut self,
        model_path: impl Into<PathBuf>,
        table_type: impl Into<String>,
    ) -> Self {
        self.table_structure_recognition_model = Some(model_path.into());
        self.table_structure_recognition_type = Some(table_type.into());
        self
    }

    /// Sets the dictionary path for table structure recognition.
    ///
    /// The dictionary file should match the model type:
    /// - `table_structure_dict_ch.txt` for Chinese
    /// - `table_structure_dict.txt` for English
    /// - `table_master_structure_dict.txt` for extended tags
    pub fn table_structure_dict_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.table_structure_dict_path = Some(path.into());
        self
    }

    /// Sets the table structure recognition configuration.
    pub fn table_structure_recognition_config(
        mut self,
        config: TableStructureRecognitionConfig,
    ) -> Self {
        self.table_structure_recognition_config = Some(config);
        self
    }

    /// Adds wired table structure recognition model.
    ///
    /// When both wired and wireless models are configured along with table classification,
    /// the system automatically selects the appropriate model based on classification results.
    pub fn with_wired_table_structure(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.wired_table_structure_model = Some(model_path.into());
        self
    }

    /// Adds wireless table structure recognition model.
    ///
    /// When both wired and wireless models are configured along with table classification,
    /// the system automatically selects the appropriate model based on classification results.
    pub fn with_wireless_table_structure(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.wireless_table_structure_model = Some(model_path.into());
        self
    }

    /// Adds wired table cell detection model.
    ///
    /// When both wired and wireless models are configured along with table classification,
    /// the system automatically selects the appropriate model based on classification results.
    pub fn with_wired_table_cell_detection(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.wired_table_cell_model = Some(model_path.into());
        self
    }

    /// Adds wireless table cell detection model.
    ///
    /// When both wired and wireless models are configured along with table classification,
    /// the system automatically selects the appropriate model based on classification results.
    pub fn with_wireless_table_cell_detection(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.wireless_table_cell_model = Some(model_path.into());
        self
    }

    /// Adds formula recognition to the pipeline.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the formula recognition model
    /// * `tokenizer_path` - Path to the tokenizer JSON file
    /// * `model_type` - Type of formula model: "pp_formulanet" or "unimernet"
    ///
    /// This component recognizes mathematical formulas and outputs LaTeX.
    pub fn with_formula_recognition(
        mut self,
        model_path: impl Into<PathBuf>,
        tokenizer_path: impl Into<PathBuf>,
        model_type: impl Into<String>,
    ) -> Self {
        self.formula_recognition_model = Some(model_path.into());
        self.formula_tokenizer_path = Some(tokenizer_path.into());
        self.formula_recognition_type = Some(model_type.into());
        self
    }

    /// Sets the formula recognition configuration.
    pub fn formula_recognition_config(mut self, config: FormulaRecognitionConfig) -> Self {
        self.formula_recognition_config = Some(config);
        self
    }

    /// Integrates OCR into the pipeline for text extraction.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model` - Path to the text detection model
    /// * `text_recognition_model` - Path to the text recognition model
    /// * `character_dict_path` - Path to the character dictionary file
    pub fn with_ocr(
        mut self,
        text_detection_model: impl Into<PathBuf>,
        text_recognition_model: impl Into<PathBuf>,
        character_dict_path: impl Into<PathBuf>,
    ) -> Self {
        self.text_detection_model = Some(text_detection_model.into());
        self.text_recognition_model = Some(text_recognition_model.into());
        self.character_dict_path = Some(character_dict_path.into());
        self
    }

    /// Adds text line orientation detection to the OCR pipeline.
    ///
    /// This component detects whether text lines are upright (0°) or inverted (180°),
    /// which helps improve OCR accuracy for documents with mixed text orientations.
    ///
    /// # PP-StructureV3 Integration
    ///
    /// When enabled, detected text lines are classified before recognition:
    /// - Lines classified as 180° rotated are flipped before OCR
    /// - This improves accuracy for documents scanned upside-down or with mixed orientations
    pub fn with_text_line_orientation(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.text_line_orientation_model = Some(model_path.into());
        self
    }

    /// Sets the text detection configuration.
    pub fn text_detection_config(mut self, config: TextDetectionConfig) -> Self {
        self.text_detection_config = Some(config);
        self
    }

    /// Sets the text recognition configuration.
    pub fn text_recognition_config(mut self, config: TextRecognitionConfig) -> Self {
        self.text_recognition_config = Some(config);
        self
    }

    /// Builds the structure analyzer runtime.
    ///
    /// This method instantiates all adapters and returns a ready-to-use structure analyzer.
    pub fn build(self) -> Result<OARStructure, OCRError> {
        // Load character dictionary if OCR is enabled
        let char_dict = if let Some(ref dict_path) = self.character_dict_path {
            Some(
                std::fs::read_to_string(dict_path).map_err(|e| OCRError::InvalidInput {
                    message: format!(
                        "Failed to read character dictionary from '{}': {}",
                        dict_path.display(),
                        e
                    ),
                })?,
            )
        } else {
            None
        };

        // Build document orientation adapter if enabled
        let document_orientation_adapter = if let Some(ref model_path) =
            self.document_orientation_model
        {
            use crate::domain::adapters::DocumentOrientationAdapterBuilder;

            let mut builder = DocumentOrientationAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::document_orientation(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build document rectification adapter if enabled
        let rectification_adapter = if let Some(ref model_path) = self.document_rectification_model
        {
            use crate::domain::adapters::UVDocRectifierAdapterBuilder;

            let mut builder = UVDocRectifierAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::document_rectification(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build layout detection adapter (required)
        let mut layout_builder = LayoutDetectionAdapterBuilder::new();

        // Use explicit model name or default
        let layout_model_config = if let Some(name) = &self.layout_model_name {
            use crate::domain::adapters::LayoutModelConfig;
            match name.as_str() {
                "picodet_layout_1x" => LayoutModelConfig::picodet_layout_1x(),
                "picodet_layout_1x_table" => LayoutModelConfig::picodet_layout_1x_table(),
                "picodet_s_layout_3cls" => LayoutModelConfig::picodet_s_layout_3cls(),
                "picodet_l_layout_3cls" => LayoutModelConfig::picodet_l_layout_3cls(),
                "picodet_s_layout_17cls" => LayoutModelConfig::picodet_s_layout_17cls(),
                "picodet_l_layout_17cls" => LayoutModelConfig::picodet_l_layout_17cls(),
                "rt-detr-h_layout_3cls" => LayoutModelConfig::rtdetr_h_layout_3cls(),
                "rt-detr-h_layout_17cls" => LayoutModelConfig::rtdetr_h_layout_17cls(),
                "pp-docblocklayout" => LayoutModelConfig::pp_docblocklayout(),
                "pp-doclayout-s" => LayoutModelConfig::pp_doclayout_s(),
                "pp-doclayout-m" => LayoutModelConfig::pp_doclayout_m(),
                "pp-doclayout-l" => LayoutModelConfig::pp_doclayout_l(),
                "pp-doclayout_plus-l" => LayoutModelConfig::pp_doclayout_plus_l(),
                _ => LayoutModelConfig::pp_doclayout_plus_l(),
            }
        } else {
            // Default fallback
            crate::domain::adapters::LayoutModelConfig::pp_doclayout_plus_l()
        };

        layout_builder = layout_builder.model_config(layout_model_config);

        // If caller didn't provide an explicit layout config, fall back to PP-StructureV3 defaults.
        let effective_layout_cfg = self
            .layout_detection_config
            .clone()
            .unwrap_or_else(LayoutDetectionConfig::with_pp_structurev3_defaults);
        layout_builder = layout_builder.with_config(effective_layout_cfg);

        if let Some(ref ort_config) = self.ort_session_config {
            layout_builder = layout_builder.with_ort_config(ort_config.clone());
        }

        let layout_detection_adapter = Arc::new(TaskAdapter::layout_detection(
            layout_builder.build(&self.layout_detection_model)?,
        )) as Arc<dyn DynModelAdapter>;

        // Build region detection adapter if enabled (PP-DocBlockLayout)
        let region_detection_adapter = if let Some(ref model_path) = self.region_detection_model {
            use crate::domain::adapters::LayoutModelConfig;
            let mut region_builder = LayoutDetectionAdapterBuilder::new();

            // Use model name to select configuration, default to PP-DocBlockLayout
            let region_model_config = if let Some(ref name) = self.region_model_name {
                match name.to_lowercase().replace("-", "_").as_str() {
                    "pp_docblocklayout" => LayoutModelConfig::pp_docblocklayout(),
                    _ => LayoutModelConfig::pp_docblocklayout(),
                }
            } else {
                LayoutModelConfig::pp_docblocklayout()
            };
            region_builder = region_builder.model_config(region_model_config);

            // PP-StructureV3 region detection uses merge_bboxes_mode="small".
            let mut region_cfg = LayoutDetectionConfig::default();
            let mut merge_modes = std::collections::HashMap::new();
            merge_modes.insert(
                "region".to_string(),
                crate::domain::tasks::layout_detection::MergeBboxMode::Small,
            );
            region_cfg.class_merge_modes = Some(merge_modes);
            region_builder = region_builder.with_config(region_cfg);

            if let Some(ref ort_config) = self.ort_session_config {
                region_builder = region_builder.with_ort_config(ort_config.clone());
            }

            let adapter = region_builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::layout_detection(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build table classification adapter if enabled
        let table_classification_adapter = if let Some(ref model_path) =
            self.table_classification_model
        {
            let mut builder = TableClassificationAdapterBuilder::new();

            if let Some(ref config) = self.table_classification_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::table_classification(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build table orientation adapter if enabled (reuses document orientation model)
        // This detects rotated tables (0°, 90°, 180°, 270°) before structure recognition
        let table_orientation_adapter = if let Some(ref model_path) = self.table_orientation_model {
            use crate::domain::adapters::DocumentOrientationAdapterBuilder;

            let mut builder = DocumentOrientationAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::document_orientation(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build table cell detection adapter if enabled
        let table_cell_detection_adapter = if let Some(ref model_path) =
            self.table_cell_detection_model
        {
            let cell_type = self.table_cell_detection_type.as_deref().unwrap_or("wired");

            use crate::domain::adapters::table_cell_detection_adapter::TableCellModelConfig;

            let model_config = match cell_type {
                "wired" => TableCellModelConfig::rtdetr_l_wired_table_cell_det(),
                "wireless" => TableCellModelConfig::rtdetr_l_wireless_table_cell_det(),
                _ => {
                    return Err(OCRError::config_error_detailed(
                        "table_cell_detection",
                        format!(
                            "Invalid cell type '{}': must be 'wired' or 'wireless'",
                            cell_type
                        ),
                    ));
                }
            };

            let mut builder = TableCellDetectionAdapterBuilder::new().model_config(model_config);

            if let Some(ref config) = self.table_cell_detection_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::table_cell_detection(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build table structure recognition adapter if enabled
        let table_structure_recognition_adapter = if let Some(ref model_path) =
            self.table_structure_recognition_model
        {
            let table_type = self
                .table_structure_recognition_type
                .as_deref()
                .unwrap_or("wired");
            let dict_path = self
                    .table_structure_dict_path
                    .clone()
                    .ok_or_else(|| {
                        OCRError::config_error_detailed(
                            "table_structure_recognition",
                            "Dictionary path is required. Call table_structure_dict_path() when enabling table structure recognition.".to_string(),
                        )
                    })?;

            let adapter: Arc<dyn DynModelAdapter> = match table_type {
                "wired" => {
                    let mut builder = SLANetWiredAdapterBuilder::new().dict_path(dict_path.clone());

                    if let Some(ref config) = self.table_structure_recognition_config {
                        builder = builder.with_config(config.clone());
                    }

                    if let Some(ref ort_config) = self.ort_session_config {
                        builder = builder.with_ort_config(ort_config.clone());
                    }

                    let adapter = builder.build(model_path)?;
                    Arc::new(TaskAdapter::table_structure_recognition(adapter))
                }
                "wireless" => {
                    let mut builder =
                        SLANetWirelessAdapterBuilder::new().dict_path(dict_path.clone());

                    if let Some(ref config) = self.table_structure_recognition_config {
                        builder = builder.with_config(config.clone());
                    }

                    if let Some(ref ort_config) = self.ort_session_config {
                        builder = builder.with_ort_config(ort_config.clone());
                    }

                    let adapter = builder.build(model_path)?;
                    Arc::new(TaskAdapter::table_structure_recognition(adapter))
                }
                _ => {
                    return Err(OCRError::config_error_detailed(
                        "table_structure_recognition",
                        format!(
                            "Invalid table type '{}': must be 'wired' or 'wireless'",
                            table_type
                        ),
                    ));
                }
            };

            Some(adapter)
        } else {
            None
        };

        // Build wired/wireless table structure adapters for auto-switch (PP-StructureV3)
        let wired_table_structure_adapter = if let Some(ref model_path) =
            self.wired_table_structure_model
        {
            let dict_path = self.table_structure_dict_path.clone().ok_or_else(|| {
                OCRError::config_error_detailed(
                    "wired_table_structure",
                    "Dictionary path is required. Call table_structure_dict_path() when enabling table structure recognition.".to_string(),
                )
            })?;

            let mut builder = SLANetWiredAdapterBuilder::new().dict_path(dict_path);

            if let Some(ref config) = self.table_structure_recognition_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::table_structure_recognition(adapter))
                as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        let wireless_table_structure_adapter = if let Some(ref model_path) =
            self.wireless_table_structure_model
        {
            let dict_path = self.table_structure_dict_path.clone().ok_or_else(|| {
                OCRError::config_error_detailed(
                    "wireless_table_structure",
                    "Dictionary path is required. Call table_structure_dict_path() when enabling table structure recognition.".to_string(),
                )
            })?;

            let mut builder = SLANetWirelessAdapterBuilder::new().dict_path(dict_path);

            if let Some(ref config) = self.table_structure_recognition_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::table_structure_recognition(adapter))
                as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build wired/wireless table cell detection adapters for auto-switch
        let wired_table_cell_adapter = if let Some(ref model_path) = self.wired_table_cell_model {
            use crate::domain::adapters::table_cell_detection_adapter::TableCellModelConfig;

            let model_config = TableCellModelConfig::rtdetr_l_wired_table_cell_det();
            let mut builder = TableCellDetectionAdapterBuilder::new().model_config(model_config);

            if let Some(ref config) = self.table_cell_detection_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::table_cell_detection(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        let wireless_table_cell_adapter = if let Some(ref model_path) =
            self.wireless_table_cell_model
        {
            use crate::domain::adapters::table_cell_detection_adapter::TableCellModelConfig;

            let model_config = TableCellModelConfig::rtdetr_l_wireless_table_cell_det();
            let mut builder = TableCellDetectionAdapterBuilder::new().model_config(model_config);

            if let Some(ref config) = self.table_cell_detection_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::table_cell_detection(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build formula recognition adapter if enabled
        let formula_recognition_adapter = if let Some(ref model_path) =
            self.formula_recognition_model
        {
            let tokenizer_path = self.formula_tokenizer_path.as_ref().ok_or_else(|| {
                OCRError::config_error_detailed(
                    "formula_recognition",
                    "Tokenizer path is required for formula recognition".to_string(),
                )
            })?;

            let model_type = self.formula_recognition_type.as_deref().ok_or_else(|| {
                OCRError::config_error_detailed(
                    "formula_recognition",
                    "Model type is required (must be 'pp_formulanet' or 'unimernet')".to_string(),
                )
            })?;

            let adapter: Arc<dyn DynModelAdapter> = match model_type.to_lowercase().as_str() {
                "pp_formulanet" | "pp-formulanet" => {
                    let mut builder = PPFormulaNetAdapterBuilder::new();

                    builder = builder.tokenizer_path(tokenizer_path);

                    // Note: region_batch_size batching not yet implemented for structure analysis

                    if let Some(ref config) = self.formula_recognition_config {
                        builder = builder.task_config(config.clone());
                    }

                    if let Some(ref ort_config) = self.ort_session_config {
                        builder = builder.with_ort_config(ort_config.clone());
                    }

                    let adapter = builder.build(model_path)?;
                    Arc::new(TaskAdapter::formula_recognition(adapter))
                }
                "unimernet" => {
                    let mut builder = UniMERNetFormulaAdapterBuilder::new();

                    builder = builder.tokenizer_path(tokenizer_path);

                    // Note: region_batch_size batching not yet implemented for structure analysis

                    if let Some(ref config) = self.formula_recognition_config {
                        builder = builder.task_config(config.clone());
                    }

                    if let Some(ref ort_config) = self.ort_session_config {
                        builder = builder.with_ort_config(ort_config.clone());
                    }

                    let adapter = builder.build(model_path)?;
                    Arc::new(TaskAdapter::formula_recognition(adapter))
                }
                _ => {
                    return Err(OCRError::config_error_detailed(
                        "formula_recognition",
                        format!(
                            "Invalid model type '{}': must be 'pp_formulanet' or 'unimernet'",
                            model_type
                        ),
                    ));
                }
            };

            Some(adapter)
        } else {
            None
        };

        // Build seal text detection adapter if enabled
        let seal_text_detection_adapter = if let Some(ref model_path) =
            self.seal_text_detection_model
        {
            use crate::domain::adapters::SealTextDetectionAdapterBuilder;

            let mut builder = SealTextDetectionAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::seal_text_detection(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build text detection adapter if enabled.
        //
        // PP-StructureV3 overall OCR uses DB preprocess with:
        // - limit_side_len=736
        // - limit_type="min"
        // We fill these defaults here (only for the structure pipeline) unless the caller
        // explicitly overrides them via `text_detection_config`.
        let text_detection_adapter = if let Some(ref model_path) = self.text_detection_model {
            let mut builder = TextDetectionAdapterBuilder::new();

            // Note: image_batch_size batching not yet implemented for structure analysis

            let mut effective_cfg = self.text_detection_config.clone().unwrap_or_default();
            if effective_cfg.limit_side_len.is_none() {
                effective_cfg.limit_side_len = Some(736);
            }
            if effective_cfg.limit_type.is_none() {
                effective_cfg.limit_type = Some(crate::processors::LimitType::Min);
            }
            builder = builder.with_config(effective_cfg);

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::text_detection(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build text line orientation adapter if enabled (PP-StructureV3)
        let text_line_orientation_adapter = if let Some(ref model_path) =
            self.text_line_orientation_model
        {
            use crate::domain::adapters::TextLineOrientationAdapterBuilder;

            let mut builder = TextLineOrientationAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::text_line_orientation(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build text recognition adapter if enabled
        let text_recognition_adapter = if let Some(ref model_path) = self.text_recognition_model {
            let dict = char_dict.ok_or_else(|| OCRError::InvalidInput {
                message: "Character dictionary is required for text recognition".to_string(),
            })?;

            // Parse dict into Vec<String> - one character per line
            let char_vec: Vec<String> = dict.lines().map(|s| s.to_string()).collect();

            let mut builder = TextRecognitionAdapterBuilder::new().character_dict(char_vec);

            // Note: region_batch_size batching not yet implemented for structure analysis

            if let Some(ref config) = self.text_recognition_config {
                builder = builder.with_config(config.clone());
            }

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(model_path)?;
            Some(Arc::new(TaskAdapter::text_recognition(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        let pipeline = StructurePipeline {
            document_orientation_adapter,
            rectification_adapter,
            layout_detection_adapter,
            region_detection_adapter,
            table_classification_adapter,
            table_orientation_adapter,
            table_cell_detection_adapter,
            table_structure_recognition_adapter,
            wired_table_structure_adapter,
            wireless_table_structure_adapter,
            wired_table_cell_adapter,
            wireless_table_cell_adapter,
            use_e2e_wired_table_rec: self.use_e2e_wired_table_rec,
            use_e2e_wireless_table_rec: self.use_e2e_wireless_table_rec,
            formula_recognition_adapter,
            seal_text_detection_adapter,
            text_detection_adapter,
            text_line_orientation_adapter,
            text_recognition_adapter,
            region_batch_size: self.region_batch_size,
        };

        Ok(OARStructure { pipeline })
    }
}

/// Runtime for document structure analysis.
///
/// This struct represents a configured and ready-to-use document structure analyzer.
#[derive(Debug)]
pub struct OARStructure {
    pipeline: StructurePipeline,
}

impl OARStructure {
    /// Refinement of overall OCR results using layout boxes.
    ///
    /// This mirrors two behaviors in `layout_parsing/pipeline_v2.py`:
    /// 1) If a single overall OCR box overlaps multiple layout blocks, re-recognize
    ///    the intersection crop per block and replace/append OCR entries.
    /// 2) If a non-vision layout block has no matched OCR text, run recognition
    ///    on the layout bbox crop as a fallback.
    ///
    /// We approximate poly handling with AABB intersections. The resulting
    /// crops are stored into `TextRegion` with `dt_poly/rec_poly` set to the crop box.
    fn refine_overall_ocr_with_layout(
        text_regions: &mut Vec<crate::oarocr::TextRegion>,
        layout_elements: &[crate::domain::structure::LayoutElement],
        region_blocks: Option<&[crate::domain::structure::RegionBlock]>,
        page_image: &image::RgbImage,
        text_recognition_adapter: &Arc<dyn DynModelAdapter>,
        region_batch_size: usize,
    ) -> Result<(), OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::domain::structure::LayoutElementType;
        use crate::processors::BoundingBox;
        use crate::utils::BBoxCrop;

        if text_regions.is_empty() || layout_elements.is_empty() {
            return Ok(());
        }

        fn aabb_intersection(b1: &BoundingBox, b2: &BoundingBox) -> Option<BoundingBox> {
            let x1 = b1.x_min().max(b2.x_min());
            let y1 = b1.y_min().max(b2.y_min());
            let x2 = b1.x_max().min(b2.x_max());
            let y2 = b1.y_max().min(b2.y_max());
            if x2 - x1 <= 1.0 || y2 - y1 <= 1.0 {
                None
            } else {
                Some(BoundingBox::from_coords(x1, y1, x2, y2))
            }
        }

        // Layout boxes that participate in OCR matching (exclude specialized types).
        let is_excluded_layout = |t: LayoutElementType| {
            matches!(
                t,
                LayoutElementType::Formula
                    | LayoutElementType::FormulaNumber
                    | LayoutElementType::Table
                    | LayoutElementType::Seal
            )
        };

        // Build overlap maps: ocr_idx -> layout_idxes.
        // `get_sub_regions_ocr_res` uses get_overlap_boxes_idx:
        // any overlap with intersection width/height >3px counts as a match (no ratio threshold).
        let min_pixels = 3.0;
        let mut matched_ocr: Vec<Vec<usize>> = vec![Vec::new(); text_regions.len()];
        for (ocr_idx, region) in text_regions.iter().enumerate() {
            for (layout_idx, elem) in layout_elements.iter().enumerate() {
                if is_excluded_layout(elem.element_type) {
                    continue;
                }
                let inter_x_min = region.bounding_box.x_min().max(elem.bbox.x_min());
                let inter_y_min = region.bounding_box.y_min().max(elem.bbox.y_min());
                let inter_x_max = region.bounding_box.x_max().min(elem.bbox.x_max());
                let inter_y_max = region.bounding_box.y_max().min(elem.bbox.y_max());
                if inter_x_max - inter_x_min > min_pixels && inter_y_max - inter_y_min > min_pixels
                {
                    matched_ocr[ocr_idx].push(layout_idx);
                }
            }
        }

        // 1) Cross-layout re-recognition for OCR boxes matched to multiple blocks.
        let mut appended_regions: Vec<crate::oarocr::TextRegion> = Vec::new();
        let original_ocr_len = text_regions.len();
        let mut multi_layout_ocr_count = 0usize;
        let mut multi_layout_crop_count = 0usize;

        for ocr_idx in 0..original_ocr_len {
            let layout_ids = matched_ocr[ocr_idx].clone();
            if layout_ids.len() <= 1 {
                continue;
            }
            multi_layout_ocr_count += 1;

            let ocr_box = text_regions[ocr_idx].bounding_box.clone();

            let mut crops: Vec<image::RgbImage> = Vec::new();
            let mut crop_boxes: Vec<(BoundingBox, bool)> = Vec::new(); // (bbox, is_first)

            for (j, layout_idx) in layout_ids.iter().enumerate() {
                let layout_box = &layout_elements[*layout_idx].bbox;
                let Some(crop_box) = aabb_intersection(&ocr_box, layout_box) else {
                    continue;
                };

                // Suppress existing OCR text fully covered by this crop (IoU > 0.8).
                for (other_idx, other_region) in text_regions.iter_mut().enumerate() {
                    if other_idx == ocr_idx {
                        continue;
                    }
                    if other_region.bounding_box.iou(&crop_box) > 0.8 {
                        other_region.text = None;
                    }
                }

                if let Ok(crop_img) = BBoxCrop::crop_bounding_box(page_image, &crop_box) {
                    crops.push(crop_img);
                    crop_boxes.push((crop_box, j == 0));
                }
            }
            multi_layout_crop_count += crop_boxes.len();

            if crops.is_empty() {
                continue;
            }

            // Run recognition on all crops (batched).
            let mut rec_texts: Vec<String> = Vec::with_capacity(crops.len());
            let mut rec_scores: Vec<f32> = Vec::with_capacity(crops.len());

            for batch_start in (0..crops.len()).step_by(region_batch_size.max(1)) {
                let batch_end = (batch_start + region_batch_size).min(crops.len());
                let batch: Vec<_> = crops[batch_start..batch_end].to_vec();
                let rec_input = DynTaskInput::from_text_recognition(
                    crate::domain::tasks::TextRecognitionInput::new(batch),
                );
                let rec_output = text_recognition_adapter.execute_dyn(rec_input)?;
                if let Ok(rec_result) = rec_output.into_text_recognition() {
                    rec_texts.extend(rec_result.texts);
                    rec_scores.extend(rec_result.scores);
                }
            }

            for ((crop_box, is_first), (text, score)) in crop_boxes
                .into_iter()
                .zip(rec_texts.into_iter().zip(rec_scores.into_iter()))
            {
                if text.is_empty() {
                    continue;
                }
                if is_first {
                    text_regions[ocr_idx].bounding_box = crop_box.clone();
                    text_regions[ocr_idx].dt_poly = Some(crop_box.clone());
                    text_regions[ocr_idx].rec_poly = Some(crop_box.clone());
                    text_regions[ocr_idx].text = Some(Arc::from(text));
                    text_regions[ocr_idx].confidence = Some(score);
                } else {
                    appended_regions.push(crate::oarocr::TextRegion {
                        bounding_box: crop_box.clone(),
                        dt_poly: Some(crop_box.clone()),
                        rec_poly: Some(crop_box),
                        text: Some(Arc::from(text)),
                        confidence: Some(score),
                        orientation_angle: None,
                        word_boxes: None,
                    });
                }
            }
        }

        if !appended_regions.is_empty() {
            text_regions.extend(appended_regions);
        }

        // 2) Layout-bbox fallback OCR for blocks with no matched text.
        // Prefer region blocks for hierarchy if present, but OCR fallback is driven by layout boxes.
        let mut fallback_blocks = 0usize;
        for elem in layout_elements.iter() {
            if is_excluded_layout(elem.element_type) {
                continue;
            }
            if matches!(
                elem.element_type,
                LayoutElementType::Image | LayoutElementType::Chart
            ) {
                continue;
            }

            let mut has_text = false;
            for region in text_regions.iter() {
                if !region.text.as_ref().map(|t| !t.is_empty()).unwrap_or(false) {
                    continue;
                }
                let inter_x_min = region.bounding_box.x_min().max(elem.bbox.x_min());
                let inter_y_min = region.bounding_box.y_min().max(elem.bbox.y_min());
                let inter_x_max = region.bounding_box.x_max().min(elem.bbox.x_max());
                let inter_y_max = region.bounding_box.y_max().min(elem.bbox.y_max());
                if inter_x_max - inter_x_min > min_pixels && inter_y_max - inter_y_min > min_pixels
                {
                    has_text = true;
                    break;
                }
            }

            if has_text {
                continue;
            }
            fallback_blocks += 1;

            // Crop layout bbox and run recognition.
            if let Ok(crop_img) = BBoxCrop::crop_bounding_box(page_image, &elem.bbox) {
                let rec_input = DynTaskInput::from_text_recognition(
                    crate::domain::tasks::TextRecognitionInput::new(vec![crop_img]),
                );
                let rec_output = text_recognition_adapter.execute_dyn(rec_input)?;
                if let Ok(rec_result) = rec_output.into_text_recognition()
                    && let (Some(text), Some(score)) =
                        (rec_result.texts.first(), rec_result.scores.first())
                    && !text.is_empty()
                {
                    let crop_box = elem.bbox.clone();
                    text_regions.push(crate::oarocr::TextRegion {
                        bounding_box: crop_box.clone(),
                        dt_poly: Some(crop_box.clone()),
                        rec_poly: Some(crop_box),
                        text: Some(Arc::from(text.as_str())),
                        confidence: Some(*score),
                        orientation_angle: None,
                        word_boxes: None,
                    });
                }
            }
        }

        tracing::info!(
            "overall OCR refine: multi-layout OCR boxes={}, crops={}, fallback layout blocks={}",
            multi_layout_ocr_count,
            multi_layout_crop_count,
            fallback_blocks
        );

        // Region blocks currently do not require special handling here; they are only
        // used for ordering later. Kept as a parameter for future parity work.
        let _ = region_blocks;

        Ok(())
    }

    /// Split OCR bounding boxes based on table cell boundaries when they span multiple cells.
    ///
    /// This mirrors `split_ocr_bboxes_by_table_cells`:
    /// - For each OCR box that overlaps >= k cells (by intersection / cell_area > 0.5),
    ///   split the box vertically at cell boundaries
    /// - Re-run text recognition on each split crop
    /// - Replace the original OCR box/text with the split boxes/texts
    fn split_ocr_bboxes_by_table_cells(
        tables: &[TableResult],
        text_regions: &mut Vec<crate::oarocr::TextRegion>,
        page_image: &image::RgbImage,
        text_recognition_adapter: &Arc<dyn DynModelAdapter>,
    ) -> Result<(), OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::processors::BoundingBox;

        // Collect all cell boxes in [x1, y1, x2, y2] format
        let mut cell_boxes: Vec<[f32; 4]> = Vec::new();
        for table in tables {
            for cell in &table.cells {
                let x1 = cell.bbox.x_min();
                let y1 = cell.bbox.y_min();
                let x2 = cell.bbox.x_max();
                let y2 = cell.bbox.y_max();
                if x2 > x1 && y2 > y1 {
                    cell_boxes.push([x1, y1, x2, y2]);
                }
            }
        }

        if cell_boxes.is_empty() || text_regions.is_empty() {
            return Ok(());
        }

        // Calculate intersection / cell_area (matches calculate_iou in split_ocr_bboxes_by_table_cells)
        fn overlap_ratio_box_over_cell(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
            let x_left = box1[0].max(box2[0]);
            let y_top = box1[1].max(box2[1]);
            let x_right = box1[2].min(box2[2]);
            let y_bottom = box1[3].min(box2[3]);

            if x_right <= x_left || y_bottom <= y_top {
                return 0.0;
            }

            let inter_area = (x_right - x_left) * (y_bottom - y_top);
            let cell_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
            if cell_area <= 0.0 {
                0.0
            } else {
                inter_area / cell_area
            }
        }

        // Find cells that significantly overlap with an OCR box
        fn get_overlapping_cells(
            ocr_box: &[f32; 4],
            cells: &[[f32; 4]],
            threshold: f32,
        ) -> Vec<usize> {
            let mut overlapping = Vec::new();
            for (idx, cell) in cells.iter().enumerate() {
                if overlap_ratio_box_over_cell(ocr_box, cell) > threshold {
                    overlapping.push(idx);
                }
            }
            // Sort by cell x1 (left to right)
            overlapping.sort_by(|&i, &j| {
                cells[i][0]
                    .partial_cmp(&cells[j][0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            overlapping
        }

        // Split an OCR box vertically at cell boundaries.
        fn split_box_by_cells(
            ocr_box: &[f32; 4],
            cell_indices: &[usize],
            cells: &[[f32; 4]],
        ) -> Vec<[f32; 4]> {
            if cell_indices.is_empty() {
                return vec![*ocr_box];
            }

            let mut split_boxes: Vec<[f32; 4]> = Vec::new();
            let cells_to_split: Vec<[f32; 4]> = cell_indices.iter().map(|&i| cells[i]).collect();

            // Leading segment before first cell
            if ocr_box[0] < cells_to_split[0][0] {
                split_boxes.push([ocr_box[0], ocr_box[1], cells_to_split[0][0], ocr_box[3]]);
            }

            // Segments overlapping each cell and gaps between cells
            for (i, current_cell) in cells_to_split.iter().enumerate() {
                // Cell overlap segment
                split_boxes.push([
                    ocr_box[0].max(current_cell[0]),
                    ocr_box[1],
                    ocr_box[2].min(current_cell[2]),
                    ocr_box[3],
                ]);

                // Gap between this cell and the next cell
                if i + 1 < cells_to_split.len() {
                    let next_cell = cells_to_split[i + 1];
                    if current_cell[2] < next_cell[0] {
                        split_boxes.push([current_cell[2], ocr_box[1], next_cell[0], ocr_box[3]]);
                    }
                }
            }

            // Trailing segment after last cell
            let last_cell = cells_to_split[cells_to_split.len() - 1];
            if last_cell[2] < ocr_box[2] {
                split_boxes.push([last_cell[2], ocr_box[1], ocr_box[2], ocr_box[3]]);
            }

            // Deduplicate boxes
            let mut unique = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for b in split_boxes {
                let key = (
                    b[0].to_bits(),
                    b[1].to_bits(),
                    b[2].to_bits(),
                    b[3].to_bits(),
                );
                if seen.insert(key) {
                    unique.push(b);
                }
            }
            unique
        }

        let k_min_cells = 2usize;
        let overlap_threshold = CELL_OVERLAP_IOU_THRESHOLD;

        let mut new_regions: Vec<crate::oarocr::TextRegion> =
            Vec::with_capacity(text_regions.len());

        for region in text_regions.iter() {
            let ocr_box = [
                region.bounding_box.x_min(),
                region.bounding_box.y_min(),
                region.bounding_box.x_max(),
                region.bounding_box.y_max(),
            ];

            let overlapping_cells = get_overlapping_cells(&ocr_box, &cell_boxes, overlap_threshold);

            // If OCR box does not span multiple cells, keep as-is
            if overlapping_cells.len() < k_min_cells {
                new_regions.push(region.clone());
                continue;
            }

            let split_boxes = split_box_by_cells(&ocr_box, &overlapping_cells, &cell_boxes);

            for box_coords in split_boxes {
                // Convert to integer crop coordinates, clamp to image bounds
                let img_w = page_image.width() as i32;
                let img_h = page_image.height() as i32;

                let mut x1 = box_coords[0].floor() as i32;
                let mut y1 = box_coords[1].floor() as i32;
                let mut x2 = box_coords[2].ceil() as i32;
                let mut y2 = box_coords[3].ceil() as i32;

                x1 = x1.clamp(0, img_w.saturating_sub(1));
                y1 = y1.clamp(0, img_h.saturating_sub(1));
                x2 = x2.clamp(0, img_w);
                y2 = y2.clamp(0, img_h);

                if x2 - x1 <= 1 || y2 - y1 <= 1 {
                    continue;
                }

                let crop_w = (x2 - x1) as u32;
                let crop_h = (y2 - y1) as u32;
                if crop_w <= 1 || crop_h <= 1 {
                    continue;
                }

                let x1u = x1 as u32;
                let y1u = y1 as u32;
                if x1u >= page_image.width() || y1u >= page_image.height() {
                    continue;
                }
                let crop_w = crop_w.min(page_image.width() - x1u);
                let crop_h = crop_h.min(page_image.height() - y1u);
                if crop_w <= 1 || crop_h <= 1 {
                    continue;
                }

                let crop =
                    image::imageops::crop_imm(page_image, x1u, y1u, crop_w, crop_h).to_image();

                let rec_input = DynTaskInput::from_text_recognition(
                    crate::domain::tasks::TextRecognitionInput::new(vec![crop]),
                );
                let rec_output = text_recognition_adapter.execute_dyn(rec_input)?;
                if let Ok(rec_result) = rec_output.into_text_recognition()
                    && let (Some(text), Some(score)) =
                        (rec_result.texts.first(), rec_result.scores.first())
                    && !text.is_empty()
                {
                    let bbox = BoundingBox::from_coords(
                        box_coords[0],
                        box_coords[1],
                        box_coords[2],
                        box_coords[3],
                    );
                    new_regions.push(crate::oarocr::TextRegion {
                        bounding_box: bbox.clone(),
                        dt_poly: Some(bbox.clone()),
                        rec_poly: Some(bbox),
                        text: Some(Arc::from(text.as_str())),
                        confidence: Some(*score),
                        orientation_angle: None,
                        word_boxes: None,
                    });
                }
            }
        }

        *text_regions = new_regions;
        Ok(())
    }

    fn detect_layout_and_regions(
        &self,
        page_image: &image::RgbImage,
    ) -> Result<
        (
            Vec<crate::domain::structure::LayoutElement>,
            Option<Vec<crate::domain::structure::RegionBlock>>,
        ),
        OCRError,
    > {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;
        use crate::domain::structure::{LayoutElement, LayoutElementType, RegionBlock};

        let input = DynTaskInput::from_images(ImageTaskInput::new(vec![page_image.clone()]));
        let layout_output = self.pipeline.layout_detection_adapter.execute_dyn(input)?;
        let layout_result = layout_output.into_layout_detection()?;

        let mut layout_elements: Vec<LayoutElement> = Vec::new();
        if let Some(elements) = layout_result.elements.first() {
            for element in elements {
                let element_type_enum = LayoutElementType::from_label(&element.element_type);
                layout_elements.push(
                    LayoutElement::new(element.bbox.clone(), element_type_enum, element.score)
                        .with_label(element.element_type.clone()),
                );
            }
        }

        let mut detected_region_blocks: Option<Vec<RegionBlock>> = None;
        if let Some(ref region_adapter) = self.pipeline.region_detection_adapter {
            let region_input =
                DynTaskInput::from_images(ImageTaskInput::new(vec![page_image.clone()]));
            if let Ok(region_output) = region_adapter.execute_dyn(region_input)
                && let Ok(region_result) = region_output.into_layout_detection()
                && let Some(region_elements) = region_result.elements.first()
                && !region_elements.is_empty()
            {
                let blocks: Vec<RegionBlock> = region_elements
                    .iter()
                    .map(|e| RegionBlock {
                        bbox: e.bbox.clone(),
                        confidence: e.score,
                        order_index: None,
                        element_indices: Vec::new(),
                    })
                    .collect();
                detected_region_blocks = Some(blocks);
            }
        }

        if layout_elements.len() > 1 {
            let removed = crate::domain::structure::remove_overlapping_layout_elements(
                &mut layout_elements,
                LAYOUT_OVERLAP_IOU_THRESHOLD,
            );
            if removed > 0 {
                tracing::info!(
                    "Removing {} overlapping layout elements (threshold={})",
                    removed,
                    LAYOUT_OVERLAP_IOU_THRESHOLD
                );
            }
        }

        crate::domain::structure::apply_standardized_layout_label_fixes(&mut layout_elements);

        Ok((layout_elements, detected_region_blocks))
    }

    fn recognize_formulas(
        &self,
        page_image: &image::RgbImage,
        layout_elements: &[crate::domain::structure::LayoutElement],
    ) -> Result<Vec<crate::domain::structure::FormulaResult>, OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;
        use crate::domain::structure::FormulaResult;
        use crate::utils::BBoxCrop;

        let Some(ref formula_adapter) = self.pipeline.formula_recognition_adapter else {
            return Ok(Vec::new());
        };

        let formula_elements: Vec<_> = layout_elements
            .iter()
            .filter(|e| e.element_type.is_formula())
            .collect();

        if formula_elements.is_empty() {
            tracing::debug!(
                "Formula recognition skipped: no formula regions from layout detection"
            );
            return Ok(Vec::new());
        }

        let mut crops = Vec::new();
        let mut bboxes = Vec::new();

        for elem in &formula_elements {
            match BBoxCrop::crop_bounding_box(page_image, &elem.bbox) {
                Ok(crop) => {
                    crops.push(crop);
                    bboxes.push(elem.bbox.clone());
                }
                Err(err) => {
                    tracing::warn!("Formula region crop failed: {}", err);
                }
            }
        }

        if crops.is_empty() {
            tracing::debug!(
                "Formula recognition skipped: all formula crops failed for {} regions",
                formula_elements.len()
            );
            return Ok(Vec::new());
        }

        let input = DynTaskInput::from_images(ImageTaskInput::new(crops));
        let formula_output = formula_adapter.execute_dyn(input)?;

        let Ok(formula_result) = formula_output.into_formula_recognition() else {
            return Ok(Vec::new());
        };

        let mut formulas = Vec::new();
        for ((bbox, formula), score) in bboxes
            .into_iter()
            .zip(formula_result.formulas.into_iter())
            .zip(formula_result.scores.into_iter())
        {
            let width = bbox.x_max() - bbox.x_min();
            let height = bbox.y_max() - bbox.y_min();
            if width <= 0.0 || height <= 0.0 {
                tracing::warn!(
                    "Skipping formula with non-positive bbox dimensions: w={:.2}, h={:.2}",
                    width,
                    height
                );
                continue;
            }

            formulas.push(FormulaResult {
                bbox,
                latex: formula,
                confidence: score.unwrap_or(0.0),
            });
        }

        Ok(formulas)
    }

    fn detect_seal_text(
        &self,
        page_image: &image::RgbImage,
        layout_elements: &mut Vec<crate::domain::structure::LayoutElement>,
    ) -> Result<(), OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;
        use crate::domain::structure::{LayoutElement, LayoutElementType};
        use crate::processors::Point;
        use crate::utils::BBoxCrop;

        let Some(ref seal_adapter) = self.pipeline.seal_text_detection_adapter else {
            return Ok(());
        };

        let seal_regions: Vec<_> = layout_elements
            .iter()
            .filter(|e| e.element_type == LayoutElementType::Seal)
            .map(|e| e.bbox.clone())
            .collect();

        if seal_regions.is_empty() {
            tracing::debug!("Seal detection skipped: no seal regions from layout detection");
            return Ok(());
        }

        let mut seal_crops = Vec::new();
        let mut crop_offsets = Vec::new();

        for region_bbox in &seal_regions {
            match BBoxCrop::crop_bounding_box(page_image, region_bbox) {
                Ok(crop) => {
                    seal_crops.push(crop);
                    crop_offsets.push((region_bbox.x_min(), region_bbox.y_min()));
                }
                Err(err) => {
                    tracing::warn!("Seal region crop failed: {}", err);
                }
            }
        }

        if seal_crops.is_empty() {
            return Ok(());
        }

        let input = DynTaskInput::from_images(ImageTaskInput::new(seal_crops));
        let seal_output = seal_adapter.execute_dyn(input)?;

        if let Ok(seal_result) = seal_output.into_seal_text_detection() {
            for ((dx, dy), detections) in
                crop_offsets.iter().zip(seal_result.detections.into_iter())
            {
                for detection in detections {
                    let translated_bbox = crate::processors::BoundingBox::new(
                        detection
                            .bbox
                            .points
                            .iter()
                            .map(|p| Point::new(p.x + dx, p.y + dy))
                            .collect(),
                    );

                    layout_elements.push(
                        LayoutElement::new(
                            translated_bbox,
                            LayoutElementType::Seal,
                            detection.score,
                        )
                        .with_label("seal".to_string()),
                    );
                }
            }
        }

        Ok(())
    }

    fn sort_layout_elements_enhanced(
        layout_elements: &mut Vec<crate::domain::structure::LayoutElement>,
        page_width: f32,
        page_height: f32,
    ) {
        use crate::processors::layout_sorting::sort_layout_enhanced;

        if layout_elements.is_empty() {
            return;
        }

        let sortable_elements: Vec<_> = layout_elements
            .iter()
            .map(|e| (e.bbox.clone(), e.element_type))
            .collect();

        let sorted_indices = sort_layout_enhanced(&sortable_elements, page_width, page_height);
        if sorted_indices.len() != layout_elements.len() {
            return;
        }

        let sorted_elements: Vec<_> = sorted_indices
            .into_iter()
            .map(|idx| layout_elements[idx].clone())
            .collect();
        *layout_elements = sorted_elements;
    }

    fn assign_region_block_membership(
        region_blocks: &mut [crate::domain::structure::RegionBlock],
        layout_elements: &[crate::domain::structure::LayoutElement],
    ) {
        use std::cmp::Ordering;

        if region_blocks.is_empty() {
            return;
        }

        region_blocks.sort_by(|a, b| {
            a.bbox
                .y_min()
                .partial_cmp(&b.bbox.y_min())
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    a.bbox
                        .x_min()
                        .partial_cmp(&b.bbox.x_min())
                        .unwrap_or(Ordering::Equal)
                })
        });

        for (i, region) in region_blocks.iter_mut().enumerate() {
            region.order_index = Some((i + 1) as u32);
            region.element_indices.clear();
        }

        if layout_elements.is_empty() {
            return;
        }

        for (elem_idx, elem) in layout_elements.iter().enumerate() {
            let elem_area = elem.bbox.area();
            if elem_area <= 0.0 {
                continue;
            }

            let mut best_region: Option<usize> = None;
            let mut best_ioa = 0.0f32;

            for (region_idx, region) in region_blocks.iter().enumerate() {
                let intersection = elem.bbox.intersection_area(&region.bbox);
                if intersection <= 0.0 {
                    continue;
                }
                let ioa = intersection / elem_area;
                if ioa > best_ioa {
                    best_ioa = ioa;
                    best_region = Some(region_idx);
                }
            }

            if let Some(region_idx) = best_region
                && best_ioa >= REGION_MEMBERSHIP_IOA_THRESHOLD
            {
                region_blocks[region_idx].element_indices.push(elem_idx);
            }
        }
    }

    fn run_overall_ocr(
        &self,
        page_image: &image::RgbImage,
        layout_elements: &[crate::domain::structure::LayoutElement],
        region_blocks: Option<&[crate::domain::structure::RegionBlock]>,
    ) -> Result<Vec<crate::oarocr::TextRegion>, OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;
        use crate::oarocr::TextRegion;
        use std::sync::Arc;

        let Some(ref text_detection_adapter) = self.pipeline.text_detection_adapter else {
            return Ok(Vec::new());
        };
        let Some(ref text_recognition_adapter) = self.pipeline.text_recognition_adapter else {
            return Ok(Vec::new());
        };

        let mut text_regions = Vec::new();

        // Mask formula regions before text detection (PP-StructureV3 behavior).
        let mut ocr_image = page_image.clone();
        let mask_bboxes: Vec<crate::processors::BoundingBox> = layout_elements
            .iter()
            .filter(|e| e.element_type.is_formula())
            .map(|e| e.bbox.clone())
            .collect();

        if !mask_bboxes.is_empty() {
            crate::utils::mask_regions(&mut ocr_image, &mask_bboxes, [255, 255, 255]);
        }

        // Text detection (on masked image).
        let input = DynTaskInput::from_images(ImageTaskInput::new(vec![ocr_image.clone()]));
        let det_output = text_detection_adapter.execute_dyn(input)?;

        let mut detection_boxes = if let Ok(det_result) = det_output.clone().into_text_detection()
            && let Some(detections) = det_result.detections.first()
        {
            detections
                .iter()
                .map(|d| d.bbox.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        // Debug: raw text detection boxes from overall OCR (before any splitting).
        let raw_detection_boxes = detection_boxes.clone();
        if tracing::enabled!(tracing::Level::DEBUG) && !raw_detection_boxes.is_empty() {
            let raw_rects: Vec<[f32; 4]> = raw_detection_boxes
                .iter()
                .map(|b| [b.x_min(), b.y_min(), b.x_max(), b.y_max()])
                .collect();
            tracing::debug!("overall OCR text det boxes (raw): {:?}", raw_rects);
        }

        // Cross-layout re-recognition: split text det boxes that span multiple layout/region boxes.
        if !detection_boxes.is_empty() {
            let mut split_boxes = Vec::new();
            let mut split_count = 0usize;

            let container_boxes: Vec<crate::processors::BoundingBox> =
                if let Some(regions) = region_blocks {
                    regions.iter().map(|r| r.bbox.clone()).collect()
                } else {
                    layout_elements
                        .iter()
                        .filter(|e| {
                            matches!(
                            e.element_type,
                            crate::domain::structure::LayoutElementType::DocTitle
                                | crate::domain::structure::LayoutElementType::ParagraphTitle
                                | crate::domain::structure::LayoutElementType::Text
                                | crate::domain::structure::LayoutElementType::Content
                                | crate::domain::structure::LayoutElementType::Abstract
                                | crate::domain::structure::LayoutElementType::Header
                                | crate::domain::structure::LayoutElementType::Footer
                                | crate::domain::structure::LayoutElementType::Footnote
                                | crate::domain::structure::LayoutElementType::Number
                                | crate::domain::structure::LayoutElementType::Reference
                                | crate::domain::structure::LayoutElementType::ReferenceContent
                                | crate::domain::structure::LayoutElementType::Algorithm
                                | crate::domain::structure::LayoutElementType::AsideText
                                | crate::domain::structure::LayoutElementType::List
                                | crate::domain::structure::LayoutElementType::FigureTitle
                                | crate::domain::structure::LayoutElementType::TableTitle
                                | crate::domain::structure::LayoutElementType::ChartTitle
                                | crate::domain::structure::LayoutElementType::FigureTableChartTitle
                        )
                        })
                        .map(|e| e.bbox.clone())
                        .collect()
                };

            if !container_boxes.is_empty() {
                for bbox in detection_boxes.into_iter() {
                    let mut intersections: Vec<crate::processors::BoundingBox> = Vec::new();
                    let self_area = bbox.area();
                    if self_area <= 0.0 {
                        split_boxes.push(bbox);
                        continue;
                    }

                    for container in &container_boxes {
                        let inter_x_min = bbox.x_min().max(container.x_min());
                        let inter_y_min = bbox.y_min().max(container.y_min());
                        let inter_x_max = bbox.x_max().min(container.x_max());
                        let inter_y_max = bbox.y_max().min(container.y_max());

                        if inter_x_max - inter_x_min <= 2.0 || inter_y_max - inter_y_min <= 2.0 {
                            continue;
                        }

                        let inter_bbox = crate::processors::BoundingBox::from_coords(
                            inter_x_min,
                            inter_y_min,
                            inter_x_max,
                            inter_y_max,
                        );
                        let inter_area = inter_bbox.area();
                        if inter_area <= 0.0 {
                            continue;
                        }

                        let ioa = inter_area / self_area;
                        if ioa >= TEXT_BOX_SPLIT_IOA_THRESHOLD {
                            intersections.push(inter_bbox);
                        }
                    }

                    if intersections.len() >= 2 {
                        split_count += intersections.len();
                        split_boxes.extend(intersections);
                    } else {
                        split_boxes.push(bbox);
                    }
                }

                if split_count > 0 {
                    tracing::debug!(
                        "Cross-layout re-recognition: split {} text boxes into {} sub-boxes",
                        split_count,
                        split_boxes.len()
                    );
                }

                detection_boxes = split_boxes;
            }
        }

        // Debug: boxes actually used for recognition cropping (after cross-layout splitting).
        if tracing::enabled!(tracing::Level::DEBUG) && !detection_boxes.is_empty() {
            let pre_rec_rects: Vec<[f32; 4]> = detection_boxes
                .iter()
                .map(|b| [b.x_min(), b.y_min(), b.x_max(), b.y_max()])
                .collect();
            tracing::debug!(
                "overall OCR boxes pre-recognition (after splitting): {:?}",
                pre_rec_rects
            );
        }

        if !detection_boxes.is_empty() {
            use crate::oarocr::processors::{EdgeProcessor, TextCroppingProcessor};

            let processor = TextCroppingProcessor::new(true);
            let cropped =
                processor.process((Arc::new(page_image.clone()), detection_boxes.clone()))?;

            let mut cropped_images: Vec<image::RgbImage> = Vec::new();
            let mut valid_indices: Vec<usize> = Vec::new();

            for (idx, crop_result) in cropped.into_iter().enumerate() {
                if let Some(img) = crop_result {
                    cropped_images.push((*img).clone());
                    valid_indices.push(idx);
                }
            }

            if !cropped_images.is_empty() {
                let mut items: Vec<(usize, f32, image::RgbImage)> = cropped_images
                    .into_iter()
                    .zip(valid_indices)
                    .map(|(img, det_idx)| {
                        let wh_ratio = img.width() as f32 / img.height().max(1) as f32;
                        (det_idx, wh_ratio, img)
                    })
                    .collect();

                items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let batch_size = self.pipeline.region_batch_size.unwrap_or(8).max(1);

                while !items.is_empty() {
                    let take_n = batch_size.min(items.len());
                    let mut batch_items: Vec<(usize, f32, image::RgbImage)> =
                        items.drain(0..take_n).collect();

                    if let Some(ref tlo_adapter) = self.pipeline.text_line_orientation_adapter {
                        let tlo_imgs: Vec<_> =
                            batch_items.iter().map(|(_, _, img)| img.clone()).collect();
                        let tlo_input = DynTaskInput::from_images(ImageTaskInput::new(tlo_imgs));
                        if let Ok(tlo_output) = tlo_adapter.execute_dyn(tlo_input)
                            && let Ok(tlo_result) = tlo_output.into_text_line_orientation()
                        {
                            for (i, classifications) in
                                tlo_result.classifications.iter().enumerate()
                            {
                                if i >= batch_items.len() {
                                    break;
                                }
                                if let Some(top_cls) = classifications.first()
                                    && top_cls.class_id == 1
                                {
                                    batch_items[i].2 =
                                        image::imageops::rotate180(&batch_items[i].2);
                                }
                            }
                        }
                    }

                    let mut det_indices: Vec<usize> = Vec::with_capacity(batch_items.len());
                    let mut rec_imgs: Vec<image::RgbImage> = Vec::with_capacity(batch_items.len());
                    for (det_idx, _ratio, img) in batch_items {
                        det_indices.push(det_idx);
                        rec_imgs.push(img);
                    }

                    let rec_input = DynTaskInput::from_text_recognition(
                        crate::domain::tasks::TextRecognitionInput::new(rec_imgs),
                    );
                    if let Ok(rec_output) = text_recognition_adapter.execute_dyn(rec_input)
                        && let Ok(rec_result) = rec_output.into_text_recognition()
                    {
                        for ((det_idx, text), score) in det_indices
                            .into_iter()
                            .zip(rec_result.texts.into_iter())
                            .zip(rec_result.scores.into_iter())
                        {
                            if text.is_empty() {
                                continue;
                            }

                            let bbox = detection_boxes[det_idx].clone();
                            text_regions.push(TextRegion {
                                bounding_box: bbox.clone(),
                                dt_poly: Some(bbox.clone()),
                                rec_poly: Some(bbox),
                                text: Some(Arc::from(text)),
                                confidence: Some(score),
                                orientation_angle: None,
                                word_boxes: None,
                            });
                        }
                    }
                }
            }
        }

        let batch_size = self.pipeline.region_batch_size.unwrap_or(8).max(1);
        Self::refine_overall_ocr_with_layout(
            &mut text_regions,
            layout_elements,
            region_blocks,
            page_image,
            text_recognition_adapter,
            batch_size,
        )?;

        Ok(text_regions)
    }

    /// Analyzes the structure of a document image from a path.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the input image
    ///
    /// # Returns
    ///
    /// A `StructureResult` containing detected layout elements, tables, formulas, and text.
    pub fn predict(&self, image_path: impl Into<PathBuf>) -> Result<StructureResult, OCRError> {
        let image_path = image_path.into();

        // Load the image
        let image = image::open(&image_path).map_err(|e| OCRError::InvalidInput {
            message: format!(
                "failed to load image from '{}': {}",
                image_path.display(),
                e
            ),
        })?;

        let mut result = self.predict_image(image.to_rgb8())?;
        result.input_path = std::sync::Arc::from(image_path.to_string_lossy().as_ref());
        Ok(result)
    }

    /// Analyzes the structure of a document image.
    ///
    /// This method is the core implementation for structure analysis and can be called
    /// directly with an in-memory image.
    ///
    /// # Arguments
    ///
    /// * `image` - The input RGB image
    ///
    /// # Returns
    ///
    /// A `StructureResult` containing detected layout elements, tables, formulas, and text.
    pub fn predict_image(&self, image: image::RgbImage) -> Result<StructureResult, OCRError> {
        use crate::oarocr::preprocess::DocumentPreprocessor;

        let preprocessor = DocumentPreprocessor::new(
            self.pipeline.document_orientation_adapter.clone(),
            self.pipeline.rectification_adapter.clone(),
        );
        let preprocess = preprocessor.preprocess(image)?;
        let current_image = preprocess.image;
        let orientation_angle = preprocess.orientation_angle;
        let rectified_img = preprocess.rectified_img;
        let rotation = preprocess.rotation;

        let (mut layout_elements, mut detected_region_blocks) =
            self.detect_layout_and_regions(&current_image)?;

        let mut tables = Vec::new();
        let mut formulas = self.recognize_formulas(&current_image, &layout_elements)?;

        self.detect_seal_text(&current_image, &mut layout_elements)?;

        // Sort layout elements after all detection/augmentation steps (formulas/seals)
        // so reading order includes any injected blocks.
        if !layout_elements.is_empty() {
            let (width, height) = if let Some(img) = &rectified_img {
                (img.width() as f32, img.height() as f32)
            } else {
                (current_image.width() as f32, current_image.height() as f32)
            };
            Self::sort_layout_elements_enhanced(&mut layout_elements, width, height);
        }

        if let Some(ref mut regions) = detected_region_blocks {
            Self::assign_region_block_membership(regions, &layout_elements);
        }

        let mut text_regions = self.run_overall_ocr(
            &current_image,
            &layout_elements,
            detected_region_blocks.as_deref(),
        )?;

        {
            let analyzer = crate::oarocr::table_analyzer::TableAnalyzer::new(
                crate::oarocr::table_analyzer::TableAnalyzerConfig {
                    table_classification_adapter: self
                        .pipeline
                        .table_classification_adapter
                        .clone(),
                    table_orientation_adapter: self.pipeline.table_orientation_adapter.clone(),
                    table_structure_recognition_adapter: self
                        .pipeline
                        .table_structure_recognition_adapter
                        .clone(),
                    wired_table_structure_adapter: self
                        .pipeline
                        .wired_table_structure_adapter
                        .clone(),
                    wireless_table_structure_adapter: self
                        .pipeline
                        .wireless_table_structure_adapter
                        .clone(),
                    table_cell_detection_adapter: self
                        .pipeline
                        .table_cell_detection_adapter
                        .clone(),
                    wired_table_cell_adapter: self.pipeline.wired_table_cell_adapter.clone(),
                    wireless_table_cell_adapter: self.pipeline.wireless_table_cell_adapter.clone(),
                    use_e2e_wired_table_rec: self.pipeline.use_e2e_wired_table_rec,
                    use_e2e_wireless_table_rec: self.pipeline.use_e2e_wireless_table_rec,
                },
            );
            tables.extend(analyzer.analyze_tables(
                &current_image,
                &layout_elements,
                &formulas,
                &text_regions,
            )?);
        }

        // 5b. Optional OCR box splitting by table cell boundaries.
        //
        // Split OCR boxes that span multiple table cells horizontally and re-recognize
        // the smaller segments. This mirrors `split_ocr_bboxes_by_table_cells`:
        // - For each OCR box that overlaps >= k table cells, split at cell boundaries
        // - Re-run recognition on each split crop
        // - Replace the original OCR box with the split boxes + texts
        if !tables.is_empty()
            && !text_regions.is_empty()
            && let Some(ref text_rec_adapter) = self.pipeline.text_recognition_adapter
        {
            Self::split_ocr_bboxes_by_table_cells(
                &tables,
                &mut text_regions,
                &current_image,
                text_rec_adapter,
            )?;
        }

        // Transform bounding boxes back to original coordinate system if rotation was applied.
        // If rectification was applied, keep coordinates in rectified space (UVDoc can't be inverted).
        if let Some(rot) = rotation {
            let rotated_width = rot.rotated_width;
            let rotated_height = rot.rotated_height;
            let angle = rot.angle;

            // Transform layout elements
            for element in &mut layout_elements {
                element.bbox =
                    element
                        .bbox
                        .rotate_back_to_original(angle, rotated_width, rotated_height);
            }

            // Transform table bounding boxes and cells
            for table in &mut tables {
                table.bbox =
                    table
                        .bbox
                        .rotate_back_to_original(angle, rotated_width, rotated_height);

                // Transform cell bounding boxes
                for cell in &mut table.cells {
                    cell.bbox =
                        cell.bbox
                            .rotate_back_to_original(angle, rotated_width, rotated_height);
                }
            }

            // Transform formula bounding boxes
            for formula in &mut formulas {
                formula.bbox =
                    formula
                        .bbox
                        .rotate_back_to_original(angle, rotated_width, rotated_height);
            }

            // Transform text region polygons, bounding boxes, and word boxes
            for region in &mut text_regions {
                region.dt_poly = region
                    .dt_poly
                    .take()
                    .map(|poly| poly.rotate_back_to_original(angle, rotated_width, rotated_height));
                region.rec_poly = region
                    .rec_poly
                    .take()
                    .map(|poly| poly.rotate_back_to_original(angle, rotated_width, rotated_height));
                region.bounding_box = region.bounding_box.rotate_back_to_original(
                    angle,
                    rotated_width,
                    rotated_height,
                );

                if let Some(ref word_boxes) = region.word_boxes {
                    let transformed_word_boxes: Vec<_> = word_boxes
                        .iter()
                        .map(|wb| wb.rotate_back_to_original(angle, rotated_width, rotated_height))
                        .collect();
                    region.word_boxes = Some(transformed_word_boxes);
                }
            }

            // Transform region block bounding boxes
            if let Some(ref mut regions) = detected_region_blocks {
                for region in regions.iter_mut() {
                    region.bbox =
                        region
                            .bbox
                            .rotate_back_to_original(angle, rotated_width, rotated_height);
                }
            }
        }

        // Construct and return result
        let mut result = StructureResult {
            input_path: Arc::from("memory"),
            index: 0,
            layout_elements,
            tables,
            formulas,
            text_regions: if text_regions.is_empty() {
                None
            } else {
                Some(text_regions)
            },
            orientation_angle,
            region_blocks: detected_region_blocks,
            rectified_img,
        };

        // Stitch text results into layout elements and tables
        // Note: When region_blocks is present, stitching preserves the hierarchical order
        use crate::oarocr::stitching::{ResultStitcher, StitchConfig};
        let stitch_cfg = StitchConfig::default();
        ResultStitcher::stitch_with_config(&mut result, &stitch_cfg);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_builder_new() {
        let builder = OARStructureBuilder::new("models/layout.onnx");
        assert_eq!(
            builder.layout_detection_model,
            PathBuf::from("models/layout.onnx")
        );
        assert!(builder.table_classification_model.is_none());
        assert!(builder.formula_recognition_model.is_none());
    }

    #[test]
    fn test_structure_builder_with_table_components() {
        let builder = OARStructureBuilder::new("models/layout.onnx")
            .with_table_classification("models/table_cls.onnx")
            .with_table_cell_detection("models/table_cell.onnx", "wired")
            .with_table_structure_recognition("models/table_struct.onnx", "wired")
            .table_structure_dict_path("models/table_structure_dict.txt");

        assert!(builder.table_classification_model.is_some());
        assert!(builder.table_cell_detection_model.is_some());
        assert!(builder.table_structure_recognition_model.is_some());
        assert_eq!(builder.table_cell_detection_type, Some("wired".to_string()));
        assert_eq!(
            builder.table_structure_recognition_type,
            Some("wired".to_string())
        );
        assert_eq!(
            builder.table_structure_dict_path,
            Some(PathBuf::from("models/table_structure_dict.txt"))
        );
    }

    #[test]
    fn test_structure_builder_with_formula() {
        let builder = OARStructureBuilder::new("models/layout.onnx").with_formula_recognition(
            "models/formula.onnx",
            "models/tokenizer.json",
            "pp_formulanet",
        );

        assert!(builder.formula_recognition_model.is_some());
        assert!(builder.formula_tokenizer_path.is_some());
        assert_eq!(
            builder.formula_recognition_type,
            Some("pp_formulanet".to_string())
        );
    }

    #[test]
    fn test_structure_builder_with_ocr() {
        let builder = OARStructureBuilder::new("models/layout.onnx").with_ocr(
            "models/det.onnx",
            "models/rec.onnx",
            "models/dict.txt",
        );

        assert!(builder.text_detection_model.is_some());
        assert!(builder.text_recognition_model.is_some());
        assert!(builder.character_dict_path.is_some());
    }

    #[test]
    fn test_structure_builder_with_configuration() {
        let layout_config = LayoutDetectionConfig {
            score_threshold: 0.5,
            max_elements: 100,
            ..Default::default()
        };

        let builder = OARStructureBuilder::new("models/layout.onnx")
            .layout_detection_config(layout_config.clone())
            .image_batch_size(4)
            .region_batch_size(64);

        assert!(builder.layout_detection_config.is_some());
        assert_eq!(builder.image_batch_size, Some(4));
        assert_eq!(builder.region_batch_size, Some(64));
    }
}
