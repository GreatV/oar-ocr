//! Document Structure Analysis Example
//!
//! This example demonstrates how to run the document structure pipeline built with
//! `OARStructureBuilder`. It performs layout detection and can optionally add table
//! analysis, formula recognition, seal detection, and integrated OCR.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features cuda --example structure -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Common Options
//!
//! * `--layout-model` - Path to the layout detection model (required)
//! * `--layout-model-name` - Layout model preset name (default: PP-DocLayout_plus-L)
//! * `--orientation-model` / `--rectification-model` - Optional document preprocessing
//! * `--region-model` - Optional region detection (PP-DocBlockLayout) for hierarchical ordering
//! * Table analysis (PP-StructureV3 auto-switch mode):
//!   - `--table-cls-model` - Table classification model (wired/wireless)
//!   - `--wired-structure-model` / `--wireless-structure-model` - Table structure models
//!   - `--wired-structure-model-name` / `--wireless-structure-model-name` - Model names
//!   - `--wired-cell-model` / `--wireless-cell-model` - Table cell detection models
//!   - `--wired-cell-model-name` / `--wireless-cell-model-name` - Model names
//!   - `--table-structure-dict` - Table structure dictionary
//! * Formula recognition:
//!   - `--formula-model` / `--formula-tokenizer` / `--formula-type` (pp_formulanet or unimernet)
//! * OCR integration:
//!   - `--text-det-model`, `--text-det-model-name` - Text detection model
//!   - `--text-rec-model`, `--text-rec-model-name` - Text recognition model
//!   - `--text-dict-path` - Character dictionary
//! * `--device` - Device to use (`cpu`, `cuda`, `cuda:0`, etc.) - Default: cuda
//!
//! # Supported Model Names
//!
//! ## Layout Detection Models
//! - `PP-DocLayout_plus-L` (default, 800x800, 20 classes)
//! - `PP-DocLayout-L`, `PP-DocLayout-M`, `PP-DocLayout-S` (640x640/480x480, 23 classes)
//! - `PP-DocBlockLayout` (640x640, region detection)
//! - `PicoDet-L_layout_17cls`, `PicoDet-S_layout_17cls` (640x640/480x480, 17 classes)
//! - `PicoDet-L_layout_3cls`, `PicoDet-S_layout_3cls` (640x640/480x480, 3 classes)
//! - `RT-DETR-H_layout_17cls`, `RT-DETR-H_layout_3cls` (640x640)
//!
//! ## Table Structure Models
//! - `SLANeXt_wired` (wired tables, default for wired)
//! - `SLANet_plus` (wireless tables, default for wireless)
//! - `SLANet` (basic)
//!
//! ## Table Cell Detection Models
//! - `RT-DETR-L_wired_table_cell_det` (default for wired)
//! - `RT-DETR-L_wireless_table_cell_det` (default for wireless)
//!
//! ## Text Detection Models
//! - `PP-OCRv5_server_det` (default, high accuracy)
//! - `PP-OCRv5_mobile_det` (faster, mobile)
//! - `PP-OCRv4_server_det`, `PP-OCRv4_mobile_det`
//!
//! ## Text Recognition Models
//! - `PP-OCRv5_server_rec` (default, high accuracy)
//! - `PP-OCRv5_mobile_rec` (faster, mobile)
//! - `PP-OCRv4_server_rec`, `PP-OCRv4_mobile_rec`
//!
//! # Examples
//!
//! ## Minimal (Layout Detection Only)
//!
//! ```bash
//! cargo run --release --features cuda --example structure -- \
//!   --layout-model models/PP-DocLayout_plus-L.onnx \
//!   --region-model models/PP-DocBlockLayout.onnx \
//!   document.jpg
//! ```
//!
//! ## Layout + OCR
//!
//! ```bash
//! cargo run --release --features cuda --example structure -- \
//!   --layout-model models/PP-DocLayout_plus-L.onnx \
//!   --region-model models/PP-DocBlockLayout.onnx \
//!   --text-det-model models/PP-OCRv5_server_det.onnx \
//!   --text-rec-model models/PP-OCRv5_server_rec.onnx \
//!   --text-dict-path models/ppocrv5_dict.txt \
//!   document.jpg
//! ```
//!
//! ## Using PicoDet Layout Model
//!
//! ```bash
//! cargo run --release --features cuda --example structure -- \
//!   --layout-model models/PicoDet-L_layout_17cls.onnx \
//!   --layout-model-name PicoDet-L_layout_17cls \
//!   document.jpg
//! ```
//!
//! ## Full PP-StructureV3 Pipeline
//!
//! ```bash
//! cargo run --release --features cuda --example structure -- \
//!   --layout-model models/PP-DocLayout_plus-L.onnx \
//!   --region-model models/PP-DocBlockLayout.onnx \
//!   --orientation-model models/PP-LCNet_x1_0_doc_ori.onnx \
//!   --rectification-model models/UVDoc.onnx \
//!   --table-cls-model models/PP-LCNet_x1_0_table_cls.onnx \
//!   --wired-structure-model models/SLANeXt_wired.onnx \
//!   --wireless-structure-model models/SLANet_plus.onnx \
//!   --wired-cell-model models/RT-DETR-L_wired_table_cell_det.onnx \
//!   --wireless-cell-model models/RT-DETR-L_wireless_table_cell_det.onnx \
//!   --table-structure-dict models/table_structure_dict_ch.txt \
//!   --formula-model models/PP-FormulaNet_plus-L.onnx \
//!   --formula-tokenizer models/pp_formulanet_tokenizer.json \
//!   --formula-type pp_formulanet \
//!   --seal-model models/PP-OCRv4_server_seal_det.onnx \
//!   --text-det-model models/PP-OCRv5_server_det.onnx \
//!   --text-rec-model models/PP-OCRv5_server_rec.onnx \
//!   --text-dict-path models/ppocrv5_dict.txt \
//!   --to-json --to-markdown \
//!   -o output/structure \
//!   document.jpg
//! ```
//!
//! # Model Reference (PP-StructureV3 Defaults)
//!
//! | Component | Model Name | Model Path Arg | Model Name Arg |
//! |-----------|------------|----------------|----------------|
//! | Layout Detection | PP-DocLayout_plus-L | `--layout-model` | `--layout-model-name` |
//! | Region Detection | PP-DocBlockLayout | `--region-model` | `--region-model-name` |
//! | Document Orientation | PP-LCNet_x1_0_doc_ori | `--orientation-model` | - |
//! | Document Rectification | UVDoc | `--rectification-model` | - |
//! | Table Classification | PP-LCNet_x1_0_table_cls | `--table-cls-model` | - |
//! | Wired Table Structure | SLANeXt_wired | `--wired-structure-model` | `--wired-structure-model-name` |
//! | Wireless Table Structure | SLANet_plus | `--wireless-structure-model` | `--wireless-structure-model-name` |
//! | Wired Cell Detection | RT-DETR-L_wired_table_cell_det | `--wired-cell-model` | `--wired-cell-model-name` |
//! | Wireless Cell Detection | RT-DETR-L_wireless_table_cell_det | `--wireless-cell-model` | `--wireless-cell-model-name` |
//! | Table Structure Dict | table_structure_dict_ch.txt | `--table-structure-dict` | - |
//! | Formula Recognition | PP-FormulaNet_plus-L | `--formula-model` | `--formula-type` |
//! | Seal Detection | PP-OCRv4_server_seal_det | `--seal-model` | - |
//! | Text Detection | PP-OCRv5_server_det | `--text-det-model` | `--text-det-model-name` |
//! | Text Recognition | PP-OCRv5_server_rec | `--text-rec-model` | `--text-rec-model-name` |
//! | Character Dict | ppocrv5_dict.txt | `--text-dict-path` | - |

mod utils;

use clap::Parser;
use oar_ocr::domain::structure::TableType;
use oar_ocr::domain::tasks::{
    FormulaRecognitionConfig, LayoutDetectionConfig, TextDetectionConfig, TextRecognitionConfig,
};
use oar_ocr::oarocr::OARStructureBuilder;
use oar_ocr::processors::LimitType;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use tracing::{error, info, warn};
use utils::parse_device_config;

/// Command-line arguments for the structure analysis example
#[derive(Parser)]
#[command(name = "structure")]
#[command(about = "Run document structure analysis with optional table/formula/OCR components")]
struct Args {
    /// Layout detection model path (required)
    #[arg(long = "layout-model")]
    layout_model: PathBuf,

    /// Layout model preset name.
    ///
    /// Used to select the correct built-in preprocessing/postprocessing preset.
    /// Supported values:
    /// - `PP-DocLayout_plus-L` (default, 800x800, 20 classes)
    /// - `PP-DocLayout-L`, `PP-DocLayout-M`, `PP-DocLayout-S`
    /// - `PP-DocBlockLayout` (region detection)
    /// - `PicoDet-L_layout_17cls`, `PicoDet-S_layout_17cls`
    /// - `PicoDet-L_layout_3cls`, `PicoDet-S_layout_3cls`
    /// - `RT-DETR-H_layout_17cls`, `RT-DETR-H_layout_3cls`
    #[arg(long = "layout-model-name", default_value = "PP-DocLayout_plus-L")]
    layout_model_name: String,

    /// Input images to analyze
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Optional document orientation classification model
    #[arg(long)]
    orientation_model: Option<PathBuf>,

    /// Optional document rectification model
    #[arg(long)]
    rectification_model: Option<PathBuf>,

    /// Optional region detection model (PP-DocBlockLayout) for hierarchical ordering
    #[arg(long = "region-model")]
    region_model: Option<PathBuf>,

    /// Region detection model name (default: PP-DocBlockLayout)
    #[arg(long = "region-model-name", default_value = "PP-DocBlockLayout")]
    region_model_name: String,

    /// Table classification model (wired vs wireless)
    #[arg(long = "table-cls-model")]
    table_cls_model: Option<PathBuf>,

    /// Table orientation detection model (reuses doc orientation model PP-LCNet_x1_0_doc_ori)
    /// Detects if tables are rotated (0°, 90°, 180°, 270°) before structure recognition
    #[arg(long = "table-orientation-model")]
    table_orientation_model: Option<PathBuf>,

    /// Wired table structure recognition model
    #[arg(long = "wired-structure-model")]
    wired_structure_model: Option<PathBuf>,

    /// Wired table structure model name (default: SLANeXt_wired)
    #[arg(long = "wired-structure-model-name", default_value = "SLANeXt_wired")]
    wired_structure_model_name: String,

    /// Wireless table structure recognition model
    #[arg(long = "wireless-structure-model")]
    wireless_structure_model: Option<PathBuf>,

    /// Wireless table structure model name (default: SLANet_plus)
    #[arg(long = "wireless-structure-model-name", default_value = "SLANet_plus")]
    wireless_structure_model_name: String,

    /// Wired table cell detection model
    #[arg(long = "wired-cell-model")]
    wired_cell_model: Option<PathBuf>,

    /// Wired table cell detection model name (default: RT-DETR-L_wired_table_cell_det)
    #[arg(
        long = "wired-cell-model-name",
        default_value = "RT-DETR-L_wired_table_cell_det"
    )]
    wired_cell_model_name: String,

    /// Wireless table cell detection model
    #[arg(long = "wireless-cell-model")]
    wireless_cell_model: Option<PathBuf>,

    /// Wireless table cell detection model name (default: RT-DETR-L_wireless_table_cell_det)
    #[arg(
        long = "wireless-cell-model-name",
        default_value = "RT-DETR-L_wireless_table_cell_det"
    )]
    wireless_cell_model_name: String,

    /// Table structure dictionary path (required when table models are provided)
    #[arg(long = "table-structure-dict")]
    table_structure_dict: Option<PathBuf>,

    /// Use end-to-end mode for wired table recognition (skip cell detection, default: false)
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    use_e2e_wired_table_rec: bool,

    /// Use end-to-end mode for wireless table recognition (skip cell detection, default: true)
    /// Use --use-e2e-wireless-table-rec=false to disable E2E mode and enable cell detection
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    use_e2e_wireless_table_rec: bool,

    /// Formula recognition model
    #[arg(long = "formula-model")]
    formula_model: Option<PathBuf>,

    /// Tokenizer path for formula recognition
    #[arg(long = "formula-tokenizer")]
    formula_tokenizer: Option<PathBuf>,

    /// Formula model type: `pp_formulanet` or `unimernet`
    #[arg(long = "formula-type")]
    formula_type: Option<String>,

    /// Seal text detection model
    #[arg(long = "seal-model")]
    seal_model: Option<PathBuf>,

    /// Text detection model for OCR integration
    #[arg(long = "text-det-model")]
    text_det_model: Option<PathBuf>,

    /// Text detection model name (default: PP-OCRv5_server_det)
    /// Supported: PP-OCRv5_server_det, PP-OCRv5_mobile_det, PP-OCRv4_server_det, PP-OCRv4_mobile_det
    #[arg(long = "text-det-model-name", default_value = "PP-OCRv5_server_det")]
    text_det_model_name: String,

    /// Text recognition model for OCR integration
    #[arg(long = "text-rec-model")]
    text_rec_model: Option<PathBuf>,

    /// Text recognition model name (default: PP-OCRv5_server_rec)
    /// Supported: PP-OCRv5_server_rec, PP-OCRv5_mobile_rec, PP-OCRv4_server_rec, PP-OCRv4_mobile_rec
    #[arg(long = "text-rec-model-name", default_value = "PP-OCRv5_server_rec")]
    text_rec_model_name: String,

    /// Character dictionary for OCR integration
    #[arg(long = "text-dict-path")]
    text_dict_path: Option<PathBuf>,

    /// Optional text line orientation model (PP-LCNet_x1_0_textline_ori).
    /// When provided, upright/180° text lines are corrected before recognition.
    #[arg(long = "textline-orientation-model")]
    textline_orientation_model: Option<PathBuf>,

    /// Device to use for inference (default: cuda)
    /// Supported: cpu, cuda, cuda:0, cuda:1, etc.
    #[arg(long, default_value = "cuda")]
    device: String,

    /// Layout detection score threshold (varies by class, 0.3-0.5)
    #[arg(long, default_value = "0.5")]
    layout_score_thresh: f32,

    /// Layout detection NMS threshold
    #[arg(long, default_value = "0.5")]
    layout_nms_thresh: f32,

    /// Enable NMS for layout detection (default: true)
    #[arg(long, default_value = "true")]
    layout_nms: bool,

    /// Formula recognition score threshold
    #[arg(long, default_value_t = 0.0)]
    formula_score_thresh: f32,

    /// Maximum formula length in tokens
    #[arg(long, default_value_t = 1536)]
    formula_max_length: usize,

    /// Text detection score threshold (DB thresh, default: 0.3)
    #[arg(long, default_value = "0.3")]
    det_score_thresh: f32,

    /// Text detection box threshold (GeneralOCR: 0.6, Table: 0.4)
    #[arg(long, default_value = "0.6")]
    det_box_thresh: f32,

    /// Text detection unclip ratio (default: 1.5)
    #[arg(long, default_value = "1.5")]
    det_unclip_ratio: f32,

    /// Max text detection candidates (default: 1000)
    #[arg(long, default_value = "1000")]
    det_max_candidates: usize,

    /// Text recognition score threshold (default: 0.0)
    #[arg(long, default_value = "0.0")]
    rec_score_thresh: f32,

    /// Max text length for recognition
    #[arg(long, default_value_t = 320)]
    text_rec_max_length: usize,

    /// Seal detection score threshold (default: 0.2, lower than general text)
    #[arg(long, default_value = "0.2")]
    seal_det_score_thresh: f32,

    /// Seal detection box threshold (default: 0.6)
    #[arg(long, default_value = "0.6")]
    seal_det_box_thresh: f32,

    /// Seal detection unclip ratio (default: 0.5, smaller than general text)
    #[arg(long, default_value = "0.5")]
    seal_det_unclip_ratio: f32,

    /// Table text detection box threshold (default: 0.4, lower than general)
    #[arg(long, default_value = "0.4")]
    table_det_box_thresh: f32,

    /// Output directory for the exported results
    #[arg(short, long, default_value = "output/structure_analysis")]
    output_dir: PathBuf,

    /// Save results as JSON
    #[arg(long = "to-json", default_value_t = false)]
    to_json: bool,

    /// Save results as Markdown
    #[arg(long = "to-markdown", default_value_t = false)]
    to_markdown: bool,

    /// Save results as HTML
    #[arg(long = "to-html", default_value_t = false)]
    to_html: bool,

    /// Save visualization image with labeled bounding boxes
    #[arg(long = "visualize", default_value_t = true)]
    visualize: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    info!("Running document structure analysis example");

    // Validate required layout model
    if !args.layout_model.exists() {
        error!("Layout model not found: {}", args.layout_model.display());
        return Err("Layout model not found".into());
    }

    // Validate optional model paths when provided
    validate_optional_path("orientation model", args.orientation_model.as_ref())?;
    validate_optional_path("rectification model", args.rectification_model.as_ref())?;
    validate_optional_path("region model", args.region_model.as_ref())?;
    validate_optional_path("table classification model", args.table_cls_model.as_ref())?;
    validate_optional_path("wired structure model", args.wired_structure_model.as_ref())?;
    validate_optional_path(
        "wireless structure model",
        args.wireless_structure_model.as_ref(),
    )?;
    validate_optional_path("wired cell model", args.wired_cell_model.as_ref())?;
    validate_optional_path("wireless cell model", args.wireless_cell_model.as_ref())?;
    validate_optional_path("table structure dict", args.table_structure_dict.as_ref())?;
    validate_optional_path("formula model", args.formula_model.as_ref())?;
    validate_optional_path("formula tokenizer", args.formula_tokenizer.as_ref())?;
    validate_optional_path("seal model", args.seal_model.as_ref())?;
    validate_optional_path("text detection model", args.text_det_model.as_ref())?;
    validate_optional_path("text recognition model", args.text_rec_model.as_ref())?;
    validate_optional_path("text dict", args.text_dict_path.as_ref())?;
    validate_optional_path(
        "text line orientation model",
        args.textline_orientation_model.as_ref(),
    )?;

    // Filter input images that exist
    let existing_images: Vec<PathBuf> = args
        .images
        .iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                error!("Image not found: {}", path.display());
            }
            exists
        })
        .cloned()
        .collect();

    if existing_images.is_empty() {
        return Err("No valid images provided".into());
    }

    // Validate table recognition: structure models require dictionary
    let has_table_structure =
        args.wired_structure_model.is_some() || args.wireless_structure_model.is_some();

    if has_table_structure && args.table_structure_dict.is_none() {
        return Err("Table structure recognition requires --table-structure-dict".into());
    }

    if args.formula_model.is_some()
        && (args.formula_tokenizer.is_none() || args.formula_type.is_none())
    {
        return Err("Formula recognition requires --formula-tokenizer and --formula-type".into());
    }

    let has_partial_ocr = args.text_det_model.is_some()
        || args.text_rec_model.is_some()
        || args.text_dict_path.is_some();
    if has_partial_ocr
        && (args.text_det_model.is_none()
            || args.text_rec_model.is_none()
            || args.text_dict_path.is_none())
    {
        warn!(
            "OCR integration ignored because detection/recognition/dictionary are not all provided"
        );
    }

    // Build layout config (PP-StructureV3 defaults + CLI overrides)
    let mut layout_config = LayoutDetectionConfig::with_pp_structurev3_defaults();
    layout_config.score_threshold = args.layout_score_thresh;
    layout_config.layout_nms = args.layout_nms;

    let formula_config = FormulaRecognitionConfig {
        score_threshold: args.formula_score_thresh,
        max_length: args.formula_max_length,
    };

    let text_det_config = TextDetectionConfig {
        score_threshold: args.det_score_thresh,
        box_threshold: args.det_box_thresh,
        unclip_ratio: args.det_unclip_ratio,
        max_candidates: args.det_max_candidates,
        // PP-StructureV3 overall OCR defaults
        limit_side_len: Some(736),
        limit_type: Some(LimitType::Min),
        max_side_len: None,
    };

    let text_rec_config = TextRecognitionConfig {
        score_threshold: args.rec_score_thresh,
        max_text_length: args.text_rec_max_length,
    };

    // Build structure pipeline
    let mut builder =
        OARStructureBuilder::new(&args.layout_model).layout_detection_config(layout_config);

    // Always set layout model name (has default value)
    builder = builder.layout_model_name(&args.layout_model_name);

    if let Some(config) = parse_device_config(&args.device)? {
        builder = builder.ort_session(config);
    }

    if let Some(path) = args.orientation_model {
        builder = builder.with_document_orientation(path);
    }

    if let Some(path) = args.rectification_model {
        builder = builder.with_document_rectification(path);
    }

    if let Some(path) = args.region_model {
        builder = builder
            .with_region_detection(path)
            .region_model_name(&args.region_model_name);
    }

    // Table recognition: auto-switch based on classification when both wired/wireless models are provided
    if let Some(path) = args.table_cls_model {
        builder = builder.with_table_classification(path);
    }
    if let Some(path) = args.table_orientation_model {
        builder = builder.with_table_orientation(path);
    }
    if let Some(path) = args.wired_structure_model {
        builder = builder
            .with_wired_table_structure(path)
            .wired_table_structure_model_name(&args.wired_structure_model_name);
    }
    if let Some(path) = args.wireless_structure_model {
        builder = builder
            .with_wireless_table_structure(path)
            .wireless_table_structure_model_name(&args.wireless_structure_model_name);
    }
    if let Some(path) = args.wired_cell_model {
        builder = builder
            .with_wired_table_cell_detection(path)
            .wired_table_cell_model_name(&args.wired_cell_model_name);
    }
    if let Some(path) = args.wireless_cell_model {
        builder = builder
            .with_wireless_table_cell_detection(path)
            .wireless_table_cell_model_name(&args.wireless_cell_model_name);
    }
    if let Some(path) = args.table_structure_dict {
        builder = builder.table_structure_dict_path(path);
    }
    // E2E mode settings (defaults: wired=false, wireless=true)
    builder = builder.use_e2e_wired_table_rec(args.use_e2e_wired_table_rec);
    builder = builder.use_e2e_wireless_table_rec(args.use_e2e_wireless_table_rec);

    if let Some(path) = args.formula_model {
        builder = builder
            .with_formula_recognition(
                path,
                args.formula_tokenizer.as_ref().expect("validated above"),
                args.formula_type.as_ref().expect("validated above"),
            )
            .formula_recognition_config(formula_config);
    }

    if let Some(path) = args.seal_model {
        builder = builder.with_seal_text_detection(path);
    }

    if let Some(path) = args.textline_orientation_model {
        builder = builder.with_text_line_orientation(path);
    }

    if args.text_det_model.is_some()
        && args.text_rec_model.is_some()
        && args.text_dict_path.is_some()
    {
        builder = builder
            .with_ocr(
                args.text_det_model.as_ref().unwrap(),
                args.text_rec_model.as_ref().unwrap(),
                args.text_dict_path.as_ref().unwrap(),
            )
            .text_detection_model_name(&args.text_det_model_name)
            .text_recognition_model_name(&args.text_rec_model_name)
            .text_detection_config(text_det_config)
            .text_recognition_config(text_rec_config);
    }

    let analyzer = builder.build()?;

    // Process each image individually
    for (idx, image_path) in existing_images.iter().enumerate() {
        info!("\nProcessing image {}: {}", idx + 1, image_path.display());

        // Load image using the utility function
        let image = match load_image(image_path) {
            Ok(img) => img,
            Err(err) => {
                error!("Failed to load {}: {}", image_path.display(), err);
                continue;
            }
        };

        let mut result = match analyzer.predict_image(image) {
            Ok(res) => res,
            Err(err) => {
                error!("Failed to analyze {}: {}", image_path.display(), err);
                continue;
            }
        };
        result.input_path = std::sync::Arc::from(image_path.to_string_lossy().as_ref());

        // Save results to output directory
        if let Err(err) = result.save_results(
            &args.output_dir,
            args.to_json,
            args.to_markdown,
            args.to_html,
        ) {
            error!(
                "Failed to save results for {}: {}",
                image_path.display(),
                err
            );
        }

        // Save visualization if requested
        #[cfg(feature = "visualization")]
        if args.visualize {
            let stem = image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("result");
            let ext = image_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("png");
            let vis_path = args.output_dir.join(format!("{}.{}", stem, ext));

            if let Err(err) =
                oar_ocr::utils::visualization::visualize_structure_results(&result, &vis_path, None)
            {
                error!("Failed to save visualization: {}", err);
            } else {
                info!("  Visualization saved to: {}", vis_path.display());
            }
        }

        if let Some(angle) = result.orientation_angle {
            info!("  Orientation corrected by {:.0} degrees", angle);
        }

        info!("  Layout elements: {}", result.layout_elements.len());
        for (elem_idx, elem) in result.layout_elements.iter().enumerate() {
            let label = elem
                .label
                .as_deref()
                .unwrap_or_else(|| elem.element_type.as_str());
            info!(
                "    [{}] {} ({:.1}%) at [{:.1},{:.1}] - [{:.1},{:.1}]",
                elem_idx + 1,
                label,
                elem.confidence * 100.0,
                elem.bbox.x_min(),
                elem.bbox.y_min(),
                elem.bbox.x_max(),
                elem.bbox.y_max()
            );
        }

        if let Some(regions) = &result.region_blocks {
            info!("  Region blocks: {}", regions.len());
            for (region_idx, region) in regions.iter().enumerate() {
                let order = region
                    .order_index
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "n/a".to_string());
                info!(
                    "    [{}] order={} elements={} ({:.1}%) at [{:.1},{:.1}] - [{:.1},{:.1}]",
                    region_idx + 1,
                    order,
                    region.element_indices.len(),
                    region.confidence * 100.0,
                    region.bbox.x_min(),
                    region.bbox.y_min(),
                    region.bbox.x_max(),
                    region.bbox.y_max()
                );
            }
        } else {
            info!("  Region blocks: not enabled");
        }

        info!("  Tables: {}", result.tables.len());
        for (table_idx, table) in result.tables.iter().enumerate() {
            let table_type = match table.table_type {
                TableType::Wired => "wired",
                TableType::Wireless => "wireless",
                TableType::Unknown => "unknown",
            };

            let cls_conf = table
                .classification_confidence
                .map(|c| format!("{:.1}%", c * 100.0))
                .unwrap_or_else(|| "n/a".to_string());

            let html_info = table
                .html_structure
                .as_ref()
                .map(|html| format!("html len {}", html.len()))
                .unwrap_or_else(|| "no structure".to_string());

            info!(
                "    [{}] type={} cls={} cells={} {}",
                table_idx + 1,
                table_type,
                cls_conf,
                table.cells.len(),
                html_info
            );
        }

        info!("  Formulas: {}", result.formulas.len());
        for (formula_idx, formula) in result.formulas.iter().enumerate() {
            info!(
                "    [{}] {} ({:.1}%)",
                formula_idx + 1,
                formula.latex,
                formula.confidence * 100.0
            );
        }

        if let Some(text_regions) = &result.text_regions {
            info!("  OCR regions: {}", text_regions.len());
            for (region_idx, region) in text_regions.iter().enumerate() {
                let text = region
                    .text
                    .as_ref()
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "<no text>".to_string());
                let score = region.confidence.unwrap_or(0.0) * 100.0;

                info!(
                    "    [{}] \"{}\" ({:.1}%) at [{:.1},{:.1}] - [{:.1},{:.1}]",
                    region_idx + 1,
                    text,
                    score,
                    region.bounding_box.x_min(),
                    region.bounding_box.y_min(),
                    region.bounding_box.x_max(),
                    region.bounding_box.y_max()
                );
            }
        } else {
            info!("  OCR regions: not enabled");
        }
    }

    Ok(())
}

fn validate_optional_path(
    label: &str,
    path: Option<&PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(p) = path
        && !p.exists()
    {
        return Err(format!("{label} not found: {}", p.display()).into());
    }
    Ok(())
}
