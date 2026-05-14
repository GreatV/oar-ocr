//! Run HSD on OmniDocBench v1.5 with a real per-page draft.
//!
//! For each page (up to `--max-pages`):
//!   1. Load the page image from `<bench>/images/<image_path>`.
//!   2. Build a GT draft from `layout_dets`: markdown/plain text for document
//!      parsers, or `text<|LOC_x|><|LOC_y|>...` spotting streams for
//!      PaddleOCR-VL spotting.
//!   3. Run baseline generation and time it.
//!   4. Run backend `generate_hsd(...)` with the draft, capture stats.
//!   5. Aggregate `SR_decode`, `SR_e2e`, AAL, fallback ratio across pages.
//!
//! ```bash
//! cargo run -p oar-ocr-vl --release --features hsd,download-binaries \
//!     --example hsd_omnidocbench -- \
//!         --bench-dir data/omnidocbench_v1.5 \
//!         --model-dir models/HunyuanOCR \
//!         --max-pages 20
//!
//! cargo run -p oar-ocr-vl --release --features hsd,download-binaries \
//!     --example hsd_omnidocbench -- \
//!         --backend paddleocr_vl --task spotting \
//!         --bench-dir data/omnidocbench_v1.5 \
//!         --model-dir models/PaddleOCR-VL-1.5 \
//!         --max-pages 20
//! ```

mod utils;

#[cfg(not(feature = "hsd"))]
fn main() {
    eprintln!("This example requires the `hsd` feature. Re-run with `--features hsd`.");
    std::process::exit(1);
}

#[cfg(feature = "hsd")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::run()
}

#[cfg(feature = "hsd")]
mod imp {

use super::utils;

use clap::{Parser, ValueEnum};
use image::imageops::FilterType;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

use oar_ocr::prelude::{OARStructure, OARStructureBuilder};
use oar_ocr_core::core::{OCRError, config::OrtSessionConfig};
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType, StructureResult};
use oar_ocr_core::domain::tasks::FormulaRecognitionConfig;
use oar_ocr_core::predictors::TextRecognitionPredictor;
use oar_ocr_core::processors::{BoundingBox, Point};
use oar_ocr_core::utils::{BBoxCrop, load_image};
use oar_ocr_vl::hsd::drafting::{
    TargetDraftAdapter, bbox_xyxy, page_markdown_for, region_markdowns_for,
};
use oar_ocr_vl::hsd::types::{Draft, DsvConfig, HsdConfig, HsdStats, RegionKind, SpecDecodeStats};
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{GlmOcr, HunyuanOcr, MinerU, PaddleOcrVl, PaddleOcrVlTask};

use utils::structure_match::{MatchThresholds, match_region};

const HUNYUAN_CHINESE_PARSING_PROMPT: &str = "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。";
const HUNYUAN_REGION_PROMPT: &str = "Extract all information from the document region image and represent it in markdown format. Tables should be expressed in HTML format, and formulas should be represented using LaTeX format.";
const HUNYUAN_CHINESE_REGION_PROMPT: &str = "提取文档区域图片中的所有信息用 markdown 格式表示，表格用 html 格式表达，公式用 latex 格式表示。";
const GLMOCR_TEXT_RECOGNITION_PROMPT: &str = "Text Recognition:";
const MINERU_TEXT_RECOGNITION_PROMPT: &str = "\nText Recognition:";

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Backend {
    #[value(name = "hunyuanocr")]
    HunyuanOcr,
    #[value(name = "paddleocr_vl")]
    PaddleOcrVl,
    #[value(name = "mineru")]
    MinerU,
    #[value(name = "glmocr")]
    GlmOcr,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Self::HunyuanOcr => "hunyuanocr",
            Self::PaddleOcrVl => "paddleocr_vl",
            Self::MinerU => "mineru",
            Self::GlmOcr => "glmocr",
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Task {
    Ocr,
    Table,
    Chart,
    Formula,
    Spotting,
    Seal,
}

impl Task {
    fn to_native(self) -> PaddleOcrVlTask {
        match self {
            Task::Ocr => PaddleOcrVlTask::Ocr,
            Task::Table => PaddleOcrVlTask::Table,
            Task::Chart => PaddleOcrVlTask::Chart,
            Task::Formula => PaddleOcrVlTask::Formula,
            Task::Spotting => PaddleOcrVlTask::Spotting,
            Task::Seal => PaddleOcrVlTask::Seal,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    /// Full-page generation / verification.
    Page,
    /// PaddleOCR-VL region-level verification using OmniDocBench layout crops.
    Region,
}

enum BackendModel {
    HunyuanOcr(HunyuanOcr),
    PaddleOcrVl(PaddleOcrVl),
    MinerU(MinerU),
    GlmOcr(GlmOcr),
}

struct RegionDrafts {
    joined: String,
    per_element: Vec<Option<String>>,
    per_element_tokens: Vec<Option<Vec<u32>>>,
}

#[derive(Parser)]
#[command(name = "hsd_omnidocbench")]
struct Args {
    /// Root of the unzipped OmniDocBench dataset (must contain
    /// `OmniDocBench.json` and `images/`).
    #[arg(long)]
    bench_dir: PathBuf,
    /// Backend weights directory.
    #[arg(long)]
    model_dir: PathBuf,
    /// Target backend to benchmark.
    #[arg(long, value_enum, default_value_t = Backend::HunyuanOcr)]
    backend: Backend,
    /// PaddleOCR-VL task. Ignored by HunyuanOCR.
    #[arg(long, value_enum, default_value_t = Task::Spotting)]
    task: Task,
    /// Benchmark mode. `region` currently supports HunyuanOCR and PaddleOCR-VL.
    #[arg(long, value_enum, default_value_t = Mode::Page)]
    mode: Mode,
    #[arg(long, default_value = "cuda:0")]
    device: String,
    /// Device for ONNX-based PP-OCRv5 / PP-StructureV3 drafter models.
    /// Defaults to `--device`.
    #[arg(long)]
    drafter_device: Option<String>,
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    /// Number of pages to evaluate. Default 5 for a quick smoke run.
    #[arg(long, default_value_t = 5)]
    max_pages: usize,
    /// Index of the first entry to evaluate (skip this many before counting).
    #[arg(long, default_value_t = 0)]
    start_idx: usize,
    /// Optional substring filter — only run entries whose `image_path` contains this.
    #[arg(long)]
    filter: Option<String>,
    /// Restrict to a specific OmniDocBench subset (e.g. `v1.5`, `equation_hard`).
    #[arg(long)]
    subset: Option<String>,
    /// Restrict to a specific page language (e.g. `english`, `chinese`).
    #[arg(long)]
    language: Option<String>,
    /// Use the matching official Chinese-language Parsing prompt for pages
    /// whose language is `chinese`. (English pages still use --instruction.)
    #[arg(long, default_value_t = true)]
    auto_prompt_lang: bool,
    /// Skip pages whose image fails to load (vs. aborting).
    #[arg(long, default_value_t = true)]
    skip_missing: bool,
    /// HunyuanOCR instruction prompt. Defaults to HunyuanOCR's official
    /// "Parsing" task prompt, which elicits markdown output matching the
    /// OmniDocBench GT format. (Source: HunyuanOCR README under Quick Start
    /// → Tasks.)
    #[arg(
        long,
        default_value = "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."
    )]
    instruction: String,
    /// HunyuanOCR region-level instruction used for Stage 1 crop verification.
    #[arg(long, default_value = HUNYUAN_REGION_PROMPT)]
    hunyuanocr_region_instruction: String,
    /// GLM-OCR instruction prompt. Defaults to the model-native OCR expert
    /// prompt documented in the local README.
    #[arg(long, default_value = GLMOCR_TEXT_RECOGNITION_PROMPT)]
    glmocr_instruction: String,
    /// GLM-OCR region-level instruction used for Stage 1 crop verification.
    #[arg(long, default_value = GLMOCR_TEXT_RECOGNITION_PROMPT)]
    glmocr_region_instruction: String,
    /// MinerU2.5 instruction prompt. Defaults to the model-native content
    /// extraction prompt used by the MinerU two-step example.
    #[arg(long, default_value = MINERU_TEXT_RECOGNITION_PROMPT)]
    mineru_instruction: String,
    /// MinerU2.5 region-level instruction used for Stage 1 crop verification.
    /// Pass an empty string to opt into two-step mode where each layout
    /// element gets its native MinerU prompt (`\nText Recognition:`,
    /// `\nTable Recognition:`, `\nFormula Recognition:`, `\nImage Analysis:`),
    /// matching MinerU's official `two_step_extract` flow.
    #[arg(long, default_value = MINERU_TEXT_RECOGNITION_PROMPT)]
    mineru_region_instruction: String,
    /// Convenience flag for `--mineru-region-instruction ""` — forces MinerU
    /// Stage 1 into per-element prompt dispatch (`two_step_extract`-style).
    /// Overrides any explicit `--mineru-region-instruction` value.
    #[arg(long, default_value_t = false)]
    mineru_two_step: bool,
    /// Path to a JSON file containing pre-postprocess raw drafts from a
    /// *different* VLM. Activates `--draft-source cross-vlm-file`. Schema:
    /// `{"source_backend": "<name>", "pages": {"<image>": [{"bbox": [...], "raw_text": "..."}]}}`.
    /// Use the source backend's `decode_tokens_raw` to populate `raw_text`.
    /// The target adapter handles any per-target surface conversion
    /// (HTML↔OTSL, formula wrapping, etc.) — no explicit source hint needed.
    #[arg(long)]
    cross_vlm_draft_file: Option<PathBuf>,
    /// IoU floor when matching cross-VLM regions onto layout elements.
    /// Defaults to 0.5, matching the structure/formula IoU thresholds used
    /// elsewhere in the bench.
    #[arg(long, default_value_t = 0.5)]
    cross_vlm_iou_threshold: f32,
    #[arg(long, default_value_t = 0.75)]
    tau: f32,
    /// Override `DsvConfig::window_len` (paper §3.2 `n`). Default 0 = honour
    /// the preset's value (3 for all presets).
    ///
    /// **Use with care.** Empirical 2026-05-13 result on HunyuanOCR page+gt: a
    /// 3-page smoke with `--dsv-window-len 2` *regressed* SR_e2e from 0.45×
    /// to 0.17× because the matcher then found many more candidates per step
    /// but most were stale matches — leading to bigger trees that the
    /// verifier had to forward through anyway, then reject. When the matrix
    /// reports high `dsv empty tree calls / page`, the right answer is
    /// usually NOT a smaller window — it's a better drafter (closer
    /// byte-level alignment with the target VLM's natural output).
    #[arg(long, default_value_t = 0)]
    dsv_window_len: usize,
    /// Override `DsvConfig::max_candidates_per_step`. Default 0 = honour the
    /// preset's value (32 for default, 128 for omnibench). Lower this when
    /// `dsv avg tree nodes / page` blows up — each verify_tree forward
    /// processes the whole packed tree even if all paths get rejected, so
    /// big trees on divergent drafts pay full compute for no acceptance.
    #[arg(long, default_value_t = 0)]
    dsv_max_candidates: usize,
    /// Override `DsvConfig::max_suffix_len`. Default 0 = honour the preset's
    /// value (256 for all presets).
    #[arg(long, default_value_t = 0)]
    dsv_max_suffix_len: usize,
    /// Print the first N chars of baseline + draft for each page (debug).
    #[arg(long, default_value_t = 0)]
    preview: usize,
    /// Write baseline-vs-draft token alignment diagnostics to this file.
    #[arg(long)]
    token_diff_output: Option<PathBuf>,
    /// Number of leading tokens to include in token diagnostics.
    #[arg(long, default_value_t = 200)]
    token_diff_limit: usize,
    /// N-gram/window length used for token-match diagnostics.
    #[arg(long, default_value_t = 3)]
    token_diff_window_len: usize,
    /// Stop after writing token diagnostics, before running HSD.
    #[arg(long, default_value_t = false)]
    token_diff_only: bool,
    /// In page mode, run a full Stage-1 + Stage-2 HSD path for backends that
    /// support it when region elements are available from the drafter. Use
    /// `--page-dual-stage=false` to keep the page-level-only ablation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    page_dual_stage: bool,
    /// Count pages with AAL at or below this threshold as low-AAL outliers.
    #[arg(long, default_value_t = 0.5)]
    outlier_aal_threshold: f32,
    /// Count pages with SR_e2e below this threshold as speed-regression
    /// outliers.
    #[arg(long, default_value_t = 1.0)]
    outlier_sr_e2e_threshold: f64,
    /// Draft source: `gt` (build from OmniDocBench layout_dets), `baseline`
    /// (re-use baseline output as an oracle draft), `ppocr-rec` (region-mode
    /// PP-OCRv5 recognition model output), or `structure` (OARStructureBuilder
    /// / PP-StructureV3-style markdown in page mode; IoU-matched
    /// text/table/formula drafts in region mode).
    #[arg(long, default_value = "gt")]
    draft_source: String,
    /// Preserve legacy HunyuanOCR GT draft markdown heuristics (`# title` and
    /// `$$\nformula\n$$`). The default is tuned for token overlap with
    /// HunyuanOCR baseline output.
    #[arg(long, default_value_t = false)]
    hunyuanocr_legacy_gt_format: bool,
    /// Apply lightweight PaddleOCR-VL region GT surface normalization. Only
    /// affects `--backend paddleocr_vl --mode region --draft-source gt`.
    #[arg(long, default_value_t = false)]
    normalize_draft: bool,
    /// Pre-resize each page so its longer side fits this many pixels.
    /// HunyuanOCR's vit encoder fails on very large pages even though the
    /// preprocessor's clamp logic claims to handle them; 1280 is a safe
    /// default that keeps text readable. Set to 0 to disable.
    #[arg(long, default_value_t = 1280)]
    resize_max: u32,
    /// PP-OCR recognition model for `--draft-source ppocr-rec`. Default
    /// resolves to `pp-ocrv5_mobile_rec.onnx` in CWD; pass a full path to
    /// override.
    #[arg(long, default_value = "pp-ocrv5_mobile_rec.onnx")]
    ppocr_rec_model: PathBuf,
    /// PP-OCR recognition dictionary for `--draft-source ppocr-rec`. Default
    /// resolves to `ppocrv5_dict.txt` in CWD; pass a full path to override.
    #[arg(long, default_value = "ppocrv5_dict.txt")]
    ppocr_dict_path: PathBuf,
    /// PP-OCR recognition score threshold.
    #[arg(long, default_value_t = 0.0)]
    ppocr_score_thresh: f32,
    /// PP-OCR recognition max text length.
    #[arg(long, default_value_t = 200)]
    ppocr_max_text_length: usize,
    /// Layout model for `--draft-source structure`. Default resolves to
    /// `pp-doclayout_plus-l.onnx` in CWD; pass a full path to override.
    #[arg(long, default_value = "pp-doclayout_plus-l.onnx")]
    structure_layout_model: PathBuf,
    /// Layout model preset for `--draft-source structure`.
    #[arg(long, default_value = "pp-doclayout_plus-l")]
    structure_layout_model_name: String,
    /// PP-DocBlockLayout model for structure reading order. **Required for
    /// paper-equivalent multi-column AAL** — without it, reading order
    /// degrades to bbox `(y, x)` sort and Stage-2 acceptance collapses on
    /// multi-column pages. Default resolves to `pp-docblocklayout.onnx` in
    /// the current working directory; pass a full path (e.g.
    /// `/some/dir/pp-docblocklayout.onnx`) to use a model placed elsewhere,
    /// or pass an empty string (`--structure-region-model ""`) to explicitly
    /// disable.
    #[arg(long, default_value = "pp-docblocklayout.onnx")]
    structure_region_model: PathBuf,
    /// PP-OCR detection model for structure OCR. Default resolves to
    /// `pp-ocrv5_mobile_det.onnx` in CWD; pass a full path to override.
    #[arg(long, default_value = "pp-ocrv5_mobile_det.onnx")]
    structure_ocr_det_model: PathBuf,
    /// PP-OCR recognition model for structure OCR. Default resolves to
    /// `pp-ocrv5_mobile_rec.onnx` in CWD; pass a full path to override.
    #[arg(long, default_value = "pp-ocrv5_mobile_rec.onnx")]
    structure_ocr_rec_model: PathBuf,
    /// PP-OCR dictionary for structure OCR. Default resolves to
    /// `ppocrv5_dict.txt` in CWD; pass a full path to override.
    #[arg(long, default_value = "ppocrv5_dict.txt")]
    structure_ocr_dict_path: PathBuf,
    /// Table classifier for structure table routing.
    ///
    /// Default resolves to `pp-lcnet_x1_0_table_cls.onnx` in CWD; pass a full
    /// path to load from elsewhere. Pass an empty string to opt out — but
    /// doing so forces every detected table region to a single structure
    /// model (wired *or* wireless) and produces no draft for the unused
    /// branch. Without table coverage the matrix Stage-1 region kind table
    /// shows 0/N drafts for `table`, leaving acceptance to drop to 0% on
    /// table-heavy pages.
    #[arg(long, default_value = "pp-lcnet_x1_0_table_cls.onnx")]
    structure_table_cls_model: PathBuf,
    /// Wired table structure model (SLANeXt) for structure table HTML.
    /// Default: `slanext_wired.onnx` in CWD; pass empty string to skip.
    #[arg(long, default_value = "slanext_wired.onnx")]
    structure_wired_table_model: PathBuf,
    /// Wireless table structure model (SLANet+) for structure table HTML.
    /// Default: `slanet_plus.onnx` in CWD; pass empty string to skip.
    #[arg(long, default_value = "slanet_plus.onnx")]
    structure_wireless_table_model: PathBuf,
    /// Table structure dictionary for structure table HTML. Default:
    /// `table_structure_dict_ch.txt` (PaddleX-compatible bilingual dict) in
    /// CWD; pass empty string to skip.
    #[arg(long, default_value = "table_structure_dict_ch.txt")]
    structure_table_dict_path: PathBuf,
    /// Wired table cell detection model (RT-DETR-L). Default:
    /// `rt-detr-l_wired_table_cell_det.onnx` in CWD; pass empty string to
    /// skip cell detection (table structure still works without it, but
    /// reduced fidelity on complex tables).
    #[arg(long, default_value = "rt-detr-l_wired_table_cell_det.onnx")]
    structure_wired_cell_model: PathBuf,
    /// Wireless table cell detection model (RT-DETR-L). Default:
    /// `rt-detr-l_wireless_table_cell_det.onnx` in CWD; pass empty string to
    /// skip.
    #[arg(long, default_value = "rt-detr-l_wireless_table_cell_det.onnx")]
    structure_wireless_cell_model: PathBuf,
    /// Formula model for structure LaTeX drafts.
    ///
    /// Default: `pp-formulanet_plus-l.onnx` in CWD (732MB) for accuracy on the
    /// quality matrix — Plus-S has noticeably worse argmax behavior on
    /// OmniDocBench academic pages (e.g. `\breve` vs `\check`, dropped
    /// subscripts) and is recommended only for smoke / perf runs. Pass
    /// `pp-formulanet_plus-s.onnx` (232MB) explicitly if disk/RAM is tight.
    /// Empty string skips formula drafting and drops formula AAL to 0 on
    /// academic pages.
    #[arg(long, default_value = "pp-formulanet_plus-l.onnx")]
    structure_formula_model: PathBuf,
    /// Formula tokenizer for structure LaTeX drafts. Default:
    /// `pp-formulanet-tokenizer.json` in CWD; pass empty string to skip
    /// (must match the choice of `--structure-formula-model`).
    #[arg(long, default_value = "pp-formulanet-tokenizer.json")]
    structure_formula_tokenizer: PathBuf,
    /// Formula model type for structure LaTeX drafts.
    #[arg(long, default_value = "pp_formulanet")]
    structure_formula_type: String,
    /// Device for structure formula recognition. Defaults to the drafter
    /// device through the global structure ORT session; pass `cpu`, `cuda`,
    /// or `cuda:N` to override just formula recognition.
    #[arg(long)]
    structure_formula_device: Option<String>,
    /// Preferred formula recognition batch size for structure drafts.
    #[arg(long, default_value_t = 8)]
    structure_formula_batch_size: usize,
    /// Maximum decoded formula length for structure drafts.
    #[arg(long, default_value_t = 1536)]
    structure_formula_max_length: usize,
    /// Strict IoU floor for cross-category structure → region matches. The
    /// previous "max IoU wins regardless of type" policy is preserved at this
    /// floor as a safety net for cases where the structure pipeline assigns
    /// an unexpected type to the matching region.
    #[arg(long, default_value_t = 0.8)]
    structure_iou_threshold: f32,
    /// Relaxed IoU floor for same-`semantic_category` structure → region
    /// matches. Since the type pre-filter bounds poisoning risk, a lower
    /// floor here can improve coverage on partially-overlapping regions where
    /// the structure pipeline and OmniDocBench layout disagree on the exact
    /// bbox, but the 2026-05-06 30-page run regressed AAL/SR at 0.5. The
    /// conservative default therefore matches `--structure-iou-threshold`.
    #[arg(long, default_value_t = 0.8)]
    structure_same_category_iou: f32,
    /// Allow table/formula/chart regions to fall back to generic layout OCR
    /// text when specialized structure output is missing.
    #[arg(long, default_value_t = false)]
    structure_allow_generic_fallback: bool,
    /// Batch size for structure region OCR.
    #[arg(long, default_value_t = 8)]
    structure_region_batch_size: usize,
    /// Where to write a per-page CSV log. Default: `<bench-dir>/hsd_results.csv`.
    #[arg(long)]
    output_csv: Option<PathBuf>,
    /// Where to write an aggregate markdown summary. Default: `<output-csv>.md`.
    #[arg(long)]
    output_summary: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct OmniEntry {
    layout_dets: Vec<LayoutDet>,
    page_info: PageInfo,
    #[serde(default)]
    #[allow(dead_code)]
    extra: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct LayoutDet {
    #[serde(default)]
    category_type: String,
    #[serde(default)]
    ignore: bool,
    /// Reading-order index. May be `null` for skipped/abandoned regions.
    order: Option<i64>,
    /// Recognised text. Often `""` for non-text regions (figure/table).
    #[serde(default)]
    text: String,
    /// OmniDocBench quadrilateral `[x1,y1,x2,y2,x3,y3,x4,y4]`.
    #[serde(default)]
    poly: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct PageInfo {
    image_path: String,
    #[allow(dead_code)]
    page_no: Option<i64>,
    #[allow(dead_code)]
    height: Option<f64>,
    #[allow(dead_code)]
    width: Option<f64>,
    #[serde(default)]
    page_attribute: serde_json::Value,
}

/// JSON format consumed by `--cross-vlm-draft-file`. Lets the bench exercise
/// the [`crate::hsd::drafting::convert_raw_to_target_adapter`] un-postprocess
/// path without loading a second VLM in-process.
///
/// The producer is expected to be a separate run of another VLM (e.g.
/// PaddleOCR-VL) that called `decode_tokens_raw` per region and serialized
/// the pre-postprocess raw text. The bench then assigns those texts onto the
/// target backend's layout elements (matched by bbox IoU), and the target's
/// `TargetDraftAdapter` does the rest of the surface conversion (e.g.
/// HTML↔OTSL, `$$ ... $$` wrapping).
///
/// Minimal schema:
/// ```json
/// {
///   "source_backend": "paddleocr_vl",
///   "pages": {
///     "page-001.png": [
///       {"bbox": [10.0, 20.0, 200.0, 50.0], "raw_text": "$$x = 1$$"}
///     ]
///   }
/// }
/// ```
/// `source_backend` is informational only — the target adapter handles
/// per-element form so the source hint is not required for correctness.
#[derive(Debug, Clone, Deserialize)]
struct CrossVlmDraftFile {
    #[allow(dead_code)]
    #[serde(default)]
    source_backend: Option<String>,
    pages: HashMap<String, Vec<CrossVlmRegion>>,
}

#[derive(Debug, Clone, Deserialize)]
struct CrossVlmRegion {
    /// `[x_min, y_min, x_max, y_max]` in original image pixel coordinates.
    bbox: [f32; 4],
    /// Pre-postprocess decoded string from the source backend (use that
    /// backend's `decode_tokens_raw`, not `decode_tokens`).
    raw_text: String,
}

impl CrossVlmDraftFile {
    fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path).map_err(|e| {
            format!(
                "failed to read --cross-vlm-draft-file {}: {e}",
                path.display()
            )
        })?;
        let parsed: Self = serde_json::from_slice(&bytes).map_err(|e| {
            format!(
                "failed to parse --cross-vlm-draft-file {}: {e}",
                path.display()
            )
        })?;
        Ok(parsed)
    }

    /// Look up the per-page region list. Tries the full image path first,
    /// then the basename (so callers can use either convention).
    fn lookup_page(&self, image_path: &str) -> Option<&[CrossVlmRegion]> {
        if let Some(regions) = self.pages.get(image_path) {
            return Some(regions.as_slice());
        }
        let basename = std::path::Path::new(image_path)
            .file_name()
            .and_then(|n| n.to_str())?;
        self.pages.get(basename).map(Vec::as_slice)
    }
}

/// Axis-aligned IoU between two `[x_min, y_min, x_max, y_max]` rectangles.
/// Returns 0.0 when either box has zero area.
fn axis_aligned_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (ax0, ay0, ax1, ay1) = (a[0], a[1], a[2], a[3]);
    let (bx0, by0, bx1, by1) = (b[0], b[1], b[2], b[3]);
    let area_a = ((ax1 - ax0).max(0.0)) * ((ay1 - ay0).max(0.0));
    let area_b = ((bx1 - bx0).max(0.0)) * ((by1 - by0).max(0.0));
    if area_a <= 0.0 || area_b <= 0.0 {
        return 0.0;
    }
    let ix0 = ax0.max(bx0);
    let iy0 = ay0.max(by0);
    let ix1 = ax1.min(bx1);
    let iy1 = ay1.min(by1);
    let iw = (ix1 - ix0).max(0.0);
    let ih = (iy1 - iy0).max(0.0);
    let inter = iw * ih;
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Find the cross-VLM region whose bbox best matches `elem_bbox` (max IoU
/// above `iou_threshold`). Returns `None` if no candidate clears the bar.
fn match_cross_vlm_region<'a>(
    elem_bbox: &[f32; 4],
    candidates: &'a [CrossVlmRegion],
    iou_threshold: f32,
) -> Option<&'a CrossVlmRegion> {
    let mut best: Option<(&CrossVlmRegion, f32)> = None;
    for cand in candidates {
        let iou = axis_aligned_iou(elem_bbox, &cand.bbox);
        if iou < iou_threshold {
            continue;
        }
        match best {
            Some((_, best_iou)) if best_iou >= iou => {}
            _ => best = Some((cand, iou)),
        }
    }
    best.map(|(c, _)| c)
}

#[derive(Clone, Copy)]
struct DraftFormat {
    heading_prefix: bool,
    wrap_formulas: bool,
    formula_newlines: bool,
    space_after_sec_dot: bool,
    separator_after_page_number: bool,
}

impl DraftFormat {
    fn markdown() -> Self {
        Self {
            heading_prefix: true,
            wrap_formulas: true,
            formula_newlines: true,
            space_after_sec_dot: false,
            separator_after_page_number: false,
        }
    }

    fn hunyuanocr_aligned() -> Self {
        Self {
            heading_prefix: false,
            wrap_formulas: false,
            formula_newlines: false,
            space_after_sec_dot: true,
            separator_after_page_number: true,
        }
    }
}

fn align_hunyuan_heading(text: &str, fmt: DraftFormat) -> String {
    if !fmt.space_after_sec_dot {
        return text.to_string();
    }
    if let Some(rest) = text.strip_prefix("SEC.")
        && rest.chars().next().is_some_and(|ch| ch.is_ascii_digit())
    {
        return format!("SEC. {rest}");
    }
    text.to_string()
}

/// Heuristic markdown-style serialisation for one page's draft.
fn build_draft(entry: &OmniEntry, fmt: DraftFormat) -> String {
    let mut dets: Vec<&LayoutDet> = entry
        .layout_dets
        .iter()
        .filter(|d| d.order.is_some())
        .filter(|d| !matches!(d.category_type.as_str(), "abandon" | "text_mask"))
        .collect();
    dets.sort_by_key(|d| d.order.unwrap());

    let mut out = String::new();
    for d in dets {
        let text = d.text.trim();
        if text.is_empty() {
            continue;
        }
        let category = d.category_type.as_str();
        let formatted = match category {
            "title" | "header" if fmt.heading_prefix => format!("# {text}"),
            "title" | "header" => align_hunyuan_heading(text, fmt),
            "equation_isolated" | "equation_semantic" => {
                if !fmt.wrap_formulas || text.starts_with("$$") || text.starts_with("\\[") {
                    text.to_string()
                } else if fmt.formula_newlines {
                    format!("$$\n{text}\n$$")
                } else {
                    format!("$${text}$$")
                }
            }
            _ => text.to_string(),
        };
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&formatted);
        if fmt.separator_after_page_number && category == "page_number" {
            out.push_str("\n\n---");
        }
    }
    out
}

fn build_plain_draft(entry: &OmniEntry) -> String {
    let mut dets: Vec<&LayoutDet> = entry
        .layout_dets
        .iter()
        .filter(|d| d.order.is_some())
        .filter(|d| !d.ignore)
        .filter(|d| !is_mask_or_abandon(d.category_type.as_str()))
        .collect();
    dets.sort_by_key(|d| d.order.unwrap());

    dets.into_iter()
        .map(|d| d.text.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_spotting_draft(entry: &OmniEntry, image_width: u32, image_height: u32) -> String {
    let mut dets: Vec<&LayoutDet> = entry
        .layout_dets
        .iter()
        .filter(|d| d.order.is_some())
        .filter(|d| !d.ignore)
        .filter(|d| !is_mask_or_abandon(d.category_type.as_str()))
        .collect();
    dets.sort_by_key(|d| d.order.unwrap());

    let mut out = String::new();
    for d in dets {
        let text = d.text.trim();
        if text.is_empty() {
            continue;
        }
        let Some(loc_tokens) = poly_loc_tokens(&d.poly, image_width, image_height) else {
            continue;
        };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(text);
        out.push_str(&loc_tokens);
    }
    out
}

fn build_gt_draft(
    entry: &OmniEntry,
    backend: Backend,
    task: Task,
    image_size: (u32, u32),
    hunyuanocr_legacy_gt_format: bool,
) -> String {
    match backend {
        Backend::HunyuanOcr => build_draft(
            entry,
            if hunyuanocr_legacy_gt_format {
                DraftFormat::markdown()
            } else {
                DraftFormat::hunyuanocr_aligned()
            },
        ),
        Backend::MinerU | Backend::GlmOcr => build_plain_draft(entry),
        Backend::PaddleOcrVl => match task {
            Task::Spotting => build_spotting_draft(entry, image_size.0, image_size.1),
            Task::Ocr | Task::Seal => build_plain_draft(entry),
            Task::Table | Task::Chart | Task::Formula => {
                build_draft(entry, DraftFormat::markdown())
            }
        },
    }
}

fn target_draft_adapter(backend: Backend, task: Task) -> TargetDraftAdapter {
    // Pick the target VLM's natural draft surface so structure / cross-VLM
    // drafts get auto-normalized via the adapter (HTML↔OTSL for tables,
    // formula wrapper handling, heading shell, etc.). PaddleOCR-VL is
    // element-only and uses the same `PaddleOcrVl` adapter regardless of
    // task — the adapter dispatches on element kind, not task.
    let _ = task;
    match backend {
        Backend::HunyuanOcr => TargetDraftAdapter::HunyuanOcr,
        Backend::MinerU => TargetDraftAdapter::MinerU,
        Backend::GlmOcr => TargetDraftAdapter::GlmOcr,
        Backend::PaddleOcrVl => TargetDraftAdapter::PaddleOcrVl,
    }
}

fn tokenize_draft(
    tokenizer: &Tokenizer,
    draft: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    tokenizer
        .encode(draft, false)
        .map(|enc| enc.get_ids().to_vec())
        .map_err(|e| format!("tokenizer encode failed: {e}").into())
}

fn token_piece(tokenizer: &Tokenizer, token: u32) -> String {
    tokenizer
        .decode(&[token], false)
        .unwrap_or_else(|_| format!("<decode-error:{token}>"))
        .replace('\n', "\\n")
}

fn count_window_hits(reference: &[u32], target: &[u32], window_len: usize) -> (usize, usize) {
    if window_len == 0 || reference.len() < window_len {
        return (0, 0);
    }
    let total = reference.len() - window_len + 1;
    if target.len() < window_len {
        return (0, total);
    }
    let hits = reference
        .windows(window_len)
        .filter(|w| target.windows(window_len).any(|dw| dw == *w))
        .count();
    (hits, total)
}

fn best_per_draft_window_hits(
    reference: &[u32],
    drafts: &[Vec<u32>],
    window_len: usize,
) -> (usize, usize) {
    drafts
        .iter()
        .map(|draft| count_window_hits(reference, draft, window_len))
        .max_by_key(|(hits, _)| *hits)
        .unwrap_or((
            0,
            reference.len().saturating_sub(window_len).saturating_add(1),
        ))
}

struct TokenDiffInput<'a> {
    run_row_idx: usize,
    candidate_idx: usize,
    image_path: &'a str,
    backend: Backend,
    mode: Mode,
    draft_source: &'a str,
    baseline_text: &'a str,
    draft_text: &'a str,
    baseline_tokens: &'a [u32],
    draft_tokens: &'a [u32],
    structure_elements: Option<&'a [LayoutElement]>,
    hsd_page_draft_count: usize,
    region_draft_count: usize,
    per_draft_max_hits: Option<(usize, usize)>,
    limit: usize,
    window_len: usize,
}

fn preview_text(input: &str, limit: usize) -> String {
    input
        .chars()
        .take(limit)
        .collect::<String>()
        .replace('\n', "\\n")
}

fn append_token_diff_report(out: &mut String, tokenizer: &Tokenizer, input: TokenDiffInput<'_>) {
    let common = input
        .baseline_tokens
        .iter()
        .zip(input.draft_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let (hits, total) =
        count_window_hits(input.baseline_tokens, input.draft_tokens, input.window_len);
    let hit_rate = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };
    out.push_str(&format!(
        "# HSD Token Diff\n\n\
         | field | value |\n\
         |---|---|\n\
         | run row idx | {} |\n\
         | candidate idx | {} |\n\
         | image | {} |\n\
         | backend | {} |\n\
         | mode | {:?} |\n\
         | draft source | {} |\n\
         | baseline chars | {} |\n\
         | draft chars | {} |\n\
         | baseline tokens | {} |\n\
         | draft tokens | {} |\n\
         | common token prefix | {} |\n\
         | HSD page draft count | {} |\n\
         | diagnostic region draft count | {} |\n\
         | baseline {}-gram hits in concatenated/page draft | {}/{} ({:.3}) |\n",
        input.run_row_idx,
        input.candidate_idx,
        input.image_path,
        input.backend.as_str(),
        input.mode,
        input.draft_source,
        input.baseline_text.chars().count(),
        input.draft_text.chars().count(),
        input.baseline_tokens.len(),
        input.draft_tokens.len(),
        common,
        input.hsd_page_draft_count,
        input.region_draft_count,
        input.window_len,
        hits,
        total,
        hit_rate
    ));
    if let Some((per_hits, per_total)) = input.per_draft_max_hits {
        let per_rate = if per_total > 0 {
            per_hits as f64 / per_total as f64
        } else {
            0.0
        };
        out.push_str(&format!(
            "| best single-region {}-gram hits | {}/{} ({:.3}) |\n",
            input.window_len, per_hits, per_total, per_rate
        ));
    }
    out.push('\n');

    if let Some(elements) = input.structure_elements {
        out.push_str("## Structure Element Order\n\n");
        out.push_str("| idx | type | bbox | text preview |\n");
        out.push_str("|---:|---|---|---|\n");
        for (idx, elem) in elements.iter().take(40).enumerate() {
            let text = elem
                .text
                .as_deref()
                .map(|s| preview_text(s, 120))
                .unwrap_or_default();
            out.push_str(&format!(
                "| {} | {:?} | [{:.0},{:.0},{:.0},{:.0}] | `{}` |\n",
                idx,
                elem.element_type,
                elem.bbox.x_min(),
                elem.bbox.y_min(),
                elem.bbox.x_max(),
                elem.bbox.y_max(),
                text.replace('`', "\\`")
            ));
        }
        out.push('\n');
    }

    out.push_str("## First Differing Tokens\n\n");
    out.push_str("| idx | baseline id | baseline piece | draft id | draft piece |\n");
    out.push_str("|---:|---:|---|---:|---|\n");
    let n = input
        .limit
        .min(input.baseline_tokens.len().max(input.draft_tokens.len()));
    for i in common.saturating_sub(5)..n {
        let b = input.baseline_tokens.get(i).copied();
        let d = input.draft_tokens.get(i).copied();
        let bp = b
            .map(|t| token_piece(tokenizer, t))
            .unwrap_or_else(|| "".to_string());
        let dp = d
            .map(|t| token_piece(tokenizer, t))
            .unwrap_or_else(|| "".to_string());
        out.push_str(&format!(
            "| {} | {} | `{}` | {} | `{}` |\n",
            i,
            b.map(|t| t.to_string()).unwrap_or_default(),
            bp.replace('`', "\\`"),
            d.map(|t| t.to_string()).unwrap_or_default(),
            dp.replace('`', "\\`")
        ));
    }

    let bp: String = input.baseline_text.chars().take(1000).collect();
    let dp: String = input.draft_text.chars().take(1000).collect();
    out.push_str("\n## Baseline Text Preview\n\n```text\n");
    out.push_str(&bp);
    out.push_str("\n```\n\n## Draft Text Preview\n\n```text\n");
    out.push_str(&dp);
    out.push_str("\n```\n");
}

fn page_attr<'a>(entry: &'a OmniEntry, key: &str) -> &'a str {
    entry
        .page_info
        .page_attribute
        .as_object()
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_str())
        .unwrap_or("")
}

fn prompt_for_entry<'a>(entry: &OmniEntry, args: &'a Args) -> (&'a str, &'static str) {
    if args.auto_prompt_lang && page_attr(entry, "language").eq_ignore_ascii_case("chinese") {
        (HUNYUAN_CHINESE_PARSING_PROMPT, "hunyuanocr_parsing_zh")
    } else {
        (args.instruction.as_str(), "hunyuanocr_parsing_en")
    }
}

fn hunyuanocr_region_prompt<'a>(entry: &OmniEntry, args: &'a Args) -> (&'a str, &'static str) {
    if args.auto_prompt_lang && page_attr(entry, "language").eq_ignore_ascii_case("chinese") {
        (HUNYUAN_CHINESE_REGION_PROMPT, "hunyuanocr_region_zh")
    } else {
        (
            args.hunyuanocr_region_instruction.as_str(),
            "hunyuanocr_region_en",
        )
    }
}

fn glmocr_prompt(args: &Args) -> (&str, &'static str) {
    (args.glmocr_instruction.as_str(), "glmocr_text_recognition")
}

fn glmocr_region_prompt(args: &Args) -> (&str, &'static str) {
    (
        args.glmocr_region_instruction.as_str(),
        "glmocr_region_text_recognition",
    )
}

fn mineru_prompt(args: &Args) -> (&str, &'static str) {
    (args.mineru_instruction.as_str(), "mineru_text_recognition")
}

fn mineru_region_prompt(args: &Args) -> (&str, &'static str) {
    // `--mineru-two-step` forces empty region prompt, which MinerU's
    // `generate_hsd_full` interprets as "dispatch per-element via
    // `MinerUTaskPrompt::for_layout`" (matches the official `two_step_extract`
    // flow).
    if args.mineru_two_step {
        ("", "mineru_two_step_per_element")
    } else {
        (
            args.mineru_region_instruction.as_str(),
            "mineru_region_text_recognition",
        )
    }
}

fn csv_escape(value: impl AsRef<str>) -> String {
    let value = value.as_ref();
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_row(fields: &[String]) -> String {
    let mut row = fields.iter().map(csv_escape).collect::<Vec<_>>().join(",");
    row.push('\n');
    row
}

fn require_hsd_elements<'a>(
    elements: Option<&'a [LayoutElement]>,
    backend: &'static str,
) -> Result<&'a [LayoutElement], OCRError> {
    let Some(elements) = elements.filter(|elements| !elements.is_empty()) else {
        return Err(OCRError::InvalidInput {
            message: format!("{backend} page dual-stage requires non-empty HSD layout elements"),
        });
    };
    Ok(elements)
}

fn layout_kind_bucket(t: LayoutElementType) -> &'static str {
    match t {
        LayoutElementType::Table => "table",
        LayoutElementType::Formula | LayoutElementType::FormulaNumber => "formula",
        LayoutElementType::Image
        | LayoutElementType::Chart
        | LayoutElementType::Seal
        | LayoutElementType::HeaderImage
        | LayoutElementType::FooterImage => "visual",
        LayoutElementType::DocTitle
        | LayoutElementType::ParagraphTitle
        | LayoutElementType::FigureTitle
        | LayoutElementType::TableTitle
        | LayoutElementType::ChartTitle
        | LayoutElementType::FigureTableChartTitle => "title",
        LayoutElementType::Header | LayoutElementType::Footer | LayoutElementType::Number => {
            "page_artifact"
        }
        LayoutElementType::List => "list",
        LayoutElementType::Text
        | LayoutElementType::Content
        | LayoutElementType::Abstract
        | LayoutElementType::AsideText
        | LayoutElementType::Reference
        | LayoutElementType::ReferenceContent
        | LayoutElementType::Footnote => "text",
        _ => "other",
    }
}

fn region_kind_buckets(elements: &[LayoutElement]) -> String {
    let mut counts: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for elem in elements {
        let entry = counts
            .entry(layout_kind_bucket(elem.element_type))
            .or_insert((0, 0));
        entry.0 += 1;
        if elem
            .text
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        {
            entry.1 += 1;
        }
    }
    counts
        .into_iter()
        .map(|(kind, (total, drafted))| format!("{kind}:{drafted}/{total}"))
        .collect::<Vec<_>>()
        .join(";")
}

fn region_kind_name(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::Text => "text",
        RegionKind::Title => "title",
        RegionKind::List => "list",
        RegionKind::Table => "table",
        RegionKind::Formula => "formula",
        RegionKind::Figure => "visual",
        RegionKind::Header | RegionKind::Footer => "page_artifact",
        RegionKind::Other => "other",
    }
}

fn stage1_region_kind_stats(stats: &HsdStats) -> String {
    let mut by_kind: BTreeMap<&'static str, (u32, u32, u32, u32)> = BTreeMap::new();
    for region in &stats.stage1_regions {
        let entry = by_kind.entry(region_kind_name(region.kind)).or_default();
        entry.0 += 1;
        entry.1 += region.stats.accept.num_steps;
        entry.2 += region.stats.accept.num_fallbacks;
        entry.3 += region
            .stats
            .accept
            .per_step_accepted
            .iter()
            .copied()
            .sum::<u32>();
    }
    by_kind
        .into_iter()
        .map(|(kind, (regions, steps, fallbacks, accepted_sum))| {
            let aal = if steps == 0 {
                0.0
            } else {
                accepted_sum as f32 / steps as f32
            };
            let fallback_rate = if steps == 0 {
                0.0
            } else {
                fallbacks as f32 / steps as f32
            };
            format!("{kind}:regions={regions},aal={aal:.2},fallback={fallback_rate:.3}")
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn is_mask_or_abandon(category: &str) -> bool {
    matches!(category, "abandon" | "text_mask") || category.ends_with("_mask")
}

fn poly_loc_tokens(poly: &[f32], image_width: u32, image_height: u32) -> Option<String> {
    if poly.len() < 8 {
        return None;
    }
    if image_width == 0 || image_height == 0 {
        return None;
    }
    let mut out = String::new();
    for i in 0..4 {
        let x = loc_index(poly[2 * i], image_width);
        let y = loc_index(poly[2 * i + 1], image_height);
        out.push_str(&format!("<|LOC_{x}|><|LOC_{y}|>"));
    }
    Some(out)
}

fn loc_index(coord: f32, extent: u32) -> i32 {
    ((coord * 1000.0 / extent as f32).round() as i32).clamp(0, 1000)
}

fn build_layout_elements(
    entry: &OmniEntry,
    x_scale: f32,
    y_scale: f32,
    normalize_text: bool,
    require_text: bool,
) -> Vec<LayoutElement> {
    let mut dets: Vec<&LayoutDet> = entry
        .layout_dets
        .iter()
        .filter(|d| d.order.is_some())
        .filter(|d| !d.ignore)
        .filter(|d| !is_mask_or_abandon(d.category_type.as_str()))
        .filter(|d| !require_text || !d.text.trim().is_empty())
        .filter(|d| d.poly.len() >= 8)
        .filter(|d| valid_scaled_poly(&d.poly, x_scale, y_scale))
        .collect();
    dets.sort_by_key(|d| d.order.unwrap());

    dets.into_iter()
        .map(|d| {
            let bbox = BoundingBox::new(
                (0..4)
                    .map(|i| Point::new(d.poly[2 * i], d.poly[2 * i + 1]))
                    .map(|p| Point::new(p.x * x_scale, p.y * y_scale))
                    .collect(),
            );
            let element_type = layout_type_from_omni_category(d.category_type.as_str());
            let mut elem = LayoutElement::new(bbox, element_type, 1.0);
            elem.label = Some(d.category_type.clone());
            let text = d.text.trim();
            if !text.is_empty() {
                elem.text = Some(if normalize_text {
                    normalize_paddleocr_vl_region_draft(text)
                } else {
                    text.to_string()
                });
            }
            elem.order_index = d.order.map(|x| x as u32);
            elem
        })
        .collect()
}

fn normalize_paddleocr_vl_region_draft(input: &str) -> String {
    let mut s = input.to_string();
    let replacements = [
        ("\u{00a0}", " "),
        ("\t", " "),
        ("\r\n", "\n"),
        ("\r", "\n"),
        ("\u{2010}", "-"),
        ("\u{2011}", "-"),
        ("\u{2012}", "-"),
        ("\u{2013}", "-"),
        ("\u{2014}", "-"),
        ("\u{2212}", "-"),
        ("\u{2026}", "..."),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
        ("\u{201c}", "\""),
        ("\u{201d}", "\""),
        ("\u{2217}", "*"),
        ("\u{00d7}", "x"),
        ("\u{2022}", "-"),
        ("\u{25cf}", "-"),
        ("\u{25aa}", "-"),
    ];
    for (from, to) in replacements {
        s = s.replace(from, to);
    }

    s = collapse_horizontal_space(&s);
    s = normalize_math_operator_spaces(&s);
    s = normalize_dash_between_alnums(&s);
    s = normalize_latex_inline_wrappers(&s);
    s.trim().to_string()
}

fn collapse_horizontal_space(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_was_space = false;
    for ch in input.chars() {
        if ch == ' ' {
            if !last_was_space {
                out.push(ch);
            }
            last_was_space = true;
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out
}

fn normalize_math_operator_spaces(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ' ' && should_remove_space_around_operator(&chars, i) {
            i += 1;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn should_remove_space_around_operator(chars: &[char], i: usize) -> bool {
    let prev = previous_non_space(chars, i);
    let next = next_non_space(chars, i + 1);
    match (prev, next) {
        (Some(a), Some(b)) => {
            is_math_operator(a) || is_math_operator(b) || (is_numericish(a) && is_numericish(b))
        }
        _ => false,
    }
}

fn previous_non_space(chars: &[char], mut i: usize) -> Option<char> {
    while i > 0 {
        i -= 1;
        if chars[i] != ' ' {
            return Some(chars[i]);
        }
    }
    None
}

fn next_non_space(chars: &[char], mut i: usize) -> Option<char> {
    while i < chars.len() {
        if chars[i] != ' ' {
            return Some(chars[i]);
        }
        i += 1;
    }
    None
}

fn is_math_operator(ch: char) -> bool {
    matches!(
        ch,
        '<' | '>' | '=' | '+' | '-' | '±' | '≤' | '≥' | '×' | '÷' | '/' | '*'
    )
}

fn is_numericish(ch: char) -> bool {
    ch.is_ascii_digit() || matches!(ch, '.' | ',' | '%' | '<' | '>' | '=' | '+' | '-')
}

fn normalize_dash_between_alnums(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    for (i, ch) in chars.iter().copied().enumerate() {
        if ch == '-'
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1].is_ascii_alphanumeric()
            && chars[i + 1].is_ascii_alphanumeric()
        {
            out.push(' ');
        } else {
            out.push(ch);
        }
    }
    out
}

fn normalize_latex_inline_wrappers(input: &str) -> String {
    let mut s = input.trim().to_string();
    loop {
        let trimmed = s.trim();
        let unwrapped = if trimmed.starts_with("$") && trimmed.ends_with("$") && trimmed.len() > 2 {
            Some(trimmed[1..trimmed.len() - 1].trim())
        } else if trimmed.starts_with("\\(") && trimmed.ends_with("\\)") && trimmed.len() > 4 {
            Some(trimmed[2..trimmed.len() - 2].trim())
        } else {
            None
        };
        match unwrapped {
            Some(inner) => s = inner.to_string(),
            None => break,
        }
    }
    s
}

fn valid_scaled_poly(poly: &[f32], x_scale: f32, y_scale: f32) -> bool {
    if poly.len() < 8 {
        return false;
    }
    let xs = [
        poly[0] * x_scale,
        poly[2] * x_scale,
        poly[4] * x_scale,
        poly[6] * x_scale,
    ];
    let ys = [
        poly[1] * y_scale,
        poly[3] * y_scale,
        poly[5] * y_scale,
        poly[7] * y_scale,
    ];
    let w = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - xs.iter().copied().fold(f32::INFINITY, f32::min);
    let h = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - ys.iter().copied().fold(f32::INFINITY, f32::min);
    if w < 2.0 || h < 2.0 {
        return false;
    }
    let ratio = (w / h).max(h / w);
    ratio <= 200.0
}

fn layout_type_from_omni_category(category: &str) -> LayoutElementType {
    match category {
        "title" => LayoutElementType::DocTitle,
        "header" => LayoutElementType::Header,
        "footer" => LayoutElementType::Footer,
        "page_number" => LayoutElementType::Number,
        "text_block" => LayoutElementType::Text,
        "list_group" => LayoutElementType::List,
        "table" => LayoutElementType::Table,
        "table_caption" => LayoutElementType::TableTitle,
        "table_footnote" => LayoutElementType::Footnote,
        "figure" => LayoutElementType::Image,
        "figure_caption" | "figure_footnote" => LayoutElementType::FigureTitle,
        "chart" => LayoutElementType::Chart,
        "equation_isolated" | "equation_semantic" => LayoutElementType::Formula,
        "equation_caption" | "equation_explanation" => LayoutElementType::Text,
        "reference" => LayoutElementType::Reference,
        "code_txt" | "code_txt_caption" => LayoutElementType::Text,
        _ => LayoutElementType::Other,
    }
}

fn paddleocr_vl_task_for_layout_type(t: LayoutElementType) -> Option<PaddleOcrVlTask> {
    match t {
        LayoutElementType::Table => Some(PaddleOcrVlTask::Table),
        LayoutElementType::Chart => Some(PaddleOcrVlTask::Chart),
        LayoutElementType::Formula => Some(PaddleOcrVlTask::Formula),
        LayoutElementType::Image
        | LayoutElementType::HeaderImage
        | LayoutElementType::FooterImage
        | LayoutElementType::Seal => None,
        _ => Some(PaddleOcrVlTask::Ocr),
    }
}

fn run_paddleocr_vl_region_baseline(
    model: &PaddleOcrVl,
    image: &image::RgbImage,
    elements: &[LayoutElement],
    max_tokens: usize,
) -> Result<RegionDrafts, Box<dyn std::error::Error>> {
    let mut outputs = Vec::new();
    let mut per_element = vec![None; elements.len()];
    let mut per_element_tokens = vec![None; elements.len()];
    for (idx, elem) in elements.iter().enumerate() {
        let Some(task) = paddleocr_vl_task_for_layout_type(elem.element_type) else {
            continue;
        };
        let crop = BBoxCrop::crop_bounding_box(image, &elem.bbox)?;
        let tokens = model
            .generate_tokens(&[crop], &[task], max_tokens)
            .into_iter()
            .next()
            .ok_or("PaddleOCR-VL region baseline returned no result")??;
        let (_, result) = model.decode_tokens(&tokens, task)?;
        if !result.trim().is_empty() {
            let trimmed = result.trim().to_string();
            per_element[idx] = Some(trimmed.clone());
            per_element_tokens[idx] = Some(tokens);
            outputs.push(trimmed);
        }
    }
    Ok(RegionDrafts {
        joined: outputs.join("\n\n"),
        per_element,
        per_element_tokens,
    })
}

fn run_ppocr_rec_drafter(
    predictor: &TextRecognitionPredictor,
    image: &image::RgbImage,
    elements: &[LayoutElement],
) -> Result<RegionDrafts, Box<dyn std::error::Error>> {
    let mut crops = Vec::new();
    let mut indices = Vec::new();
    for (idx, elem) in elements.iter().enumerate() {
        if elem
            .text
            .as_ref()
            .map(|s| s.trim().is_empty())
            .unwrap_or(true)
        {
            continue;
        }
        let crop = BBoxCrop::crop_rotated_bounding_box(image, &elem.bbox)
            .or_else(|_| BBoxCrop::crop_bounding_box(image, &elem.bbox))?;
        crops.push(crop);
        indices.push(idx);
    }

    let mut per_element = vec![None; elements.len()];
    if crops.is_empty() {
        return Ok(RegionDrafts {
            joined: String::new(),
            per_element,
            per_element_tokens: vec![None; elements.len()],
        });
    }

    let output = predictor.predict(crops)?;
    let mut joined = Vec::new();
    for ((idx, text), score) in indices.into_iter().zip(output.texts).zip(output.scores) {
        let text = text.trim().to_string();
        if text.is_empty() || score <= 0.0 {
            continue;
        }
        per_element[idx] = Some(text.clone());
        joined.push(text);
    }
    Ok(RegionDrafts {
        joined: joined.join("\n\n"),
        per_element,
        per_element_tokens: vec![None; elements.len()],
    })
}

fn build_structure_drafter(args: &Args) -> Result<OARStructure, Box<dyn std::error::Error>> {
    let mut builder = OARStructureBuilder::new(&args.structure_layout_model)
        .layout_model_name(&args.structure_layout_model_name)
        .with_ocr(
            &args.structure_ocr_det_model,
            &args.structure_ocr_rec_model,
            &args.structure_ocr_dict_path,
        )
        .region_batch_size(args.structure_region_batch_size);

    if let Some(ort_cfg) = parse_ort_device(drafter_device(args))? {
        builder = builder.ort_session(ort_cfg);
    }
    // PP-DocBlockLayout reading-order model. Required for paper-equivalent
    // multi-column AAL. Empty path = explicit user opt-out; missing file =
    // warn-and-fallback to bbox (y, x) sort.
    let region_path = &args.structure_region_model;
    if region_path.as_os_str().is_empty() {
        eprintln!(
            "[WARN] --structure-region-model is empty: reading-order falls back to bbox (y,x) sort. \
             Multi-column pages may miss Stage-2 acceptance; pass the PP-DocBlockLayout ONNX path to recover paper AAL."
        );
    } else if !region_path.exists() {
        eprintln!(
            "[WARN] --structure-region-model '{}' not found on disk: reading-order falls back to bbox (y,x) sort. \
             Multi-column pages may miss Stage-2 acceptance.",
            region_path.display()
        );
    } else {
        builder = builder
            .with_region_detection(region_path)
            .region_model_name("pp-docblocklayout");
    }
    // Sub-model resolver: empty path = explicit user opt-out, non-empty +
    // missing file = warn-and-skip. Returns None when the path should be
    // dropped so the call-site can branch.
    fn resolve_submodel<'a>(label: &str, path: &'a PathBuf) -> Option<&'a PathBuf> {
        if path.as_os_str().is_empty() {
            return None;
        }
        if !path.exists() {
            eprintln!(
                "[WARN] {label} '{}' not found on disk: structure drafter will skip this sub-model. \
                 Affected regions produce 0 drafts and target VLM autoregresses through them.",
                path.display()
            );
            return None;
        }
        Some(path)
    }

    if let Some(path) = resolve_submodel(
        "--structure-table-cls-model",
        &args.structure_table_cls_model,
    ) {
        builder = builder.with_table_classification(path);
    }
    // Table structure dict applies to both wired and wireless models. Resolve
    // once so the wired/wireless branches can borrow it without re-checking.
    let table_dict = resolve_submodel(
        "--structure-table-dict-path",
        &args.structure_table_dict_path,
    );
    if let Some(path) = resolve_submodel(
        "--structure-wired-table-model",
        &args.structure_wired_table_model,
    ) {
        let Some(dict) = table_dict else {
            return Err(
                "--structure-wired-table-model requires --structure-table-dict-path (or pass empty string to skip wired tables)".into(),
            );
        };
        builder = builder
            .with_wired_table_structure(path)
            .wired_table_structure_model_name("slanext_wired")
            .table_structure_dict_path(dict);
    }
    if let Some(path) = resolve_submodel(
        "--structure-wireless-table-model",
        &args.structure_wireless_table_model,
    ) {
        let Some(dict) = table_dict else {
            return Err(
                "--structure-wireless-table-model requires --structure-table-dict-path (or pass empty string to skip wireless tables)".into(),
            );
        };
        builder = builder
            .with_wireless_table_structure(path)
            .wireless_table_structure_model_name("slanet_plus")
            .table_structure_dict_path(dict);
    }
    if let Some(path) = resolve_submodel(
        "--structure-wired-cell-model",
        &args.structure_wired_cell_model,
    ) {
        builder = builder
            .with_wired_table_cell_detection(path)
            .wired_table_cell_model_name("rtdetr-l_wired_table_cell_det");
    }
    if let Some(path) = resolve_submodel(
        "--structure-wireless-cell-model",
        &args.structure_wireless_cell_model,
    ) {
        builder = builder
            .with_wireless_table_cell_detection(path)
            .wireless_table_cell_model_name("rtdetr-l_wireless_table_cell_det");
    }
    // Formula recognition requires both the ONNX model and its tokenizer.
    // Treat them as a single unit: both present → enable; either missing → skip.
    let formula_model =
        resolve_submodel("--structure-formula-model", &args.structure_formula_model);
    let formula_tokenizer = resolve_submodel(
        "--structure-formula-tokenizer",
        &args.structure_formula_tokenizer,
    );
    match (formula_model, formula_tokenizer) {
        (Some(path), Some(tokenizer)) => {
            builder = builder
                .with_formula_recognition(path, tokenizer, &args.structure_formula_type)
                .formula_recognition_config(FormulaRecognitionConfig {
                    score_threshold: 0.0,
                    max_length: args.structure_formula_max_length,
                    batch_size: args.structure_formula_batch_size,
                });
            if let Some(formula_device) = &args.structure_formula_device {
                builder = builder.formula_ort_session(parse_required_ort_device(formula_device)?);
            }
        }
        (Some(_), None) => {
            return Err(
                "--structure-formula-model requires --structure-formula-tokenizer (or pass both as empty strings to skip formula drafting)".into(),
            );
        }
        (None, Some(_)) => {
            // Tokenizer without model is a no-op; emit a hint but don't fail.
            eprintln!(
                "[WARN] --structure-formula-tokenizer set but --structure-formula-model is empty/missing: \
                 skipping formula drafting. Formula regions will produce 0 drafts."
            );
        }
        (None, None) => {} // both opt-out
    }

    Ok(builder.build()?)
}

fn run_structure_drafter(
    structure: &OARStructure,
    image: &image::RgbImage,
    elements: &[LayoutElement],
    th: MatchThresholds,
) -> Result<RegionDrafts, Box<dyn std::error::Error>> {
    let result = structure.predict_image(image.clone())?;
    Ok(match_structure_to_regions(&result, elements, th))
}

fn run_structure_page_drafter(
    structure: &OARStructure,
    image: &image::RgbImage,
) -> Result<StructureResult, Box<dyn std::error::Error>> {
    Ok(structure.predict_image(image.clone())?)
}

// Thin wrapper kept for readability at call sites; the real implementation
// lives in `oar_ocr_vl::hsd::drafting::structure_result_to_layout_elements`
// so other consumers can share the same OAR-structure → HSD-element bridge.
fn structure_result_hsd_elements(result: &StructureResult) -> Vec<LayoutElement> {
    oar_ocr_vl::hsd::drafting::structure_result_to_layout_elements(result)
}

fn match_structure_to_regions(
    result: &StructureResult,
    elements: &[LayoutElement],
    th: MatchThresholds,
) -> RegionDrafts {
    let mut per_element = vec![None; elements.len()];
    let mut joined = Vec::new();

    for (idx, elem) in elements.iter().enumerate() {
        let draft = match_region(result, elem, th)
            .map(|m| m.text.trim().to_string())
            .filter(|s| !s.is_empty());
        if let Some(text) = draft {
            per_element[idx] = Some(text.clone());
            joined.push(text);
        }
    }

    RegionDrafts {
        joined: joined.join("\n\n"),
        per_element,
        per_element_tokens: vec![None; elements.len()],
    }
}

fn parse_ort_device(device: &str) -> Result<Option<OrtSessionConfig>, Box<dyn std::error::Error>> {
    let device_lower = device.to_lowercase();
    if device_lower == "cpu" {
        return Ok(None);
    }

    #[cfg(feature = "cuda")]
    {
        use oar_ocr_core::core::config::OrtExecutionProvider;
        if device_lower.starts_with("cuda") {
            let device_id = if device_lower == "cuda" {
                0
            } else if let Some(id_str) = device_lower.strip_prefix("cuda:") {
                id_str.parse::<i32>()?
            } else {
                return Err(format!("invalid device: {device}").into());
            };
            return Ok(Some(OrtSessionConfig::new().with_execution_providers(
                vec![
                    OrtExecutionProvider::CUDA {
                        device_id: Some(device_id),
                        gpu_mem_limit: None,
                        arena_extend_strategy: None,
                        cudnn_conv_algo_search: None,
                        cudnn_conv_use_max_workspace: None,
                    },
                    OrtExecutionProvider::CPU,
                ],
            )));
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        if device_lower.starts_with("cuda") {
            return Err("CUDA requested for PP-OCR but cuda feature is not enabled".into());
        }
    }

    Err(format!("unsupported device for PP-OCR drafter: {device}").into())
}

fn parse_required_ort_device(device: &str) -> Result<OrtSessionConfig, Box<dyn std::error::Error>> {
    let device_lower = device.to_lowercase();
    if device_lower == "cpu" {
        use oar_ocr_core::core::config::OrtExecutionProvider;
        return Ok(
            OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU])
        );
    }

    parse_ort_device(device)?
        .ok_or_else(|| format!("unsupported explicit ONNX Runtime device: {device}").into())
}

fn drafter_device(args: &Args) -> &str {
    args.drafter_device
        .as_deref()
        .unwrap_or(args.device.as_str())
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    // Parse the bench JSON.
    let json_path = args.bench_dir.join("OmniDocBench.json");
    println!("Reading {}", json_path.display());
    let bytes = std::fs::read(&json_path)?;
    let entries: Vec<OmniEntry> = serde_json::from_slice(&bytes)?;
    println!("Loaded {} entries", entries.len());

    // Load backend.
    let device = parse_device(&args.device)?;
    println!(
        "Loading {} from {}",
        args.backend.as_str(),
        args.model_dir.display()
    );
    let model = match args.backend {
        Backend::HunyuanOcr => {
            BackendModel::HunyuanOcr(HunyuanOcr::from_dir(&args.model_dir, device)?)
        }
        Backend::PaddleOcrVl => {
            BackendModel::PaddleOcrVl(PaddleOcrVl::from_dir(&args.model_dir, device)?)
        }
        Backend::MinerU => BackendModel::MinerU(MinerU::from_dir(&args.model_dir, device)?),
        Backend::GlmOcr => BackendModel::GlmOcr(GlmOcr::from_dir(&args.model_dir, device)?),
    };
    let paddleocr_vl_task = args.task.to_native();
    let ppocr_rec = if args.draft_source == "ppocr-rec" {
        if !matches!(args.mode, Mode::Region) || !matches!(args.backend, Backend::PaddleOcrVl) {
            return Err(
                "--draft-source ppocr-rec requires --backend paddleocr_vl --mode region".into(),
            );
        }
        println!(
            "Loading PP-OCR rec drafter from {}",
            args.ppocr_rec_model.display()
        );
        let mut builder = TextRecognitionPredictor::builder()
            .score_threshold(args.ppocr_score_thresh)
            .max_text_length(args.ppocr_max_text_length)
            .dict_path(&args.ppocr_dict_path);
        if let Some(ort_cfg) = parse_ort_device(drafter_device(&args))? {
            builder = builder.with_ort_config(ort_cfg);
        }
        Some(builder.build(&args.ppocr_rec_model)?)
    } else {
        None
    };
    let structure_drafter = if args.draft_source == "structure" {
        if matches!(args.mode, Mode::Region)
            && !matches!(args.backend, Backend::HunyuanOcr | Backend::PaddleOcrVl)
        {
            return Err(
                "--draft-source structure requires --backend hunyuanocr or paddleocr_vl with --mode region".into(),
            );
        }
        println!(
            "Loading structure drafter with layout model {}",
            args.structure_layout_model.display()
        );
        Some(build_structure_drafter(&args)?)
    } else {
        None
    };

    let cross_vlm_drafts = if args.draft_source == "cross-vlm-file" {
        let path = args
            .cross_vlm_draft_file
            .as_ref()
            .ok_or("--draft-source cross-vlm-file requires --cross-vlm-draft-file <path>")?;
        println!("Loading cross-VLM drafts from {}", path.display());
        let parsed = CrossVlmDraftFile::load(path)?;
        if let Some(name) = parsed.source_backend.as_deref() {
            println!("  source_backend = {name}, pages = {}", parsed.pages.len());
        } else {
            println!("  pages = {}", parsed.pages.len());
        }
        Some(parsed)
    } else {
        if args.cross_vlm_draft_file.is_some() {
            eprintln!(
                "[warn] --cross-vlm-draft-file is set but --draft-source is not cross-vlm-file; ignoring."
            );
        }
        None
    };

    // Per-page log.
    let csv_path = args
        .output_csv
        .clone()
        .unwrap_or_else(|| args.bench_dir.join("hsd_results.csv"));
    let summary_path = args.output_summary.clone().unwrap_or_else(|| {
        let mut p = csv_path.clone().into_os_string();
        p.push(".md");
        PathBuf::from(p)
    });
    let mut csv = String::from(
        "page_idx,image,subset,language,backend,mode,task,device,drafter_device,draft_source,page_dual_stage,hsd_entry,regions,draft_regions,draft_coverage,region_kind_buckets,stage1_region_kind_stats,draft_3gram_hit_rate,tau,max_tokens,resize_max,start_idx,prompt_kind,prompt,region_prompt_kind,region_prompt,baseline_ms,hsd_ms,drafter_ms,decode_ms,prefill_ms,stage1_decode_ms,stage1_prefill_ms,stage1_verify_steps,stage1_fallback_steps,stage1_aal,stage2_decode_ms,stage2_prefill_ms,stage2_verify_steps,stage2_fallback_steps,stage2_aal,dsv_candidate_ms,dsv_verify_ms,dsv_traverse_ms,dsv_commit_ms,dsv_step_one_ms,dsv_fallback_argmax_ms,dsv_verify_calls,dsv_step_one_calls,dsv_fallback_argmax_calls,dsv_avg_candidates,dsv_max_candidates,dsv_empty_tree_calls,dsv_rejected_tree_calls,dsv_accepted_tree_calls,dsv_avg_tree_nodes,dsv_max_tree_nodes,emitted_tokens,verify_steps,fallback_steps,aal,sr_decode,sr_e2e\n",
    );

    let mut dsv = DsvConfig {
        tau: args.tau,
        ..Default::default()
    };
    // Per-knob CLI overrides. Each defaults to 0 = "honour the preset"; any
    // non-zero value wins so a user can do e.g. `--dsv-window-len 2` on top of
    // `--config-preset omnibench` to relax matching on divergent drafts
    // without having to copy the rest of the preset.
    if args.dsv_window_len > 0 {
        dsv.window_len = args.dsv_window_len;
    }
    if args.dsv_max_candidates > 0 {
        dsv.max_candidates_per_step = args.dsv_max_candidates;
    }
    if args.dsv_max_suffix_len > 0 {
        dsv.max_suffix_len = args.dsv_max_suffix_len;
    }
    let cfg = HsdConfig {
        dsv,
        enable_stage1: matches!(args.mode, Mode::Region)
            || (matches!(args.mode, Mode::Page) && args.page_dual_stage),
        enable_stage2: true,
        max_page_tokens: args.max_tokens,
        max_region_tokens: args.max_tokens,
    };
    if matches!(args.mode, Mode::Region)
        && !matches!(args.backend, Backend::HunyuanOcr | Backend::PaddleOcrVl)
    {
        return Err("--mode region currently supports --backend hunyuanocr or paddleocr_vl".into());
    }

    // Build the candidate pool with optional substring + subset filter.
    let candidates: Vec<&OmniEntry> = entries
        .iter()
        .filter(|e| match &args.filter {
            Some(s) => e.page_info.image_path.contains(s),
            None => true,
        })
        .filter(|e| match &args.subset {
            Some(want) => {
                let got = e
                    .page_info
                    .page_attribute
                    .as_object()
                    .and_then(|m| m.get("subset"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                got == want
            }
            None => true,
        })
        .filter(|e| match &args.language {
            Some(want) => {
                let got = e
                    .page_info
                    .page_attribute
                    .as_object()
                    .and_then(|m| m.get("language"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                got == want
            }
            None => true,
        })
        .skip(args.start_idx)
        .collect();
    let n_pages = args.max_pages.min(candidates.len());
    println!(
        "Running HSD on {} of {} candidates (backend = {}, mode = {:?}, task = {:?}, draft = {}, filter = {:?}, start = {}, τ = {}, max_tokens = {})",
        n_pages,
        candidates.len(),
        args.backend.as_str(),
        args.mode,
        args.task,
        args.draft_source,
        args.filter,
        args.start_idx,
        args.tau,
        args.max_tokens
    );

    let mut sum_baseline_ms: f64 = 0.0;
    let mut sum_hsd_ms: f64 = 0.0;
    let mut sum_decode_ms: f64 = 0.0;
    let mut sum_prefill_ms: f64 = 0.0;
    let mut sum_drafter_ms: f64 = 0.0;
    let mut sum_aal: f64 = 0.0;
    let mut sum_sr_decode: f64 = 0.0;
    let mut sum_sr_e2e: f64 = 0.0;
    let mut sum_emitted: u64 = 0;
    let mut sum_steps: u64 = 0;
    let mut sum_fallbacks: u64 = 0;
    let mut sum_dsv = SpecDecodeStats::default();
    let mut low_aal_outliers = 0u32;
    let mut sr_e2e_outliers = 0u32;
    let mut worst_aal = f32::INFINITY;
    let mut worst_sr_e2e = f64::INFINITY;
    let mut worst_aal_page = String::new();
    let mut worst_sr_e2e_page = String::new();
    let mut hsd_entry_counts: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut counted = 0u32;
    let mut skipped = 0u32;

    println!(
        "\n{:>4} {:>9} {:>9} {:>5} {:>5} {:>5} | {:<60}",
        "idx", "base_ms", "hsd_ms", "AAL", "ndec", "fb", "page"
    );
    println!("{}", "-".repeat(110));

    for (i, entry) in candidates.iter().take(n_pages).enumerate() {
        let candidate_idx = args.start_idx + i;
        let img_path = args
            .bench_dir
            .join("images")
            .join(&entry.page_info.image_path);
        if !img_path.exists() {
            if args.skip_missing {
                skipped += 1;
                continue;
            } else {
                return Err(format!("missing image: {}", img_path.display()).into());
            }
        }
        let mut image = match load_image(&img_path) {
            Ok(img) => img,
            Err(e) => {
                if args.skip_missing {
                    eprintln!("[skip] {} ({e})", entry.page_info.image_path);
                    skipped += 1;
                    continue;
                } else {
                    return Err(e.into());
                }
            }
        };
        let gt_image_size = image.dimensions();
        let mut x_scale = 1.0f32;
        let mut y_scale = 1.0f32;
        if args.resize_max > 0 {
            let long = image.width().max(image.height());
            if long > args.resize_max {
                let scale = args.resize_max as f32 / long as f32;
                let nw = (image.width() as f32 * scale).round() as u32;
                let nh = (image.height() as f32 * scale).round() as u32;
                x_scale = nw as f32 / image.width() as f32;
                y_scale = nh as f32 / image.height() as f32;
                // Match upstream Python's pre-resize filter (PIL.Image.LANCZOS)
                // — using CatmullRom here was the dominant source of pixel
                // drift vs the Python pipeline (per-patch cos 0.998 →
                // 1.000 after this change), which compounded into ~18 %
                // divergence at the prefill's last-token logits.
                image = image::imageops::resize(&image, nw, nh, FilterType::Lanczos3);
            }
        }
        let need_layout_elements = matches!(args.mode, Mode::Region)
            || (matches!(args.mode, Mode::Page)
                && args.page_dual_stage
                && args.draft_source == "gt");
        let elements = if need_layout_elements {
            let normalize_text = args.normalize_draft && args.draft_source == "gt";
            let require_text = matches!(args.draft_source.as_str(), "gt" | "ppocr-rec");
            build_layout_elements(entry, x_scale, y_scale, normalize_text, require_text)
        } else {
            Vec::new()
        };
        let draft = match args.mode {
            Mode::Page => build_gt_draft(
                entry,
                args.backend,
                args.task,
                gt_image_size,
                args.hunyuanocr_legacy_gt_format,
            ),
            Mode::Region => elements
                .iter()
                .filter_map(|e| e.text.as_ref())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n"),
        };
        if draft.trim().is_empty() {
            let draft_filled_later = matches!(
                args.draft_source.as_str(),
                "baseline" | "structure" | "cross-vlm-file"
            ) && (matches!(args.mode, Mode::Page)
                || matches!(args.mode, Mode::Region));
            if !draft_filled_later {
                // No usable draft text — skip rather than running with empty draft
                // (which would just be a baseline run with extra overhead).
                skipped += 1;
                continue;
            }
        }
        let (hunyuanocr_prompt, hunyuanocr_prompt_kind) = prompt_for_entry(entry, &args);
        let (hunyuanocr_region_prompt, hunyuanocr_region_prompt_kind) =
            hunyuanocr_region_prompt(entry, &args);
        let (mineru_prompt, mineru_prompt_kind) = mineru_prompt(&args);
        let (mineru_region_prompt, mineru_region_prompt_kind) = mineru_region_prompt(&args);
        let (glmocr_prompt, glmocr_prompt_kind) = glmocr_prompt(&args);
        let (glmocr_region_prompt, glmocr_region_prompt_kind) = glmocr_region_prompt(&args);
        let (prompt_text, prompt_kind) = match args.backend {
            Backend::HunyuanOcr => (hunyuanocr_prompt, hunyuanocr_prompt_kind),
            Backend::PaddleOcrVl => (paddleocr_vl_task.prompt(), "paddleocr_vl_task"),
            Backend::MinerU => (mineru_prompt, mineru_prompt_kind),
            Backend::GlmOcr => (glmocr_prompt, glmocr_prompt_kind),
        };
        let (region_prompt_text, region_prompt_kind) = match args.backend {
            Backend::HunyuanOcr => (hunyuanocr_region_prompt, hunyuanocr_region_prompt_kind),
            Backend::PaddleOcrVl => (paddleocr_vl_task.prompt(), "paddleocr_vl_region_task"),
            Backend::MinerU => (mineru_region_prompt, mineru_region_prompt_kind),
            Backend::GlmOcr => (glmocr_region_prompt, glmocr_region_prompt_kind),
        };

        // Baseline.
        let t0 = Instant::now();
        let mut region_baseline_drafts: Option<Vec<Option<String>>> = None;
        let mut region_baseline_token_drafts: Option<Vec<Option<Vec<u32>>>> = None;
        let mut baseline_tokens: Option<Vec<u32>> = None;
        let baseline_result: Result<String, Box<dyn std::error::Error>> = match (&model, args.mode)
        {
            (BackendModel::HunyuanOcr(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[hunyuanocr_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (BackendModel::PaddleOcrVl(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[paddleocr_vl_task], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model
                    .decode_tokens(&toks, paddleocr_vl_task)
                    .map(|(_, processed)| processed)
                    .map_err(|e| e.into())
            }
            (BackendModel::MinerU(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[mineru_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (BackendModel::GlmOcr(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[glmocr_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (BackendModel::PaddleOcrVl(model), Mode::Region) => {
                let drafts =
                    run_paddleocr_vl_region_baseline(model, &image, &elements, args.max_tokens)?;
                region_baseline_drafts = Some(drafts.per_element);
                region_baseline_token_drafts = Some(drafts.per_element_tokens);
                Ok(drafts.joined)
            }
            (BackendModel::HunyuanOcr(model), Mode::Region) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[hunyuanocr_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (_, Mode::Region) => {
                Err("--mode region currently supports --backend hunyuanocr or paddleocr_vl".into())
            }
        };
        let baseline_dur = t0.elapsed();
        let baseline_text = match baseline_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "[skip] baseline failed for {}: {}",
                    entry.page_info.image_path, e
                );
                skipped += 1;
                continue;
            }
        };
        if args.preview > 0 {
            let bp: String = baseline_text.chars().take(args.preview).collect();
            let dp: String = draft.chars().take(args.preview).collect();
            println!(
                "--- BASELINE ({} chars total) ---\n{bp}\n",
                baseline_text.len()
            );
            println!("--- DRAFT ({} chars total) ---\n{dp}\n", draft.len());
        }

        // Pick draft per --draft-source.
        let mut hsd_elements: Option<Vec<LayoutElement>> = None;
        let mut external_drafter_dur = Duration::ZERO;
        let mut diagnostic_region_drafts: Vec<String> = Vec::new();
        let actual_draft = match args.draft_source.as_str() {
            "gt" => {
                if matches!(args.mode, Mode::Page) && args.page_dual_stage {
                    hsd_elements = Some(elements.clone());
                }
                draft.clone()
            }
            "baseline" if matches!(args.mode, Mode::Region) => {
                let per_element = region_baseline_drafts
                    .as_ref()
                    .ok_or("missing region baseline drafts")?;
                let mut oracle_elements = elements.clone();
                for (elem, baseline) in oracle_elements.iter_mut().zip(per_element.iter()) {
                    if let Some(text) = baseline {
                        elem.text = Some(text.clone());
                    }
                }
                hsd_elements = Some(oracle_elements);
                baseline_text.clone()
            }
            "ppocr-rec" if matches!(args.mode, Mode::Region) => {
                let predictor = ppocr_rec
                    .as_ref()
                    .ok_or("missing PP-OCR recognition drafter")?;
                let t_drafter = Instant::now();
                let ppocr_drafts = run_ppocr_rec_drafter(predictor, &image, &elements)?;
                external_drafter_dur += t_drafter.elapsed();
                let mut drafter_elements = elements.clone();
                for (elem, draft) in drafter_elements
                    .iter_mut()
                    .zip(ppocr_drafts.per_element.iter())
                {
                    if let Some(text) = draft {
                        elem.text = Some(text.clone());
                    }
                }
                hsd_elements = Some(drafter_elements);
                ppocr_drafts.joined
            }
            "structure" if matches!(args.mode, Mode::Region) => {
                let structure = structure_drafter
                    .as_ref()
                    .ok_or("missing structure drafter")?;
                let t_drafter = Instant::now();
                let structure_drafts = run_structure_drafter(
                    structure,
                    &image,
                    &elements,
                    MatchThresholds::new(
                        args.structure_same_category_iou,
                        args.structure_iou_threshold,
                        args.structure_allow_generic_fallback,
                    ),
                )?;
                external_drafter_dur += t_drafter.elapsed();
                let mut drafter_elements = elements.clone();
                for (elem, draft) in drafter_elements
                    .iter_mut()
                    .zip(structure_drafts.per_element.iter())
                {
                    elem.text = draft.clone();
                }
                hsd_elements = Some(drafter_elements);
                structure_drafts.joined
            }
            "structure" if matches!(args.mode, Mode::Page) => {
                let structure = structure_drafter
                    .as_ref()
                    .ok_or("missing structure drafter")?;
                let t_drafter = Instant::now();
                let result = run_structure_page_drafter(structure, &image)?;
                external_drafter_dur += t_drafter.elapsed();
                let elems = structure_result_hsd_elements(&result);
                let adapter = target_draft_adapter(args.backend, args.task);
                diagnostic_region_drafts = region_markdowns_for(&elems, &[], adapter);
                hsd_elements = Some(elems);
                page_markdown_for(hsd_elements.as_deref().unwrap_or(&[]), &[], adapter)
            }
            "baseline" => baseline_text.clone(),
            "cross-vlm-file" => {
                // Per-page raw drafts from another VLM. The target backend's
                // adapter handles surface conversion (HTML↔OTSL, formula
                // wrapping); we just match elements by bbox IoU and stash the
                // raw text on `elem.text` so downstream `generate_hsd_full`
                // picks it up via `region_markdown_for`.
                let cross = cross_vlm_drafts
                    .as_ref()
                    .ok_or("missing cross-VLM drafts (loaded earlier?)")?;
                let regions = cross.lookup_page(&entry.page_info.image_path);
                let mut drafter_elements = elements.clone();
                let mut matched = 0usize;
                if let Some(regions) = regions {
                    for elem in drafter_elements.iter_mut() {
                        let elem_bbox = bbox_xyxy(&elem.bbox);
                        if let Some(region) = match_cross_vlm_region(
                            &elem_bbox,
                            regions,
                            args.cross_vlm_iou_threshold,
                        ) {
                            elem.text = Some(region.raw_text.clone());
                            matched += 1;
                        } else {
                            // No matching cross-VLM region — drop the text so
                            // the element won't contribute a stale draft to
                            // Stage 1 / Stage 2.
                            elem.text = None;
                        }
                    }
                } else {
                    // No page entry — clear all element texts so the bench
                    // reports `draft_regions=0` rather than silently reusing
                    // the OmniDocBench GT text on this page.
                    for elem in drafter_elements.iter_mut() {
                        elem.text = None;
                    }
                }
                if regions.is_none() {
                    eprintln!(
                        "[warn] cross-vlm-file has no entry for {}",
                        entry.page_info.image_path
                    );
                }
                let adapter = target_draft_adapter(args.backend, args.task);
                diagnostic_region_drafts = region_markdowns_for(&drafter_elements, &[], adapter);
                let joined = page_markdown_for(&drafter_elements, &[], adapter);
                hsd_elements = Some(drafter_elements);
                println!(
                    "  cross-vlm-file matched {matched}/{} regions for {}",
                    elements.len(),
                    entry.page_info.image_path,
                );
                joined
            }
            other => return Err(format!("unknown --draft-source: {other}").into()),
        };
        if actual_draft.trim().is_empty() {
            skipped += 1;
            continue;
        }

        // Build the Stage-2 draft set `Ỹ^pg` per paper Eq. 3. Most draft
        // sources are inherently single-document (gt = ground-truth page,
        // baseline = the VLM's own page output), so they form a 1-element
        // set. The structure+page route splits per layout element so the
        // matcher can scan each region draft independently (Eqs. 1+2),
        // preserving per-region n-gram locality even when the drafter's
        // page-level format diverges from the target VLM's.
        let actual_drafts: Vec<String> = if matches!(args.mode, Mode::Page)
            && args.draft_source == "structure"
            && !diagnostic_region_drafts.is_empty()
        {
            diagnostic_region_drafts.clone()
        } else {
            vec![actual_draft.clone()]
        };

        let page_draft_tokens = if matches!(args.mode, Mode::Page) {
            Some(match &model {
                BackendModel::HunyuanOcr(model) => {
                    tokenize_draft(model.tokenizer(), &actual_draft)?
                }
                BackendModel::PaddleOcrVl(model) => {
                    tokenize_draft(model.tokenizer(), &actual_draft)?
                }
                BackendModel::MinerU(model) => tokenize_draft(model.tokenizer(), &actual_draft)?,
                BackendModel::GlmOcr(model) => tokenize_draft(model.tokenizer(), &actual_draft)?,
            })
        } else {
            None
        };
        let draft_3gram_hit_rate = if let (Some(baseline_tokens), Some(draft_tokens)) =
            (baseline_tokens.as_ref(), page_draft_tokens.as_ref())
        {
            let (hits, total) =
                count_window_hits(baseline_tokens, draft_tokens, args.token_diff_window_len);
            if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            }
        } else {
            0.0
        };
        let diagnostic_region_draft_tokens =
            if matches!(args.mode, Mode::Page) && !diagnostic_region_drafts.is_empty() {
                let tokenizer = match &model {
                    BackendModel::HunyuanOcr(model) => model.tokenizer(),
                    BackendModel::PaddleOcrVl(model) => model.tokenizer(),
                    BackendModel::MinerU(model) => model.tokenizer(),
                    BackendModel::GlmOcr(model) => model.tokenizer(),
                };
                Some(
                    diagnostic_region_drafts
                        .iter()
                        .map(|d| tokenize_draft(tokenizer, d))
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
        let per_draft_max_hits = if let (Some(baseline_tokens), Some(region_drafts)) = (
            baseline_tokens.as_ref(),
            diagnostic_region_draft_tokens.as_ref(),
        ) {
            Some(best_per_draft_window_hits(
                baseline_tokens,
                region_drafts,
                args.token_diff_window_len,
            ))
        } else {
            None
        };

        if let Some(path) = &args.token_diff_output {
            if !matches!(args.mode, Mode::Page) {
                return Err("--token-diff-output currently supports --mode page only".into());
            }
            let baseline_tokens = baseline_tokens
                .as_ref()
                .ok_or("--token-diff-output requires page-mode baseline tokens")?;
            let draft_tokens = page_draft_tokens
                .as_ref()
                .ok_or("--token-diff-output requires page-mode draft tokens")?;
            let tokenizer = match &model {
                BackendModel::HunyuanOcr(model) => model.tokenizer(),
                BackendModel::PaddleOcrVl(model) => model.tokenizer(),
                BackendModel::MinerU(model) => model.tokenizer(),
                BackendModel::GlmOcr(model) => model.tokenizer(),
            };
            let mut report = String::new();
            append_token_diff_report(
                &mut report,
                tokenizer,
                TokenDiffInput {
                    run_row_idx: i,
                    candidate_idx,
                    image_path: &entry.page_info.image_path,
                    backend: args.backend,
                    mode: args.mode,
                    draft_source: &args.draft_source,
                    baseline_text: &baseline_text,
                    draft_text: &actual_draft,
                    baseline_tokens,
                    draft_tokens,
                    structure_elements: hsd_elements.as_deref(),
                    hsd_page_draft_count: actual_drafts.len(),
                    region_draft_count: diagnostic_region_drafts.len(),
                    per_draft_max_hits,
                    limit: args.token_diff_limit,
                    window_len: args.token_diff_window_len,
                },
            );
            std::fs::write(path, report)?;
            println!("Token diff report -> {}", path.display());
            if args.token_diff_only {
                return Ok(());
            }
        }

        // For page mode, `draft_region_count` is the size of `Ỹ^pg` (the
        // Stage-2 draft set per paper Eq. 3) — `actual_drafts.len()` after
        // the multi-draft refactor, NOT a 0/1 indicator of "is the draft
        // non-empty?". For region mode it's the count of elements with
        // non-empty draft text, unchanged.
        let hsd_element_count = hsd_elements.as_ref().map_or(elements.len(), Vec::len);
        let draft_region_count = if matches!(args.mode, Mode::Page) {
            if let Some(elems) = hsd_elements.as_ref().filter(|elems| !elems.is_empty()) {
                elems
                    .iter()
                    .filter(|e| e.text.as_deref().is_some_and(|s| !s.trim().is_empty()))
                    .count()
            } else {
                actual_drafts
                    .iter()
                    .filter(|d| !d.trim().is_empty())
                    .count()
            }
        } else {
            hsd_elements
                .as_ref()
                .unwrap_or(&elements)
                .iter()
                .filter(|e| e.text.as_deref().is_some_and(|s| !s.trim().is_empty()))
                .count()
        };
        let draft_coverage = if hsd_element_count == 0 {
            if matches!(args.mode, Mode::Page) && draft_region_count > 0 {
                1.0
            } else {
                0.0
            }
        } else {
            draft_region_count as f64 / hsd_element_count as f64
        };
        let region_kind_buckets = hsd_elements
            .as_deref()
            .map(region_kind_buckets)
            .unwrap_or_else(|| region_kind_buckets(&elements));

        // HSD with the draft.
        let t1 = Instant::now();
        let oracle_draft = (args.draft_source == "baseline")
            .then(|| {
                baseline_tokens
                    .as_ref()
                    .map(|t| vec![Draft::new(t.clone())])
            })
            .flatten();
        let (hsd_entry, hsd) = match (&model, args.mode, oracle_draft.as_deref()) {
            (BackendModel::HunyuanOcr(model), Mode::Page, Some(token_drafts)) => (
                "hunyuanocr.generate_hsd_with_token_drafts",
                model.generate_hsd_with_token_drafts(&image, hunyuanocr_prompt, token_drafts, &cfg),
            ),
            (BackendModel::HunyuanOcr(model), Mode::Page, None) => {
                if args.page_dual_stage {
                    let elems = require_hsd_elements(hsd_elements.as_deref(), "hunyuanocr")?;
                    (
                        "hunyuanocr.generate_hsd_full",
                        model.generate_hsd_full(
                            &image,
                            oar_ocr_vl::HunyuanHsdPrompts {
                                page: hunyuanocr_prompt,
                                region: hunyuanocr_region_prompt,
                            },
                            elems,
                            &[],
                            |elem| elem.text.iter().cloned().collect(),
                            &cfg,
                        ),
                    )
                } else {
                    (
                        "hunyuanocr.generate_hsd",
                        model.generate_hsd(&image, hunyuanocr_prompt, &actual_drafts, &cfg),
                    )
                }
            }
            (BackendModel::PaddleOcrVl(model), Mode::Page, Some(token_drafts)) => (
                "paddleocr_vl.generate_hsd_with_token_drafts",
                model.generate_hsd_with_token_drafts(&image, paddleocr_vl_task, token_drafts, &cfg),
            ),
            (BackendModel::PaddleOcrVl(model), Mode::Page, None) => (
                "paddleocr_vl.generate_hsd",
                model.generate_hsd(&image, paddleocr_vl_task, &actual_drafts, &cfg),
            ),
            (BackendModel::MinerU(model), Mode::Page, Some(token_drafts)) => (
                "mineru.generate_hsd_with_token_drafts",
                model.generate_hsd_with_token_drafts(&image, mineru_prompt, token_drafts, &cfg),
            ),
            (BackendModel::MinerU(model), Mode::Page, None) => {
                if args.page_dual_stage {
                    let elems = require_hsd_elements(hsd_elements.as_deref(), "mineru")?;
                    (
                        "mineru.generate_hsd_full",
                        model.generate_hsd_full(
                            &image,
                            elems,
                            &[],
                            mineru_prompt,
                            mineru_region_prompt,
                            &cfg,
                        ),
                    )
                } else {
                    (
                        "mineru.generate_hsd",
                        model.generate_hsd(&image, mineru_prompt, &actual_drafts, &cfg),
                    )
                }
            }
            (BackendModel::GlmOcr(model), Mode::Page, Some(token_drafts)) => (
                "glmocr.generate_hsd_with_token_drafts",
                model.generate_hsd_with_token_drafts(&image, glmocr_prompt, token_drafts, &cfg),
            ),
            (BackendModel::GlmOcr(model), Mode::Page, None) => {
                if args.page_dual_stage {
                    let elems = require_hsd_elements(hsd_elements.as_deref(), "glmocr")?;
                    (
                        "glmocr.generate_hsd_full",
                        model.generate_hsd_full(
                            &image,
                            elems,
                            &[],
                            glmocr_prompt,
                            glmocr_region_prompt,
                            &cfg,
                        ),
                    )
                } else {
                    (
                        "glmocr.generate_hsd",
                        model.generate_hsd(&image, glmocr_prompt, &actual_drafts, &cfg),
                    )
                }
            }
            (BackendModel::PaddleOcrVl(model), Mode::Region, _) => {
                let elems = hsd_elements.as_deref().unwrap_or(elements.as_slice());
                if args.draft_source == "baseline" {
                    let token_drafts = region_baseline_token_drafts.as_ref().ok_or_else(|| {
                        oar_ocr_core::core::OCRError::InvalidInput {
                            message: "missing region baseline token drafts".to_string(),
                        }
                    })?;
                    (
                        "paddleocr_vl.generate_hsd_full_with_token_drafts",
                        model.generate_hsd_full_with_token_drafts(
                            &image,
                            elems,
                            &[],
                            token_drafts,
                            &cfg,
                        ),
                    )
                } else {
                    (
                        "paddleocr_vl.generate_hsd_full",
                        model.generate_hsd_full(&image, elems, &[], &cfg),
                    )
                }
            }
            (BackendModel::HunyuanOcr(model), Mode::Region, _) => {
                let elems = hsd_elements.as_deref().unwrap_or(elements.as_slice());
                (
                    "hunyuanocr.generate_hsd_full",
                    model.generate_hsd_full(
                        &image,
                        oar_ocr_vl::HunyuanHsdPrompts {
                            page: hunyuanocr_prompt,
                            region: hunyuanocr_region_prompt,
                        },
                        elems,
                        &[],
                        |elem| elem.text.iter().cloned().collect(),
                        &cfg,
                    ),
                )
            }
            (_, Mode::Region, _) => {
                unreachable!("--mode region is rejected for unsupported backends before the loop")
            }
        };
        let hsd = match hsd {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[skip] HSD failed for {}: {e}", entry.page_info.image_path);
                let mut cur: Option<&dyn std::error::Error> = std::error::Error::source(&e);
                while let Some(s) = cur {
                    eprintln!("    caused by: {s}");
                    cur = s.source();
                }
                skipped += 1;
                continue;
            }
        };
        let hsd_dur = t1.elapsed() + external_drafter_dur;
        let (_text, mut stats) = hsd;
        stats.drafter += external_drafter_dur;
        *hsd_entry_counts.entry(hsd_entry).or_insert(0) += 1;

        let baseline_ms = baseline_dur.as_secs_f64() * 1000.0;
        let hsd_ms = hsd_dur.as_secs_f64() * 1000.0;
        let drafter_ms = stats.drafter.as_secs_f64() * 1000.0;
        let stage1_decode_ms = stats.stage1.decode.as_secs_f64() * 1000.0;
        let stage1_prefill_ms = stats.stage1.vision_prefill.as_secs_f64() * 1000.0;
        let stage1_aal = stats.stage1.accept.aal();
        let stage2_decode_ms = stats.stage2.decode.as_secs_f64() * 1000.0;
        let stage2_prefill_ms = stats.stage2.vision_prefill.as_secs_f64() * 1000.0;
        let stage2_aal = stats.stage2.accept.aal();
        let stage1_region_kind_stats = stage1_region_kind_stats(&stats);
        let stage = match args.mode {
            Mode::Page => &stats.stage2,
            Mode::Region => &stats.stage1,
        };
        let decode_ms = stage.decode.as_secs_f64() * 1000.0;
        let prefill_ms = stage.vision_prefill.as_secs_f64() * 1000.0;
        let baseline_decode_estimate_ms = (baseline_ms - prefill_ms).max(0.0);
        let sr_decode = if decode_ms > 0.0 {
            baseline_decode_estimate_ms / decode_ms
        } else {
            0.0
        };
        let sr_e2e = if hsd_ms > 0.0 {
            baseline_ms / hsd_ms
        } else {
            0.0
        };
        let aal = stage.accept.aal();

        sum_baseline_ms += baseline_ms;
        sum_hsd_ms += hsd_ms;
        sum_decode_ms += decode_ms;
        sum_prefill_ms += prefill_ms;
        sum_drafter_ms += drafter_ms;
        sum_aal += aal as f64;
        sum_sr_decode += sr_decode;
        sum_sr_e2e += sr_e2e;
        sum_emitted += stage.emitted_tokens as u64;
        sum_steps += stage.accept.num_steps as u64;
        sum_fallbacks += stage.accept.num_fallbacks as u64;
        sum_dsv.add_assign(&stage.dsv);
        if aal <= args.outlier_aal_threshold {
            low_aal_outliers += 1;
        }
        if sr_e2e < args.outlier_sr_e2e_threshold {
            sr_e2e_outliers += 1;
        }
        if aal < worst_aal {
            worst_aal = aal;
            worst_aal_page = entry.page_info.image_path.clone();
        }
        if sr_e2e < worst_sr_e2e {
            worst_sr_e2e = sr_e2e;
            worst_sr_e2e_page = entry.page_info.image_path.clone();
        }
        counted += 1;

        let subset = page_attr(entry, "subset");
        let language = page_attr(entry, "language");
        let img_short: String = entry.page_info.image_path.chars().take(60).collect();
        println!(
            "{:>4} {:>9.0} {:>9.0} {:>5.1} {:>5} {:>5} | {:<60}",
            i,
            baseline_ms,
            hsd_ms,
            aal,
            stage.accept.num_steps,
            stage.accept.num_fallbacks,
            img_short
        );

        csv.push_str(&csv_row(&[
            i.to_string(),
            entry.page_info.image_path.clone(),
            subset.to_string(),
            language.to_string(),
            args.backend.as_str().to_string(),
            format!("{:?}", args.mode).to_lowercase(),
            format!("{:?}", args.task).to_lowercase(),
            args.device.clone(),
            drafter_device(&args).to_string(),
            args.draft_source.clone(),
            args.page_dual_stage.to_string(),
            hsd_entry.to_string(),
            hsd_element_count.to_string(),
            draft_region_count.to_string(),
            format!("{draft_coverage:.3}"),
            region_kind_buckets,
            stage1_region_kind_stats,
            format!("{draft_3gram_hit_rate:.3}"),
            format!("{:.3}", args.tau),
            args.max_tokens.to_string(),
            args.resize_max.to_string(),
            args.start_idx.to_string(),
            prompt_kind.to_string(),
            prompt_text.to_string(),
            region_prompt_kind.to_string(),
            region_prompt_text.to_string(),
            format!("{baseline_ms:.1}"),
            format!("{hsd_ms:.1}"),
            format!("{drafter_ms:.1}"),
            format!("{decode_ms:.1}"),
            format!("{prefill_ms:.1}"),
            format!("{stage1_decode_ms:.1}"),
            format!("{stage1_prefill_ms:.1}"),
            stats.stage1.accept.num_steps.to_string(),
            stats.stage1.accept.num_fallbacks.to_string(),
            format!("{stage1_aal:.2}"),
            format!("{stage2_decode_ms:.1}"),
            format!("{stage2_prefill_ms:.1}"),
            stats.stage2.accept.num_steps.to_string(),
            stats.stage2.accept.num_fallbacks.to_string(),
            format!("{stage2_aal:.2}"),
            format_duration_ms(stage.dsv.candidate_build),
            format_duration_ms(stage.dsv.verify_tree),
            format_duration_ms(stage.dsv.traverse),
            format_duration_ms(stage.dsv.commit),
            format_duration_ms(stage.dsv.step_one),
            format_duration_ms(stage.dsv.fallback_argmax),
            stage.dsv.verify_tree_calls.to_string(),
            stage.dsv.step_one_calls.to_string(),
            stage.dsv.fallback_argmax_calls.to_string(),
            format!("{:.1}", stage.dsv.avg_candidates()),
            stage.dsv.candidates_max.to_string(),
            stage.dsv.empty_tree_calls.to_string(),
            stage.dsv.rejected_tree_calls.to_string(),
            stage.dsv.accepted_tree_calls.to_string(),
            format!("{:.1}", stage.dsv.avg_tree_nodes()),
            stage.dsv.tree_nodes_max.to_string(),
            stage.emitted_tokens.to_string(),
            stage.accept.num_steps.to_string(),
            stage.accept.num_fallbacks.to_string(),
            format!("{aal:.2}"),
            format!("{sr_decode:.3}"),
            format!("{sr_e2e:.3}"),
        ]));
        std::fs::write(&csv_path, &csv)?;
    }

    if counted == 0 {
        eprintln!("No pages produced valid measurements (skipped: {skipped}).");
        return Err("nothing measured".into());
    }
    let n = counted as f64;

    println!("{}", "-".repeat(110));
    println!(
        "\n=== AGGREGATE ({} pages, {} skipped) ===",
        counted, skipped
    );
    println!("baseline e2e (mean):   {:.1} ms", sum_baseline_ms / n);
    println!("HSD e2e (mean):        {:.1} ms", sum_hsd_ms / n);
    println!("drafter (mean):        {:.1} ms", sum_drafter_ms / n);
    println!("HSD decode (mean):     {:.1} ms", sum_decode_ms / n);
    println!("HSD prefill (mean):    {:.1} ms", sum_prefill_ms / n);
    println!("emitted tokens (mean): {}", sum_emitted / counted as u64);
    println!("verify steps (mean):   {}", sum_steps / counted as u64);
    println!("fallback steps (mean): {}", sum_fallbacks / counted as u64);
    println!("AAL (mean):            {:.2}", sum_aal / n);
    println!(
        "low-AAL outliers:      {} (AAL <= {:.2})",
        low_aal_outliers, args.outlier_aal_threshold
    );
    println!(
        "SR_e2e outliers:       {} (SR_e2e < {:.2})",
        sr_e2e_outliers, args.outlier_sr_e2e_threshold
    );
    println!(
        "DSV candidate (mean):  {:.1} ms",
        sum_dsv.candidate_build.as_secs_f64() * 1000.0 / n
    );
    println!(
        "DSV verify_tree mean:  {:.1} ms (calls/page {:.1}, avg nodes {:.1}, max nodes {})",
        sum_dsv.verify_tree.as_secs_f64() * 1000.0 / n,
        sum_dsv.verify_tree_calls as f64 / n,
        sum_dsv.avg_tree_nodes(),
        sum_dsv.tree_nodes_max
    );
    println!(
        "DSV candidates:        avg {:.1}, max {}, empty/reject/accept {} / {} / {}",
        sum_dsv.avg_candidates(),
        sum_dsv.candidates_max,
        sum_dsv.empty_tree_calls,
        sum_dsv.rejected_tree_calls,
        sum_dsv.accepted_tree_calls
    );
    println!(
        "DSV traverse/commit:   {:.1} / {:.1} ms",
        sum_dsv.traverse.as_secs_f64() * 1000.0 / n,
        sum_dsv.commit.as_secs_f64() * 1000.0 / n
    );
    println!(
        "DSV step_one mean:     {:.1} ms (calls/page {:.1})",
        sum_dsv.step_one.as_secs_f64() * 1000.0 / n,
        sum_dsv.step_one_calls as f64 / n
    );
    println!();
    println!("SR_decode (mean):      {:.2}×", sum_sr_decode / n);
    println!("SR_e2e (mean):         {:.2}×", sum_sr_e2e / n);
    // Throughput-style aggregate (sum baseline / sum HSD).
    let sr_e2e_total = sum_baseline_ms / sum_hsd_ms;
    let sr_decode_total = (sum_baseline_ms - sum_prefill_ms).max(0.0) / sum_decode_ms;
    println!("SR_e2e (total time):   {:.2}×", sr_e2e_total);
    println!("SR_decode (total):     {:.2}×", sr_decode_total);

    let fallback_rate = if sum_steps > 0 {
        sum_fallbacks as f64 / sum_steps as f64
    } else {
        0.0
    };
    let hsd_entries = hsd_entry_counts
        .iter()
        .map(|(entry, count)| format!("{entry}={count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let summary = format!(
        "# HSD OmniDocBench Summary\n\n\
         ## Run\n\n\
         | field | value |\n\
         |---|---|\n\
         | backend | {backend} |\n\
         | mode | {mode:?} |\n\
         | task | {task:?} |\n\
         | device | {device} |\n\
         | drafter device | {drafter_device} |\n\
         | draft source | {draft_source} |\n\
         | page dual stage | {page_dual_stage} |\n\
         | HSD entries | {hsd_entries} |\n\
         | region prompt | {region_prompt_kind} |\n\
         | tau | {tau:.3} |\n\
         | max tokens | {max_tokens} |\n\
         | resize max | {resize_max} |\n\
         | start idx | {start_idx} |\n\
         | max pages | {max_pages} |\n\
         | subset filter | {subset_filter} |\n\
         | language filter | {language_filter} |\n\
         | outlier AAL threshold | {outlier_aal_threshold:.2} |\n\
         | outlier SR_e2e threshold | {outlier_sr_e2e_threshold:.2} |\n\
         | CSV | {csv_path} |\n\n\
         ## Aggregate\n\n\
         | metric | value |\n\
         |---|---:|\n\
         | measured pages | {counted} |\n\
         | skipped pages | {skipped} |\n\
         | baseline e2e mean ms | {baseline_mean:.1} |\n\
         | HSD e2e mean ms | {hsd_mean:.1} |\n\
         | drafter mean ms | {drafter_mean:.1} |\n\
         | HSD decode mean ms | {decode_mean:.1} |\n\
         | HSD prefill mean ms | {prefill_mean:.1} |\n\
         | DSV candidate mean ms | {dsv_candidate_mean:.1} |\n\
         | DSV verify_tree mean ms | {dsv_verify_mean:.1} |\n\
         | DSV traverse mean ms | {dsv_traverse_mean:.1} |\n\
         | DSV commit mean ms | {dsv_commit_mean:.1} |\n\
         | DSV step_one mean ms | {dsv_step_one_mean:.1} |\n\
         | DSV verify calls/page | {dsv_verify_calls_mean:.1} |\n\
         | DSV step_one calls/page | {dsv_step_one_calls_mean:.1} |\n\
         | DSV avg candidates | {dsv_avg_candidates:.1} |\n\
         | DSV max candidates | {dsv_max_candidates} |\n\
         | DSV empty tree calls | {dsv_empty_tree_calls} |\n\
         | DSV rejected tree calls | {dsv_rejected_tree_calls} |\n\
         | DSV accepted tree calls | {dsv_accepted_tree_calls} |\n\
         | DSV avg tree nodes | {dsv_avg_tree_nodes:.1} |\n\
         | DSV max tree nodes | {dsv_max_tree_nodes} |\n\
         | emitted tokens mean | {emitted_mean} |\n\
         | verify steps mean | {steps_mean} |\n\
         | fallback steps mean | {fallback_mean} |\n\
         | fallback total | {sum_fallbacks} |\n\
         | fallback rate | {fallback_rate:.3} |\n\
         | AAL mean | {aal_mean:.2} |\n\
         | low-AAL outliers | {low_aal_outliers} |\n\
         | low-AAL outlier rate | {low_aal_outlier_rate:.3} |\n\
         | worst AAL | {worst_aal:.2} |\n\
         | worst AAL page | {worst_aal_page} |\n\
         | SR_e2e outliers | {sr_e2e_outliers} |\n\
         | SR_e2e outlier rate | {sr_e2e_outlier_rate:.3} |\n\
         | worst SR_e2e | {worst_sr_e2e:.2}x |\n\
         | worst SR_e2e page | {worst_sr_e2e_page} |\n\
         | SR_decode mean | {sr_decode_mean:.2}x |\n\
         | SR_e2e mean | {sr_e2e_mean:.2}x |\n\
         | SR_decode total | {sr_decode_total:.2}x |\n\
         | SR_e2e total time | {sr_e2e_total:.2}x |\n",
        backend = args.backend.as_str(),
        mode = args.mode,
        task = args.task,
        device = args.device,
        drafter_device = drafter_device(&args),
        draft_source = args.draft_source,
        page_dual_stage = args.page_dual_stage,
        hsd_entries = hsd_entries,
        region_prompt_kind = match args.backend {
            Backend::HunyuanOcr => "hunyuanocr_region",
            Backend::PaddleOcrVl => "paddleocr_vl_region_task",
            Backend::MinerU => "mineru_region_text_recognition",
            Backend::GlmOcr => "glmocr_region_text_recognition",
        },
        tau = args.tau,
        max_tokens = args.max_tokens,
        resize_max = args.resize_max,
        start_idx = args.start_idx,
        max_pages = args.max_pages,
        subset_filter = args.subset.as_deref().unwrap_or(""),
        language_filter = args.language.as_deref().unwrap_or(""),
        outlier_aal_threshold = args.outlier_aal_threshold,
        outlier_sr_e2e_threshold = args.outlier_sr_e2e_threshold,
        csv_path = csv_path.display(),
        baseline_mean = sum_baseline_ms / n,
        hsd_mean = sum_hsd_ms / n,
        drafter_mean = sum_drafter_ms / n,
        decode_mean = sum_decode_ms / n,
        prefill_mean = sum_prefill_ms / n,
        dsv_candidate_mean = sum_dsv.candidate_build.as_secs_f64() * 1000.0 / n,
        dsv_verify_mean = sum_dsv.verify_tree.as_secs_f64() * 1000.0 / n,
        dsv_traverse_mean = sum_dsv.traverse.as_secs_f64() * 1000.0 / n,
        dsv_commit_mean = sum_dsv.commit.as_secs_f64() * 1000.0 / n,
        dsv_step_one_mean = sum_dsv.step_one.as_secs_f64() * 1000.0 / n,
        dsv_verify_calls_mean = sum_dsv.verify_tree_calls as f64 / n,
        dsv_step_one_calls_mean = sum_dsv.step_one_calls as f64 / n,
        dsv_avg_candidates = sum_dsv.avg_candidates(),
        dsv_max_candidates = sum_dsv.candidates_max,
        dsv_empty_tree_calls = sum_dsv.empty_tree_calls,
        dsv_rejected_tree_calls = sum_dsv.rejected_tree_calls,
        dsv_accepted_tree_calls = sum_dsv.accepted_tree_calls,
        dsv_avg_tree_nodes = sum_dsv.avg_tree_nodes(),
        dsv_max_tree_nodes = sum_dsv.tree_nodes_max,
        emitted_mean = sum_emitted / counted as u64,
        steps_mean = sum_steps / counted as u64,
        fallback_mean = sum_fallbacks / counted as u64,
        aal_mean = sum_aal / n,
        low_aal_outliers = low_aal_outliers,
        low_aal_outlier_rate = low_aal_outliers as f64 / n,
        worst_aal = worst_aal,
        worst_aal_page = worst_aal_page,
        sr_e2e_outliers = sr_e2e_outliers,
        sr_e2e_outlier_rate = sr_e2e_outliers as f64 / n,
        worst_sr_e2e = worst_sr_e2e,
        worst_sr_e2e_page = worst_sr_e2e_page,
        sr_decode_mean = sum_sr_decode / n,
        sr_e2e_mean = sum_sr_e2e / n,
    );

    std::fs::write(&csv_path, csv)?;
    std::fs::write(&summary_path, summary)?;
    println!("\nPer-page CSV → {}", csv_path.display());
    println!("Markdown summary → {}", summary_path.display());
    Ok(())
}

fn format_duration_ms(duration: std::time::Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_aligned_iou_self_is_one() {
        let b = [0.0, 0.0, 10.0, 10.0];
        assert!((axis_aligned_iou(&b, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn axis_aligned_iou_disjoint_is_zero() {
        let a = [0.0, 0.0, 5.0, 5.0];
        let b = [10.0, 10.0, 20.0, 20.0];
        assert_eq!(axis_aligned_iou(&a, &b), 0.0);
    }

    #[test]
    fn axis_aligned_iou_half_overlap() {
        // a = 10x10 at origin; b = 10x10 shifted right by 5 → 5x10 overlap.
        // intersection = 50, union = 100 + 100 - 50 = 150, IoU = 1/3.
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 0.0, 15.0, 10.0];
        let iou = axis_aligned_iou(&a, &b);
        assert!((iou - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn axis_aligned_iou_zero_area_returns_zero() {
        let degenerate = [5.0, 5.0, 5.0, 10.0]; // zero width
        let b = [0.0, 0.0, 10.0, 10.0];
        assert_eq!(axis_aligned_iou(&degenerate, &b), 0.0);
        assert_eq!(axis_aligned_iou(&b, &degenerate), 0.0);
    }

    fn region(bbox: [f32; 4], raw: &str) -> CrossVlmRegion {
        CrossVlmRegion {
            bbox,
            raw_text: raw.to_string(),
        }
    }

    #[test]
    fn match_cross_vlm_picks_best_iou_above_threshold() {
        let elem = [0.0, 0.0, 10.0, 10.0];
        let regions = vec![
            region([100.0, 100.0, 200.0, 200.0], "far"),
            region([0.0, 0.0, 9.0, 10.0], "best"), // IoU = 0.9
            region([2.0, 2.0, 12.0, 12.0], "ok"),  // IoU ≈ 0.471
        ];
        let m = match_cross_vlm_region(&elem, &regions, 0.5).expect("match");
        assert_eq!(m.raw_text, "best");
    }

    #[test]
    fn match_cross_vlm_returns_none_below_threshold() {
        let elem = [0.0, 0.0, 10.0, 10.0];
        let regions = vec![region([100.0, 100.0, 200.0, 200.0], "far")];
        assert!(match_cross_vlm_region(&elem, &regions, 0.5).is_none());
    }

    #[test]
    fn cross_vlm_draft_file_parses_minimal_json() {
        let json = r#"{
            "source_backend": "paddleocr_vl",
            "pages": {
                "page-001.png": [
                    {"bbox": [10.0, 20.0, 200.0, 50.0], "raw_text": "$$x = 1$$"}
                ]
            }
        }"#;
        let parsed: CrossVlmDraftFile = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.source_backend.as_deref(), Some("paddleocr_vl"));
        let regions = parsed.lookup_page("page-001.png").expect("page");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].raw_text, "$$x = 1$$");
        assert_eq!(regions[0].bbox, [10.0, 20.0, 200.0, 50.0]);
    }

    #[test]
    fn cross_vlm_draft_file_falls_back_to_basename() {
        let json = r#"{"pages": {"page-001.png": [{"bbox": [0,0,1,1], "raw_text": "x"}]}}"#;
        let parsed: CrossVlmDraftFile = serde_json::from_str(json).expect("parse");
        // Caller may pass a nested path; lookup should fall back to basename.
        let regions = parsed
            .lookup_page("images/subset/page-001.png")
            .expect("page by basename");
        assert_eq!(regions.len(), 1);
    }
}

} // mod imp
