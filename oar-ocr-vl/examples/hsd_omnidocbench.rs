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
//! cargo run -p oar-ocr-vl --release --features cuda,download-binaries \
//!     --example hsd_omnidocbench -- \
//!         --bench-dir data/omnidocbench_v1.5 \
//!         --model-dir models/HunyuanOCR \
//!         --max-pages 20
//!
//! cargo run -p oar-ocr-vl --release --features cuda,download-binaries \
//!     --example hsd_omnidocbench -- \
//!         --backend paddleocr_vl --task spotting \
//!         --bench-dir data/omnidocbench_v1.5 \
//!         --model-dir models/PaddleOCR-VL-1.5 \
//!         --max-pages 20
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use image::imageops::FilterType;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Instant;

use oar_ocr::prelude::{OARStructure, OARStructureBuilder};
use oar_ocr_core::core::config::OrtSessionConfig;
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType, StructureResult};
use oar_ocr_core::predictors::TextRecognitionPredictor;
use oar_ocr_core::processors::{BoundingBox, Point};
use oar_ocr_core::utils::{BBoxCrop, load_image};
use oar_ocr_vl::hsd::types::{Draft, DsvConfig, HsdConfig, SpecDecodeStats};
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{GlmOcr, HunyuanOcr, MinerU, PaddleOcrVl, PaddleOcrVlTask};

use utils::structure_match::{MatchThresholds, match_region};

const HUNYUAN_CHINESE_PARSING_PROMPT: &str = "提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。";

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Backend {
    Hunyuan,
    #[value(name = "paddleocr_vl", alias = "paddleocr-vl")]
    PaddleOcrVl,
    #[value(name = "mineru", alias = "mineru2_5", alias = "mineru2.5")]
    MinerU,
    #[value(name = "glmocr", alias = "glm_ocr", alias = "glm-ocr")]
    GlmOcr,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hunyuan => "hunyuan",
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
    Hunyuan(HunyuanOcr),
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
    #[arg(long, value_enum, default_value_t = Backend::Hunyuan)]
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
    /// Instruction prompt. Defaults to HunyuanOCR's official "Parsing" task
    /// prompt, which elicits markdown output matching the OmniDocBench GT
    /// format. (Source: HunyuanOCR README under Quick Start → Tasks.)
    #[arg(
        long,
        default_value = "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."
    )]
    instruction: String,
    #[arg(long, default_value_t = 0.75)]
    tau: f32,
    /// Print the first N chars of baseline + draft for each page (debug).
    #[arg(long, default_value_t = 0)]
    preview: usize,
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
    /// PP-OCR recognition model for `--draft-source ppocr-rec`.
    #[arg(long, default_value = ".oar/pp-ocrv5_mobile_rec.onnx")]
    ppocr_rec_model: PathBuf,
    /// PP-OCR recognition dictionary for `--draft-source ppocr-rec`.
    #[arg(long, default_value = ".oar/ppocrv5_dict.txt")]
    ppocr_dict_path: PathBuf,
    /// PP-OCR recognition score threshold.
    #[arg(long, default_value_t = 0.0)]
    ppocr_score_thresh: f32,
    /// PP-OCR recognition max text length.
    #[arg(long, default_value_t = 200)]
    ppocr_max_text_length: usize,
    /// Layout model for `--draft-source structure`.
    #[arg(long, default_value = ".oar/pp-doclayout_plus-l.onnx")]
    structure_layout_model: PathBuf,
    /// Layout model preset for `--draft-source structure`.
    #[arg(long, default_value = "pp-doclayout_plus-l")]
    structure_layout_model_name: String,
    /// Optional PP-DocBlockLayout model for structure reading order.
    #[arg(long)]
    structure_region_model: Option<PathBuf>,
    /// Optional PP-OCR detection model for structure OCR.
    #[arg(long, default_value = ".oar/pp-ocrv5_mobile_det.onnx")]
    structure_ocr_det_model: PathBuf,
    /// Optional PP-OCR recognition model for structure OCR.
    #[arg(long, default_value = ".oar/pp-ocrv5_mobile_rec.onnx")]
    structure_ocr_rec_model: PathBuf,
    /// Optional PP-OCR dictionary for structure OCR.
    #[arg(long, default_value = ".oar/ppocrv5_dict.txt")]
    structure_ocr_dict_path: PathBuf,
    /// Optional table classifier for structure table routing.
    #[arg(long)]
    structure_table_cls_model: Option<PathBuf>,
    /// Optional wired table structure model for structure table HTML.
    #[arg(long)]
    structure_wired_table_model: Option<PathBuf>,
    /// Optional wireless table structure model for structure table HTML.
    #[arg(long)]
    structure_wireless_table_model: Option<PathBuf>,
    /// Optional table structure dictionary for structure table HTML.
    #[arg(long)]
    structure_table_dict_path: Option<PathBuf>,
    /// Optional wired table cell detection model.
    #[arg(long)]
    structure_wired_cell_model: Option<PathBuf>,
    /// Optional wireless table cell detection model.
    #[arg(long)]
    structure_wireless_cell_model: Option<PathBuf>,
    /// Optional formula model for structure LaTeX drafts.
    #[arg(long)]
    structure_formula_model: Option<PathBuf>,
    /// Optional formula tokenizer for structure LaTeX drafts.
    #[arg(long)]
    structure_formula_tokenizer: Option<PathBuf>,
    /// Formula model type for structure LaTeX drafts.
    #[arg(long, default_value = "pp_formulanet")]
    structure_formula_type: String,
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

/// Heuristic markdown-style serialisation for one page's draft.
fn build_draft(entry: &OmniEntry) -> String {
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
        let formatted = match d.category_type.as_str() {
            "title" | "header" => format!("# {text}"),
            "equation_isolated" | "equation_semantic" => {
                if text.starts_with("$$") || text.starts_with("\\[") {
                    text.to_string()
                } else {
                    format!("$$\n{text}\n$$")
                }
            }
            _ => text.to_string(),
        };
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&formatted);
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
) -> String {
    match backend {
        Backend::Hunyuan => build_draft(entry),
        Backend::MinerU | Backend::GlmOcr => build_plain_draft(entry),
        Backend::PaddleOcrVl => match task {
            Task::Spotting => build_spotting_draft(entry, image_size.0, image_size.1),
            Task::Ocr | Task::Seal => build_plain_draft(entry),
            Task::Table | Task::Chart | Task::Formula => build_draft(entry),
        },
    }
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
        (HUNYUAN_CHINESE_PARSING_PROMPT, "hunyuan_parsing_zh")
    } else {
        (args.instruction.as_str(), "hunyuan_parsing_en")
    }
}

fn instruction_prompt<'a>(entry: &OmniEntry, args: &'a Args) -> (&'a str, &'static str) {
    if args.auto_prompt_lang && page_attr(entry, "language").eq_ignore_ascii_case("chinese") {
        (HUNYUAN_CHINESE_PARSING_PROMPT, "instruction_zh")
    } else {
        (args.instruction.as_str(), "instruction")
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
                    normalize_paddle_region_draft(text)
                } else {
                    text.to_string()
                });
            }
            elem.order_index = d.order.map(|x| x as u32);
            elem
        })
        .collect()
}

fn normalize_paddle_region_draft(input: &str) -> String {
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

fn paddle_task_for_layout_type(t: LayoutElementType) -> Option<PaddleOcrVlTask> {
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

fn run_paddle_region_baseline(
    model: &PaddleOcrVl,
    image: &image::RgbImage,
    elements: &[LayoutElement],
    max_tokens: usize,
) -> Result<RegionDrafts, Box<dyn std::error::Error>> {
    let mut outputs = Vec::new();
    let mut per_element = vec![None; elements.len()];
    let mut per_element_tokens = vec![None; elements.len()];
    for (idx, elem) in elements.iter().enumerate() {
        let Some(task) = paddle_task_for_layout_type(elem.element_type) else {
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
    for ((idx, text), score) in indices
        .into_iter()
        .zip(output.texts.into_iter())
        .zip(output.scores.into_iter())
    {
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
    if let Some(path) = &args.structure_region_model {
        builder = builder
            .with_region_detection(path)
            .region_model_name("pp-docblocklayout");
    }
    if let Some(path) = &args.structure_table_cls_model {
        builder = builder.with_table_classification(path);
    }
    if let Some(path) = &args.structure_wired_table_model {
        let Some(dict) = &args.structure_table_dict_path else {
            return Err(
                "--structure-wired-table-model requires --structure-table-dict-path".into(),
            );
        };
        builder = builder
            .with_wired_table_structure(path)
            .wired_table_structure_model_name("slanext_wired")
            .table_structure_dict_path(dict);
    }
    if let Some(path) = &args.structure_wireless_table_model {
        let Some(dict) = &args.structure_table_dict_path else {
            return Err(
                "--structure-wireless-table-model requires --structure-table-dict-path".into(),
            );
        };
        builder = builder
            .with_wireless_table_structure(path)
            .wireless_table_structure_model_name("slanet_plus")
            .table_structure_dict_path(dict);
    }
    if let Some(path) = &args.structure_wired_cell_model {
        builder = builder
            .with_wired_table_cell_detection(path)
            .wired_table_cell_model_name("rtdetr-l_wired_table_cell_det");
    }
    if let Some(path) = &args.structure_wireless_cell_model {
        builder = builder
            .with_wireless_table_cell_detection(path)
            .wireless_table_cell_model_name("rtdetr-l_wireless_table_cell_det");
    }
    if let Some(path) = &args.structure_formula_model {
        let Some(tokenizer) = &args.structure_formula_tokenizer else {
            return Err("--structure-formula-model requires --structure-formula-tokenizer".into());
        };
        builder = builder.with_formula_recognition(path, tokenizer, &args.structure_formula_type);
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
) -> Result<String, Box<dyn std::error::Error>> {
    let result = structure.predict_image(image.clone())?;
    Ok(result.to_markdown())
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

fn drafter_device(args: &Args) -> &str {
    args.drafter_device
        .as_deref()
        .unwrap_or(args.device.as_str())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        Backend::Hunyuan => BackendModel::Hunyuan(HunyuanOcr::from_dir(&args.model_dir, device)?),
        Backend::PaddleOcrVl => {
            BackendModel::PaddleOcrVl(PaddleOcrVl::from_dir(&args.model_dir, device)?)
        }
        Backend::MinerU => BackendModel::MinerU(MinerU::from_dir(&args.model_dir, device)?),
        Backend::GlmOcr => BackendModel::GlmOcr(GlmOcr::from_dir(&args.model_dir, device)?),
    };
    let paddle_task = args.task.to_native();
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
            && !matches!(args.backend, Backend::Hunyuan | Backend::PaddleOcrVl)
        {
            return Err(
                "--draft-source structure requires --backend hunyuan or paddleocr_vl with --mode region".into(),
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
        "page_idx,image,subset,language,backend,mode,task,device,drafter_device,draft_source,regions,draft_regions,draft_coverage,tau,max_tokens,resize_max,start_idx,prompt_kind,prompt,baseline_ms,hsd_ms,decode_ms,prefill_ms,dsv_candidate_ms,dsv_verify_ms,dsv_traverse_ms,dsv_commit_ms,dsv_step_one_ms,dsv_fallback_argmax_ms,dsv_verify_calls,dsv_step_one_calls,dsv_fallback_argmax_calls,dsv_avg_tree_nodes,dsv_max_tree_nodes,emitted_tokens,verify_steps,fallback_steps,aal,sr_decode,sr_e2e\n",
    );

    let cfg = HsdConfig {
        dsv: DsvConfig {
            tau: args.tau,
            window_len: 3,
            max_candidates_per_step: 32,
            max_suffix_len: 64,
        },
        enable_stage1: matches!(args.mode, Mode::Region),
        enable_stage2: true,
        max_page_tokens: args.max_tokens,
        max_region_tokens: args.max_tokens,
    };
    if matches!(args.mode, Mode::Region)
        && !matches!(args.backend, Backend::Hunyuan | Backend::PaddleOcrVl)
    {
        return Err("--mode region currently supports --backend hunyuan or paddleocr_vl".into());
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
    let mut counted = 0u32;
    let mut skipped = 0u32;

    println!(
        "\n{:>4} {:>9} {:>9} {:>5} {:>5} {:>5} | {:<60}",
        "idx", "base_ms", "hsd_ms", "AAL", "ndec", "fb", "page"
    );
    println!("{}", "-".repeat(110));

    for (i, entry) in candidates.iter().take(n_pages).enumerate() {
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
        let elements = if matches!(args.mode, Mode::Region) {
            let normalize_text = args.normalize_draft && args.draft_source == "gt";
            let require_text = matches!(args.draft_source.as_str(), "gt" | "ppocr-rec");
            build_layout_elements(entry, x_scale, y_scale, normalize_text, require_text)
        } else {
            Vec::new()
        };
        let draft = match args.mode {
            Mode::Page => build_gt_draft(entry, args.backend, args.task, gt_image_size),
            Mode::Region => elements
                .iter()
                .filter_map(|e| e.text.as_ref())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n"),
        };
        if draft.trim().is_empty() {
            let draft_filled_later = matches!(args.draft_source.as_str(), "baseline" | "structure")
                && (matches!(args.mode, Mode::Page) || matches!(args.mode, Mode::Region));
            if !draft_filled_later {
                // No usable draft text — skip rather than running with empty draft
                // (which would just be a baseline run with extra overhead).
                skipped += 1;
                continue;
            }
        }
        let (hunyuan_prompt, hunyuan_prompt_kind) = prompt_for_entry(entry, &args);
        let (generic_prompt, generic_prompt_kind) = instruction_prompt(entry, &args);
        let (prompt_text, prompt_kind) = match args.backend {
            Backend::Hunyuan => (hunyuan_prompt, hunyuan_prompt_kind),
            Backend::PaddleOcrVl => (paddle_task.prompt(), "paddleocr_vl_task"),
            Backend::MinerU => (generic_prompt, generic_prompt_kind),
            Backend::GlmOcr => (generic_prompt, generic_prompt_kind),
        };

        // Baseline.
        let t0 = Instant::now();
        let mut region_baseline_drafts: Option<Vec<Option<String>>> = None;
        let mut region_baseline_token_drafts: Option<Vec<Option<Vec<u32>>>> = None;
        let mut baseline_tokens: Option<Vec<u32>> = None;
        let baseline_result: Result<String, Box<dyn std::error::Error>> = match (&model, args.mode)
        {
            (BackendModel::Hunyuan(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[hunyuan_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (BackendModel::PaddleOcrVl(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[paddle_task], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model
                    .decode_tokens(&toks, paddle_task)
                    .map(|(_, processed)| processed)
                    .map_err(|e| e.into())
            }
            (BackendModel::MinerU(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[generic_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (BackendModel::GlmOcr(model), Mode::Page) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[generic_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (BackendModel::PaddleOcrVl(model), Mode::Region) => {
                let drafts = run_paddle_region_baseline(model, &image, &elements, args.max_tokens)?;
                region_baseline_drafts = Some(drafts.per_element);
                region_baseline_token_drafts = Some(drafts.per_element_tokens);
                Ok(drafts.joined)
            }
            (BackendModel::Hunyuan(model), Mode::Region) => {
                let toks_result = model
                    .generate_tokens(&[image.clone()], &[hunyuan_prompt], args.max_tokens)
                    .into_iter()
                    .next()
                    .ok_or("baseline returned no results")?;
                let toks = toks_result?;
                baseline_tokens = Some(toks.clone());
                model.decode_tokens(&toks).map_err(|e| e.into())
            }
            (_, Mode::Region) => {
                Err("--mode region currently supports --backend hunyuan or paddleocr_vl".into())
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
        let actual_draft = match args.draft_source.as_str() {
            "gt" => draft.clone(),
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
                let ppocr_drafts = run_ppocr_rec_drafter(predictor, &image, &elements)?;
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
                run_structure_page_drafter(structure, &image)?
            }
            "baseline" => baseline_text.clone(),
            other => return Err(format!("unknown --draft-source: {other}").into()),
        };
        if actual_draft.trim().is_empty() {
            skipped += 1;
            continue;
        }
        let draft_region_count = if matches!(args.mode, Mode::Page) {
            usize::from(!actual_draft.trim().is_empty())
        } else {
            hsd_elements
                .as_ref()
                .unwrap_or(&elements)
                .iter()
                .filter(|e| e.text.as_deref().is_some_and(|s| !s.trim().is_empty()))
                .count()
        };
        let draft_coverage = if elements.is_empty() {
            if matches!(args.mode, Mode::Page) && !actual_draft.trim().is_empty() {
                1.0
            } else {
                0.0
            }
        } else {
            draft_region_count as f64 / elements.len() as f64
        };

        // HSD with the draft.
        let t1 = Instant::now();
        let oracle_draft = (args.draft_source == "baseline")
            .then(|| {
                baseline_tokens
                    .as_ref()
                    .map(|t| vec![Draft::new(t.clone())])
            })
            .flatten();
        let hsd = match (&model, args.mode, oracle_draft.as_deref()) {
            (BackendModel::Hunyuan(model), Mode::Page, Some(token_drafts)) => {
                model.generate_hsd_with_token_drafts(&image, hunyuan_prompt, token_drafts, &cfg)
            }
            (BackendModel::Hunyuan(model), Mode::Page, None) => model.generate_hsd(
                &image,
                hunyuan_prompt,
                std::slice::from_ref(&actual_draft),
                &cfg,
            ),
            (BackendModel::PaddleOcrVl(model), Mode::Page, Some(token_drafts)) => {
                model.generate_hsd_with_token_drafts(&image, paddle_task, token_drafts, &cfg)
            }
            (BackendModel::PaddleOcrVl(model), Mode::Page, None) => model.generate_hsd(
                &image,
                paddle_task,
                std::slice::from_ref(&actual_draft),
                &cfg,
            ),
            (BackendModel::MinerU(model), Mode::Page, Some(token_drafts)) => {
                model.generate_hsd_with_token_drafts(&image, generic_prompt, token_drafts, &cfg)
            }
            (BackendModel::MinerU(model), Mode::Page, None) => model.generate_hsd(
                &image,
                generic_prompt,
                std::slice::from_ref(&actual_draft),
                &cfg,
            ),
            (BackendModel::GlmOcr(model), Mode::Page, Some(token_drafts)) => {
                model.generate_hsd_with_token_drafts(&image, generic_prompt, token_drafts, &cfg)
            }
            (BackendModel::GlmOcr(model), Mode::Page, None) => model.generate_hsd(
                &image,
                generic_prompt,
                std::slice::from_ref(&actual_draft),
                &cfg,
            ),
            (BackendModel::PaddleOcrVl(model), Mode::Region, _) => {
                let elems = hsd_elements
                    .as_ref()
                    .map(|v| v.as_slice())
                    .unwrap_or_else(|| elements.as_slice());
                if args.draft_source == "baseline" {
                    let token_drafts = region_baseline_token_drafts.as_ref().ok_or_else(|| {
                        oar_ocr_core::core::OCRError::InvalidInput {
                            message: "missing region baseline token drafts".to_string(),
                        }
                    })?;
                    model.generate_hsd_full_with_token_drafts(
                        &image,
                        elems,
                        &[],
                        token_drafts,
                        &cfg,
                    )
                } else {
                    model.generate_hsd_full(&image, elems, &[], &cfg)
                }
            }
            (BackendModel::Hunyuan(model), Mode::Region, _) => {
                let elems = hsd_elements
                    .as_ref()
                    .map(|v| v.as_slice())
                    .unwrap_or_else(|| elements.as_slice());
                model.generate_hsd_full(&image, hunyuan_prompt, elems, &[], &cfg)
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
        let hsd_dur = t1.elapsed();
        let (_text, stats) = hsd;

        let baseline_ms = baseline_dur.as_secs_f64() * 1000.0;
        let hsd_ms = hsd_dur.as_secs_f64() * 1000.0;
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
            elements.len().to_string(),
            draft_region_count.to_string(),
            format!("{draft_coverage:.3}"),
            format!("{:.3}", args.tau),
            args.max_tokens.to_string(),
            args.resize_max.to_string(),
            args.start_idx.to_string(),
            prompt_kind.to_string(),
            prompt_text.to_string(),
            format!("{baseline_ms:.1}"),
            format!("{hsd_ms:.1}"),
            format!("{decode_ms:.1}"),
            format!("{prefill_ms:.1}"),
            format_duration_ms(stage.dsv.candidate_build),
            format_duration_ms(stage.dsv.verify_tree),
            format_duration_ms(stage.dsv.traverse),
            format_duration_ms(stage.dsv.commit),
            format_duration_ms(stage.dsv.step_one),
            format_duration_ms(stage.dsv.fallback_argmax),
            stage.dsv.verify_tree_calls.to_string(),
            stage.dsv.step_one_calls.to_string(),
            stage.dsv.fallback_argmax_calls.to_string(),
            format!("{:.1}", stage.dsv.avg_tree_nodes()),
            stage.dsv.tree_nodes_max.to_string(),
            stage.emitted_tokens.to_string(),
            stage.accept.num_steps.to_string(),
            stage.accept.num_fallbacks.to_string(),
            format!("{aal:.2}"),
            format!("{sr_decode:.3}"),
            format!("{sr_e2e:.3}"),
        ]));
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
         | HSD decode mean ms | {decode_mean:.1} |\n\
         | HSD prefill mean ms | {prefill_mean:.1} |\n\
         | DSV candidate mean ms | {dsv_candidate_mean:.1} |\n\
         | DSV verify_tree mean ms | {dsv_verify_mean:.1} |\n\
         | DSV traverse mean ms | {dsv_traverse_mean:.1} |\n\
         | DSV commit mean ms | {dsv_commit_mean:.1} |\n\
         | DSV step_one mean ms | {dsv_step_one_mean:.1} |\n\
         | DSV verify calls/page | {dsv_verify_calls_mean:.1} |\n\
         | DSV step_one calls/page | {dsv_step_one_calls_mean:.1} |\n\
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
        decode_mean = sum_decode_ms / n,
        prefill_mean = sum_prefill_ms / n,
        dsv_candidate_mean = sum_dsv.candidate_build.as_secs_f64() * 1000.0 / n,
        dsv_verify_mean = sum_dsv.verify_tree.as_secs_f64() * 1000.0 / n,
        dsv_traverse_mean = sum_dsv.traverse.as_secs_f64() * 1000.0 / n,
        dsv_commit_mean = sum_dsv.commit.as_secs_f64() * 1000.0 / n,
        dsv_step_one_mean = sum_dsv.step_one.as_secs_f64() * 1000.0 / n,
        dsv_verify_calls_mean = sum_dsv.verify_tree_calls as f64 / n,
        dsv_step_one_calls_mean = sum_dsv.step_one_calls as f64 / n,
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
