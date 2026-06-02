//! Speed + accuracy benchmark on OmniDocBench v1.5 for the MinerU family.
//!
//! Runs a full-page `Text Recognition` pass per page and reports:
//!   * **Speed** — mean seconds/page and output characters/second.
//!   * **Accuracy** — mean normalized edit distance (NED) against the
//!     reading-order ground-truth text, and the derived score `1 - NED`.
//!
//! Both backends are driven with the same task so the numbers are comparable:
//!   * `--backend mineru`    — MinerU2.5 / MinerU2.5-Pro (autoregressive).
//!   * `--backend diffusion` — MinerU-Diffusion-V1 (block-diffusion decode).
//!
//! ```bash
//! cargo run -p oar-ocr-vl --release --features cuda,download-binaries \
//!     --example bench_omnidocbench -- \
//!         --backend diffusion \
//!         --model-dir /path/to/MinerU-Diffusion-V1-0320-2.5B \
//!         --bench-dir /home/greatx/data/ocr_bench/omnidocbench_v1.5 \
//!         --device cuda:0 --max-pages 30
//! ```

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use image::{RgbImage, imageops};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use tracing::{error, info, warn};

use oar_ocr_core::core::OCRError;
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::mineru_diffusion::DEFAULT_PROMPT;
use oar_ocr_vl::utils::image::resize_for_mineru;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::utils::{convert_otsl_to_html, truncate_repetitive_content};
use oar_ocr_vl::{DiffusionGenerationConfig, MinerU, MinerUDiffusion};

// --- MinerU2.5 two-step extraction prompts (mirrors the `mineru` example) ---
const LAYOUT_PROMPT: &str = "\nLayout Detection:";
const TABLE_PROMPT: &str = "\nTable Recognition:";
const EQUATION_PROMPT: &str = "\nFormula Recognition:";
const MINERU_TEXT_PROMPT: &str = "\nText Recognition:";
const LAYOUT_IMAGE_SIZE: u32 = 1036;
const LAYOUT_RE: &str = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$";
static LAYOUT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(LAYOUT_RE).expect("layout regex"));

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Backend {
    /// MinerU2.5 / MinerU2.5-Pro (autoregressive Qwen2-VL).
    Mineru,
    /// MinerU-Diffusion-V1 (block-diffusion decode).
    Diffusion,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum MineruMode {
    /// Native two-step extraction: layout detection -> per-region recognition
    /// -> reading-order assembly. This is MinerU2.5's intended pipeline.
    TwoStep,
    /// Single full-page `Text Recognition` prompt (simpler, weaker for MinerU2.5).
    Page,
}

#[derive(Parser)]
#[command(name = "bench_omnidocbench")]
#[command(about = "Speed + accuracy benchmark on OmniDocBench v1.5")]
struct Args {
    #[arg(long, value_enum)]
    backend: Backend,

    #[arg(short, long)]
    model_dir: PathBuf,

    /// OmniDocBench v1.5 root (contains OmniDocBench.json and images/).
    #[arg(long)]
    bench_dir: PathBuf,

    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Number of pages to evaluate (after stride reordering).
    #[arg(long, default_value_t = 30)]
    max_pages: usize,

    /// Visit pages spread across the dataset in `stride` buckets (1 = in-order).
    /// With stride>1 the first pages processed are evenly spaced over the whole
    /// dataset, so a partial run is still a representative sample.
    #[arg(long, default_value_t = 1)]
    stride: usize,

    /// Instruction prompt (default: "\nText Recognition:").
    #[arg(long)]
    prompt: Option<String>,

    /// MinerU autoregressive token budget.
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// MinerU inference mode: two-step (native) or single full-page pass.
    #[arg(long, value_enum, default_value_t = MineruMode::TwoStep)]
    mineru_mode: MineruMode,

    /// Two-step: minimum edge length for cropped regions.
    #[arg(long, default_value_t = 28)]
    min_image_edge: u32,

    /// Two-step: max edge ratio before padding.
    #[arg(long, default_value_t = 50.0)]
    max_image_edge_ratio: f32,

    /// If set, write each page's markdown prediction to
    /// `<dump-dir>/<image-basename>.md` for the official OmniDocBench eval.
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// Diffusion total generation length (multiple of block-length).
    #[arg(long, default_value_t = 2048)]
    gen_length: usize,

    /// Diffusion block length.
    #[arg(long, default_value_t = 32)]
    block_length: usize,

    /// Diffusion denoising steps per block.
    #[arg(long, default_value_t = 32)]
    denoising_steps: usize,

    /// Diffusion confidence threshold.
    #[arg(long, default_value_t = 0.95)]
    dynamic_threshold: f32,
}

#[derive(Debug, Deserialize)]
struct Page {
    page_info: PageInfo,
    #[serde(default)]
    layout_dets: Vec<Det>,
}

#[derive(Debug, Deserialize)]
struct PageInfo {
    image_path: String,
}

#[derive(Debug, Deserialize)]
struct Det {
    category_type: String,
    #[serde(default)]
    order: Option<i64>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    ignore: Option<bool>,
}

/// Categories that contribute body text to the reading-order ground truth.
/// Headers, footers, page numbers and `abandon` regions are excluded, matching
/// the "ignore headers and footers" framing of the recognition task.
fn is_body_category(cat: &str) -> bool {
    matches!(
        cat,
        "text_block"
            | "title"
            | "text"
            | "equation_isolated"
            | "equation_semantic"
            | "equation"
            | "list"
            | "table"
            | "caption"
    )
}

fn build_gt_text(page: &Page) -> String {
    let mut dets: Vec<&Det> = page
        .layout_dets
        .iter()
        .filter(|d| !d.ignore.unwrap_or(false))
        .filter(|d| is_body_category(&d.category_type))
        .filter(|d| d.text.as_deref().is_some_and(|t| !t.trim().is_empty()))
        .collect();
    dets.sort_by_key(|d| d.order.unwrap_or(i64::MAX));
    dets.iter()
        .map(|d| d.text.as_deref().unwrap_or("").trim())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collapse runs of whitespace to a single space and trim — a light
/// normalization so spacing differences don't dominate the edit distance.
fn normalize(s: &str) -> Vec<char> {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out.chars().collect()
}

/// Levenshtein edit distance over character slices (two-row DP).
fn edit_distance(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Normalized edit distance in [0, 1]: `dist / max(len_pred, len_gt)`.
fn ned(pred: &[char], gt: &[char]) -> f64 {
    let denom = pred.len().max(gt.len());
    if denom == 0 {
        return 0.0;
    }
    edit_distance(pred, gt) as f64 / denom as f64
}

// ============================= MinerU2.5 two-step =============================

/// One detected layout region.
struct ContentBlock {
    block_type: String,
    bbox: [f32; 4],
    angle: Option<u16>,
    content: Option<String>,
}

/// Layout types excluded from the assembled body text, to parallel the GT
/// builder (which drops headers, footers, page numbers and abandoned regions).
fn is_excluded_block(t: &str) -> bool {
    matches!(
        t,
        "header" | "footer" | "page_number" | "page_footnote" | "aside_text" | "image"
    )
}

/// Run MinerU2.5's native two-step extraction and return reading-order body
/// text (tables as HTML, formulas as LaTeX), comparable to [`build_gt_text`].
fn two_step_text(
    model: &MinerU,
    image: &RgbImage,
    max_tokens: usize,
    min_image_edge: u32,
    max_image_edge_ratio: f32,
) -> Result<String, OCRError> {
    // Step 1 — layout detection on a square-resized page.
    let layout_image = imageops::resize(
        image,
        LAYOUT_IMAGE_SIZE,
        LAYOUT_IMAGE_SIZE,
        imageops::FilterType::CatmullRom,
    );
    let layout = model
        .generate(&[layout_image], &[LAYOUT_PROMPT], max_tokens)
        .into_iter()
        .next()
        .ok_or_else(|| OCRError::InvalidInput {
            message: "layout detection returned no result".to_string(),
        })??;
    let mut blocks = parse_layout_output(&layout);
    if blocks.is_empty() {
        return Ok(String::new());
    }

    // Step 2 — per-region recognition on cropped blocks (one at a time).
    let (block_images, prompts, indices) =
        prepare_for_extract(image, &blocks, min_image_edge, max_image_edge_ratio);
    for (i, (block_image, prompt)) in block_images.into_iter().zip(prompts.iter()).enumerate() {
        let idx = indices[i];
        if let Some(Ok(content)) = model
            .generate(&[block_image], &[prompt.as_str()], max_tokens)
            .into_iter()
            .next()
        {
            let cleaned = truncate_repetitive_content(&content, 10, 10, 10);
            let content = if blocks[idx].block_type == "table" {
                convert_otsl_to_html(&cleaned)
            } else {
                cleaned.trim().to_string()
            };
            blocks[idx].content = Some(content);
        }
    }

    // Reading-order assembly (layout output order == reading order).
    let text = blocks
        .iter()
        .filter(|b| !is_excluded_block(&b.block_type))
        .filter_map(|b| b.content.as_deref())
        .map(|c| c.trim())
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(text)
}

fn parse_layout_output(output: &str) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    for line in output.lines() {
        let Some(caps) = LAYOUT_REGEX.captures(line) else {
            continue;
        };
        let Some((x1, y1, x2, y2)) = (|| {
            Some((
                caps.get(1)?.as_str().parse().ok()?,
                caps.get(2)?.as_str().parse().ok()?,
                caps.get(3)?.as_str().parse().ok()?,
                caps.get(4)?.as_str().parse().ok()?,
            ))
        })() else {
            continue;
        };
        let ref_type = caps.get(5).map(|m| m.as_str().to_lowercase()).unwrap_or_default();
        let tail = caps.get(6).map(|m| m.as_str()).unwrap_or("");
        let Some((x1, y1, x2, y2)) = normalize_bbox(x1, y1, x2, y2) else {
            continue;
        };
        if !is_block_type(&ref_type) {
            continue;
        }
        blocks.push(ContentBlock {
            block_type: ref_type,
            bbox: [x1, y1, x2, y2],
            angle: parse_angle(tail),
            content: None,
        });
    }
    blocks
}

fn normalize_bbox(x1: i32, y1: i32, x2: i32, y2: i32) -> Option<(f32, f32, f32, f32)> {
    if [x1, y1, x2, y2].iter().any(|&v| !(0..=1000).contains(&v)) {
        return None;
    }
    let (x1, x2) = if x2 < x1 { (x2, x1) } else { (x1, x2) };
    let (y1, y2) = if y2 < y1 { (y2, y1) } else { (y1, y2) };
    if x1 == x2 || y1 == y2 {
        return None;
    }
    Some((
        x1 as f32 / 1000.0,
        y1 as f32 / 1000.0,
        x2 as f32 / 1000.0,
        y2 as f32 / 1000.0,
    ))
}

fn parse_angle(tail: &str) -> Option<u16> {
    if tail.contains("<|rotate_up|>") {
        Some(0)
    } else if tail.contains("<|rotate_right|>") {
        Some(90)
    } else if tail.contains("<|rotate_down|>") {
        Some(180)
    } else if tail.contains("<|rotate_left|>") {
        Some(270)
    } else {
        None
    }
}

fn is_block_type(value: &str) -> bool {
    matches!(
        value,
        "text" | "title" | "table" | "image" | "code" | "algorithm" | "header" | "footer"
            | "page_number" | "page_footnote" | "aside_text" | "equation" | "equation_block"
            | "ref_text" | "list" | "phonetic" | "table_caption" | "image_caption"
            | "code_caption" | "table_footnote" | "image_footnote" | "unknown"
    )
}

fn prepare_for_extract(
    image: &RgbImage,
    blocks: &[ContentBlock],
    min_image_edge: u32,
    max_image_edge_ratio: f32,
) -> (Vec<RgbImage>, Vec<String>, Vec<usize>) {
    let mut block_images = Vec::new();
    let mut prompts = Vec::new();
    let mut indices = Vec::new();
    let width = image.width() as f32;
    let height = image.height() as f32;

    for (idx, block) in blocks.iter().enumerate() {
        if matches!(block.block_type.as_str(), "image" | "list" | "equation_block") {
            continue;
        }
        let x1 = (block.bbox[0] * width).round().clamp(0.0, width - 1.0) as u32;
        let y1 = (block.bbox[1] * height).round().clamp(0.0, height - 1.0) as u32;
        let x2 = (block.bbox[2] * width).round().clamp(0.0, width) as u32;
        let y2 = (block.bbox[3] * height).round().clamp(0.0, height) as u32;
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        let mut crop = imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
        if let Some(angle) = block.angle {
            crop = match angle {
                90 => imageops::rotate90(&crop),
                180 => imageops::rotate180(&crop),
                270 => imageops::rotate270(&crop),
                _ => crop,
            };
        }
        crop = resize_for_mineru(&crop, min_image_edge, max_image_edge_ratio);
        block_images.push(crop);
        prompts.push(prompt_for_block(&block.block_type).to_string());
        indices.push(idx);
    }
    (block_images, prompts, indices)
}

fn prompt_for_block(block_type: &str) -> &'static str {
    match block_type {
        "table" => TABLE_PROMPT,
        "equation" => EQUATION_PROMPT,
        _ => MINERU_TEXT_PROMPT,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let device = parse_device(&args.device)?;
    info!("Device: {device:?}, backend: {:?}", args.backend);

    let json_path = args.bench_dir.join("OmniDocBench.json");
    let images_dir = args.bench_dir.join("images");
    let pages: Vec<Page> = serde_json::from_str(&std::fs::read_to_string(&json_path)?)?;
    info!(
        "Loaded {} pages from {}; evaluating up to {}",
        pages.len(),
        json_path.display(),
        args.max_pages
    );

    let prompt = args.prompt.clone().unwrap_or_else(|| DEFAULT_PROMPT.to_string());

    if let Some(dir) = &args.dump_dir {
        std::fs::create_dir_all(dir)?;
        info!("Writing markdown predictions to {}", dir.display());
    }

    // Load the requested backend.
    let load_start = Instant::now();
    let mineru = match args.backend {
        Backend::Mineru => Some(MinerU::from_dir(&args.model_dir, device.clone())?),
        Backend::Diffusion => None,
    };
    let diffusion = match args.backend {
        Backend::Diffusion => Some(MinerUDiffusion::from_dir(&args.model_dir, device.clone())?),
        Backend::Mineru => None,
    };
    let gen_cfg = DiffusionGenerationConfig {
        gen_length: args.gen_length,
        block_length: args.block_length,
        denoising_steps: args.denoising_steps,
        dynamic_threshold: args.dynamic_threshold,
        ..Default::default()
    };
    info!("Model loaded in {:.2?}", load_start.elapsed());

    let mut n = 0usize;
    let mut total_time = Duration::ZERO;
    let mut total_chars = 0usize;
    let mut sum_ned = 0.0f64;

    // Stride ordering: bucket 0 first (indices 0, stride, 2*stride, ...), then
    // bucket 1, etc. The leading prefix is spread across the whole dataset.
    let stride = args.stride.max(1);
    let mut order: Vec<usize> = Vec::with_capacity(pages.len());
    for start in 0..stride {
        let mut i = start;
        while i < pages.len() {
            order.push(i);
            i += stride;
        }
    }
    order.truncate(args.max_pages);

    for &pi in &order {
        let page = &pages[pi];
        let img_path = images_dir.join(&page.page_info.image_path);
        let image = match load_image(&img_path) {
            Ok(img) => img,
            Err(e) => {
                warn!("skip {}: {e}", img_path.display());
                continue;
            }
        };

        let start = Instant::now();
        let pred = match args.backend {
            Backend::Mineru => {
                let m = mineru.as_ref().unwrap();
                let res = match args.mineru_mode {
                    MineruMode::Page => m
                        .generate(std::slice::from_ref(&image), &[prompt.as_str()], args.max_tokens)
                        .into_iter()
                        .next()
                        .unwrap(),
                    MineruMode::TwoStep => two_step_text(
                        m,
                        &image,
                        args.max_tokens,
                        args.min_image_edge,
                        args.max_image_edge_ratio,
                    ),
                };
                match res {
                    Ok(t) => t,
                    Err(e) => {
                        error!("infer failed {}: {e}", img_path.display());
                        continue;
                    }
                }
            }
            Backend::Diffusion => {
                let d = diffusion.as_ref().unwrap();
                match d.generate(&image, &prompt, &gen_cfg) {
                    Ok(t) => t,
                    Err(e) => {
                        error!("infer failed {}: {e}", img_path.display());
                        continue;
                    }
                }
            }
        };
        let elapsed = start.elapsed();

        // Dump the raw markdown prediction for the official OmniDocBench eval.
        if let Some(dir) = &args.dump_dir {
            let stem = std::path::Path::new(&page.page_info.image_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&page.page_info.image_path);
            if let Err(e) = std::fs::write(dir.join(format!("{stem}.md")), &pred) {
                warn!("failed to write prediction for {stem}: {e}");
            }
        }

        let gt = build_gt_text(page);
        let pred_n = normalize(&pred);
        let gt_n = normalize(&gt);
        let page_ned = ned(&pred_n, &gt_n);

        n += 1;
        total_time += elapsed;
        total_chars += pred_n.len();
        sum_ned += page_ned;

        info!(
            "[{n}] {} | {:.2}s | pred {}c gt {}c | NED {:.3} (acc {:.3})",
            page.page_info.image_path,
            elapsed.as_secs_f64(),
            pred_n.len(),
            gt_n.len(),
            page_ned,
            1.0 - page_ned,
        );
    }

    if n == 0 {
        error!("No pages evaluated.");
        return Ok(());
    }

    let mean_s = total_time.as_secs_f64() / n as f64;
    let chars_per_s = total_chars as f64 / total_time.as_secs_f64();
    let mean_ned = sum_ned / n as f64;
    println!("\n================ OmniDocBench v1.5 — {:?} ================", args.backend);
    println!("model_dir       : {}", args.model_dir.display());
    println!("pages evaluated : {n}");
    println!("speed           : {mean_s:.3} s/page, {chars_per_s:.1} output chars/s");
    println!("mean NED        : {mean_ned:.4}");
    println!("accuracy (1-NED): {:.4}", 1.0 - mean_ned);
    println!("=========================================================");

    Ok(())
}
