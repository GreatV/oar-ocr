//! Indra OCR: Datalab's CPU-first page reader on ONNX Runtime.
//!
//! [Indra OCR](https://huggingface.co/datalab-to/indra-ocr) is a set of small
//! ONNX models that turn a page image into lines with boxes, layout blocks
//! in reading order, and Markdown: a page orientation classifier, a line
//! detector, a script classifier that routes each line to the recognizer
//! of its script group (Latin, Cyrillic and Greek; Chinese, Japanese and
//! Korean; Arabic and Hebrew; Indic, Southeast Asian and other scripts),
//! n-gram language models for lines a recognizer is unsure of, a
//! reading-order model with layout heads, and an optional math reader that
//! writes lines with math as inline LaTeX.
//!
//! [`IndraOcr`] loads the models from a local copy of the Hugging Face
//! repository: `det.onnx`, `rec.onnx`, `order.onnx` and `meta.json`, with
//! `orient.onnx`, `script.onnx`, the group recognizers `rec_<group>.onnx`
//! and the language models `lm.npz` and `lm_<group>.npz` used when present.
//! It reproduces the reference pipeline's OCR mode (`indra-ocr` 1.1.2;
//! the earlier v3.1 model set loads too) with two fixes: token positions
//! of a tall-window recognizer map back through the crop it read, and a
//! table built from the table heads never spans over a neighbouring cell. Reading a PDF's own text layer is
//! not supported: render PDF pages to images first.
//!
//! [`IndraOcrBuilder::workers`] enables CPU page concurrency with separate
//! model sessions and shared language models. Select execution providers
//! through [`IndraOcrBuilder::ort_session`]; accelerators require one worker.
//!
//! The model weights use a modified OpenRAIL-M license; see the model
//! repository before using them commercially.
//!
//! ```no_run
//! use oar_ocr::indra::IndraOcrBuilder;
//! use oar_ocr::utils::load_image;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ocr = IndraOcrBuilder::new("models/datalab-to/indra-ocr").build()?;
//! let page = ocr.predict(vec![load_image("page.png")?])?.remove(0);
//! println!("{}", page.markdown);
//! for line in &page.lines {
//!     println!("{:?} {}", line.bbox, line.text);
//! }
//! # Ok(())
//! # }
//! ```

mod cv;
mod detect;
mod layout;
mod lm;
mod math;
mod order;
mod recognize;
mod rtl;

use image::{GrayImage, RgbImage};
use oar_ocr_core::core::OCRError;
use oar_ocr_core::core::config::{ModelInferenceConfig, OrtSessionConfig};
use oar_ocr_core::core::inference::OrtInfer;
use ort::session::{SessionInputValue, SessionOutputs};
use ort::value::Tensor;
use rayon::prelude::*;
use recognize::{Crop, LineText, Token};
use serde::Serialize;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use unicode_normalization::UnicodeNormalization;

/// A median line height (detector input pixels) under this triggers a
/// second detection pass at a larger scale.
const ADAPTIVE_MIN_H: f32 = 9.0;
const ADAPTIVE_TARGET_H: f32 = 12.0;
const ADAPTIVE_MAX_SIDE: usize = 2048;
/// Cut a detected line at a gap this many line heights wide (between CTC
/// character centers).
const SPLIT_GAP: f64 = 3.0;
/// Cut a table line into cells at a gap this many line heights wide.
const CELL_GAP: f64 = 1.8;
/// `orient.onnx` input side: the page thumbnail, padded square.
const ORIENT_SIDE: usize = 768;
/// A quarter or half turn is applied only above this probability.
const ORIENT_MIN: f32 = 0.95;
/// The `orient.onnx` class (of five) of a page set in vertical columns.
const VERTICAL: u32 = 4;
/// The skew search and the orientation model read the page shrunk to at
/// most this many pixels a side.
const PREVIEW_SIDE: f64 = 1600.0;
/// A recognized line under this mean CTC peak probability gets no text.
const MIN_CONF: f64 = 0.5;
/// Degrees of tilt searched by the deskew (quarter turns cover the rest).
const DESKEW_MAX: f64 = 45.0;
/// Smaller tilts are left alone.
const DESKEW_MIN: f64 = 0.3;
/// Lines per recognizer batch.
const REC_BATCH: usize = 64;
/// A line goes to one group's recognizer when the script classifier is
/// surer than this; otherwise the main recognizer reads it as well and the
/// more confident reading wins.
const ROUTE_SURE: f32 = 0.95;
/// Script groups in `script.onnx`'s class order unless `meta.json` names
/// them (`script_groups`). `base` is `rec.onnx`.
const GROUPS: [&str; 4] = ["base", "cjk", "rtl", "other"];
/// Language-model defaults (`meta.json` `lm_params` overrides them per
/// group): lines under the gate confidence are read by beam search.
const LM_GATE: f64 = 0.9;
const LM_ALPHA: f64 = 0.5;
const LM_BETA: f64 = 1.5;
const LM_BEAM: usize = 8;
/// A class under this probability at a step is not tried by the beam search.
const LM_CAND_FLOOR: f64 = 3e-3;

/// Builder for [`IndraOcr`].
#[derive(Debug, Clone)]
pub struct IndraOcrBuilder {
    model_dir: PathBuf,
    math: bool,
    keep_furniture: bool,
    table_heads: bool,
    workers: usize,
    ort_session: Option<OrtSessionConfig>,
}

impl IndraOcrBuilder {
    /// Starts a builder for the models in `model_dir`, a local copy of the
    /// `datalab-to/indra-ocr` Hugging Face repository.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            math: false,
            keep_furniture: false,
            table_heads: false,
            workers: 1,
            ort_session: None,
        }
    }

    /// Reads lines that hold math as text with inline LaTeX (off by default;
    /// slower on math pages). Needs `mathcls.onnx`, `math_enc.onnx`,
    /// `math_dec_init.onnx`, `math_dec_step.onnx` and `math_meta.json`.
    pub fn math(mut self, enable: bool) -> Self {
        self.math = enable;
        self
    }

    /// Keeps page headers and footers in the Markdown (left out by default;
    /// they are always listed in [`IndraPage::blocks`]).
    pub fn keep_furniture(mut self, enable: bool) -> Self {
        self.keep_furniture = enable;
        self
    }

    /// Builds tables from the order model's row, column and cell heads
    /// instead of from line geometry (off by default, as in the reference).
    /// Table lines are first split into cells and the order model runs
    /// again; a table whose cell head joins side-by-side lines falls back
    /// to geometry. Needs an `order.onnx` with the `tab` output; without
    /// it, tables come from geometry.
    pub fn table_heads(mut self, enable: bool) -> Self {
        self.table_heads = enable;
        self
    }

    /// Sets the ONNX Runtime session configuration (execution providers,
    /// threads) applied to every model.
    pub fn ort_session(mut self, config: OrtSessionConfig) -> Self {
        self.ort_session = Some(config);
        self
    }

    /// Sets the number of CPU page workers (default: one).
    ///
    /// Each worker owns a copy of every ONNX session; vocabularies and
    /// language models are shared. More workers increase memory use for
    /// sessions and page buffers. With multiple workers, unset ONNX intra-
    /// and inter-op thread counts default to one. Explicit settings are
    /// retained: keep workers times threads within the available CPU cores.
    /// A process-wide ORT thread pool retains its own thread configuration.
    /// Accelerator execution providers currently require one worker.
    /// Zero workers is a configuration error.
    pub fn workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    /// Loads the models.
    pub fn build(self) -> Result<IndraOcr, OCRError> {
        if self.workers == 0 {
            return Err(OCRError::config_error(
                "Indra OCR needs at least one worker",
            ));
        }
        let dir = self.model_dir.as_path();
        let mut common = ModelInferenceConfig::new();
        common.ort_session = self.ort_session.map(OrtSessionConfig::resolve_auto);
        let pool = if self.workers > 1 {
            let config = common.ort_session.get_or_insert_with(OrtSessionConfig::new);
            if config.has_accelerator_provider() {
                return Err(OCRError::config_error(
                    "Indra OCR accelerator execution requires workers(1); multiple workers are CPU-only",
                ));
            }
            config.intra_threads.get_or_insert(1);
            config.inter_threads.get_or_insert(1);
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(self.workers)
                    .thread_name(|i| format!("indra-{i}"))
                    .build()
                    .map_err(|e| OCRError::config_error(format!("Indra OCR worker pool: {e}")))?,
            )
        } else {
            None
        };
        let meta = read_json(&dir.join("meta.json"))?;
        let vocab = |key: &str| -> Result<Vec<String>, OCRError> {
            let inner: serde_json::Value = serde_json::from_str(meta[key].as_str().unwrap_or(""))
                .map_err(|e| {
                OCRError::config_error(format!("meta.json: {key} is not a vocabulary: {e}"))
            })?;
            string_list(&inner["chars"], &format!("meta.json: {key}.chars"))
        };
        let size = |key: &str, default: Option<usize>| {
            match meta.get(key) {
                None => default,
                Some(v) => v.as_u64().filter(|&v| v > 0).map(|v| v as usize),
            }
            .ok_or_else(|| {
                OCRError::config_error(format!("meta.json: {key} must be a positive integer"))
            })
        };
        // A recognizer, with its language model when the folder has one and
        // the recognizer exports the beam search's inputs.
        let recognizer = |name: &str, file: &str, lm_file: &str, vocab: Vec<String>| {
            let model = model(dir, file, &common)?;
            let lm_path = dir.join(lm_file);
            let outputs: Vec<String> = model.output_shapes().into_iter().map(|o| o.0).collect();
            let lm = if lm_path.exists()
                && ["blank", "topv", "topi"]
                    .iter()
                    .all(|o| outputs.iter().any(|n| n == o))
            {
                let params = &meta["lm_params"][name];
                let param = |key: &str, default: f64| params[key].as_f64().unwrap_or(default);
                Some(Arc::new(Lm {
                    model: lm::NgramLm::load(&lm_path)?,
                    gate: param("gate", LM_GATE),
                    weights: lm::Weights {
                        alpha: param("alpha", LM_ALPHA),
                        beta: param("beta", LM_BETA),
                        beam: LM_BEAM,
                        cand_floor: LM_CAND_FLOOR,
                    },
                }))
            } else {
                None
            };
            let jamo = vocab.iter().any(|c| c == "\u{1100}");
            Ok::<_, OCRError>(Recognizer {
                model,
                vocab: vocab.into(),
                jamo,
                lm,
            })
        };
        // A script group is read by its own recognizer when the folder has
        // it; otherwise by rec.onnx.
        let mut groups = Vec::new();
        if let Some(map) = meta.as_object() {
            for name in map.keys().filter_map(|k| k.strip_prefix("vocab_")) {
                let file = format!("rec_{name}.onnx");
                if dir.join(&file).exists() {
                    let rec = recognizer(
                        name,
                        &file,
                        &format!("lm_{name}.npz"),
                        vocab(&format!("vocab_{name}"))?,
                    )?;
                    groups.push((name.to_string(), rec));
                }
            }
        }
        let script = if !groups.is_empty() && dir.join("script.onnx").exists() {
            Some(model(dir, "script.onnx", &common)?)
        } else {
            None
        };
        let script_groups = match &meta["script_groups"] {
            serde_json::Value::Null => GROUPS.map(str::to_string).to_vec(),
            v => string_list(v, "meta.json: script_groups")?,
        };
        let optional = |name: &str| -> Result<Option<OrtInfer>, OCRError> {
            if dir.join(name).exists() {
                model(dir, name, &common).map(Some)
            } else {
                Ok(None)
            }
        };
        let first = Engine {
            det: model(dir, "det.onnx", &common)?,
            base: recognizer("base", "rec.onnx", "lm.npz", vocab("vocab")?)?,
            order: model(dir, "order.onnx", &common)?,
            orient: optional("orient.onnx")?,
            groups,
            script,
            script_groups,
            math: if self.math {
                Some(math::MathReader::load(dir, &common)?)
            } else {
                None
            },
            long_side: size("det_long_side", None)?,
            rec_height: size("rec_height", None)?,
            window: size("rec_window", Some(1))?,
            keep_furniture: self.keep_furniture,
            table_heads: self.table_heads,
        };
        let mut engines = Vec::with_capacity(self.workers);
        engines.push(first);
        for _ in 1..self.workers {
            engines.push(engines[0].new_worker(&common)?);
        }
        Ok(IndraOcr { engines, pool })
    }
}

/// A CTC line recognizer and its vocabulary.
#[derive(Debug)]
struct Recognizer {
    model: OrtInfer,
    vocab: Arc<[String]>,
    /// The vocabulary reads Hangul as jamo, composed back into syllables.
    jamo: bool,
    lm: Option<Arc<Lm>>,
}

impl Recognizer {
    fn new_worker(&self, common: &ModelInferenceConfig) -> Result<Self, OCRError> {
        Ok(Self {
            model: reload_model(&self.model, common)?,
            vocab: Arc::clone(&self.vocab),
            jamo: self.jamo,
            lm: self.lm.clone(),
        })
    }
}

/// A recognizer's language model and when to use it.
#[derive(Debug)]
struct Lm {
    model: lm::NgramLm,
    /// Lines under this confidence are read by beam search.
    gate: f64,
    weights: lm::Weights,
}

/// The Indra OCR pipeline. Build it with [`IndraOcrBuilder`].
#[derive(Debug)]
pub struct IndraOcr {
    engines: Vec<Engine>,
    pool: Option<rayon::ThreadPool>,
}

/// One worker's ONNX sessions and shared recognition data.
#[derive(Debug)]
struct Engine {
    det: OrtInfer,
    /// `rec.onnx`: Latin, Cyrillic and Greek, and every line no group
    /// recognizer reads.
    base: Recognizer,
    order: OrtInfer,
    orient: Option<OrtInfer>,
    /// The script groups' recognizers.
    groups: Vec<(String, Recognizer)>,
    script: Option<OrtInfer>,
    script_groups: Vec<String>,
    math: Option<math::MathReader>,
    long_side: usize,
    rec_height: usize,
    /// `rec.onnx` reads this many line bands: the line and the page above
    /// and below it.
    window: usize,
    keep_furniture: bool,
    table_heads: bool,
}

/// One recognized line.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct IndraLine {
    /// `[x0, y0, x1, y1]` in pixels of the upright page.
    pub bbox: [f32; 4],
    /// The line's text; empty when the recognizer was not confident.
    pub text: String,
    /// Mean CTC peak probability (1.0 for a math reading).
    pub confidence: f32,
}

/// One layout block.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct IndraBlock {
    /// The block class: `Text`, `Section-Header`, `Page-Header`,
    /// `Page-Footer`, `List-Group`, `Table`, `Caption`, `Footnote`,
    /// `Table-Of-Contents` or `Form`.
    pub label: String,
    /// `[x0, y0, x1, y1]` in pixels of the upright page: the union of the
    /// block's lines, or on a straightened page that union's rectangle
    /// mapped back across the skew (as the reference does), which can be
    /// slightly larger than the union of the mapped line boxes.
    pub bbox: [f32; 4],
    /// The block's lines, newline-joined.
    pub text: String,
    /// The table HTML of a table block.
    #[serde(skip)]
    html: Option<String>,
}

/// One page read by [`IndraOcr`].
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct IndraPage {
    /// Position of the page in the [`IndraOcr::predict`] batch.
    pub index: usize,
    /// Width of the upright page the boxes are in.
    pub width: u32,
    /// Height of the upright page the boxes are in.
    pub height: u32,
    /// Degrees counter-clockwise the input was turned (0, 90, 180 or 270);
    /// the page was turned upright before reading.
    pub rotation: u32,
    /// Degrees of tilt straightened before reading; boxes are mapped back
    /// to the page before the straightening.
    pub skew: f32,
    /// The page is set in vertical columns (Chinese, Japanese): it was read
    /// turned a quarter turn, each column a line, and its boxes are on the
    /// page as it is ([`rotation`](Self::rotation) stays 0).
    pub vertical: bool,
    /// Lines in reading order.
    pub lines: Vec<IndraLine>,
    /// Layout blocks in reading order (empty for a page with fewer than two
    /// lines).
    pub blocks: Vec<IndraBlock>,
    /// Headings, paragraphs, lists, and tables as HTML. Page headers and
    /// footers are left out unless [`IndraOcrBuilder::keep_furniture`].
    pub markdown: String,
}

impl IndraPage {
    /// Exports this page in the shared page JSON format
    /// (`docs/page-format.md`) with `pipeline` `"indra"`.
    ///
    /// Block labels map onto the canonical vocabulary (`Section-Header` →
    /// `paragraph_title`, `Page-Header` → `header`, `Page-Footer` →
    /// `footer`, `List-Group` → `list`, `Caption` →
    /// `figure_table_chart_title`, `Table-Of-Contents` → `content`), with
    /// the original label in `label`; `Form` passes through. Table blocks
    /// carry their HTML. Boxes are in the upright page, whose size is
    /// `page.width` and `page.height`, and `page.angle` is
    /// [`rotation`](Self::rotation) when the page was turned.
    pub fn to_json(&self) -> serde_json::Value {
        let blocks: Vec<_> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| {
                let canonical = match block.label.as_str() {
                    "Text" => "text",
                    "Section-Header" => "paragraph_title",
                    "Page-Header" => "header",
                    "Page-Footer" => "footer",
                    "List-Group" => "list",
                    "Table" => "table",
                    "Caption" => "figure_table_chart_title",
                    "Footnote" => "footnote",
                    "Table-Of-Contents" => "content",
                    other => other,
                };
                let mut value = serde_json::json!({
                    "type": canonical,
                    // Boxes are rounded to 0.1 px; widen without float noise.
                    "bbox": block.bbox.map(|v| (v as f64 * 10.0).round() / 10.0),
                    "order": i + 1,
                    "confidence": null,
                    "angle": null,
                    "content": block.html.as_deref().unwrap_or(&block.text),
                });
                if !block.label.eq_ignore_ascii_case(canonical) {
                    value["label"] = serde_json::Value::String(block.label.clone());
                }
                value
            })
            .collect();
        serde_json::json!({
            "version": 1,
            "pipeline": "indra",
            "page": {
                "index": self.index,
                "width": self.width,
                "height": self.height,
                "angle": (self.rotation != 0).then_some(self.rotation),
            },
            "blocks": blocks,
            "markdown": self.markdown,
            "raw_output": null,
            "diagnostics": [],
        })
    }
}

impl IndraOcr {
    /// Reads pages; one [`IndraPage`] per image, in input order. An empty
    /// image is an [`OCRError::InvalidInput`].
    /// With multiple workers, pages are assigned as engines become free;
    /// results (or the first page error) are returned in input order.
    pub fn predict(&self, images: Vec<RgbImage>) -> Result<Vec<IndraPage>, OCRError> {
        if let Some(i) = images
            .iter()
            .position(|im| im.width() == 0 || im.height() == 0)
        {
            return Err(OCRError::invalid_input(format!(
                "image {i} has no pixels ({}x{})",
                images[i].width(),
                images[i].height()
            )));
        }
        let read = |engine: &Engine, index: usize| {
            let mut page = engine.read_page(cv::rgb_to_gray(&images[index]))?;
            page.index = index;
            Ok(page)
        };
        let Some(pool) = &self.pool else {
            return (0..images.len())
                .map(|index| read(&self.engines[0], index))
                .collect();
        };
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let next = AtomicUsize::new(self.engines.len());
        // One loop per engine. Nested Rayon image processing stays in this
        // pool, and never creates an additional global preprocessing pool.
        let mut results: Vec<_> = pool
            .broadcast(|worker| {
                let engine = &self.engines[worker.index()];
                let mut results = Vec::new();
                // Give each engine its first page before taking more work.
                // A batch of at least N pages therefore warms every engine.
                let mut index = worker.index();
                while index < images.len() {
                    results.push((index, read(engine, index)));
                    index = next.fetch_add(1, Ordering::Relaxed);
                }
                results
            })
            .into_iter()
            .flatten()
            .collect();
        results.sort_unstable_by_key(|(index, _)| *index);
        results.into_iter().map(|(_, result)| result).collect()
    }
}

impl Engine {
    fn new_worker(&self, common: &ModelInferenceConfig) -> Result<Self, OCRError> {
        let optional =
            |model: &Option<OrtInfer>| model.as_ref().map(|m| reload_model(m, common)).transpose();
        Ok(Self {
            det: reload_model(&self.det, common)?,
            base: self.base.new_worker(common)?,
            order: reload_model(&self.order, common)?,
            orient: optional(&self.orient)?,
            groups: self
                .groups
                .iter()
                .map(|(name, rec)| Ok((name.clone(), rec.new_worker(common)?)))
                .collect::<Result<_, OCRError>>()?,
            script: optional(&self.script)?,
            script_groups: self.script_groups.clone(),
            math: self
                .math
                .as_ref()
                .map(|m| m.new_worker(common))
                .transpose()?,
            long_side: self.long_side,
            rec_height: self.rec_height,
            window: self.window,
            keep_furniture: self.keep_furniture,
            table_heads: self.table_heads,
        })
    }

    fn read_page(&self, gray: GrayImage) -> Result<IndraPage, OCRError> {
        let small = preview(&gray);
        let a = estimate_skew(&small);
        // The orientation model reads the page straightened by its row
        // angle (a tilted page could read as turned).
        let k = if a.abs() >= DESKEW_MIN {
            self.orientation(&cv::rotate_u8(&small, a))?
        } else {
            self.orientation(&small)?
        };
        let vertical = k == VERTICAL;
        // Quarter turns clockwise to read the page in: a vertical page
        // turns a quarter counter-clockwise, so its columns become lines,
        // the first (right) column on top.
        let turns = if vertical { 3 } else { k % 4 };
        let gray = cv::rot90_cw(&gray, turns);
        // A half turn keeps the rows, so the angle stands; after a quarter
        // turn the rows were columns, so it is measured again.
        let mut skew = if turns % 2 == 1 {
            estimate_skew(&cv::rot90_cw(&small, turns))
        } else {
            a
        };
        let gray = if skew.abs() >= DESKEW_MIN {
            cv::rotate_u8(&gray, skew)
        } else {
            skew = 0.0;
            gray
        };
        let boxes = self.detect(&gray)?;
        let lines = self.recognize(&gray, &boxes, vertical)?;
        let mut page = self.assemble(gray.dimensions(), boxes, lines, vertical)?;
        page.rotation = if vertical { 0 } else { 90 * turns };
        page.skew = ((skew * 100.0).round_ties_even() / 100.0) as f32;
        if skew != 0.0 {
            let wh = gray.dimensions();
            for line in &mut page.lines {
                line.bbox = unrotate_box(line.bbox, skew, wh);
            }
            for block in &mut page.blocks {
                block.bbox = unrotate_box(block.bbox, skew, wh);
            }
        }
        if vertical {
            // Back on the page as it is: (x, y) -> (W - y, x).
            let w = gray.height() as f64;
            let back = |b: [f32; 4]| {
                [
                    (w - b[3] as f64) as f32,
                    b[0],
                    (w - b[1] as f64) as f32,
                    b[2],
                ]
            };
            for line in &mut page.lines {
                line.bbox = back(line.bbox);
            }
            for block in &mut page.blocks {
                block.bbox = back(block.bbox);
            }
            (page.width, page.height) = (gray.height(), gray.width());
            page.vertical = true;
        }
        Ok(page)
    }

    /// Quarter turns (counter-clockwise) the page content is turned by, or
    /// [`VERTICAL`] for a page set in vertical columns.
    fn orientation(&self, gray: &GrayImage) -> Result<u32, OCRError> {
        let Some(orient) = &self.orient else {
            return Ok(0);
        };
        let (w, h) = (gray.width() as f64, gray.height() as f64);
        let s = ORIENT_SIDE as f64 / w.max(h);
        let (sw, sh) = (
            ((w * s).round_ties_even() as u32).max(1),
            ((h * s).round_ties_even() as u32).max(1),
        );
        let small = cv::resize(gray, sw, sh, cv::Interp::Area);
        let mut x = vec![0f32; ORIENT_SIDE * ORIENT_SIDE];
        for (y, row) in small.as_raw().chunks_exact(sw as usize).enumerate() {
            for (o, &v) in x[y * ORIENT_SIDE..].iter_mut().zip(row) {
                *o = 1.0 - v as f32 / 255.0;
            }
        }
        let p = run(
            orient,
            ort::inputs!["x" => Tensor::from_array(([1, 1, ORIENT_SIDE, ORIENT_SIDE], x)).map_err(input_error)?],
            |o| Ok(o["p"].try_extract_tensor::<f32>()?.1.to_vec()),
        )?;
        let k = argmax(&p);
        Ok(if p[k] > ORIENT_MIN { k as u32 } else { 0 })
    }

    fn detect_at(
        &self,
        gray: &GrayImage,
        long_side: usize,
    ) -> Result<(Vec<[f32; 5]>, f64), OCRError> {
        let (x, h, w, scale) = detect::page_to_input(gray, long_side);
        let boxes = run(
            &self.det,
            ort::inputs!["x" => Tensor::from_array(([1, 1, h, w], x)).map_err(input_error)?],
            |o| {
                let (shape, out) = o["out"].try_extract_tensor::<f32>()?;
                Ok(detect::decode(out, shape[2] as usize, shape[3] as usize))
            },
        )?;
        Ok((boxes, scale))
    }

    /// Line boxes `[x0, y0, x1, y1, score]` in page pixels.
    fn detect(&self, gray: &GrayImage) -> Result<Vec<[f32; 5]>, OCRError> {
        let (mut boxes, mut scale) = self.detect_at(gray, self.long_side)?;
        if boxes.len() >= 8 {
            let mut heights: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
            let med_h = order::median_f32(&mut heights);
            if med_h < ADAPTIVE_MIN_H {
                let target = (self.long_side as f32 * ADAPTIVE_TARGET_H / med_h.max(1.0)) as f64;
                let ls = (target.min(ADAPTIVE_MAX_SIDE as f64) as usize) / 32 * 32;
                if ls > self.long_side {
                    (boxes, scale) = self.detect_at(gray, ls)?;
                }
            }
        }
        let (w, h) = (gray.width() as f32, gray.height() as f32);
        let s = scale as f32;
        for b in &mut boxes {
            b[0] = (b[0] / s).clamp(0.0, w);
            b[1] = (b[1] / s).clamp(0.0, h);
            b[2] = (b[2] / s).clamp(0.0, w);
            b[3] = (b[3] / s).clamp(0.0, h);
        }
        Ok(boxes)
    }

    /// Text, confidence and tokens per line; each token's x is its CTC peak
    /// mapped back through the crop. Each line goes to the recognizer of
    /// its script group (every line of a vertical page to the CJK one); a
    /// line the script classifier is unsure of is also read by `rec.onnx`,
    /// and the more confident reading wins. A line under `MIN_CONF` comes
    /// back empty.
    fn recognize(
        &self,
        gray: &GrayImage,
        boxes: &[[f32; 5]],
        vertical: bool,
    ) -> Result<Vec<LineText>, OCRError> {
        let height = self.rec_height;
        let b4 = |b: &[f32; 5]| [b[0], b[1], b[2], b[3]];
        // Light-on-dark is decided once per line box, for every crop.
        let dark: Vec<bool> = boxes
            .par_iter()
            .map(|b| recognize::is_dark_line(gray, b4(b)))
            .collect();
        let crop_all = |window: usize| -> Vec<Crop> {
            boxes
                .par_iter()
                .zip(&dark)
                .map(|(b, &d)| recognize::crop_line(gray, b4(b), height, window, d))
                .collect()
        };
        let crops = crop_all(1);
        // rec.onnx may read a tall window: the line and the page around it.
        let tall = (self.window > 1).then(|| crop_all(self.window));
        let n = crops.len();
        // (recognizer, the lines it reads, whether its tokens are characters)
        let mut reads: Vec<(&Recognizer, Vec<usize>, bool)> =
            vec![(&self.base, (0..n).collect(), false)];
        let cjk = self.groups.iter().find(|(name, _)| name == "cjk");
        if vertical && let Some((_, rec)) = cjk {
            // A vertical page's lines are CJK columns, characters on their
            // side; the CJK recognizer reads those.
            reads = vec![(rec, (0..n).collect(), true)];
        } else if let Some(script) = &self.script
            && n > 0
        {
            let (route, sure) = self.line_groups(script, &crops)?;
            reads = vec![(
                &self.base,
                (0..n).filter(|&j| route[j].is_none() || !sure[j]).collect(),
                false,
            )];
            for (g, (_, rec)) in self.groups.iter().enumerate() {
                reads.push((rec, (0..n).filter(|&j| route[j] == Some(g)).collect(), true));
            }
        }
        let mut out = vec![LineText::default(); n];
        let mut best_conf = vec![-1.0; n];
        for (rec, sel, char_tokens) in reads {
            let base = std::ptr::eq(rec, &self.base);
            let (src, rows) = match &tall {
                Some(tall) if base => (tall, height * self.window),
                _ => (&crops, height),
            };
            let widths: Vec<usize> = sel.iter().map(|&j| crops[j].width).collect();
            for part in recognize::width_batches(&widths, REC_BATCH, 1.3, 64 * 640) {
                let idx: Vec<usize> = part.iter().map(|&i| sel[i]).collect();
                let batch: Vec<&Crop> = idx.iter().map(|&j| &src[j]).collect();
                let (data, wm, crop_widths) = recognize::pad_batch(&batch, rows);
                let b = batch.len();
                let read = run(
                    &rec.model,
                    ort::inputs![
                        "x" => Tensor::from_array(([b, 1, rows, wm], data)).map_err(input_error)?,
                        "widths" => Tensor::from_array(([b], crop_widths)).map_err(input_error)?,
                    ],
                    |o| {
                        let (shape, ids) = o["ids"].try_extract_tensor::<i64>()?;
                        let (_, conf) = o["conf"].try_extract_tensor::<f32>()?;
                        let (_, lengths) = o["lengths"].try_extract_tensor::<i64>()?;
                        let lm = if rec.lm.is_some() {
                            let (_, blank) = o["blank"].try_extract_tensor::<f32>()?;
                            let (top_shape, topv) = o["topv"].try_extract_tensor::<f32>()?;
                            let (_, topi) = o["topi"].try_extract_tensor::<i64>()?;
                            Some((
                                blank.to_vec(),
                                topv.to_vec(),
                                topi.to_vec(),
                                top_shape[2] as usize,
                            ))
                        } else {
                            None
                        };
                        Ok((
                            ids.to_vec(),
                            conf.to_vec(),
                            lengths.to_vec(),
                            shape[1] as usize,
                            lm,
                        ))
                    },
                )?;
                let (ids, conf, lengths, t_len, lm_inputs) = read;
                let stride = wm as f64 / t_len.max(1) as f64;
                for (k, &j) in idx.iter().enumerate() {
                    let len = (lengths[k].max(0) as usize).min(t_len);
                    let row = k * t_len;
                    let seq = &ids[row..row + len];
                    let pm = &conf[row..row + len];
                    let mut keep: Vec<usize> = (0..len)
                        .filter(|&t| seq[t] != 0 && (t == 0 || seq[t] != seq[t - 1]))
                        .collect();
                    let [x0, y0, x1, _] =
                        [boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]].map(f64::from);
                    let h = (boxes[j][3] as f64 - y0).max(1.0);
                    let ix0 = (x0 - recognize::PAD_X * h).floor();
                    let cw = (x1 + recognize::PAD_X * h).ceil() - ix0;
                    // Timesteps map back through the crop the recognizer
                    // read (the reference scales by the one-band crop even
                    // for a tall window, which stretches token positions).
                    let f = cw / src[j].width as f64;
                    let line_conf = if keep.is_empty() {
                        0.0
                    } else {
                        mean_f32(&keep.iter().map(|&t| pm[t]).collect::<Vec<_>>())
                    };
                    let mut line_ids: Vec<i64> = keep.iter().map(|&t| seq[t]).collect();
                    if let (Some(lm), Some((blank, topv, topi, top_k))) = (&rec.lm, &lm_inputs)
                        && (MIN_CONF..lm.gate).contains(&line_conf)
                    {
                        let steps = lm::Steps {
                            blank: &blank[row..row + len],
                            topv: &topv[row * top_k..(row + len) * top_k],
                            topi: &topi[row * top_k..(row + len) * top_k],
                            k: *top_k,
                        };
                        (line_ids, keep) = lm::beam_search(&steps, &lm.model, lm.weights);
                    }
                    if line_conf <= best_conf[j] {
                        continue;
                    }
                    best_conf[j] = line_conf;
                    // An id past the vocabulary means meta.json does not
                    // belong to this recognizer.
                    let tokens: Vec<Token> = line_ids
                        .iter()
                        .zip(&keep)
                        .map(|(&id, &t)| {
                            let text = usize::try_from(id - 1)
                                .ok()
                                .and_then(|i| rec.vocab.get(i))
                                .ok_or_else(|| {
                                    OCRError::config_error(format!(
                                        "{}: token id {id} is outside its {}-entry vocabulary",
                                        rec.model.model_name(),
                                        rec.vocab.len()
                                    ))
                                })?;
                            Ok(Token {
                                text: text.clone(),
                                x: ix0 + (t as f64 + 0.5) * stride * f,
                            })
                        })
                        .collect::<Result<_, OCRError>>()?;
                    out[j] = if line_conf >= MIN_CONF {
                        let text = if char_tokens {
                            let text: String = tokens.iter().map(|t| t.text.as_str()).collect();
                            if rec.jamo { text.nfc().collect() } else { text }
                        } else {
                            recognize::join_tokens(
                                &tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>(),
                            )
                        };
                        LineText {
                            text,
                            conf: line_conf,
                            tokens,
                            char_tokens,
                        }
                    } else {
                        LineText {
                            conf: line_conf,
                            char_tokens,
                            ..LineText::default()
                        }
                    };
                }
            }
        }
        if let Some(math) = &self.math
            && !crops.is_empty()
        {
            let flagged: Vec<usize> = math
                .flags(&crops, height)?
                .iter()
                .enumerate()
                .filter_map(|(j, &f)| f.then_some(j))
                .collect();
            if !flagged.is_empty() {
                let mboxes: Vec<[f32; 4]> = flagged.iter().map(|&j| b4(&boxes[j])).collect();
                for (&j, (text, conf)) in flagged.iter().zip(math.read(gray, &mboxes)?) {
                    // Only a confident reading with math replaces the line,
                    // as one token that is never split.
                    if text.contains('$') && conf >= math::MATH_MIN_CONF {
                        let x = (boxes[j][0] + boxes[j][2]) as f64 / 2.0;
                        out[j] = LineText {
                            text: text.clone(),
                            conf: 1.0,
                            tokens: vec![Token { text, x }],
                            char_tokens: true,
                        };
                    }
                }
            }
        }
        Ok(out)
    }

    /// Per line crop, the script group whose recognizer reads it (`None`:
    /// `rec.onnx`), and whether the script classifier is sure of it. An
    /// older `script.onnx` has one output: the chance of Chinese characters.
    fn line_groups(
        &self,
        script: &OrtInfer,
        crops: &[Crop],
    ) -> Result<(Vec<Option<usize>>, Vec<bool>), OCRError> {
        let height = self.rec_height;
        let mut route = vec![None; crops.len()];
        let mut sure = vec![true; crops.len()];
        for idx in recognize::line_batches(crops) {
            let batch: Vec<&Crop> = idx.iter().map(|&j| &crops[j]).collect();
            let (data, wm, widths) = recognize::pad_batch(&batch, height);
            let n = batch.len();
            let (p, classes) = run(
                script,
                ort::inputs![
                    "x" => Tensor::from_array(([n, 1, height, wm], data)).map_err(input_error)?,
                    "widths" => Tensor::from_array(([n], widths)).map_err(input_error)?,
                ],
                |o| {
                    let (shape, p) = o["p"].try_extract_tensor::<f32>()?;
                    Ok((p.to_vec(), shape.get(1).map(|&c| c as usize)))
                },
            )?;
            for (k, &j) in idx.iter().enumerate() {
                let (name, pk) = match classes {
                    None => {
                        let pc = p[k];
                        (if pc > 0.5 { "cjk" } else { "base" }, pc.max(1.0 - pc))
                    }
                    Some(c) => {
                        let row = &p[k * c..(k + 1) * c];
                        let g = argmax(row);
                        let name = self.script_groups.get(g).ok_or_else(|| {
                            OCRError::config_error(format!(
                                "script.onnx has {c} classes; meta.json script_groups names {}",
                                self.script_groups.len()
                            ))
                        })?;
                        (name.as_str(), row[g])
                    }
                };
                route[j] = self.groups.iter().position(|(g, _)| g == name);
                sure[j] = pk > ROUTE_SURE;
            }
        }
        Ok((route, sure))
    }

    /// Lines split at wide gaps, tidied, reading order, layout blocks and
    /// Markdown. A right-to-left page reads its lines right to left, and
    /// its reading order and Markdown work on the page mirrored left to
    /// right, so the first column is the right one.
    fn assemble(
        &self,
        (pw, ph): (u32, u32),
        boxes: Vec<[f32; 5]>,
        lines: Vec<LineText>,
        vertical: bool,
    ) -> Result<IndraPage, OCRError> {
        let boxes: Vec<[f32; 4]> = boxes.iter().map(|b| [b[0], b[1], b[2], b[3]]).collect();
        let (letters, n_rtl) = lines
            .iter()
            .flat_map(|l| l.text.chars())
            .filter(|&c| rtl::is_letter(c))
            .fold((0, 0), |(n, r), c| (n + 1, r + rtl::is_rtl(c) as usize));
        let page_rtl = n_rtl > letters - n_rtl;
        let width = pw as f32;
        let geo = |b: [f32; 4]| {
            if page_rtl {
                [width - b[2], b[1], width - b[0], b[3]]
            } else {
                b
            }
        };
        let cuts = recognize::gutter_cuts(&boxes, &lines);
        let (mut nb, mut nl) = (Vec::new(), Vec::new());
        for ((b, line), at) in boxes.iter().zip(&lines).zip(&cuts) {
            for (bx, text, tokens) in recognize::split_line(*b, line, SPLIT_GAP, at, page_rtl) {
                nb.push(bx);
                nl.push(LineText {
                    text,
                    conf: line.conf,
                    tokens,
                    char_tokens: line.char_tokens,
                });
            }
        }
        let space = self.base.vocab.iter().any(|c| c == " ");
        let (mut boxes, mut lines) = recognize::tidy_lines(nb, nl, page_rtl, space);
        let mut texts: Vec<String> = lines.iter().map(|l| l.text.clone()).collect();
        let round1 = |v: f32| ((v as f64 * 10.0).round_ties_even() / 10.0) as f32;
        let mut page = IndraPage {
            index: 0,
            width: pw,
            height: ph,
            rotation: 0,
            skew: 0.0,
            vertical: false,
            lines: Vec::new(),
            blocks: Vec::new(),
            markdown: String::new(),
        };
        if boxes.len() < 2 {
            page.lines = boxes
                .iter()
                .zip(&lines)
                .map(|(b, l)| IndraLine {
                    bbox: b.map(round1),
                    text: l.text.clone(),
                    confidence: ((l.conf * 1e4).round_ties_even() / 1e4) as f32,
                })
                .collect();
            page.markdown = texts.join("\n");
            return Ok(page);
        }
        let mut geo_boxes: Vec<[f32; 4]> = boxes.iter().map(|&b| geo(b)).collect();
        let mut layout = self.layout(&geo_boxes, &texts, (pw, ph))?;
        let mut n = boxes.len();
        if vertical {
            // The order model saw no vertical pages: turned, their columns
            // of short lines look like one table read row by row. Blocks and
            // order come from XY-cut instead, without grid classes.
            let blocks = order::xycut_blocks(&boxes);
            layout.same = vec![-1.0; n * n];
            for b in &blocks {
                for &i in b {
                    for &j in b {
                        layout.same[i * n + j] = 1.0;
                    }
                }
            }
            let nc = order::BLOCK_CLASSES.len();
            for row in layout.cls.chunks_exact_mut(nc) {
                for c in order::GRID_CLASSES {
                    row[c] = -1e4;
                }
            }
            layout.order = blocks.concat();
            layout.tab = None;
        } else if layout.tab.is_some() {
            // The table heads learned one line per cell, but the detector
            // can join cells: split table lines into cells, then run the
            // order model again.
            let nc = order::BLOCK_CLASSES.len();
            let is_table: Vec<bool> = layout
                .cls
                .chunks_exact(nc)
                .map(|row| argmax(row) == layout::TABLE)
                .collect();
            if is_table.contains(&true) {
                let (mut nb, mut nl) = (Vec::new(), Vec::new());
                for ((b, line), table) in boxes.iter().zip(&lines).zip(is_table) {
                    if !table {
                        nb.push(*b);
                        nl.push(line.clone());
                        continue;
                    }
                    for (bx, text, tokens) in
                        recognize::split_line(*b, line, CELL_GAP, &[], page_rtl)
                    {
                        nb.push(bx);
                        nl.push(LineText {
                            text,
                            conf: line.conf,
                            tokens,
                            char_tokens: line.char_tokens,
                        });
                    }
                }
                if nb.len() != n {
                    (boxes, lines) = (nb, nl);
                    texts = lines.iter().map(|l| l.text.clone()).collect();
                    geo_boxes = boxes.iter().map(|&b| geo(b)).collect();
                    layout = self.layout(&geo_boxes, &texts, (pw, ph))?;
                    n = boxes.len();
                }
            }
        }
        let Layout {
            order: reading,
            cls,
            same,
            tab,
        } = layout;
        page.lines = reading
            .iter()
            .map(|&i| IndraLine {
                bbox: boxes[i].map(round1),
                text: lines[i].text.clone(),
                confidence: ((lines[i].conf * 1e4).round_ties_even() / 1e4) as f32,
            })
            .collect();
        // Inside tables, cells split at smaller gaps.
        let cells: Vec<Vec<([f32; 4], String)>> = boxes
            .iter()
            .zip(&lines)
            .map(|(b, l)| {
                recognize::split_line(*b, l, CELL_GAP, &[], page_rtl)
                    .into_iter()
                    .map(|(bx, text, _)| (geo(bx), text))
                    .collect()
            })
            .collect();
        let blocks = layout::page_blocks(
            &geo_boxes,
            &texts,
            &reading,
            &layout::Heads {
                cls: &cls,
                same: order::Square { n, data: &same },
                tab: tab.as_deref().map(|data| layout::TableHeads { n, data }),
            },
            self.keep_furniture,
            &cells,
        );
        page.markdown = layout::page_markdown(&blocks);
        let round = |v: f32| (v as f64 * 10.0).round_ties_even() / 10.0;
        page.blocks = blocks
            .into_iter()
            .map(|block| {
                let geo_boxes = &geo_boxes;
                let bx = |k: usize| block.lines.iter().map(move |&i| geo_boxes[i][k]);
                let mut bbox = [
                    round(bx(0).fold(f32::INFINITY, f32::min)),
                    round(bx(1).fold(f32::INFINITY, f32::min)),
                    round(bx(2).fold(f32::NEG_INFINITY, f32::max)),
                    round(bx(3).fold(f32::NEG_INFINITY, f32::max)),
                ];
                if page_rtl {
                    let w = pw as f64;
                    (bbox[0], bbox[2]) = (w - bbox[2], w - bbox[0]);
                }
                IndraBlock {
                    label: layout::label(block.class).to_string(),
                    bbox: bbox.map(|v| round1(v as f32)),
                    text: block
                        .lines
                        .iter()
                        .map(|&i| texts[i].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                    html: block.html,
                }
            })
            .collect();
        Ok(page)
    }
}

/// The order model's reading of a page's lines.
struct Layout {
    order: Vec<usize>,
    /// Block class logits, `[n, classes]`.
    cls: Vec<f32>,
    /// Same-block logits, `[n, n]`.
    same: Vec<f32>,
    /// Same row, column and cell logits, `[3, n, n]`, with the table heads.
    tab: Option<Vec<f32>>,
}

impl Engine {
    /// Runs the order model on line boxes (two or more) and decodes the
    /// reading order: blocks from the same-block head, lines inside a block
    /// by geometry, blocks by the order head.
    fn layout(
        &self,
        boxes: &[[f32; 4]],
        texts: &[String],
        wh: (u32, u32),
    ) -> Result<Layout, OCRError> {
        let n = boxes.len();
        let feats = order::line_features(boxes, wh, texts);
        let (a, cls, same, tab) = run(
            &self.order,
            ort::inputs![
                "feats" => Tensor::from_array(([1, n, order::N_FEATS], feats)).map_err(input_error)?,
                "mask" => Tensor::from_array(([1, n], vec![true; n])).map_err(input_error)?,
            ],
            |o| {
                let (_, a) = o["a"].try_extract_tensor::<f32>()?;
                let (_, cls) = o["cls"].try_extract_tensor::<f32>()?;
                let (_, same) = o["same"].try_extract_tensor::<f32>()?;
                let tab = match o.get("tab") {
                    Some(tab) if self.table_heads => {
                        Some(tab.try_extract_tensor::<f32>()?.1.to_vec())
                    }
                    _ => None,
                };
                Ok((a.to_vec(), cls.to_vec(), same.to_vec(), tab))
            },
        )?;
        let order = order::decode_blocks_order(
            &order::Square { n, data: &a },
            &order::Square { n, data: &same },
            boxes,
            &cls,
        );
        Ok(Layout {
            order,
            cls,
            same,
            tab,
        })
    }
}

/// The first index of the largest value.
fn argmax(values: &[f32]) -> usize {
    let mut k = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[k] {
            k = i;
        }
    }
    k
}

/// numpy's mean of float32 values: a pairwise sum and a division in
/// float32.
fn mean_f32(values: &[f32]) -> f64 {
    fn pairwise(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            a.iter().fold(0.0, |s, &v| s + v)
        } else if n <= 128 {
            let mut r: [f32; 8] = a[..8].try_into().expect("8 values");
            let full = n - n % 8;
            for chunk in a[8..full].as_chunks::<8>().0 {
                for (r, v) in r.iter_mut().zip(chunk) {
                    *r += v;
                }
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for v in &a[full..] {
                res += v;
            }
            res
        } else {
            let half = n / 2;
            let half = half - half % 8;
            pairwise(&a[..half]) + pairwise(&a[half..])
        }
    }
    (pairwise(values) / values.len() as f32) as f64
}

/// The page shrunk once to at most `PREVIEW_SIDE` pixels a side (area
/// averaging): the skew search and the orientation model read it instead
/// of the full scan.
fn preview(gray: &GrayImage) -> Cow<'_, GrayImage> {
    let (w, h) = (gray.width() as f64, gray.height() as f64);
    let s = PREVIEW_SIDE / w.max(h);
    if s >= 1.0 {
        return Cow::Borrowed(gray);
    }
    let (sw, sh) = (
        ((w * s).round_ties_even() as u32).max(1),
        ((h * s).round_ties_even() as u32).max(1),
    );
    Cow::Owned(cv::resize(gray, sw, sh, cv::Interp::Area))
}

/// The ink map (255 = ink) of the page shrunk to `side` pixels a side.
fn ink_thumb(gray: &GrayImage, side: f64) -> GrayImage {
    let (w, h) = (gray.width() as f64, gray.height() as f64);
    let s = side / w.max(h);
    let (sw, sh) = (
        ((w * s).round_ties_even() as u32).max(1),
        ((h * s).round_ties_even() as u32).max(1),
    );
    let mut small = cv::resize(gray, sw, sh, cv::Interp::Area);
    for v in small.iter_mut() {
        *v = 255 - *v;
    }
    small
}

/// How sharply the horizontal ink profile changes row to row with the page
/// turned by `a` degrees: highest when the text rows are level. Integer row
/// sums keep the score exact.
fn row_sharpness(ink: &GrayImage, a: f64) -> f64 {
    let r = cv::rotate_u8_zero(ink, a);
    let rows: Vec<i64> = r
        .as_raw()
        .chunks_exact(ink.width() as usize)
        .map(|row| row.iter().map(|&v| v as i64).sum())
        .collect();
    rows.windows(2).map(|p| (p[1] - p[0]).pow(2)).sum::<i64>() as f64
}

/// Degrees (counter-clockwise) that straighten the text rows of a page:
/// the angle whose horizontal ink profile changes most sharply row to row.
/// Searched over ±`DESKEW_MAX` in 2 degree steps on a 400 px thumbnail;
/// the two best angles and 0 are then refined in 0.5 and 0.1 degree steps
/// on an 800 px thumbnail, where small text rows are still resolved.
fn estimate_skew(gray: &GrayImage) -> f64 {
    let fine = ink_thumb(gray, 800.0);
    // A one-row thumbnail has no row profile, and a blank or flat page has
    // no rows to straighten.
    let raw = fine.as_raw();
    let mean = raw.iter().map(|&v| v as u64).sum::<u64>() as f64 / raw.len() as f64;
    if fine.height() < 2 || mean < 0.003 * 255.0 || raw.iter().all(|&v| v == raw[0]) {
        return 0.0;
    }
    let coarse = ink_thumb(gray, 400.0);
    let grid = arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0);
    let scores: Vec<f64> = grid
        .par_iter()
        .map(|&a| row_sharpness(&coarse, a))
        .collect();
    let mut best: Vec<usize> = (0..grid.len()).collect();
    best.sort_by(|&i, &j| scores[i].total_cmp(&scores[j]));
    let starts = [0.0, grid[best[best.len() - 2]], grid[best[best.len() - 1]]];
    let near: Vec<f64> = starts
        .iter()
        .flat_map(|&a0| arange(a0 - 2.0, a0 + 2.0 + 1e-6, 0.5))
        .collect();
    // The 0.5 degree ranges of nearby starts overlap: each angle is scored
    // once.
    let mut angles: Vec<f64> = Vec::new();
    for &a in &near {
        if !angles.contains(&a) {
            angles.push(a);
        }
    }
    let fine_scores: Vec<f64> = angles
        .par_iter()
        .map(|&a| row_sharpness(&fine, a))
        .collect();
    let score = |a: f64| fine_scores[angles.iter().position(|&b| b == a).expect("scored")];
    let first_best = |angles: &[f64], score: &dyn Fn(f64) -> f64| {
        let mut best = angles[0];
        for &a in angles {
            if score(a) > score(best) {
                best = a;
            }
        }
        best
    };
    let a1 = first_best(&near, &score);
    let last = arange(a1 - 0.5, a1 + 0.5 + 1e-6, 0.1);
    let last_scores: Vec<f64> = last.par_iter().map(|&a| row_sharpness(&fine, a)).collect();
    last[argmax_f64(&last_scores)]
}

/// The first index of the largest value.
fn argmax_f64(values: &[f64]) -> usize {
    let mut k = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[k] {
            k = i;
        }
    }
    k
}

/// `numpy.arange`: values `start + i * delta` with `delta` taken from the
/// first two values, so the angles (and the deskew threshold test on them)
/// match the reference to the last bit.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil().max(0.0) as usize;
    let delta = (start + step) - start;
    (0..n).map(|i| start + i as f64 * delta).collect()
}

/// A box of the page `rotate_u8(.., degrees)` made, mapped back to the page
/// before it: the bounding box of the four mapped corners, clipped to it.
fn unrotate_box(b: [f32; 4], degrees: f64, (w, h): (u32, u32)) -> [f32; 4] {
    let (w, h) = (w as f64, h as f64);
    let m = cv::rotation_matrix(w / 2.0, h / 2.0, -degrees);
    let [x0, y0, x1, y1] = b.map(f64::from);
    let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
    let xs = corners.map(|(x, y)| m[0] * x + m[1] * y + m[2]);
    let ys = corners.map(|(x, y)| m[3] * x + m[4] * y + m[5]);
    let min = |v: [f64; 4]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = |v: [f64; 4]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let r = |v: f64, hi: f64| ((v.clamp(0.0, hi) * 10.0).round_ties_even() / 10.0) as f32;
    [r(min(xs), w), r(min(ys), h), r(max(xs), w), r(max(ys), h)]
}

/// Loads one model of the folder.
fn model(dir: &Path, name: &str, common: &ModelInferenceConfig) -> Result<OrtInfer, OCRError> {
    let mut common = common.clone();
    common.model_name = Some(format!("indra {name}"));
    OrtInfer::from_config(&common, dir.join(name), None)
}

/// Creates a separate session for another worker without reloading metadata.
fn reload_model(original: &OrtInfer, common: &ModelInferenceConfig) -> Result<OrtInfer, OCRError> {
    let mut common = common.clone();
    common.model_name = Some(original.model_name().to_owned());
    OrtInfer::from_config(&common, original.model_path(), Some(original.input_name()))
}

/// A non-empty JSON array of non-empty strings (a vocabulary).
fn string_list(value: &serde_json::Value, what: &str) -> Result<Vec<String>, OCRError> {
    let list: Option<Vec<String>> = value.as_array().filter(|a| !a.is_empty()).and_then(|a| {
        a.iter()
            .map(|v| v.as_str().filter(|s| !s.is_empty()).map(str::to_string))
            .collect()
    });
    list.ok_or_else(|| {
        OCRError::config_error(format!(
            "{what} must be a non-empty list of non-empty strings"
        ))
    })
}

fn read_json(path: &Path) -> Result<serde_json::Value, OCRError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        OCRError::model_load_error(path, "cannot read", Some("check the model folder"), Some(e))
    })?;
    serde_json::from_str(&text)
        .map_err(|e| OCRError::config_error(format!("{}: {e}", path.display())))
}

/// Runs a model on named inputs and reads its outputs while the session is
/// held.
fn run<R>(
    model: &OrtInfer,
    inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)>,
    read: impl FnOnce(&SessionOutputs<'_>) -> ort::Result<R>,
) -> Result<R, OCRError> {
    let mut session = model.get_session(0)?;
    // Keep the run options (CUDA arena shrinkage) that `OrtInfer` applies.
    let outputs = match model.run_options() {
        Some(options) => session.run_with_options(inputs, options),
        None => session.run(inputs),
    }
    .map_err(|e| OCRError::inference_error(model.model_name(), "forward pass", e))?;
    read(&outputs).map_err(|e| OCRError::inference_error(model.model_name(), "reading outputs", e))
}

/// Maps an input tensor that ONNX Runtime rejects to an error.
fn input_error(e: ort::Error) -> OCRError {
    OCRError::invalid_input(format!("Indra OCR model input: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy's arange makes the third fine angle around 0 read
    /// -0.30000000000000004, which passes the deskew threshold; `-0.5 + 0.2`
    /// would not.
    #[test]
    fn deskew_angles_follow_numpy_arange() {
        let angles = arange(-0.5, 0.5 + 1e-6, 0.1);
        assert_eq!(angles.len(), 11);
        assert!(angles[2].abs() >= DESKEW_MIN);
    }

    #[test]
    fn parallel_pages_match_sequential() {
        use oar_ocr_core::core::config::OrtExecutionProvider;

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/datalab-to/indra-ocr-v1.1.0");
        if !dir.join("meta.json").is_file() {
            eprintln!("skipping Indra page concurrency test: local models are absent");
            return;
        }
        // A small raster page spelling TEST twice; no font or image fixture
        // dependency. Unequal sizes, rotation, a blank and a repeated page
        // make misplaced or lost results visible in a five-page batch.
        let glyphs = [
            [31u8, 4, 4, 4, 4, 4, 4],
            [31, 16, 16, 30, 16, 16, 31],
            [15, 16, 16, 14, 1, 1, 30],
            [31, 4, 4, 4, 4, 4, 4],
        ];
        let mut page = RgbImage::from_pixel(256, 160, image::Rgb([255; 3]));
        for y0 in [30, 90] {
            for (letter, rows) in glyphs.iter().enumerate() {
                for (row, bits) in rows.iter().enumerate() {
                    for col in 0..5 {
                        if bits & (1 << (4 - col)) != 0 {
                            for dy in 0..4 {
                                for dx in 0..4 {
                                    page.put_pixel(
                                        24 + letter as u32 * 28 + col * 4 + dx,
                                        y0 + row as u32 * 4 + dy,
                                        image::Rgb([0; 3]),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        let images = vec![
            page.clone(),
            image::imageops::rotate90(&page),
            RgbImage::from_pixel(192, 128, image::Rgb([255; 3])),
            page.clone(),
            image::imageops::rotate180(&page),
        ];
        let cpu = OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);
        let serial = IndraOcrBuilder::new(&dir)
            .ort_session(cpu.clone().with_intra_threads(1).with_inter_threads(1))
            .build()
            .unwrap();
        let expected = serial.predict(images.clone()).unwrap();
        drop(serial);
        let parallel = IndraOcrBuilder::new(&dir)
            .workers(3)
            .ort_session(cpu)
            .build()
            .unwrap();
        for _ in 0..2 {
            let actual = parallel.predict(images.clone()).unwrap();
            // Serialize all public fields, including lines and confidences;
            // the shared export additionally checks the private table HTML.
            assert_eq!(
                serde_json::to_value(&actual).unwrap(),
                serde_json::to_value(&expected).unwrap()
            );
            for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.index, index);
                assert_eq!(a.to_json(), e.to_json());
            }
        }
    }
}
