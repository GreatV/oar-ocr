//! Indra OCR: Datalab's CPU-first page reader on ONNX Runtime.
//!
//! [Indra OCR](https://huggingface.co/datalab-to/indra-ocr) is a set of small
//! ONNX models (about 18 MB) that turn a page image into lines with boxes,
//! layout blocks in reading order, and Markdown: a page orientation
//! classifier, a line detector, a script classifier that routes lines with
//! Chinese characters to their own CTC recognizer, a reading-order model
//! with layout heads, and an optional math reader that writes lines with
//! math as inline LaTeX.
//!
//! [`IndraOcr`] loads the models from a local copy of the Hugging Face
//! repository (`det.onnx`, `rec.onnx`, `order.onnx` and `meta.json`, with
//! `orient.onnx`, `script.onnx` and `rec_cjk.onnx` used when present) and
//! reproduces the reference pipeline's OCR mode. Reading a PDF's own text
//! layer is not supported: render PDF pages to images first.
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
mod math;
mod order;
mod recognize;

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
/// A recognized line under this mean CTC peak probability gets no text.
const MIN_CONF: f64 = 0.5;
/// Degrees of tilt searched by the deskew.
const DESKEW_MAX: f64 = 5.0;
/// Smaller tilts are left alone.
const DESKEW_MIN: f64 = 0.3;
/// Lines per recognizer batch.
const REC_BATCH: usize = 64;

/// Builder for [`IndraOcr`].
#[derive(Debug, Clone)]
pub struct IndraOcrBuilder {
    model_dir: PathBuf,
    math: bool,
    keep_furniture: bool,
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

    /// Sets the ONNX Runtime session configuration (execution providers,
    /// threads) applied to every model.
    pub fn ort_session(mut self, config: OrtSessionConfig) -> Self {
        self.ort_session = Some(config);
        self
    }

    /// Loads the models.
    pub fn build(self) -> Result<IndraOcr, OCRError> {
        let dir = self.model_dir.as_path();
        let mut common = ModelInferenceConfig::new();
        common.ort_session = self.ort_session;
        let meta = read_json(&dir.join("meta.json"))?;
        let vocab = |key: &str| -> Result<Vec<String>, OCRError> {
            let inner: serde_json::Value = serde_json::from_str(meta[key].as_str().unwrap_or(""))
                .map_err(|e| {
                OCRError::config_error(format!("meta.json: {key} is not a vocabulary: {e}"))
            })?;
            string_list(&inner["chars"], &format!("meta.json: {key}.chars"))
        };
        let size = |key: &str| {
            meta[key]
                .as_u64()
                .filter(|&v| v > 0)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    OCRError::config_error(format!("meta.json: {key} must be a positive integer"))
                })
        };
        let optional = |name: &str| -> Result<Option<OrtInfer>, OCRError> {
            if dir.join(name).exists() {
                model(dir, name, &common).map(Some)
            } else {
                Ok(None)
            }
        };
        let cjk = match (optional("script.onnx")?, optional("rec_cjk.onnx")?) {
            (Some(script), Some(rec)) => Some(CjkRecognizer {
                script,
                rec,
                vocab: vocab("vocab_cjk")?,
            }),
            (None, None) => None,
            // Half a pair would silently read Chinese with the Latin recognizer.
            _ => {
                return Err(OCRError::config_error(format!(
                    "{}: script.onnx and rec_cjk.onnx must be present together",
                    dir.display()
                )));
            }
        };
        Ok(IndraOcr {
            det: model(dir, "det.onnx", &common)?,
            rec: model(dir, "rec.onnx", &common)?,
            order: model(dir, "order.onnx", &common)?,
            orient: optional("orient.onnx")?,
            cjk,
            math: if self.math {
                Some(math::MathReader::load(dir, &common)?)
            } else {
                None
            },
            vocab: vocab("vocab")?,
            long_side: size("det_long_side")?,
            rec_height: size("rec_height")?,
            keep_furniture: self.keep_furniture,
        })
    }
}

/// The script classifier and the recognizer for lines with Chinese
/// characters (which also reads Latin).
#[derive(Debug)]
struct CjkRecognizer {
    script: OrtInfer,
    rec: OrtInfer,
    vocab: Vec<String>,
}

/// The Indra OCR pipeline. Build it with [`IndraOcrBuilder`].
#[derive(Debug)]
pub struct IndraOcr {
    det: OrtInfer,
    rec: OrtInfer,
    order: OrtInfer,
    orient: Option<OrtInfer>,
    cjk: Option<CjkRecognizer>,
    math: Option<math::MathReader>,
    vocab: Vec<String>,
    long_side: usize,
    rec_height: usize,
    keep_furniture: bool,
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
    /// Degrees of small tilt straightened before reading; boxes are mapped
    /// back to the page before the straightening.
    pub skew: f32,
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
        images
            .iter()
            .enumerate()
            .map(|(index, image)| {
                let mut page = self.read_page(cv::rgb_to_gray(image))?;
                page.index = index;
                Ok(page)
            })
            .collect()
    }

    fn read_page(&self, gray: GrayImage) -> Result<IndraPage, OCRError> {
        let turns = self.orientation(&gray)?;
        let gray = cv::rot90_cw(&gray, turns);
        let mut skew = estimate_skew(&gray);
        let gray = if skew.abs() >= DESKEW_MIN {
            cv::rotate_u8(&gray, skew)
        } else {
            skew = 0.0;
            gray
        };
        let boxes = self.detect(&gray)?;
        let lines = self.recognize(&gray, &boxes)?;
        let mut page = self.assemble(gray.dimensions(), boxes, lines)?;
        page.rotation = 90 * turns;
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
        Ok(page)
    }

    /// Quarter turns (counter-clockwise) the page content is turned by.
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
        let mut k = 0;
        for (i, v) in p.iter().enumerate() {
            if *v > p[k] {
                k = i;
            }
        }
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
    /// mapped back through the crop. A line under `MIN_CONF` comes back
    /// empty.
    fn recognize(&self, gray: &GrayImage, boxes: &[[f32; 5]]) -> Result<Vec<LineText>, OCRError> {
        let height = self.rec_height;
        let crops: Vec<Crop> = boxes
            .iter()
            .map(|b| recognize::crop_line(gray, [b[0], b[1], b[2], b[3]], height))
            .collect();
        let mut out = vec![LineText::default(); crops.len()];
        let all: Vec<usize> = (0..crops.len()).collect();
        let mut groups = vec![(&self.rec, &self.vocab, all, false)];
        if let Some(cjk) = &self.cjk
            && !crops.is_empty()
        {
            let hanzi = self.classify(&cjk.script, &crops, 0.5)?;
            let (yes, no): (Vec<usize>, Vec<usize>) = (0..crops.len()).partition(|&j| hanzi[j]);
            groups = vec![
                (&self.rec, &self.vocab, no, false),
                (&cjk.rec, &cjk.vocab, yes, true),
            ];
        }
        for (session, vocab, sel, char_tokens) in groups {
            let widths: Vec<usize> = sel.iter().map(|&j| crops[j].width).collect();
            for part in recognize::width_batches(&widths, REC_BATCH, 1.3, 64 * 640) {
                let idx: Vec<usize> = part.iter().map(|&i| sel[i]).collect();
                let batch: Vec<&Crop> = idx.iter().map(|&j| &crops[j]).collect();
                let (data, wm, crop_widths) = recognize::pad_batch(&batch, height);
                let n = batch.len();
                let (ids, conf, lengths, t_len) = run(
                    session,
                    ort::inputs![
                        "x" => Tensor::from_array(([n, 1, height, wm], data)).map_err(input_error)?,
                        "widths" => Tensor::from_array(([n], crop_widths)).map_err(input_error)?,
                    ],
                    |o| {
                        let (shape, ids) = o["ids"].try_extract_tensor::<i64>()?;
                        let (_, conf) = o["conf"].try_extract_tensor::<f32>()?;
                        let (_, lengths) = o["lengths"].try_extract_tensor::<i64>()?;
                        Ok((
                            ids.to_vec(),
                            conf.to_vec(),
                            lengths.to_vec(),
                            shape[1] as usize,
                        ))
                    },
                )?;
                let stride = wm as f64 / t_len.max(1) as f64;
                for (k, &j) in idx.iter().enumerate() {
                    let len = (lengths[k].max(0) as usize).min(t_len);
                    let seq = &ids[k * t_len..k * t_len + len];
                    let pm = &conf[k * t_len..k * t_len + len];
                    let keep: Vec<usize> = (0..len)
                        .filter(|&t| seq[t] != 0 && (t == 0 || seq[t] != seq[t - 1]))
                        .collect();
                    let [x0, y0, x1, _] =
                        [boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]].map(f64::from);
                    let h = (boxes[j][3] as f64 - y0).max(1.0);
                    let ix0 = (x0 - recognize::PAD_X * h).floor();
                    let cw = (x1 + recognize::PAD_X * h).ceil() - ix0;
                    let f = cw / crops[j].width as f64;
                    let conf_mean = if keep.is_empty() {
                        0.0
                    } else {
                        keep.iter().map(|&t| pm[t] as f64).sum::<f64>() / keep.len() as f64
                    };
                    // An id past the vocabulary means meta.json does not
                    // belong to this recognizer.
                    let tokens: Vec<Token> = keep
                        .iter()
                        .map(|&t| {
                            let id = seq[t];
                            let text = usize::try_from(id - 1)
                                .ok()
                                .and_then(|i| vocab.get(i))
                                .ok_or_else(|| {
                                    OCRError::config_error(format!(
                                        "{}: token id {id} is outside its {}-entry vocabulary",
                                        session.model_name(),
                                        vocab.len()
                                    ))
                                })?;
                            Ok(Token {
                                text: text.clone(),
                                x: ix0 + (t as f64 + 0.5) * stride * f,
                            })
                        })
                        .collect::<Result<_, OCRError>>()?;
                    out[j] = if conf_mean >= MIN_CONF {
                        LineText {
                            text: recognize::join_tokens(
                                &tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>(),
                            ),
                            conf: conf_mean,
                            tokens,
                            char_tokens,
                        }
                    } else {
                        LineText {
                            conf: conf_mean,
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
                let mboxes: Vec<[f32; 4]> = flagged
                    .iter()
                    .map(|&j| [boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]])
                    .collect();
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

    /// Per crop, whether a line classifier's probability is above `min`.
    fn classify(&self, model: &OrtInfer, crops: &[Crop], min: f32) -> Result<Vec<bool>, OCRError> {
        let height = self.rec_height;
        let mut out = vec![false; crops.len()];
        for idx in recognize::line_batches(crops) {
            let batch: Vec<&Crop> = idx.iter().map(|&j| &crops[j]).collect();
            let (data, wm, widths) = recognize::pad_batch(&batch, height);
            let n = batch.len();
            let p = run(
                model,
                ort::inputs![
                    "x" => Tensor::from_array(([n, 1, height, wm], data)).map_err(input_error)?,
                    "widths" => Tensor::from_array(([n], widths)).map_err(input_error)?,
                ],
                |o| Ok(o["p"].try_extract_tensor::<f32>()?.1.to_vec()),
            )?;
            for (k, &j) in idx.iter().enumerate() {
                out[j] = p[k] > min;
            }
        }
        Ok(out)
    }

    /// Lines split at wide gaps, reading order, layout blocks and Markdown.
    fn assemble(
        &self,
        (pw, ph): (u32, u32),
        boxes: Vec<[f32; 5]>,
        lines: Vec<LineText>,
    ) -> Result<IndraPage, OCRError> {
        let boxes: Vec<[f32; 4]> = boxes.iter().map(|b| [b[0], b[1], b[2], b[3]]).collect();
        let cuts = recognize::gutter_cuts(&boxes, &lines);
        let (mut nb, mut nl) = (Vec::new(), Vec::new());
        for ((b, line), at) in boxes.iter().zip(&lines).zip(&cuts) {
            for (bx, text, tokens) in recognize::split_line(*b, line, SPLIT_GAP, at) {
                nb.push(bx);
                nl.push(LineText {
                    text,
                    conf: line.conf,
                    tokens,
                    char_tokens: line.char_tokens,
                });
            }
        }
        let (boxes, lines) = (nb, nl);
        let texts: Vec<String> = lines.iter().map(|l| l.text.clone()).collect();
        let round1 = |v: f32| ((v as f64 * 10.0).round_ties_even() / 10.0) as f32;
        let mut page = IndraPage {
            index: 0,
            width: pw,
            height: ph,
            rotation: 0,
            skew: 0.0,
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
        let n = boxes.len();
        let feats = order::line_features(&boxes, (pw, ph), &texts);
        let (a, cls, same) = run(
            &self.order,
            ort::inputs![
                "feats" => Tensor::from_array(([1, n, order::N_FEATS], feats)).map_err(input_error)?,
                "mask" => Tensor::from_array(([1, n], vec![true; n])).map_err(input_error)?,
            ],
            |o| {
                let (_, a) = o["a"].try_extract_tensor::<f32>()?;
                let (_, cls) = o["cls"].try_extract_tensor::<f32>()?;
                let (_, same) = o["same"].try_extract_tensor::<f32>()?;
                Ok((a.to_vec(), cls.to_vec(), same.to_vec()))
            },
        )?;
        let a = order::Square { n, data: &a };
        let same = order::Square { n, data: &same };
        let reading = order::decode_blocks_order(&a, &same, &boxes, &cls);
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
                recognize::split_line(*b, l, CELL_GAP, &[])
                    .into_iter()
                    .map(|(bx, text, _)| (bx, text))
                    .collect()
            })
            .collect();
        let blocks = layout::page_blocks(
            &boxes,
            &texts,
            &reading,
            &cls,
            &same,
            self.keep_furniture,
            &cells,
        );
        page.markdown = layout::page_markdown(&blocks);
        page.blocks = blocks
            .into_iter()
            .map(|block| {
                let boxes = &boxes;
                let bx = |k: usize| block.lines.iter().map(move |&i| boxes[i][k]);
                IndraBlock {
                    label: layout::label(block.class).to_string(),
                    bbox: [
                        round1(bx(0).fold(f32::INFINITY, f32::min)),
                        round1(bx(1).fold(f32::INFINITY, f32::min)),
                        round1(bx(2).fold(f32::NEG_INFINITY, f32::max)),
                        round1(bx(3).fold(f32::NEG_INFINITY, f32::max)),
                    ],
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

/// Degrees (counter-clockwise) that straighten the text rows of a page: the
/// angle whose horizontal ink profile changes most sharply row to row, on an
/// 800 px thumbnail, searched in 0.5 degree steps and then 0.1 degree steps.
fn estimate_skew(gray: &GrayImage) -> f64 {
    const SIDE: f64 = 800.0;
    let (w, h) = (gray.width() as f64, gray.height() as f64);
    let s = SIDE / w.max(h);
    let (sw, sh) = (
        ((w * s).round_ties_even() as usize).max(1),
        ((h * s).round_ties_even() as usize).max(1),
    );
    // A one-row thumbnail has no row profile to sharpen (every angle would
    // score zero and the first, -5.5 degrees, would win).
    if sh < 2 {
        return 0.0;
    }
    let small = cv::resize(gray, sw as u32, sh as u32, cv::Interp::Area);
    let ink: Vec<f32> = small
        .as_raw()
        .iter()
        .map(|&v| (255.0 - v as f32) / 255.0)
        .collect();
    let mean = ink.iter().map(|&v| v as f64).sum::<f64>() / ink.len() as f64;
    // A blank page, or a flat one with no structure at all, has no rows to
    // straighten.
    if mean < 0.003 || small.as_raw().iter().all(|&v| v == small.as_raw()[0]) {
        return 0.0;
    }
    let sharpness = |a: f64| -> f64 {
        let r = cv::rotate_f32(&ink, sw, sh, a);
        let rows: Vec<f32> = r
            .chunks_exact(sw)
            .map(|row| row.iter().map(|&v| v as f64).sum::<f64>() as f32)
            .collect();
        rows.windows(2)
            .map(|p| {
                let d = p[1] - p[0];
                (d * d) as f64
            })
            .sum()
    };
    let argmax = |angles: Vec<f64>| -> f64 {
        let scores: Vec<f64> = angles.par_iter().map(|&a| sharpness(a)).collect();
        let mut best = 0;
        for (i, v) in scores.iter().enumerate() {
            if *v > scores[best] {
                best = i;
            }
        }
        angles[best]
    };
    let coarse = argmax(arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 0.5));
    argmax(arange(coarse - 0.5, coarse + 0.5 + 1e-6, 0.1))
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
}
