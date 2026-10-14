//! Math lines: a classifier flags lines that hold math, and an attention
//! decoder reads a taller crop of each as text with inline LaTeX
//! (`we have $x _ { i } \leq 1$`), one token per step with cached keys and
//! values.

use super::recognize::{Crop, crop_line, line_batches, pad_batch, width_batches};
use super::{input_error, model, run};
use image::GrayImage;
use oar_ocr_core::core::OCRError;
use oar_ocr_core::core::config::ModelInferenceConfig;
use oar_ocr_core::core::inference::OrtInfer;
use ort::value::{Tensor, TensorRef};
use std::path::Path;

/// Classifier probability above which a line is read as math.
const MATH_MIN: f32 = 0.5;
/// A math reading replaces a line only at this mean token probability.
pub(super) const MATH_MIN_CONF: f64 = 0.9;
/// Decoding stops after 8 tokens once the mean probability is this far
/// below `MATH_MIN_CONF`: the reading would be rejected anyway.
const EARLY_STOP_MARGIN: f64 = 0.1;
const BOS: i64 = 1;
const EOS: i64 = 2;
const N_SPECIAL: i64 = 4;

/// The math models and their vocabulary.
#[derive(Debug)]
pub(super) struct MathReader {
    cls: OrtInfer,
    enc: OrtInfer,
    init: OrtInfer,
    step: OrtInfer,
    height: usize,
    max_tokens: usize,
    layers: usize,
    heads: usize,
    head_dim: usize,
    /// Token strings by id; the special ids map to empty strings.
    itos: Vec<String>,
}

impl MathReader {
    pub(super) fn load(dir: &Path, common: &ModelInferenceConfig) -> Result<Self, OCRError> {
        let meta = super::read_json(&dir.join("math_meta.json"))?;
        let cfg = &meta["config"];
        let get = |key: &str| {
            cfg[key]
                .as_u64()
                .filter(|&v| v > 0)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    OCRError::config_error(format!(
                        "math_meta.json: config.{key} must be a positive integer"
                    ))
                })
        };
        let mut itos = vec![String::new(); N_SPECIAL as usize];
        itos.extend(super::string_list(
            &meta["tokens"],
            "math_meta.json: tokens",
        )?);
        let heads = get("dec_heads")?;
        let (channels, max_tokens) = (get("channels")?, get("max_tokens")?);
        if channels % heads != 0 || max_tokens < 2 {
            return Err(OCRError::config_error(
                "math_meta.json: config.channels must be a multiple of dec_heads, \
                 and max_tokens at least 2",
            ));
        }
        Ok(Self {
            cls: model(dir, "mathcls.onnx", common)?,
            enc: model(dir, "math_enc.onnx", common)?,
            init: model(dir, "math_dec_init.onnx", common)?,
            step: model(dir, "math_dec_step.onnx", common)?,
            height: get("height")?,
            max_tokens,
            layers: get("dec_layers")?,
            heads,
            head_dim: channels / heads,
            itos,
        })
    }

    /// Per line crop (the recognizer's crop): does it hold math?
    pub(super) fn flags(&self, crops: &[Crop], height: usize) -> Result<Vec<bool>, OCRError> {
        let mut out = vec![false; crops.len()];
        for idx in line_batches(crops) {
            let batch: Vec<&Crop> = idx.iter().map(|&j| &crops[j]).collect();
            let (data, wm, widths) = pad_batch(&batch, height);
            let b = batch.len();
            let p = run(
                &self.cls,
                ort::inputs![
                    "x" => Tensor::from_array(([b, 1, height, wm], data)).map_err(input_error)?,
                    "widths" => Tensor::from_array(([b], widths)).map_err(input_error)?,
                ],
                |o| Ok(o["p"].try_extract_tensor::<f32>()?.1.to_vec()),
            )?;
            for (k, &j) in idx.iter().enumerate() {
                out[j] = p[k] > MATH_MIN;
            }
        }
        Ok(out)
    }

    /// `(text with inline LaTeX, mean token probability)` per line box.
    pub(super) fn read(
        &self,
        gray: &GrayImage,
        boxes: &[[f32; 4]],
    ) -> Result<Vec<(String, f64)>, OCRError> {
        let crops: Vec<Crop> = boxes
            .iter()
            .map(|&b| crop_line(gray, b, self.height))
            .collect();
        let mut out = vec![(String::new(), 0.0); crops.len()];
        // Few, wide batches: each is one decoding loop, and a step costs about
        // the same for 8 lines as for 40.
        let widths: Vec<usize> = crops.iter().map(|c| c.width).collect();
        for idx in width_batches(&widths, 64, 4.0, 16 * 64 * 640) {
            let batch: Vec<&Crop> = idx.iter().map(|&j| &crops[j]).collect();
            for (k, reading) in self.decode_batch(&batch)?.into_iter().enumerate() {
                out[idx[k]] = reading;
            }
        }
        Ok(out)
    }

    fn decode_batch(&self, batch: &[&Crop]) -> Result<Vec<(String, f64)>, OCRError> {
        let b = batch.len();
        let (data, wm, widths) = pad_batch(batch, self.height);
        let (memory, mem_shape, mask, mask_shape) = run(
            &self.enc,
            ort::inputs![
                "x" => Tensor::from_array(([b, 1, self.height, wm], data)).map_err(input_error)?,
                "widths" => Tensor::from_array(([b], widths)).map_err(input_error)?,
            ],
            |o| {
                let (ms, m) = o["memory"].try_extract_tensor::<f32>()?;
                let (ks, k) = o["mask"].try_extract_tensor::<f32>()?;
                Ok((m.to_vec(), ms.to_vec(), k.to_vec(), ks.to_vec()))
            },
        )?;
        let (ck, cv, cross_shape) = run(
            &self.init,
            ort::inputs!["memory" => TensorRef::from_array_view((mem_shape, memory.as_slice())).map_err(input_error)?],
            |o| {
                let (s, k) = o["cross_k"].try_extract_tensor::<f32>()?;
                let (_, v) = o["cross_v"].try_extract_tensor::<f32>()?;
                Ok((k.to_vec(), v.to_vec(), s.to_vec()))
            },
        )?;
        let mut self_shape = vec![
            self.layers as i64,
            b as i64,
            self.heads as i64,
            0,
            self.head_dim as i64,
        ];
        let (mut sk, mut sv) = (Vec::<f32>::new(), Vec::<f32>::new());
        let mut tok = vec![BOS; b];
        let mut seqs: Vec<Vec<i64>> = vec![Vec::new(); b];
        let mut probs: Vec<Vec<f64>> = vec![Vec::new(); b];
        let mut done = vec![false; b];
        for t in 0..self.max_tokens.saturating_sub(1) {
            let (logits, nk, nv, nshape) = run(
                &self.step,
                ort::inputs![
                    "token" => Tensor::from_array(([b], tok.clone())).map_err(input_error)?,
                    "pos" => Tensor::from_array(((), vec![t as i64])).map_err(input_error)?,
                    "self_k" => TensorRef::from_array_view((self_shape.clone(), sk.as_slice())).map_err(input_error)?,
                    "self_v" => TensorRef::from_array_view((self_shape.clone(), sv.as_slice())).map_err(input_error)?,
                    "cross_k" => TensorRef::from_array_view((cross_shape.clone(), ck.as_slice())).map_err(input_error)?,
                    "cross_v" => TensorRef::from_array_view((cross_shape.clone(), cv.as_slice())).map_err(input_error)?,
                    "mask" => TensorRef::from_array_view((mask_shape.clone(), mask.as_slice())).map_err(input_error)?,
                ],
                |o| {
                    let (_, l) = o["logits"].try_extract_tensor::<f32>()?;
                    let (s, k) = o["new_k"].try_extract_tensor::<f32>()?;
                    let (_, v) = o["new_v"].try_extract_tensor::<f32>()?;
                    Ok((l.to_vec(), k.to_vec(), v.to_vec(), s.to_vec()))
                },
            )?;
            (sk, sv, self_shape) = (nk, nv, nshape);
            let n_vocab = logits.len() / b;
            if n_vocab != self.itos.len() {
                return Err(OCRError::config_error(format!(
                    "math_dec_step.onnx scores {n_vocab} tokens, math_meta.json lists {}",
                    self.itos.len()
                )));
            }
            for i in 0..b {
                let row = &logits[i * n_vocab..(i + 1) * n_vocab];
                let mut arg = 0;
                for (j, v) in row.iter().enumerate() {
                    if *v > row[arg] {
                        arg = j;
                    }
                }
                tok[i] = arg as i64;
                if done[i] {
                    continue;
                }
                // Softmax probability of the argmax token.
                let max = row[arg];
                let top = 1.0 / row.iter().map(|v| (v - max).exp()).sum::<f32>();
                probs[i].push(top as f64);
                let mean = probs[i].iter().sum::<f64>() / probs[i].len() as f64;
                if tok[i] == EOS
                    || (probs[i].len() >= 8 && mean < MATH_MIN_CONF - EARLY_STOP_MARGIN)
                {
                    done[i] = true;
                } else if tok[i] >= N_SPECIAL {
                    seqs[i].push(tok[i]);
                }
            }
            if done.iter().all(|&d| d) {
                break;
            }
        }
        Ok(seqs
            .iter()
            .zip(&probs)
            .map(|(seq, p)| {
                let tokens: Vec<&str> = seq
                    .iter()
                    .map(|&i| self.itos[i as usize].as_str())
                    .collect();
                let conf = if p.is_empty() {
                    0.0
                } else {
                    p.iter().sum::<f64>() / p.len() as f64
                };
                (detokenize(&tokens), conf)
            })
            .collect())
    }
}

/// Tokens -> text; inside math (between `$` tokens) they join with spaces.
fn detokenize(tokens: &[&str]) -> String {
    let mut out = String::new();
    let mut math: Vec<&str> = Vec::new();
    let mut in_math = false;
    for &t in tokens {
        if t == "$" {
            if in_math {
                out.push_str(&format!("${}$", math.join(" ")));
                math.clear();
            }
            in_math = !in_math;
        } else if in_math {
            math.push(t);
        } else {
            out.push_str(t);
        }
    }
    if in_math && !math.is_empty() {
        out.push_str(&format!("${}$", math.join(" ")));
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
