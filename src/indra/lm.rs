//! Beam search with a token n-gram language model, for lines a recognizer
//! is unsure of.
//!
//! A recognizer reads greedily, one best class per step. On a line under
//! the gate confidence, a CTC prefix beam search keeps several readings and
//! the language model breaks close calls between them:
//! `log P_ctc + alpha * log P_lm + beta * tokens`.
//!
//! `lm.npz` (and `lm_<group>.npz`) holds a backoff table: per history
//! length `n`, sorted keys of the (history, token) pairs seen in training
//! with their log-probabilities (`k<n>`, `v<n>`), and sorted history keys
//! with their backoff weights (`hk<n>`, `hb<n>`). A key packs tokens
//! `bits` bits each (0 = line start, `t + 1` = token `t`).

use oar_ocr_core::core::OCRError;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Bits per token unless the archive says more.
const BITS: u32 = 11;
/// The history length goes in the top 3 bits of a key.
const TAG_SHIFT: u32 = 61;
/// A step whose blank probability is above this only extends every beam
/// with the blank.
const BLANK_SKIP: f64 = 0.999;

/// Sorted keys, their values, and the largest value of each history length.
type Table = (Vec<u64>, Vec<f32>, Vec<f32>);

/// An n-gram language model.
pub(super) struct NgramLm {
    order: usize,
    vocab_size: usize,
    bits: u32,
    /// Every history length in one sorted table, the length in the top bits.
    p_keys: Vec<u64>,
    p_vals: Vec<f32>,
    h_keys: Vec<u64>,
    h_vals: Vec<f32>,
    /// An upper bound of every log-probability [`Self::logp`] returns.
    max_logp: f64,
}

impl std::fmt::Debug for NgramLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramLm")
            .field("order", &self.order)
            .field("vocab_size", &self.vocab_size)
            .field("entries", &self.p_keys.len())
            .finish()
    }
}

impl NgramLm {
    pub(super) fn load(path: &Path) -> Result<Self, OCRError> {
        let fail = |what: String| OCRError::config_error(format!("{}: {what}", path.display()));
        let file = std::fs::File::open(path).map_err(|e| {
            OCRError::model_load_error(path, "cannot open", Some("check the model folder"), Some(e))
        })?;
        let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
            .map_err(|e| fail(format!("not an .npz archive: {e}")))?;
        let mut array = |name: &str| -> Result<Option<Npy>, OCRError> {
            let mut entry = match zip.by_name(&format!("{name}.npy")) {
                Ok(entry) => entry,
                Err(zip::result::ZipError::FileNotFound) => return Ok(None),
                Err(e) => return Err(fail(format!("{name}: {e}"))),
            };
            let mut bytes = Vec::with_capacity(entry.size() as usize);
            entry
                .read_to_end(&mut bytes)
                .map_err(|e| fail(format!("{name}: {e}")))?;
            Npy::parse(&bytes)
                .map(Some)
                .ok_or_else(|| fail(format!("{name} is not a 1-D little-endian .npy array")))
        };
        let need = |a: Option<Npy>, name: &str| a.ok_or_else(|| fail(format!("no {name}")));
        let scalar = |a: Npy, name: &str| {
            a.int()
                .filter(|v| v.len() == 1 && v[0] > 0)
                .map(|v| v[0] as usize)
                .ok_or_else(|| fail(format!("{name} must be a positive integer")))
        };
        let order = scalar(need(array("order")?, "order")?, "order")?;
        let vocab_size = scalar(need(array("vocab_size")?, "vocab_size")?, "vocab_size")?;
        let bits = match array("bits")? {
            Some(a) => scalar(a, "bits")? as u32,
            None => BITS,
        };
        if order > 7 || order as u32 * bits > TAG_SHIFT {
            return Err(fail(format!(
                "order {order} x {bits} bits do not fit a key"
            )));
        }
        // Every history length in one sorted table; also the largest value
        // of each length.
        let mut table = |key: &str, val: &str| -> Result<Table, OCRError> {
            let (mut pairs, mut max) = (Vec::new(), Vec::new());
            for n in 0..order {
                let name = format!("{key}{n}");
                let keys = need(array(&name)?, &name)?
                    .uint()
                    .ok_or_else(|| fail(format!("{key}{n} must be uint64")))?;
                let name = format!("{val}{n}");
                let vals = need(array(&name)?, &name)?
                    .float()
                    .filter(|v| v.len() == keys.len())
                    .ok_or_else(|| fail(format!("{val}{n} must be float32 like {key}{n}")))?;
                max.extend(vals.iter().copied().reduce(f32::max));
                let tag = (n as u64) << TAG_SHIFT;
                pairs.extend(keys.into_iter().map(|k| k | tag).zip(vals));
            }
            pairs.sort_unstable_by_key(|p| p.0);
            let (keys, vals) = pairs.into_iter().unzip();
            Ok((keys, vals, max))
        };
        let (p_keys, p_vals, p_max) = table("k", "v")?;
        let (h_keys, h_vals, h_max) = table("hk", "hb")?;
        // The largest log-probability plus every positive backoff weight a
        // back-off can add.
        let top = p_max
            .iter()
            .map(|&v| v as f64)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(-(vocab_size as f64).ln());
        let backoff: f64 = h_max.iter().map(|&v| (v as f64).max(0.0)).sum();
        Ok(Self {
            order,
            vocab_size,
            bits,
            p_keys,
            p_vals,
            h_keys,
            h_vals,
            max_logp: top + backoff,
        })
    }

    fn find(keys: &[u64], vals: &[f32], key: u64) -> Option<f64> {
        keys.binary_search(&key).ok().map(|i| vals[i] as f64)
    }

    /// The log-probability of a (history, token) pair packed in one key: the
    /// last `order - 1` history tokens, then the token. The value comes from
    /// the longest seen history that saw the token, plus the backoff weights
    /// of the longer seen histories; with none, all backoffs and a uniform
    /// floor.
    fn logp(&self, key: u64) -> f64 {
        let (n_max, bits) = (self.order, self.bits);
        let token = key & ((1 << bits) - 1);
        let hist = key >> bits;
        let mut hb = [0f64; 8];
        let mut pv = [None; 8];
        // m: the longest history length seen, all shorter ones seen too.
        let mut m = 0;
        let mut counting = true;
        for n in 0..n_max {
            let tag = (n as u64) << TAG_SHIFT;
            let h = hist & ((1u64 << (bits * n as u32)) - 1);
            let found = Self::find(&self.h_keys, &self.h_vals, h | tag);
            hb[n] = found.unwrap_or(0.0);
            if n > 0 {
                counting &= found.is_some();
                if counting {
                    m = n;
                }
            }
            pv[n] = Self::find(&self.p_keys, &self.p_vals, (h << bits) | token | tag);
        }
        // Backoff weights of the histories n..=m, summed from the longest
        // as numpy's reversed cumulative sum does.
        let mut suffix = [0f64; 9];
        for n in (0..n_max).rev() {
            let x = if n <= m { hb[n] } else { 0.0 };
            suffix[n] = if n == n_max - 1 { x } else { suffix[n + 1] + x };
        }
        match (0..=m).rev().find_map(|n| pv[n].map(|v| (n, v))) {
            Some((n, v)) => suffix[n + 1] + v,
            None => suffix[0] - (self.vocab_size as f64).ln(),
        }
    }
}

/// A 1-D array of an `.npy` file.
enum Npy {
    Int(Vec<i64>),
    Uint(Vec<u64>),
    Float(Vec<f32>),
}

impl Npy {
    fn parse(bytes: &[u8]) -> Option<Self> {
        let rest = bytes.strip_prefix(b"\x93NUMPY")?;
        let (&major, rest) = rest.split_at_checked(2).map(|(v, r)| (&v[0], r))?;
        let (len, rest) = if major == 1 {
            let (l, r) = rest.split_at_checked(2)?;
            (u16::from_le_bytes([l[0], l[1]]) as usize, r)
        } else {
            let (l, r) = rest.split_at_checked(4)?;
            (u32::from_le_bytes([l[0], l[1], l[2], l[3]]) as usize, r)
        };
        let (header, data) = rest.split_at_checked(len)?;
        let header = std::str::from_utf8(header).ok()?;
        if header.contains("'fortran_order': True") {
            return None;
        }
        let descr = header.split("'descr':").nth(1)?.split('\'').nth(1)?;
        let shape = header.split("'shape':").nth(1)?;
        let shape = &shape[shape.find('(')? + 1..shape.find(')')?];
        let dims: Vec<usize> = shape
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().ok())
            .collect::<Option<_>>()?;
        if dims.len() > 1 {
            return None;
        }
        let n = dims.first().copied().unwrap_or(1);
        let width = match descr {
            "<i8" | "<u8" => 8,
            "<f4" => 4,
            _ => return None,
        };
        let data = data.get(..n * width)?;
        Some(match descr {
            "<i8" => Self::Int(
                data.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&c| i64::from_le_bytes(c))
                    .collect(),
            ),
            "<u8" => Self::Uint(
                data.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&c| u64::from_le_bytes(c))
                    .collect(),
            ),
            _ => Self::Float(
                data.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c))
                    .collect(),
            ),
        })
    }

    fn int(self) -> Option<Vec<i64>> {
        match self {
            Self::Int(v) => Some(v),
            _ => None,
        }
    }

    fn uint(self) -> Option<Vec<u64>> {
        match self {
            Self::Uint(v) => Some(v),
            _ => None,
        }
    }

    fn float(self) -> Option<Vec<f32>> {
        match self {
            Self::Float(v) => Some(v),
            _ => None,
        }
    }
}

/// `log(exp(a) + exp(b))`.
fn lse(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// A beam: log-probabilities of the prefix ending in a blank and in a
/// non-blank, its LM score, the step of each token, and its last
/// `order - 1` tokens packed.
#[derive(Clone)]
struct Beam {
    b: f64,
    nb: f64,
    s: f64,
    steps: Vec<usize>,
    hist: u64,
}

impl Beam {
    fn score(&self) -> f64 {
        lse(self.b, self.nb) + self.s
    }
}

/// Beams in insertion order, as the reference's dict keeps them (ties in
/// the ranking keep this order).
#[derive(Default)]
struct Beams {
    prefixes: Vec<Vec<i64>>,
    beams: Vec<Beam>,
    index: HashMap<Vec<i64>, usize>,
}

impl Beams {
    /// Adds probability mass to a prefix; a new prefix takes `beam`'s LM
    /// score, steps and history, an existing one keeps its own.
    fn add(&mut self, prefix: &[i64], b: f64, nb: f64, beam: impl FnOnce() -> Beam) {
        if let Some(&i) = self.index.get(prefix) {
            let old = &mut self.beams[i];
            old.b = lse(old.b, b);
            old.nb = lse(old.nb, nb);
        } else {
            let mut new = beam();
            new.b = b;
            new.nb = nb;
            self.index.insert(prefix.to_vec(), self.prefixes.len());
            self.prefixes.push(prefix.to_vec());
            self.beams.push(new);
        }
    }
}

/// A line's language-model inputs: per step, the blank log-probability and
/// the `k` best classes (blank included) with their log-probabilities.
pub(super) struct Steps<'a> {
    pub blank: &'a [f32],
    pub topv: &'a [f32],
    pub topi: &'a [i64],
    pub k: usize,
}

/// Language-model weights: `alpha` weighs the LM, `beta` rewards each token,
/// and a class under `cand_floor` probability at a step is not tried there.
#[derive(Clone, Copy, Debug)]
pub(super) struct Weights {
    pub alpha: f64,
    pub beta: f64,
    pub beam: usize,
    pub cand_floor: f64,
}

/// CTC prefix beam search over a line's steps: the token ids and the step
/// where each was first emitted.
pub(super) fn beam_search(steps: &Steps, lm: &NgramLm, w: Weights) -> (Vec<i64>, Vec<usize>) {
    let bits = lm.bits;
    let hmask = (1u64 << (bits * (lm.order as u32 - 1))) - 1;
    let log_skip = BLANK_SKIP.ln();
    let log_floor = w.cand_floor.ln();
    let reach = w.beta + w.alpha * lm.max_logp;
    let mut beams = Beams::default();
    beams.add(&[], 0.0, f64::NEG_INFINITY, || Beam {
        b: 0.0,
        nb: f64::NEG_INFINITY,
        s: 0.0,
        steps: Vec::new(),
        hist: 0,
    });
    for t in 0..steps.blank.len() {
        let b0 = steps.blank[t] as f64;
        if b0 > log_skip {
            for beam in &mut beams.beams {
                beam.b = lse(beam.b, beam.nb) + b0;
                beam.nb = f64::NEG_INFINITY;
            }
            continue;
        }
        let cands: Vec<(i64, f64)> = (0..steps.k)
            .map(|j| {
                (
                    steps.topi[t * steps.k + j],
                    steps.topv[t * steps.k + j] as f64,
                )
            })
            .filter(|&(c, v)| c != 0 && v >= log_floor)
            .collect();
        // Every beam's own entries first (the blank, and a repeat that
        // collapses): with a full beam the lowest of them is a floor that a
        // new prefix must be able to reach.
        let mut next = Beams::default();
        for (prefix, beam) in beams.prefixes.iter().zip(&beams.beams) {
            next.add(prefix, lse(beam.b, beam.nb) + b0, f64::NEG_INFINITY, || {
                beam.clone()
            });
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                if Some(c) == last {
                    next.add(prefix, f64::NEG_INFINITY, beam.nb + lp, || beam.clone());
                }
            }
        }
        let floor = if beams.beams.len() >= w.beam {
            next.beams
                .iter()
                .map(Beam::score)
                .fold(f64::INFINITY, f64::min)
        } else {
            f64::NEG_INFINITY
        };
        // (beam, class, log p of the new token)
        let mut ext: Vec<(usize, i64, f64)> = Vec::new();
        for (i, (prefix, beam)) in beams.prefixes.iter().zip(&beams.beams).enumerate() {
            let total = lse(beam.b, beam.nb);
            let last = prefix.last().copied();
            for &(c, lp) in &cands {
                // After a blank, the same class is a new token.
                let new_from = if Some(c) == last {
                    beam.b + lp
                } else {
                    total + lp
                };
                if new_from == f64::NEG_INFINITY {
                    continue;
                }
                let mut q = prefix.clone();
                q.push(c);
                if next.index.contains_key(&q) {
                    // An existing beam: it merges and keeps its LM score.
                    next.add(&q, f64::NEG_INFINITY, new_from, || beam.clone());
                } else if new_from + beam.s + reach >= floor - 1e-9 {
                    ext.push((i, c, new_from));
                }
            }
        }
        for (i, c, new_from) in ext {
            let beam = &beams.beams[i];
            let key = (beam.hist << bits) | (c as u64 + 1);
            let lmp = lm.logp(key);
            let mut q = beams.prefixes[i].clone();
            q.push(c);
            next.add(&q, f64::NEG_INFINITY, new_from, || {
                let mut steps = beam.steps.clone();
                steps.push(t);
                Beam {
                    b: 0.0,
                    nb: 0.0,
                    s: beam.s + w.alpha * lmp + w.beta,
                    steps,
                    hist: key & hmask,
                }
            });
        }
        // The best beams; a stable sort keeps insertion order among ties.
        let mut rank: Vec<usize> = (0..next.beams.len()).collect();
        rank.sort_by(|&a, &b| {
            next.beams[b]
                .score()
                .partial_cmp(&next.beams[a].score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rank.truncate(w.beam);
        let mut kept = Beams::default();
        for i in rank {
            let beam = next.beams[i].clone();
            kept.add(&next.prefixes[i], beam.b, beam.nb, || beam.clone());
        }
        beams = kept;
    }
    let mut best = 0;
    for (i, beam) in beams.beams.iter().enumerate() {
        if beam.score() > beams.beams[best].score() {
            best = i;
        }
    }
    (
        beams.prefixes[best].clone(),
        beams.beams[best].steps.clone(),
    )
}
