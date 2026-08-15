//! Text decoding utilities for OCR (Optical Character Recognition) systems.
//!
//! This module provides implementations for decoding text recognition results,
//! particularly focused on CTC (Connectionist Temporal Classification) decoding.
//! It includes structures and methods for converting model predictions into
//! readable text strings with confidence scores.

use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Decoded batch outputs along with positional metadata.
pub type PositionedDecodeResult = (
    Vec<String>,
    Vec<f32>,
    Vec<Vec<f32>>,
    Vec<Vec<usize>>,
    Vec<usize>,
);

/// Compact result of reducing CTC logits over the vocabulary dimension.
///
/// Unlike the original `(batch, time, vocab)` logits, this only retains one
/// token index and confidence per timestep, so it is cheap to move beyond the
/// lifetime of an ONNX Runtime output buffer.
#[derive(Debug, PartialEq)]
pub(crate) struct CTCArgmaxOutput {
    batch_size: usize,
    sequence_length: usize,
    indices: Vec<usize>,
    probabilities: Vec<f32>,
}

static ALPHANUMERIC_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[a-zA-Z0-9 :*./%+-]").expect("static regex: alphanumeric decoder pattern")
});

/// Argmax over a 1-D prediction row, returning `(index, value)`.
///
/// Contiguous rows (the common row-major case for the per-timestep logits) are
/// routed through the SIMD kernel in [`crate::processors::simd`]; a scalar scan
/// handles non-contiguous views. Tie-breaking matches [`Iterator::max_by`]
/// (the last maximal index wins), so decoded output is unchanged.
#[inline]
fn argmax_row(row: ndarray::ArrayView1<f32>) -> Option<(usize, f32)> {
    match row.as_slice() {
        Some(slice) => crate::processors::simd::argmax(slice),
        None => row
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)),
    }
}

/// A base decoder for text recognition that handles character mapping and basic decoding operations.
///
/// This struct is responsible for converting model predictions into readable text strings.
/// It maintains a character dictionary for mapping indices to characters and provides
/// methods for decoding text with optional duplicate removal and confidence scoring.
///
/// # Fields
/// * `reverse` - Flag indicating whether to reverse the text output
/// * `dict` - A mapping from characters to their indices in the character list
/// * `character` - A list of characters in the vocabulary, indexed by their position
pub struct BaseRecLabelDecode {
    reverse: bool,
    dict: HashMap<char, usize>,
    character: Vec<char>,
}

impl BaseRecLabelDecode {
    /// Creates a new `BaseRecLabelDecode` instance.
    ///
    /// # Arguments
    /// * `character_str` - An optional string containing the character vocabulary.
    ///   If None, a default alphanumeric character set is used.
    /// * `use_space_char` - Whether to include a space character in the vocabulary.
    ///
    /// # Returns
    /// A new `BaseRecLabelDecode` instance.
    pub fn new(character_str: Option<&str>, use_space_char: bool) -> Self {
        let mut character_list: Vec<char> = if let Some(chars) = character_str {
            chars.chars().collect()
        } else {
            "0123456789abcdefghijklmnopqrstuvwxyz".chars().collect()
        };

        if use_space_char {
            character_list.push(' ');
        }

        character_list = Self::add_special_char(character_list);

        let mut dict = HashMap::new();
        for (i, &char) in character_list.iter().enumerate() {
            dict.insert(char, i);
        }

        Self {
            reverse: false,
            dict,
            character: character_list,
        }
    }

    /// Creates a new `BaseRecLabelDecode` instance from a list of strings.
    ///
    /// # Arguments
    /// * `character_list` - An optional slice of strings containing the character vocabulary.
    ///   Only the first character of each string is used. If None, a default alphanumeric
    ///   character set is used.
    /// * `use_space_char` - Whether to include a space character in the vocabulary.
    ///
    /// # Returns
    /// A new `BaseRecLabelDecode` instance.
    pub fn from_string_list(character_list: Option<&[String]>, use_space_char: bool) -> Self {
        let mut chars: Vec<char> = if let Some(list) = character_list {
            list.iter().filter_map(|s| s.chars().next()).collect()
        } else {
            "0123456789abcdefghijklmnopqrstuvwxyz".chars().collect()
        };

        if use_space_char {
            chars.push(' ');
        }

        chars = Self::add_special_char(chars);

        let mut dict = HashMap::new();
        for (i, &char) in chars.iter().enumerate() {
            dict.insert(char, i);
        }

        Self {
            reverse: false,
            dict,
            character: chars,
        }
    }

    /// Reverses the alphanumeric parts of a string while keeping non-alphanumeric parts in place.
    ///
    /// # Arguments
    /// * `pred` - The input string to process.
    ///
    /// # Returns
    /// A new string with alphanumeric parts reversed.
    fn pred_reverse(&self, pred: &str) -> String {
        let mut pred_re = Vec::new();
        let mut c_current = String::new();

        for c in pred.chars() {
            if !ALPHANUMERIC_REGEX.is_match(&c.to_string()) {
                if !c_current.is_empty() {
                    pred_re.push(c_current.clone());
                    c_current.clear();
                }
                pred_re.push(c.to_string());
            } else {
                c_current.push(c);
            }
        }

        if !c_current.is_empty() {
            pred_re.push(c_current);
        }

        pred_re.reverse();
        pred_re.join("")
    }

    /// Adds special characters to the character list.
    ///
    /// This is a placeholder method that currently just returns the input list unchanged.
    /// It can be overridden in subclasses to add special characters.
    ///
    /// # Arguments
    /// * `character_list` - The input character list.
    ///
    /// # Returns
    /// The character list with any special characters added.
    fn add_special_char(character_list: Vec<char>) -> Vec<char> {
        character_list
    }

    /// Gets a list of token indices that should be ignored during decoding.
    ///
    /// # Returns
    /// A vector containing the indices of tokens to ignore.
    fn get_ignored_tokens(&self) -> Vec<usize> {
        vec![self.get_blank_idx()]
    }

    /// Decodes model predictions into text strings with confidence scores.
    ///
    /// # Arguments
    /// * `text_index` - A slice of vectors containing the predicted character indices.
    /// * `text_prob` - An optional slice of vectors containing the prediction probabilities.
    /// * `is_remove_duplicate` - Whether to remove consecutive duplicate characters.
    ///
    /// # Returns
    /// A vector of tuples, each containing a decoded text string and its confidence score.
    pub fn decode(
        &self,
        text_index: &[Vec<usize>],
        text_prob: Option<&[Vec<f32>]>,
        is_remove_duplicate: bool,
    ) -> Vec<(String, f32)> {
        let mut result_list = Vec::new();
        let ignored_tokens = self.get_ignored_tokens();

        for (batch_idx, indices) in text_index.iter().enumerate() {
            let mut selection = vec![true; indices.len()];

            if is_remove_duplicate && indices.len() > 1 {
                for i in 1..indices.len() {
                    if indices[i] == indices[i - 1] {
                        selection[i] = false;
                    }
                }
            }

            for &ignored_token in &ignored_tokens {
                for (i, &idx) in indices.iter().enumerate() {
                    if idx == ignored_token {
                        selection[i] = false;
                    }
                }
            }

            let char_list: Vec<char> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| selection[*i])
                .filter_map(|(_, &text_id)| self.character.get(text_id).copied())
                .collect();

            let conf_list: Vec<f32> = if let Some(probs) = text_prob {
                if batch_idx < probs.len() {
                    probs[batch_idx]
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i < selection.len() && selection[*i])
                        .map(|(_, &prob)| prob)
                        .collect()
                } else {
                    vec![1.0; char_list.len()]
                }
            } else {
                vec![1.0; char_list.len()]
            };

            let conf_list = if conf_list.is_empty() {
                vec![0.0]
            } else {
                conf_list
            };

            let mut text: String = char_list.iter().collect();

            if self.reverse {
                text = self.pred_reverse(&text);
            }

            let mean_conf = conf_list.iter().sum::<f32>() / conf_list.len() as f32;
            result_list.push((text, mean_conf));
        }

        result_list
    }

    /// Applies the decoder to a tensor of model predictions.
    ///
    /// # Arguments
    /// * `pred` - A 3D tensor containing the model predictions. Accepts any
    ///   `ndarray` storage (owned `Array3<f32>` or a zero-copy `ArrayView3<f32>`).
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of decoded text strings
    /// * A vector of confidence scores for each text string
    pub fn apply(&self, pred: &ndarray::Array3<f32>) -> (Vec<String>, Vec<f32>) {
        if pred.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let batch_size = pred.shape()[0];
        let mut all_texts = Vec::new();
        let mut all_scores = Vec::new();

        for batch_idx in 0..batch_size {
            let preds = pred.index_axis(ndarray::Axis(0), batch_idx);

            let mut sequence_idx = Vec::new();
            let mut sequence_prob = Vec::new();

            for row in preds.outer_iter() {
                if let Some((idx, prob)) = argmax_row(row) {
                    sequence_idx.push(idx);
                    sequence_prob.push(prob);
                } else {
                    sequence_idx.push(0);
                    sequence_prob.push(0.0);
                }
            }

            let text = self.decode(&[sequence_idx], Some(&[sequence_prob]), true);

            for (t, score) in text {
                all_texts.push(t);
                all_scores.push(score);
            }
        }

        (all_texts, all_scores)
    }

    /// Gets the index of the blank token.
    ///
    /// # Returns
    /// The index of the blank token (always 0 in this base implementation).
    fn get_blank_idx(&self) -> usize {
        0
    }
}

/// A decoder for CTC (Connectionist Temporal Classification) based text recognition models.
///
/// This struct extends `BaseRecLabelDecode` to provide specialized decoding for CTC models,
/// which include a blank token that needs to be handled specially during decoding.
///
/// # Fields
/// * `base` - The base decoder that handles character mapping and basic decoding operations
/// * `blank_index` - The index of the blank token in the character vocabulary
pub struct CTCLabelDecode {
    base: BaseRecLabelDecode,
    blank_index: usize,
}

impl std::fmt::Debug for CTCLabelDecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CTCLabelDecode")
            .field("character_count", &self.base.character.len())
            .field("reverse", &self.base.reverse)
            .finish()
    }
}

impl CTCLabelDecode {
    /// Creates a new `CTCLabelDecode` instance.
    ///
    /// # Arguments
    /// * `character_list` - An optional string containing the character vocabulary.
    ///   If None, a default alphanumeric character set is used.
    /// * `use_space_char` - Whether to include a space character in the vocabulary.
    ///
    /// # Returns
    /// A new `CTCLabelDecode` instance.
    pub fn new(character_list: Option<&str>, use_space_char: bool) -> Self {
        let mut base = BaseRecLabelDecode::new(character_list, use_space_char);

        // Use null char for blank to distinguish from actual space
        let mut new_character = vec!['\0'];
        new_character.extend(base.character);

        let mut new_dict = HashMap::new();
        for (i, &char) in new_character.iter().enumerate() {
            new_dict.insert(char, i);
        }

        base.character = new_character;
        base.dict = new_dict;

        let blank_index = 0;

        Self { base, blank_index }
    }

    /// Creates a new `CTCLabelDecode` instance from a list of strings.
    ///
    /// # Arguments
    /// * `character_list` - An optional slice of strings containing the character vocabulary.
    ///   Only the first character of each string is used. If None, a default alphanumeric
    ///   character set is used.
    /// * `use_space_char` - Whether to include a space character in the vocabulary.
    /// * `has_explicit_blank` - Whether the character list already includes a blank token.
    ///
    /// # Returns
    /// A new `CTCLabelDecode` instance.
    pub fn from_string_list(
        character_list: Option<&[String]>,
        use_space_char: bool,
        has_explicit_blank: bool,
    ) -> Self {
        if has_explicit_blank {
            let base = BaseRecLabelDecode::from_string_list(character_list, use_space_char);
            Self {
                base,
                blank_index: 0,
            }
        } else {
            let mut base = BaseRecLabelDecode::from_string_list(character_list, use_space_char);

            // Use null char for blank to distinguish from actual space
            let mut new_character = vec!['\0'];
            new_character.extend(base.character);

            let mut new_dict = HashMap::new();
            for (i, &char) in new_character.iter().enumerate() {
                new_dict.insert(char, i);
            }

            base.character = new_character;
            base.dict = new_dict;

            Self {
                base,
                blank_index: 0,
            }
        }
    }

    /// Gets the index of the blank token.
    ///
    /// # Returns
    /// The index of the blank token.
    pub fn get_blank_index(&self) -> usize {
        self.blank_index
    }

    /// Gets the character list used by this decoder.
    ///
    /// # Returns
    /// A slice containing the characters in the vocabulary.
    pub fn get_character_list(&self) -> &[char] {
        &self.base.character
    }

    /// Gets the number of characters in the vocabulary.
    ///
    /// # Returns
    /// The number of characters in the vocabulary.
    pub fn get_character_count(&self) -> usize {
        self.base.character.len()
    }

    /// Reduces `(batch, time, vocab)` logits to one token and confidence per
    /// timestep. This is the only part of CTC decoding that must inspect the
    /// large logits buffer.
    pub(crate) fn argmax_predictions<S>(
        &self,
        pred: &ndarray::ArrayBase<S, ndarray::Ix3>,
    ) -> CTCArgmaxOutput
    where
        S: ndarray::Data<Elem = f32> + Sync,
    {
        let [batch_size, sequence_length, vocab_size] = pred
            .shape()
            .try_into()
            .expect("CTC predictions always have three dimensions");
        let row_count = batch_size * sequence_length;

        // Preserve the public decoder's historical empty-tensor behavior: no
        // batch entries are returned when any dimension is zero.
        if pred.is_empty() {
            return CTCArgmaxOutput {
                batch_size: 0,
                sequence_length: 0,
                indices: Vec::new(),
                probabilities: Vec::new(),
            };
        }

        // ORT outputs are contiguous, so the hot path can fan all timesteps out
        // across rayon, including batch-size 1. Keep a non-contiguous fallback
        // for callers of the public ndarray-based decoder methods.
        let (indices, probabilities): (Vec<usize>, Vec<f32>) = if let Some(data) = pred.as_slice() {
            data.par_chunks_exact(vocab_size)
                .map(|row| crate::processors::simd::argmax(row).unwrap_or((self.blank_index, 0.0)))
                .unzip()
        } else {
            (0..row_count)
                .into_par_iter()
                .map(|row_idx| {
                    let batch_idx = row_idx / sequence_length;
                    let time_idx = row_idx % sequence_length;
                    argmax_row(pred.slice(ndarray::s![batch_idx, time_idx, ..]))
                        .unwrap_or((self.blank_index, 0.0))
                })
                .unzip()
        };

        CTCArgmaxOutput {
            batch_size,
            sequence_length,
            indices,
            probabilities,
        }
    }

    /// Performs CTC collapse and text construction from compact argmax data.
    /// This no longer needs access to the logits or the inference session.
    pub(crate) fn decode_argmax(&self, argmax: &CTCArgmaxOutput) -> (Vec<String>, Vec<f32>) {
        let (all_texts, all_scores): (Vec<String>, Vec<f32>) = (0..argmax.batch_size)
            .into_par_iter()
            .map(|batch_idx| {
                let start = batch_idx * argmax.sequence_length;
                let end = start + argmax.sequence_length;
                let sequence_idx = &argmax.indices[start..end];
                let sequence_prob = &argmax.probabilities[start..end];

                let mut filtered_prob = Vec::with_capacity(argmax.sequence_length);
                let mut text = String::with_capacity(argmax.sequence_length);
                let mut prev_idx = self.blank_index;
                for (i, &idx) in sequence_idx.iter().enumerate() {
                    if idx != self.blank_index
                        && idx != prev_idx
                        && let Some(&ch) = self.base.character.get(idx)
                    {
                        text.push(ch);
                        filtered_prob.push(sequence_prob[i]);
                    }
                    prev_idx = idx;
                }

                let mean_conf = if filtered_prob.is_empty() {
                    0.0
                } else {
                    filtered_prob.iter().sum::<f32>() / filtered_prob.len() as f32
                };

                (text, mean_conf)
            })
            .unzip();

        (all_texts, all_scores)
    }

    /// Performs CTC collapse while retaining character timestep positions.
    /// This no longer needs access to the logits or the inference session.
    pub(crate) fn decode_argmax_with_positions(
        &self,
        argmax: &CTCArgmaxOutput,
    ) -> PositionedDecodeResult {
        type PerItem = (String, f32, Vec<f32>, Vec<usize>, usize);
        let per: Vec<PerItem> = (0..argmax.batch_size)
            .into_par_iter()
            .map(|batch_idx| {
                let start = batch_idx * argmax.sequence_length;
                let end = start + argmax.sequence_length;
                let sequence_idx = &argmax.indices[start..end];
                let sequence_prob = &argmax.probabilities[start..end];

                let mut filtered_prob = Vec::with_capacity(argmax.sequence_length);
                let mut filtered_timesteps = Vec::with_capacity(argmax.sequence_length);
                let mut char_list = Vec::with_capacity(argmax.sequence_length);
                let mut prev_idx = self.blank_index;
                for (i, &idx) in sequence_idx.iter().enumerate() {
                    if idx != self.blank_index
                        && idx != prev_idx
                        && let Some(&ch) = self.base.character.get(idx)
                    {
                        char_list.push(ch);
                        filtered_prob.push(sequence_prob[i]);
                        filtered_timesteps.push(i);
                    }
                    prev_idx = idx;
                }

                let mean_conf = if filtered_prob.is_empty() {
                    0.0
                } else {
                    filtered_prob.iter().sum::<f32>() / filtered_prob.len() as f32
                };
                let seq_len = argmax.sequence_length as f32;
                let char_positions = filtered_timesteps
                    .iter()
                    .map(|&timestep| timestep as f32 / seq_len)
                    .collect();
                let text = char_list.iter().collect();

                (
                    text,
                    mean_conf,
                    char_positions,
                    filtered_timesteps,
                    argmax.sequence_length,
                )
            })
            .collect();

        let mut all_texts = Vec::with_capacity(argmax.batch_size);
        let mut all_scores = Vec::with_capacity(argmax.batch_size);
        let mut all_positions = Vec::with_capacity(argmax.batch_size);
        let mut all_col_indices = Vec::with_capacity(argmax.batch_size);
        let mut all_seq_lengths = Vec::with_capacity(argmax.batch_size);
        for (text, score, pos, cols, seq_len) in per {
            all_texts.push(text);
            all_scores.push(score);
            all_positions.push(pos);
            all_col_indices.push(cols);
            all_seq_lengths.push(seq_len);
        }

        (
            all_texts,
            all_scores,
            all_positions,
            all_col_indices,
            all_seq_lengths,
        )
    }

    /// Applies the CTC decoder to a tensor of model predictions with character position tracking.
    ///
    /// This method handles the special requirements of CTC decoding and additionally tracks
    /// the timestep positions of each character for word box generation.
    ///
    /// # Arguments
    /// * `pred` - A 3D tensor containing the model predictions. Accepts any
    ///   `ndarray` storage (owned `Array3<f32>` or a zero-copy `ArrayView3<f32>`).
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of decoded text strings
    /// * A vector of confidence scores for each text string
    /// * A vector of character positions (normalized 0.0-1.0) for each text string
    /// * A vector of column indices for each character in each text string
    /// * A vector of sequence lengths (total columns) for each text string
    pub fn apply_with_positions<S>(
        &self,
        pred: &ndarray::ArrayBase<S, ndarray::Ix3>,
    ) -> PositionedDecodeResult
    where
        S: ndarray::Data<Elem = f32> + Sync,
    {
        if pred.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }
        let argmax = self.argmax_predictions(pred);
        self.decode_argmax_with_positions(&argmax)
    }

    /// Applies the CTC decoder to a tensor of model predictions.
    ///
    /// This method handles the special requirements of CTC decoding:
    /// 1. Removing blank tokens
    /// 2. Removing consecutive duplicate characters
    /// 3. Converting indices to characters
    /// 4. Calculating confidence scores
    ///
    /// # Arguments
    /// * `pred` - A 3D tensor containing the model predictions. Accepts any
    ///   `ndarray` storage (owned `Array3<f32>` or a zero-copy `ArrayView3<f32>`).
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of decoded text strings
    /// * A vector of confidence scores for each text string
    pub fn apply<S>(&self, pred: &ndarray::ArrayBase<S, ndarray::Ix3>) -> (Vec<String>, Vec<f32>)
    where
        S: ndarray::Data<Elem = f32> + Sync,
    {
        if pred.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let argmax = self.argmax_predictions(pred);
        self.decode_argmax(&argmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn logits_with_winners(winners: &[&[(usize, f32)]], vocab_size: usize) -> Array3<f32> {
        let batch_size = winners.len();
        let sequence_length = winners.first().map_or(0, |sequence| sequence.len());
        let mut logits = Array3::from_elem((batch_size, sequence_length, vocab_size), -10.0);
        for (batch_idx, sequence) in winners.iter().enumerate() {
            assert_eq!(sequence.len(), sequence_length);
            for (time_idx, &(token_idx, probability)) in sequence.iter().enumerate() {
                logits[[batch_idx, time_idx, token_idx]] = probability;
            }
        }
        logits
    }

    #[test]
    fn compact_argmax_preserves_ctc_text_scores_and_positions() {
        let characters = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let decoder = CTCLabelDecode::from_string_list(Some(&characters), false, false);
        // Vocabulary slot 4 is deliberately out of the decoder's dictionary.
        let logits = logits_with_winners(
            &[
                &[
                    (0, 0.9),
                    (1, 0.8),
                    (1, 0.7),
                    (0, 0.6),
                    (1, 0.5),
                    (2, 0.4),
                    (2, 0.3),
                ],
                &[
                    (3, 0.95),
                    (3, 0.85),
                    (4, 0.75),
                    (3, 0.65),
                    (0, 0.55),
                    (2, 0.45),
                    (0, 0.35),
                ],
            ],
            5,
        );

        let argmax = decoder.argmax_predictions(&logits);
        assert_eq!(argmax.batch_size, 2);
        assert_eq!(argmax.sequence_length, 7);
        assert_eq!(argmax.indices.len(), 14);
        assert_eq!(argmax.probabilities.len(), 14);

        let (texts, scores) = decoder.decode_argmax(&argmax);
        assert_eq!(texts, ["aab", "ccb"]);
        assert_eq!(
            scores,
            [(0.8 + 0.5 + 0.4) / 3.0, (0.95 + 0.65 + 0.45) / 3.0]
        );

        let (texts, scores, positions, columns, lengths) =
            decoder.decode_argmax_with_positions(&argmax);
        assert_eq!(texts, ["aab", "ccb"]);
        assert_eq!(
            scores,
            [(0.8 + 0.5 + 0.4) / 3.0, (0.95 + 0.65 + 0.45) / 3.0]
        );
        assert_eq!(columns, [vec![1, 4, 5], vec![0, 3, 5]]);
        assert_eq!(positions[0], [1.0 / 7.0, 4.0 / 7.0, 5.0 / 7.0]);
        assert_eq!(positions[1], [0.0, 3.0 / 7.0, 5.0 / 7.0]);
        assert_eq!(lengths, [7, 7]);
    }

    #[test]
    fn compact_argmax_preserves_empty_tensor_behavior() {
        let decoder = CTCLabelDecode::new(None, false);
        let logits = Array3::<f32>::zeros((2, 0, decoder.get_character_count()));
        let argmax = decoder.argmax_predictions(&logits);

        assert_eq!(decoder.decode_argmax(&argmax), (Vec::new(), Vec::new()));
        assert_eq!(
            decoder.decode_argmax_with_positions(&argmax),
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
        );
    }
}
