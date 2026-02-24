//! Text decoding utilities for OCR (Optical Character Recognition) systems.
//!
//! This module provides implementations for decoding text recognition results,
//! particularly focused on CTC (Connectionist Temporal Classification) decoding.
//! It includes structures and methods for converting model predictions into
//! readable text strings with confidence scores.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

/// Decoded batch outputs along with positional metadata.
pub type PositionedDecodeResult = (
    Vec<String>,
    Vec<f32>,
    Vec<Vec<f32>>,
    Vec<Vec<usize>>,
    Vec<usize>,
);

static ALPHANUMERIC_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[a-zA-Z0-9 :*./%+-]")
        .unwrap_or_else(|e| panic!("Failed to compile regex pattern: {e}"))
});

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
    /// * `pred` - A 3D tensor containing the model predictions.
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
                if let Some((idx, &prob)) = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
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

    /// Applies the CTC decoder to a tensor of model predictions with character position tracking.
    ///
    /// This method handles the special requirements of CTC decoding and additionally tracks
    /// the timestep positions of each character for word box generation.
    ///
    /// # Arguments
    /// * `pred` - A 3D tensor containing the model predictions.
    ///
    /// # Returns
    /// A tuple containing:
    /// * A vector of decoded text strings
    /// * A vector of confidence scores for each text string
    /// * A vector of character positions (normalized 0.0-1.0) for each text string
    /// * A vector of column indices for each character in each text string
    /// * A vector of sequence lengths (total columns) for each text string
    pub fn apply_with_positions(&self, pred: &ndarray::Array3<f32>) -> PositionedDecodeResult {
        if pred.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        }

        let batch_size = pred.shape()[0];
        let mut all_texts = Vec::new();
        let mut all_scores = Vec::new();
        let mut all_positions = Vec::new();
        let mut all_col_indices = Vec::new();
        let mut all_seq_lengths = Vec::new();

        for batch_idx in 0..batch_size {
            let preds = pred.index_axis(ndarray::Axis(0), batch_idx);
            let seq_len = preds.shape()[0] as f32;

            let mut sequence_idx = Vec::new();
            let mut sequence_prob = Vec::new();
            let mut sequence_timesteps = Vec::new();

            for (timestep, row) in preds.outer_iter().enumerate() {
                if let Some((idx, &prob)) = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
                    sequence_idx.push(idx);
                    sequence_prob.push(prob);
                    sequence_timesteps.push(timestep);
                } else {
                    sequence_idx.push(self.blank_index);
                    sequence_prob.push(0.0);
                    sequence_timesteps.push(timestep);
                }
            }

            let mut filtered_idx = Vec::new();
            let mut filtered_prob = Vec::new();
            let mut filtered_timesteps = Vec::new();
            let mut selection = vec![true; sequence_idx.len()];

            // Remove consecutive duplicates
            if sequence_idx.len() > 1 {
                for i in 1..sequence_idx.len() {
                    if sequence_idx[i] == sequence_idx[i - 1] {
                        selection[i] = false;
                    }
                }
            }

            // Remove blanks
            for (i, &idx) in sequence_idx.iter().enumerate() {
                if idx == self.blank_index {
                    selection[i] = false;
                }
            }

            // Collect filtered results
            for (i, &idx) in sequence_idx.iter().enumerate() {
                if selection[i] {
                    filtered_idx.push(idx);
                    filtered_prob.push(sequence_prob[i]);
                    filtered_timesteps.push(sequence_timesteps[i]);
                }
            }

            let char_list: Vec<char> = filtered_idx
                .iter()
                .filter_map(|&text_id| self.base.character.get(text_id).copied())
                .collect();

            let conf_list = if filtered_prob.is_empty() {
                vec![0.0]
            } else {
                filtered_prob
            };

            // Calculate normalized character positions (0.0 to 1.0)
            let char_positions: Vec<f32> = filtered_timesteps
                .iter()
                .map(|&timestep| timestep as f32 / seq_len)
                .collect();

            // Store column indices (raw timesteps) for accurate word box generation
            let col_indices: Vec<usize> = filtered_timesteps.clone();

            let text: String = char_list.iter().collect();
            let mean_conf = conf_list.iter().sum::<f32>() / conf_list.len() as f32;

            all_texts.push(text);
            all_scores.push(mean_conf);
            all_positions.push(char_positions);
            all_col_indices.push(col_indices);
            all_seq_lengths.push(seq_len as usize);
        }

        (
            all_texts,
            all_scores,
            all_positions,
            all_col_indices,
            all_seq_lengths,
        )
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
    /// * `pred` - A 3D tensor containing the model predictions.
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
        let mut batches_with_text = 0;

        for batch_idx in 0..batch_size {
            let preds = pred.index_axis(ndarray::Axis(0), batch_idx);

            let mut sequence_idx = Vec::new();
            let mut sequence_prob = Vec::new();

            for row in preds.outer_iter() {
                if let Some((idx, &prob)) = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                {
                    sequence_idx.push(idx);
                    sequence_prob.push(prob);
                } else {
                    sequence_idx.push(self.blank_index);
                    sequence_prob.push(0.0);
                }
            }

            let mut filtered_idx = Vec::new();
            let mut filtered_prob = Vec::new();
            let mut selection = vec![true; sequence_idx.len()];

            if sequence_idx.len() > 1 {
                for i in 1..sequence_idx.len() {
                    if sequence_idx[i] == sequence_idx[i - 1] {
                        selection[i] = false;
                    }
                }
            }

            for (i, &idx) in sequence_idx.iter().enumerate() {
                if idx == self.blank_index {
                    selection[i] = false;
                }
            }

            for (i, &idx) in sequence_idx.iter().enumerate() {
                if selection[i] {
                    filtered_idx.push(idx);
                    filtered_prob.push(sequence_prob[i]);
                }
            }

            let char_list: Vec<char> = filtered_idx
                .iter()
                .filter_map(|&text_id| self.base.character.get(text_id).copied())
                .collect();

            let conf_list = if filtered_prob.is_empty() {
                vec![0.0]
            } else {
                filtered_prob
            };

            let text: String = char_list.iter().collect();
            let mean_conf = conf_list.iter().sum::<f32>() / conf_list.len() as f32;

            if !text.is_empty() {
                batches_with_text += 1;
            }

            all_texts.push(text);
            all_scores.push(mean_conf);
        }

        // Log summary of decoding results
        tracing::debug!(
            "CTC decode summary: batch_size={}, batches_with_text={}, empty_batches={}",
            batch_size,
            batches_with_text,
            batch_size - batches_with_text
        );

        (all_texts, all_scores)
    }
}
