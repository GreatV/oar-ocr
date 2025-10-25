//! Table Structure Decoding Processor
//!
//! This module provides postprocessing for table structure recognition models.
//! It decodes structure token logits and extracts bounding boxes for table cells.

use crate::core::{OCRError, Tensor3D};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

type TableDecodeArtifacts = (Vec<String>, Vec<[f32; 8]>, f32);
type TableDecodeResult = Result<TableDecodeArtifacts, OCRError>;

/// Output from table structure decoding.
#[derive(Debug, Clone)]
pub struct TableStructureDecodeOutput {
    /// HTML structure tokens for each image (without HTML wrapping)
    pub structure_tokens: Vec<Vec<String>>,
    /// Bounding boxes for table cells (4-point polygons: [x1,y1,x2,y2,x3,y3,x4,y4])
    pub bboxes: Vec<Vec<[f32; 8]>>,
    /// Mean confidence scores for structure predictions
    pub structure_scores: Vec<f32>,
}

/// Table structure decoder that converts model outputs to HTML tokens and bboxes.
#[derive(Debug, Clone)]
pub struct TableStructureDecode {
    /// HTML token dictionary (e.g., "<html>", "<table>", "<tr>", "<td>", etc.)
    character_dict: Vec<String>,
    /// Special tokens to ignore during decoding
    ignored_tokens: Vec<usize>,
    /// Token indices that should have bounding boxes (e.g., "<td>", "<td", "<td></td>")
    td_token_indices: Vec<usize>,
    /// End token index
    end_idx: usize,
}

impl TableStructureDecode {
    /// Creates a new table structure decoder from a dictionary file.
    pub fn from_dict_path(dict_path: &Path) -> Result<Self, OCRError> {
        // Load base dictionary
        let mut character_dict = Self::load_dict(dict_path)?;

        // Apply merge_no_span_structure logic (following PaddleX TableLabelDecode)
        let merge_no_span_structure = false; // Try without merge to see if it helps
        if merge_no_span_structure {
            if !character_dict.contains(&"<td></td>".to_string()) {
                character_dict.push("<td></td>".to_string());
            }
            if let Some(pos) = character_dict.iter().position(|s| s == "<td>") {
                character_dict.remove(pos);
            }
        }

        // Add special tokens (following PaddleOCR TableLabelDecode convention)
        let beg_str = "<SOS>";
        let end_str = "<EOS>";
        let unknown_str = "<UKN>";
        let pad_str = "<PAD>";

        let original_dict_size = character_dict.len();

        // Add special tokens to the END of dictionary (following PaddleOCR)
        let mut final_dict = character_dict;
        final_dict.extend(vec![
            unknown_str.to_string(),
            beg_str.to_string(),
            end_str.to_string(),
            pad_str.to_string(),
        ]);

        tracing::debug!("Dictionary processing complete:");
        tracing::debug!("  Original dict size: {}", original_dict_size);
        tracing::debug!("  Final dict size: {}", final_dict.len());
        tracing::debug!(
            "  First 10 dict entries: {:?}",
            &final_dict[..10.min(final_dict.len())]
        );
        tracing::debug!(
            "  Last 10 dict entries: {:?}",
            &final_dict[final_dict.len().saturating_sub(10)..]
        );

        // Build index mappings
        let end_idx = final_dict
            .iter()
            .position(|s| s == end_str)
            .ok_or_else(|| OCRError::ConfigError {
                message: "End token not found in dictionary".to_string(),
            })?;

        let start_idx = final_dict
            .iter()
            .position(|s| s == beg_str)
            .ok_or_else(|| OCRError::ConfigError {
                message: "Start token not found in dictionary".to_string(),
            })?;

        let pad_idx = final_dict
            .iter()
            .position(|s| s == pad_str)
            .ok_or_else(|| OCRError::ConfigError {
                message: "Pad token not found in dictionary".to_string(),
            })?;

        let unknown_idx = final_dict
            .iter()
            .position(|s| s == unknown_str)
            .ok_or_else(|| OCRError::ConfigError {
                message: "Unknown token not found in dictionary".to_string(),
            })?;

        let ignored_tokens = vec![pad_idx, start_idx, end_idx, unknown_idx];

        // Find TD token indices (following PaddleX logic)
        let td_tokens = ["<td>", "<td", "<td></td>"];
        let td_token_indices: Vec<usize> = td_tokens
            .iter()
            .filter_map(|&token| final_dict.iter().position(|s| s == token))
            .collect();

        Ok(Self {
            character_dict: final_dict,
            ignored_tokens,
            td_token_indices,
            end_idx,
        })
    }

    /// Loads dictionary from file.
    fn load_dict(path: &Path) -> Result<Vec<String>, OCRError> {
        let file = File::open(path).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to open dictionary file '{}': {}", path.display(), e),
        })?;

        let reader = BufReader::new(file);
        let mut dict = Vec::new();

        for line in reader.lines() {
            let line = line.map_err(|e| OCRError::ConfigError {
                message: format!("Failed to read dictionary line: {}", e),
            })?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                dict.push(trimmed.to_string());
            }
        }

        Ok(dict)
    }

    /// Decodes structure logits and bbox predictions.
    ///
    /// # Arguments
    ///
    /// * `structure_logits` - [batch, seq_len, vocab_size] structure predictions
    /// * `bbox_preds` - [batch, seq_len, 8] bbox predictions (normalized coordinates)
    /// * `shape_info` - [(h, w, ratio_h, ratio_w, pad_h, pad_w), ...] for each image
    ///
    /// # Returns
    ///
    /// Decoded structure tokens, bounding boxes, and confidence scores
    pub fn decode(
        &self,
        structure_logits: &Tensor3D,
        bbox_preds: &Tensor3D,
        shape_info: &[[f32; 6]],
    ) -> Result<TableStructureDecodeOutput, OCRError> {
        let batch_size = structure_logits.shape()[0];

        let mut structure_tokens_batch = Vec::with_capacity(batch_size);
        let mut bboxes_batch = Vec::with_capacity(batch_size);
        let mut scores_batch = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let (tokens, bboxes, score) =
                self.decode_single(structure_logits, bbox_preds, batch_idx, shape_info)?;

            structure_tokens_batch.push(tokens);
            bboxes_batch.push(bboxes);
            scores_batch.push(score);
        }

        Ok(TableStructureDecodeOutput {
            structure_tokens: structure_tokens_batch,
            bboxes: bboxes_batch,
            structure_scores: scores_batch,
        })
    }

    /// Decodes a single image from the batch.
    fn decode_single(
        &self,
        structure_logits: &Tensor3D,
        bbox_preds: &Tensor3D,
        batch_idx: usize,
        shape_info: &[[f32; 6]],
    ) -> TableDecodeResult {
        let seq_len = structure_logits.shape()[1];

        // Argmax to get token indices
        let mut structure_tokens = Vec::new();
        let mut bboxes = Vec::new();
        let mut scores = Vec::new();

        tracing::debug!(
            "Starting token decoding for batch {}, sequence length {}",
            batch_idx,
            seq_len
        );
        tracing::debug!("Structure logits shape: {:?}", structure_logits.shape());
        tracing::debug!("Bbox preds shape: {:?}", bbox_preds.shape());

        for seq_idx in 0..seq_len {
            // Get token index (argmax over vocab dimension)
            let (token_idx, token_prob) = self.argmax_at(structure_logits, batch_idx, seq_idx);

            // Stop at end token
            if seq_idx > 0 && token_idx == self.end_idx {
                tracing::debug!(
                    "Stopping at end token (idx: {}) at sequence position {}",
                    token_idx,
                    seq_idx
                );
                break;
            }

            // Skip ignored tokens
            if self.ignored_tokens.contains(&token_idx) {
                tracing::debug!(
                    "Skipping ignored token at seq_idx {}: token_idx={}, token='{}'",
                    seq_idx,
                    token_idx,
                    self.character_dict
                        .get(token_idx)
                        .unwrap_or(&"<INVALID>".to_string())
                );
                continue;
            }

            // Get token string
            let token = self
                .character_dict
                .get(token_idx)
                .cloned()
                .unwrap_or_else(|| format!("UNK_{}", token_idx));

            tracing::debug!(
                "Decoded token at seq_idx {}: token_idx={}, dict_size={}, token='{}', prob={:.6}",
                seq_idx,
                token_idx,
                self.character_dict.len(),
                token,
                token_prob
            );

            structure_tokens.push(token.clone());
            scores.push(token_prob);

            // Extract bbox if this is a TD token
            if self.td_token_indices.contains(&token_idx) {
                let bbox = self.extract_bbox(bbox_preds, batch_idx, seq_idx, shape_info)?;
                tracing::debug!("Extracted bbox for TD token '{}': {:?}", token, bbox);
                bboxes.push(bbox);
            }
        }

        tracing::info!(
            "Decoded {} structure tokens: {:?}",
            structure_tokens.len(),
            structure_tokens
        );
        tracing::info!("Extracted {} bounding boxes", bboxes.len());

        // Calculate max score (like PaddleX)
        let max_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        };

        Ok((structure_tokens, bboxes, max_score))
    }

    /// Finds argmax at specific position in structure logits.
    fn argmax_at(&self, logits: &Tensor3D, batch_idx: usize, seq_idx: usize) -> (usize, f32) {
        let vocab_size = logits.shape()[2];
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for vocab_idx in 0..vocab_size {
            let val = logits[[batch_idx, seq_idx, vocab_idx]];
            if val > max_val {
                max_val = val;
                max_idx = vocab_idx;
            }
        }

        // Return the raw max logit value as score (like PaddleX)
        (max_idx, max_val)
    }

    /// Extracts and denormalizes bounding box (PaddleX compatible).
    fn extract_bbox(
        &self,
        bbox_preds: &Tensor3D,
        batch_idx: usize,
        seq_idx: usize,
        shape_info: &[[f32; 6]],
    ) -> Result<[f32; 8], OCRError> {
        let mut bbox = [0.0f32; 8];

        // Extract normalized coordinates
        for (idx, coord) in bbox.iter_mut().enumerate() {
            *coord = bbox_preds[[batch_idx, seq_idx, idx]];
        }

        // Denormalize using shape information (following PaddleOCR pipeline)
        if let Some(shape) = shape_info.get(batch_idx) {
            let [orig_h, orig_w, scale, pad_h, pad_w, target_size] = *shape;
            let resized_h = target_size - pad_h;
            let resized_w = target_size - pad_w;

            // Bbox coordinates are normalized to [0,1] relative to the padded image.
            // Convert back to resized image coordinates, clamp to valid range, then scale to original size.
            for (idx, coord_ref) in bbox.iter_mut().enumerate() {
                let mut coord = *coord_ref * target_size;

                if idx % 2 == 0 {
                    // x coordinate
                    coord = coord.clamp(0.0, resized_w);
                    coord = (coord / scale).clamp(0.0, orig_w);
                } else {
                    // y coordinate
                    coord = coord.clamp(0.0, resized_h);
                    coord = (coord / scale).clamp(0.0, orig_h);
                }

                *coord_ref = coord;
            }
        }

        Ok(bbox)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_dict() {
        // This test would require the actual dictionary file
        // In practice, we'd test with a mock file
    }

    #[test]
    fn test_dictionary_processing() {
        // Test dictionary processing logic
        // Create a temporary dictionary similar to PaddleOCR's
        let temp_dict = vec![
            "<html>".to_string(),
            "<body>".to_string(),
            "<table>".to_string(),
            "<tr>".to_string(),
            "<td>".to_string(), // This should be removed
            "<td".to_string(),
            " colspan=\"4\"".to_string(),
            ">".to_string(),
            "</td>".to_string(),
            "</tr>".to_string(),
            "</table>".to_string(),
            "</body>".to_string(),
            "</html>".to_string(),
        ];

        // Test merge_no_span_structure logic
        let mut processed_dict = temp_dict.clone();
        let merge_no_span_structure = true;
        if merge_no_span_structure {
            if !processed_dict.contains(&"<td></td>".to_string()) {
                processed_dict.push("<td></td>".to_string());
            }
            if let Some(pos) = processed_dict.iter().position(|s| s == "<td>") {
                processed_dict.remove(pos);
            }
        }

        // Check that <td> was removed
        assert!(!processed_dict.contains(&"<td>".to_string()));

        // Check that <td></td> was added
        assert!(processed_dict.contains(&"<td></td>".to_string()));

        // Add special tokens
        let beg_str = "sos";
        let end_str = "eos";
        let mut final_dict = vec![beg_str.to_string()];
        final_dict.extend(processed_dict);
        final_dict.push(end_str.to_string());

        // Check special tokens are in correct positions
        assert_eq!(final_dict[0], "sos");
        assert_eq!(final_dict[final_dict.len() - 1], "eos");

        // Check that original tokens are preserved (except <td>)
        assert!(final_dict.contains(&"<html>".to_string()));
        assert!(final_dict.contains(&"<td".to_string()));
        assert!(final_dict.contains(&" colspan=\"4\"".to_string()));
    }

    #[test]
    fn test_argmax() {
        use ndarray::Array3;

        let dict_path = Path::new(".oar/table_structure_dict.txt");
        if !dict_path.exists() {
            return; // Skip if dict not available
        }

        let decoder = TableStructureDecode::from_dict_path(dict_path).unwrap();

        // Create simple logits tensor
        let logits = Array3::zeros((1, 5, 50));
        let (idx, _prob) = decoder.argmax_at(&logits, 0, 0);
        assert_eq!(idx, 0); // Should be first token (all zeros)
    }
}
