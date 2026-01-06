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

/// Wraps table structure tokens with HTML document tags.
///
/// PP-StructureV3 outputs full HTML documents with `<html><body><table>...</table></body></html>`.
/// This function wraps raw table tokens to match that format.
///
/// # Arguments
///
/// * `tokens` - Table structure tokens (e.g., `["<tr>", "<td>", "...</td>", "</tr>"]`)
///
/// # Returns
///
/// Complete HTML string with document wrapper tags.
///
/// # Example
///
/// ```
/// use oar_ocr::processors::wrap_table_html;
///
/// let tokens = vec!["<tr>".to_string(), "<td></td>".to_string(), "</tr>".to_string()];
/// let html = wrap_table_html(&tokens);
/// assert!(html.starts_with("<html><body><table>"));
/// assert!(html.ends_with("</table></body></html>"));
/// ```
pub fn wrap_table_html(tokens: &[String]) -> String {
    render_table_html(tokens, None)
}

/// Wraps table structure tokens into an HTML string with cell content filled in.
///
/// This follows standard HTML result logic:
/// - When encountering `<td></td>` or `</td>`, insert the corresponding cell text
/// - Cell texts are matched in order (td_index)
///
/// # Arguments
/// * `tokens` - Structure tokens from table structure recognition
/// * `cell_texts` - Cell texts to fill, in order of `<td>` appearance
///
/// # Example
/// ```
/// use oar_ocr::processors::wrap_table_html_with_content;
///
/// let tokens = vec![
///     "<tr>".to_string(),
///     "<td></td>".to_string(),
///     "<td></td>".to_string(),
///     "</tr>".to_string()
/// ];
/// let cell_texts = vec![Some("Cell 1".to_string()), Some("Cell 2".to_string())];
/// let html = wrap_table_html_with_content(&tokens, &cell_texts);
/// assert!(html.contains("Cell 1"));
/// assert!(html.contains("Cell 2"));
/// ```
pub fn wrap_table_html_with_content(tokens: &[String], cell_texts: &[Option<String>]) -> String {
    render_table_html(tokens, Some(cell_texts))
}

/// Renders table HTML, optionally filling cell content.
fn render_table_html(tokens: &[String], cell_texts: Option<&[Option<String>]>) -> String {
    let mut result = Vec::new();
    let mut td_index = 0;
    let mut idx = 0usize;

    result.push("<html><body>".to_string());

    // Check if table tag is already present in tokens
    let has_table_tag = tokens
        .first()
        .map(|t| t.contains("<table"))
        .unwrap_or(false);
    if !has_table_tag {
        result.push("<table>".to_string());
    }

    while idx < tokens.len() {
        let tag = tokens[idx].as_str();

        // Handle standard empty cell token
        if tag == "<td></td>" {
            result.push("<td>".to_string());
            if let Some(texts) = cell_texts
                && let Some(Some(text)) = texts.get(td_index)
            {
                result.push(text.clone());
            }
            result.push("</td>".to_string());
            td_index += 1;
            idx += 1;
            continue;
        }

        // Handle opening td tag (possibly with attributes)
        if tag.starts_with("<td") {
            let parsed = parse_td_tag(tokens, idx);
            result.push(format!("<td{}>", parsed.attrs));

            // Check for bold tag immediately following td
            // Some structure models might output <td><b>...
            let mut is_bold = false;
            let next_idx = parsed.next_index;

            // Peek ahead for <b> inside the cell (before the next tag/end)
            // Note: This simple logic assumes <b> is a distinct token if present
            if next_idx < tokens.len() && tokens[next_idx] == "<b>" {
                is_bold = true;
                // Consume <b>
                // next_idx += 1;
                // actually we don't consume it here, we let the loop handle it?
                // No, we are inside the "fill content" logic.
                // If structure has <b>, we should wrap our content in <b>.
            }

            if let Some(texts) = cell_texts
                && let Some(Some(text)) = texts.get(td_index)
            {
                if is_bold {
                    result.push("<b>".to_string());
                }
                result.push(text.clone());
                if is_bold {
                    result.push("</b>".to_string());
                }
            }

            result.push("</td>".to_string());
            td_index += 1;

            // If we detected <b> structure, we effectively "handled" it by wrapping content.
            // However, to avoid duplicating tags if they are in the token stream,
            // we should ideally consume them. But robustly parsing nested structure
            // like <td><b></b></td> is complex.
            //
            // Standard PP-StructureV3 primarily uses <thead> vs <tbody> for styling
            // and <td></td> tokens. Boldness usually comes from being in <thead>.
            // We'll stick to the parsed index.
            idx = parsed.next_index;
            continue;
        }

        // Pass through all other tokens (<thead>, <tbody>, <tfoot>, <tr>, </tr>, etc.)
        result.push(tokens[idx].clone());
        idx += 1;
    }

    if !has_table_tag {
        result.push("</table>".to_string());
    }
    result.push("</body></html>".to_string());

    result.join("")
}

/// Grid position and span information for a table cell.
#[derive(Debug, Clone, Default)]
pub struct CellGridInfo {
    /// Row index (0-based)
    pub row: usize,
    /// Column index (0-based)
    pub col: usize,
    /// Number of rows this cell spans
    pub row_span: usize,
    /// Number of columns this cell spans
    pub col_span: usize,
}

/// Parses structure tokens to extract grid position and span info for each cell.
///
/// This function walks through HTML structure tokens and tracks row/column positions,
/// accounting for colspan and rowspan attributes. The returned vector has one entry
/// per `<td>` cell in the same order as the bboxes.
///
/// # Arguments
///
/// * `tokens` - Structure tokens from table structure recognition
///
/// # Returns
///
/// A vector of `CellGridInfo` for each cell, in order of appearance.
///
/// # Example
///
/// ```ignore
/// let tokens = vec![
///     "<tr>".to_string(),
///     "<td></td>".to_string(),
///     "<td colspan=\"2\"></td>".to_string(),
///     "</tr>".to_string(),
///     "<tr>".to_string(),
///     "<td></td>".to_string(),
///     "<td></td>".to_string(),
///     "<td></td>".to_string(),
///     "</tr>".to_string(),
/// ];
/// let grid_info = parse_cell_grid_info(&tokens);
/// // First row: cell at (0,0), cell at (0,1) spanning 2 cols
/// // Second row: cells at (1,0), (1,1), (1,2)
/// ```
pub fn parse_cell_grid_info(tokens: &[String]) -> Vec<CellGridInfo> {
    let mut cells = Vec::new();
    let mut current_row: usize = 0;
    let mut current_col: usize = 0;
    let mut idx = 0usize;

    // Track which columns are occupied by rowspans from previous rows
    // Key: (row, col) -> true if occupied
    let mut occupied: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    while idx < tokens.len() {
        let token = tokens[idx].as_str();

        if token == "<tr>" {
            // Start of a new row - reset column counter
            current_col = 0;
            // Skip columns occupied by rowspans from previous rows
            while occupied.contains(&(current_row, current_col)) {
                current_col += 1;
            }
            idx += 1;
            continue;
        }

        if token == "</tr>" {
            // End of row - move to next row
            current_row += 1;
            idx += 1;
            continue;
        }

        if token == "<td></td>" {
            while occupied.contains(&(current_row, current_col)) {
                current_col += 1;
            }
            cells.push(CellGridInfo {
                row: current_row,
                col: current_col,
                row_span: 1,
                col_span: 1,
            });
            current_col += 1;
            idx += 1;
            continue;
        }

        if token.starts_with("<td") {
            let parsed = parse_td_tag(tokens, idx);

            // Skip columns occupied by rowspans
            while occupied.contains(&(current_row, current_col)) {
                current_col += 1;
            }

            // Record this cell's position
            cells.push(CellGridInfo {
                row: current_row,
                col: current_col,
                row_span: parsed.row_span,
                col_span: parsed.col_span,
            });

            // Mark cells occupied by this cell's rowspan (for future rows)
            if parsed.row_span > 1 {
                for r in 1..parsed.row_span {
                    for c in 0..parsed.col_span {
                        occupied.insert((current_row + r, current_col + c));
                    }
                }
            }

            // Advance column by colspan
            current_col += parsed.col_span;
            idx = parsed.next_index;
            continue;
        }

        idx += 1;
    }

    cells
}

/// Parses a span attribute (colspan or rowspan) from an HTML tag.
fn parse_span_attr(token: &str, attr: &str) -> Option<usize> {
    // Look for patterns like colspan="2" or rowspan="3"
    let pattern = format!("{}=\"", attr);
    if let Some(start) = token.find(&pattern) {
        let value_start = start + pattern.len();
        if let Some(end) = token[value_start..].find('"')
            && let Ok(value) = token[value_start..value_start + end].parse::<usize>()
        {
            return Some(value);
        }
    }
    None
}

/// Parsed information about a `<td>` token sequence.
#[derive(Debug, Clone)]
struct ParsedTdTag {
    /// Raw attributes (including leading spaces) to append after `<td`
    attrs: String,
    /// Rowspan value (defaults to 1)
    row_span: usize,
    /// Colspan value (defaults to 1)
    col_span: usize,
    /// Index to continue parsing from (skips attribute and closing tokens)
    next_index: usize,
}

/// Parses a `<td ...>` sequence that may be split across multiple tokens.
///
/// The Paddle table structure dictionary splits `<td` attributes into separate tokens, e.g.:
/// `["<td", " colspan=\"2\"", " rowspan=\"3\"", ">", "</td>"]`
/// This helper gathers those pieces into a single opening tag and extracts span info.
fn parse_td_tag(tokens: &[String], start_idx: usize) -> ParsedTdTag {
    let mut attrs = String::new();
    let mut col_span = 1usize;
    let mut row_span = 1usize;

    // Handle attributes that might already be embedded in the starting token (e.g., "<td colspan=\"2\">")
    if let Some(start_token) = tokens.get(start_idx)
        && let Some(stripped) = start_token.strip_prefix("<td")
        && let Some(before_gt) = stripped.split('>').next()
        && !before_gt.is_empty()
    {
        attrs.push_str(before_gt);
        if let Some(v) = parse_span_attr(before_gt, "colspan") {
            col_span = v;
        }
        if let Some(v) = parse_span_attr(before_gt, "rowspan") {
            row_span = v;
        }
    }

    let mut idx = start_idx + 1;

    // Consume subsequent attribute tokens until we hit the end of the opening tag
    while idx < tokens.len() {
        let token = tokens[idx].as_str();

        if token == ">"
            || token == "</td>"
            || token.starts_with("<td")
            || token == "<tr>"
            || token == "</tr>"
        {
            break;
        }

        attrs.push_str(token);
        if let Some(v) = parse_span_attr(token, "colspan") {
            col_span = v;
        }
        if let Some(v) = parse_span_attr(token, "rowspan") {
            row_span = v;
        }

        idx += 1;
    }

    // Skip ahead to the token after the closing `</td>` if present
    let mut next_index = idx;
    while next_index < tokens.len() {
        let token = tokens[next_index].as_str();
        if token == "</td>" {
            next_index += 1;
            break;
        }
        if token.starts_with("<td") || token == "<tr>" || token == "</tr>" {
            break;
        }
        next_index += 1;
    }

    ParsedTdTag {
        attrs,
        row_span,
        col_span,
        next_index: next_index.max(start_idx + 1),
    }
}

/// Output from table structure decoding.
#[derive(Debug, Clone)]
pub struct TableStructureDecodeOutput {
    /// HTML structure tokens for each image (without HTML wrapping)
    pub structure_tokens: Vec<Vec<String>>,
    /// Bounding boxes for table cells (4-point polygons: `[x1,y1,x2,y2,x3,y3,x4,y4]`)
    pub bboxes: Vec<Vec<[f32; 8]>>,
    /// Mean confidence scores for structure predictions
    pub structure_scores: Vec<f32>,
}

/// Table structure decoder that converts model outputs to HTML tokens and bboxes.
#[derive(Debug, Clone)]
pub struct TableStructureDecode {
    /// HTML token dictionary (e.g., `<html>`, `<table>`, `<tr>`, `<td>`, etc.)
    character_dict: Vec<String>,
    /// Special tokens to ignore during decoding
    ignored_tokens: Vec<usize>,
    /// Token indices that should have bounding boxes (e.g., `<td>`, `<td`, `<td></td>`)
    td_token_indices: Vec<usize>,
    /// End token index
    end_idx: usize,
}

impl TableStructureDecode {
    /// Creates a new table structure decoder from a dictionary file.
    ///
    /// # Alignment
    ///
    /// This follows `TableLabelDecode.add_special_char()` logic exactly:
    /// - "sos" token is prepended at index 0
    /// - "eos" token is appended at the end
    /// - The dict order is: ["sos", <original_dict...>, "eos"]
    pub fn from_dict_path(dict_path: &Path) -> Result<Self, OCRError> {
        // Load base dictionary
        let mut character_dict = Self::load_dict(dict_path)?;

        // Apply merge_no_span_structure logic
        // Default: merge_no_span_structure=True
        let merge_no_span_structure = true;
        if merge_no_span_structure {
            if !character_dict.contains(&"<td></td>".to_string()) {
                character_dict.push("<td></td>".to_string());
            }
            if let Some(pos) = character_dict.iter().position(|s| s == "<td>") {
                character_dict.remove(pos);
            }
        }

        // Add special tokens
        // CRITICAL: Use lowercase "sos" and "eos" without angle brackets
        // CRITICAL: "sos" goes at the START (index 0), "eos" goes at the END
        let beg_str = "sos";
        let end_str = "eos";

        let original_dict_size = character_dict.len();

        // Build final dict: ["sos"] + original_dict + ["eos"]
        let mut final_dict = Vec::with_capacity(original_dict_size + 2);
        final_dict.push(beg_str.to_string()); // Index 0: "sos"
        final_dict.extend(character_dict); // Index 1 to N: original dict
        final_dict.push(end_str.to_string()); // Index N+1: "eos"

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
        // "sos" is at index 0, "eos" is at the last index
        let start_idx = 0; // "sos" is always at index 0
        let end_idx = final_dict.len() - 1; // "eos" is always at the last index

        // Only ignore "sos" and "eos" tokens
        let ignored_tokens = vec![start_idx, end_idx];

        // Find TD token indices
        // Note: with merge_no_span_structure=true, "<td>" is removed and "<td></td>" is added
        let td_tokens = ["<td>", "<td", "<td></td>"];
        let td_token_indices: Vec<usize> = td_tokens
            .iter()
            .filter_map(|&token| final_dict.iter().position(|s| s == token))
            .collect();

        tracing::debug!("TD token indices: {:?}", td_token_indices);
        tracing::debug!(
            "Ignored tokens (sos={}, eos={}): {:?}",
            start_idx,
            end_idx,
            ignored_tokens
        );

        Ok(Self {
            character_dict: final_dict,
            ignored_tokens,
            td_token_indices,
            end_idx,
        })
    }

    /// Loads dictionary from file.
    ///
    /// Note: We preserve leading spaces for attribute tokens like ` colspan="2"`
    /// since they are needed to generate valid HTML like `<td colspan="2">`.
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
            // Only trim trailing whitespace, preserve leading spaces for attribute tokens
            // e.g., " colspan=\"2\"" needs the leading space for valid HTML generation
            let trimmed = line.trim_end();
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
    /// * `shape_info` - [(orig_h, orig_w, scale, pad_h, pad_w, target_size), ...] for each image.
    ///   `scale` is the ResizeByLong factor used during preprocessing: `target_size / max(orig_h, orig_w)`.
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

        // Use the mean of per-token max logits as structure score
        let mean_score = if scores.is_empty() {
            0.0
        } else {
            let sum: f32 = scores.iter().copied().sum();
            sum / (scores.len() as f32)
        };

        Ok((structure_tokens, bboxes, mean_score))
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

        // Return the raw max logit value as score
        (max_idx, max_val)
    }

    /// Extracts and denormalizes bounding box.
    ///
    /// TableLabelDecode computes bbox scales as:
    ///   `ratio = min(padded_w / orig_w, padded_h / orig_h)` and
    ///   `scale = padded_{w,h} / ratio = max(orig_w, orig_h)`.
    /// For SLANeXt / SLANet_plus models the padded input is square, so both axes
    /// use the same `scale` (the longest side of the original image).
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

        // Denormalize using shape information
        if let Some(shape) = shape_info.get(batch_idx) {
            let [orig_h, orig_w, scale, _pad_h, _pad_w, target_size] = *shape;

            if scale <= 0.0 || target_size <= 0.0 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Invalid shape info for batch {}: scale={} target_size={}",
                        batch_idx, scale, target_size
                    ),
                });
            }

            // Equivalent to TableLabelDecode _get_bbox_scales() for SLANeXt/SLANet_plus:
            // padded_w == padded_h == target_size, so target_size / scale == max(orig_w, orig_h).
            let longest_side = target_size / scale;

            // Model outputs are normalized to [0,1] w.r.t. the padded square.
            // Scale by the longest side and clamp to the original dimensions.
            for (idx, coord_ref) in bbox.iter_mut().enumerate() {
                let mut coord = *coord_ref * longest_side;

                if idx % 2 == 0 {
                    coord = coord.clamp(0.0, orig_w);
                } else {
                    coord = coord.clamp(0.0, orig_h);
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
        // Create a temporary dictionary
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

        let dict_path = Path::new("models/table_structure_dict.txt");
        if !dict_path.exists() {
            return; // Skip if dict not available
        }

        let decoder = TableStructureDecode::from_dict_path(dict_path).unwrap();

        // Create simple logits tensor
        let logits = Array3::zeros((1, 5, 50));
        let (idx, _prob) = decoder.argmax_at(&logits, 0, 0);
        assert_eq!(idx, 0); // Should be first token (all zeros)
    }

    #[test]
    fn test_parse_cell_grid_info_simple() {
        // Simple 2x2 table
        let tokens = vec![
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
        ];

        let grid = parse_cell_grid_info(&tokens);
        assert_eq!(grid.len(), 4);

        // First row
        assert_eq!(grid[0].row, 0);
        assert_eq!(grid[0].col, 0);
        assert_eq!(grid[0].row_span, 1);
        assert_eq!(grid[0].col_span, 1);

        assert_eq!(grid[1].row, 0);
        assert_eq!(grid[1].col, 1);

        // Second row
        assert_eq!(grid[2].row, 1);
        assert_eq!(grid[2].col, 0);

        assert_eq!(grid[3].row, 1);
        assert_eq!(grid[3].col, 1);
    }

    #[test]
    fn test_parse_cell_grid_info_colspan() {
        // Table with colspan
        let tokens = vec![
            "<tr>".to_string(),
            "<td colspan=\"2\"></td>".to_string(),
            "</tr>".to_string(),
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
        ];

        let grid = parse_cell_grid_info(&tokens);
        assert_eq!(grid.len(), 3);

        // First row: single cell spanning 2 columns
        assert_eq!(grid[0].row, 0);
        assert_eq!(grid[0].col, 0);
        assert_eq!(grid[0].col_span, 2);

        // Second row: two cells
        assert_eq!(grid[1].row, 1);
        assert_eq!(grid[1].col, 0);

        assert_eq!(grid[2].row, 1);
        assert_eq!(grid[2].col, 1);
    }

    #[test]
    fn test_parse_cell_grid_info_rowspan() {
        // Table with rowspan
        let tokens = vec![
            "<tr>".to_string(),
            "<td rowspan=\"2\"></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
            "<tr>".to_string(),
            "<td></td>".to_string(), // Should be at col 1, not col 0
            "</tr>".to_string(),
        ];

        let grid = parse_cell_grid_info(&tokens);
        assert_eq!(grid.len(), 3);

        // First row
        assert_eq!(grid[0].row, 0);
        assert_eq!(grid[0].col, 0);
        assert_eq!(grid[0].row_span, 2);

        assert_eq!(grid[1].row, 0);
        assert_eq!(grid[1].col, 1);

        // Second row: cell should skip col 0 (occupied by rowspan)
        assert_eq!(grid[2].row, 1);
        assert_eq!(grid[2].col, 1);
    }

    #[test]
    fn test_parse_cell_grid_info_split_tokens_with_spans() {
        // Tokens are split like Paddle's dictionary: "<td", " colspan=\"2\"", ">", "</td>"
        let tokens = vec![
            "<tr>",
            "<td",
            " colspan=\"2\"",
            ">",
            "</td>",
            "</tr>", // first row, single cell span 2 cols
            "<tr>",
            "<td",
            " rowspan=\"2\"",
            ">",
            "</td>",
            "<td></td>",
            "</tr>", // second row, first cell spans 2 rows
            "<tr>",
            "<td></td>",
            "</tr>", // third row should skip col 0 due to rowspan
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();

        let grid = parse_cell_grid_info(&tokens);
        assert_eq!(grid.len(), 4);

        // Row 0: one cell spanning two columns
        assert_eq!(grid[0].row, 0);
        assert_eq!(grid[0].col_span, 2);

        // Row 1: first cell has rowspan=2, so next cell should be at col 1
        assert_eq!(grid[1].row, 1);
        assert_eq!(grid[1].col, 0);
        assert_eq!(grid[1].row_span, 2);

        assert_eq!(grid[2].row, 1);
        assert_eq!(grid[2].col, 1);

        // Row 2: colspan from row 0 should not affect, but rowspan from row 1 should shift to col 1
        assert_eq!(grid[3].row, 2);
        assert_eq!(grid[3].col, 1);
    }

    #[test]
    fn test_wrap_table_html_with_split_tokens() {
        let tokens = vec!["<tr>", "<td", " colspan=\"2\"", ">", "</td>", "</tr>"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();

        let cell_texts = vec![Some("Cell A".to_string())];
        let html = wrap_table_html_with_content(&tokens, &cell_texts);

        assert!(html.contains("<td colspan=\"2\">Cell A</td>"));
        assert!(html.starts_with("<html><body><table>"));
        assert!(html.ends_with("</table></body></html>"));
    }

    #[test]
    fn test_parse_span_attr() {
        assert_eq!(parse_span_attr("<td colspan=\"2\">", "colspan"), Some(2));
        assert_eq!(parse_span_attr("<td rowspan=\"3\">", "rowspan"), Some(3));
        assert_eq!(
            parse_span_attr("<td colspan=\"2\" rowspan=\"3\">", "colspan"),
            Some(2)
        );
        assert_eq!(
            parse_span_attr("<td colspan=\"2\" rowspan=\"3\">", "rowspan"),
            Some(3)
        );
        assert_eq!(parse_span_attr("<td></td>", "colspan"), None);
        assert_eq!(parse_span_attr("<td>", "rowspan"), None);
    }

    #[test]
    fn test_extract_bbox_longest_side_scaling_matches_standard() {
        let decoder = TableStructureDecode {
            character_dict: Vec::new(),
            ignored_tokens: Vec::new(),
            td_token_indices: Vec::new(),
            end_idx: 0,
        };

        // Simulate normalized bbox predictions for a portrait image (orig 300x600)
        let mut bbox_preds = ndarray::Array3::<f32>::zeros((1, 1, 8));
        let preds = [0.45f32, 0.25, 0.9, 0.25, 0.45, 0.8, 0.9, 0.8];
        for (i, val) in preds.iter().enumerate() {
            bbox_preds[[0, 0, i]] = *val;
        }

        let orig_h: f32 = 600.0;
        let orig_w: f32 = 300.0;
        let target_size: f32 = 512.0;
        let scale = target_size / orig_h.max(orig_w); // ResizeByLong scale
        let pad_h = 0.0;
        let pad_w = target_size - (orig_w * scale); // Padding on the right
        let shape_info = [[orig_h, orig_w, scale, pad_h, pad_w, target_size]];

        let bbox = decoder
            .extract_bbox(&bbox_preds, 0, 0, &shape_info)
            .expect("bbox decode");

        // TableLabelDecode _get_bbox_scales for SLANeXt/SLANet_plus:
        // ratio = target_size / max(orig_w, orig_h) => denorm factor = max dim.
        let longest_side = orig_h.max(orig_w);
        let expected = [
            (preds[0] * longest_side).clamp(0.0, orig_w),
            (preds[1] * longest_side).clamp(0.0, orig_h),
            (preds[2] * longest_side).clamp(0.0, orig_w),
            (preds[3] * longest_side).clamp(0.0, orig_h),
            (preds[4] * longest_side).clamp(0.0, orig_w),
            (preds[5] * longest_side).clamp(0.0, orig_h),
            (preds[6] * longest_side).clamp(0.0, orig_w),
            (preds[7] * longest_side).clamp(0.0, orig_h),
        ];

        for (idx, (got, exp)) in bbox.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "bbox coord {} mismatch: got {}, expected {}",
                idx,
                got,
                exp
            );
        }
    }
}
