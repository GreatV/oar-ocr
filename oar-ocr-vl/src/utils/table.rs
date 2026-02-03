use once_cell::sync::Lazy;
use regex::Regex;

// Shared regex patterns
pub static TABLE_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"</?(table|tr|th|td|thead|tbody|tfoot)[^>]*>").expect("static regex"));
static OTSL_TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(<fcel>|<lcel>|<ucel>|<xcel>|<ecel>|<nl>)").expect("static regex"));

const OTSL_NL: &str = "<nl>";
const OTSL_FCEL: &str = "<fcel>";
const OTSL_ECEL: &str = "<ecel>";
const OTSL_LCEL: &str = "<lcel>";
const OTSL_UCEL: &str = "<ucel>";
const OTSL_XCEL: &str = "<xcel>";

/// Convert OTSL table tokens (or TSV text) to HTML table.
pub fn convert_otsl_to_html(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // If it's already HTML, just clean it
    if trimmed.contains("<table") {
        return clean_html_table(trimmed);
    }

    // If it contains OTSL tokens, try PaddleX-compatible conversion first.
    if looks_like_table_tokens(trimmed) {
        if let Some(html) = try_convert_table_tokens_to_html(trimmed) {
            return html;
        }
        // Fallback if parsing fails
        return strip_table_tokens_fallback(trimmed);
    }

    // If no tags, treat as simple TSV
    simple_otsl_conversion(trimmed)
}

pub fn clean_html_table(text: &str) -> String {
    let mut result = text.to_string();
    result = result.replace("<tdcolspan=", "<td colspan=");
    result = result.replace("<tdrowspan=", "<td rowspan=");
    result = result.replace("colspan=", " colspan=");
    result = result.replace("<|sn|>", "");
    result = result.replace("<|unk|>", "");
    result = result.replace('\u{FFFF}', "");
    result
}

fn simple_otsl_conversion(text: &str) -> String {
    let mut html = String::from("<table>");
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        html.push_str("<tr>");
        for cell in line.split('\t') {
            html.push_str("<td>");
            html.push_str(&html_escape::encode_text(cell.trim()));
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

pub fn looks_like_table_tokens(input: &str) -> bool {
    input.contains("<fcel>")
        || input.contains("<lcel>")
        || input.contains("<ucel>")
        || input.contains("<xcel>")
        || input.contains("<ecel>")
        || input.contains("<nl>")
}

fn strip_table_tokens_fallback(input: &str) -> String {
    let mut out = input.replace("<ecel>", "\n").replace("<nl>", "\n");
    out = out
        .replace("<fcel>", "\t")
        .replace("<lcel>", "")
        .replace("<ucel>", "")
        .replace("<xcel>", "");
    out.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn try_convert_table_tokens_to_html(input: &str) -> Option<String> {
    let padded = otsl_pad_to_sqr_v2(input);
    let (tokens, texts) = otsl_extract_tokens_and_text(&padded);
    if tokens.is_empty() {
        return None;
    }
    let (table_cells, split_row_tokens) = otsl_parse_texts(&texts, &tokens);
    let num_rows = split_row_tokens.len();
    let num_cols = split_row_tokens
        .iter()
        .map(|row| row.len())
        .max()
        .unwrap_or(0);
    if num_rows == 0 || num_cols == 0 {
        return None;
    }
    let html = otsl_export_to_html(&table_cells, num_rows, num_cols);
    if html.is_empty() { None } else { Some(html) }
}

#[derive(Debug, Clone)]
struct TableCell {
    row_span: usize,
    col_span: usize,
    start_row_offset_idx: usize,
    end_row_offset_idx: usize,
    start_col_offset_idx: usize,
    end_col_offset_idx: usize,
    text: String,
}

fn otsl_pad_to_sqr_v2(otsl_str: &str) -> String {
    let otsl_str = otsl_str.trim();
    if !otsl_str.contains(OTSL_NL) {
        return format!("{otsl_str}{OTSL_NL}");
    }
    let lines: Vec<&str> = otsl_str.split(OTSL_NL).collect();
    let mut row_segments: Vec<Vec<String>> = Vec::new();
    let mut row_lengths: Vec<usize> = Vec::new();
    let mut row_min_lengths: Vec<usize> = Vec::new();

    for line in lines {
        if line.is_empty() {
            continue;
        }
        let segments = split_otsl_segments(line);
        if segments.is_empty() {
            continue;
        }
        let total_len = segments.len();
        let mut min_len = 0;
        for (i, segment) in segments.iter().enumerate() {
            if segment.starts_with(OTSL_FCEL) {
                min_len = i + 1;
            }
        }
        row_segments.push(segments.into_iter().map(|s| s.to_string()).collect());
        row_lengths.push(total_len);
        row_min_lengths.push(min_len);
    }

    if row_segments.is_empty() {
        return OTSL_NL.to_string();
    }

    let global_min_width = row_min_lengths.into_iter().max().unwrap_or(0);
    let max_total_len = row_lengths.iter().copied().max().unwrap_or(0);
    let search_start = global_min_width;
    let search_end = std::cmp::max(global_min_width, max_total_len);
    let mut min_total_cost = usize::MAX;
    let mut optimal_width = search_end;
    for width in search_start..=search_end {
        let cost: usize = row_lengths.iter().map(|len| (*len).abs_diff(width)).sum();
        if cost < min_total_cost {
            min_total_cost = cost;
            optimal_width = width;
        }
    }

    let mut repaired_lines = Vec::new();
    for mut segments in row_segments {
        if segments.len() > optimal_width {
            segments.truncate(optimal_width);
        } else if segments.len() < optimal_width {
            segments.extend(std::iter::repeat_n(
                OTSL_ECEL.to_string(),
                optimal_width - segments.len(),
            ));
        }
        repaired_lines.push(segments.concat());
    }
    let mut output = repaired_lines.join(OTSL_NL);
    output.push_str(OTSL_NL);
    output
}

/// Split an OTSL line into segments, where each segment starts with a token
/// and includes any text up to the next token.
/// Any leading text before the first token is prepended to the first segment.
fn split_otsl_segments(line: &str) -> Vec<&str> {
    let matches: Vec<regex::Match<'_>> = OTSL_TOKEN_RE.find_iter(line).collect();
    if matches.is_empty() {
        return Vec::new();
    }
    let mut segments = Vec::with_capacity(matches.len());
    let first_token_start = matches[0].start();
    for (idx, mat) in matches.iter().enumerate() {
        // For the first segment, include any leading text before the first token
        let start = if idx == 0 { 0 } else { mat.start() };
        let end = matches
            .get(idx + 1)
            .map(|next| next.start())
            .unwrap_or_else(|| line.len());
        // Skip empty leading prefix (when first token is at position 0)
        if idx == 0 && first_token_start == 0 {
            segments.push(&line[mat.start()..end]);
        } else {
            segments.push(&line[start..end]);
        }
    }
    segments
}

fn otsl_extract_tokens_and_text(input: &str) -> (Vec<String>, Vec<String>) {
    let mut tokens = Vec::new();
    let mut parts = Vec::new();
    let mut last = 0usize;
    for mat in OTSL_TOKEN_RE.find_iter(input) {
        let before = &input[last..mat.start()];
        if !before.trim().is_empty() {
            parts.push(before.to_string());
        }
        let token = mat.as_str();
        tokens.push(token.to_string());
        if !token.trim().is_empty() {
            parts.push(token.to_string());
        }
        last = mat.end();
    }
    let trailing = &input[last..];
    if !trailing.trim().is_empty() {
        parts.push(trailing.to_string());
    }
    (tokens, parts)
}

fn otsl_parse_texts(texts: &[String], tokens: &[String]) -> (Vec<TableCell>, Vec<Vec<String>>) {
    let mut split_row_tokens = Vec::new();
    let mut current_row = Vec::new();
    for token in tokens {
        if token == OTSL_NL {
            if !current_row.is_empty() {
                split_row_tokens.push(std::mem::take(&mut current_row));
            }
        } else {
            current_row.push(token.clone());
        }
    }
    if !current_row.is_empty() {
        split_row_tokens.push(current_row);
    }

    let mut normalized_texts = texts.to_vec();
    if !split_row_tokens.is_empty() {
        let max_cols = split_row_tokens
            .iter()
            .map(|row| row.len())
            .max()
            .unwrap_or(0);
        for row in split_row_tokens.iter_mut() {
            while row.len() < max_cols {
                row.push(OTSL_ECEL.to_string());
            }
        }

        let mut new_texts = Vec::new();
        let mut text_idx = 0usize;
        for row in split_row_tokens.iter() {
            for token in row.iter() {
                new_texts.push(token.clone());
                if text_idx < normalized_texts.len() && normalized_texts[text_idx] == *token {
                    text_idx += 1;
                    if text_idx < normalized_texts.len()
                        && !is_otsl_tag(&normalized_texts[text_idx])
                    {
                        new_texts.push(normalized_texts[text_idx].clone());
                        text_idx += 1;
                    }
                }
            }
            new_texts.push(OTSL_NL.to_string());
            if text_idx < normalized_texts.len() && normalized_texts[text_idx] == OTSL_NL {
                text_idx += 1;
            }
        }
        normalized_texts = new_texts;
    }

    fn is_l_or_x(token: &str) -> bool {
        token == OTSL_LCEL || token == OTSL_XCEL
    }

    fn is_u_or_x(token: &str) -> bool {
        token == OTSL_UCEL || token == OTSL_XCEL
    }

    fn count_right(tokens: &[Vec<String>], c_idx: usize, r_idx: usize) -> usize {
        let mut span = 0usize;
        let mut c = c_idx;
        while r_idx < tokens.len() && c < tokens[r_idx].len() && is_l_or_x(&tokens[r_idx][c]) {
            span += 1;
            c += 1;
        }
        span
    }

    fn count_down(tokens: &[Vec<String>], c_idx: usize, r_idx: usize) -> usize {
        let mut span = 0usize;
        let mut r = r_idx;
        while r < tokens.len() && c_idx < tokens[r].len() && is_u_or_x(&tokens[r][c_idx]) {
            span += 1;
            r += 1;
        }
        span
    }

    let mut table_cells = Vec::new();
    let mut r_idx = 0usize;
    let mut c_idx = 0usize;
    for i in 0..normalized_texts.len() {
        let text = normalized_texts[i].as_str();
        if text == OTSL_FCEL || text == OTSL_ECEL {
            let mut row_span = 1usize;
            let mut col_span = 1usize;
            let mut right_offset = 1usize;
            let mut cell_text = String::new();
            if text != OTSL_ECEL {
                cell_text = normalized_texts.get(i + 1).cloned().unwrap_or_default();
                right_offset = 2;
            }

            let next_right_cell = normalized_texts
                .get(i + right_offset)
                .map(String::as_str)
                .unwrap_or("");
            let next_bottom_cell = if r_idx + 1 < split_row_tokens.len()
                && c_idx < split_row_tokens[r_idx + 1].len()
            {
                split_row_tokens[r_idx + 1][c_idx].as_str()
            } else {
                ""
            };

            if is_l_or_x(next_right_cell) {
                col_span += count_right(&split_row_tokens, c_idx + 1, r_idx);
            }
            if is_u_or_x(next_bottom_cell) {
                row_span += count_down(&split_row_tokens, c_idx, r_idx + 1);
            }

            table_cells.push(TableCell {
                text: cell_text.trim().to_string(),
                row_span,
                col_span,
                start_row_offset_idx: r_idx,
                end_row_offset_idx: r_idx + row_span,
                start_col_offset_idx: c_idx,
                end_col_offset_idx: c_idx + col_span,
            });
        }
        if text == OTSL_FCEL
            || text == OTSL_ECEL
            || text == OTSL_LCEL
            || text == OTSL_UCEL
            || text == OTSL_XCEL
        {
            c_idx += 1;
        }
        if text == OTSL_NL {
            r_idx += 1;
            c_idx = 0;
        }
    }

    (table_cells, split_row_tokens)
}

fn is_otsl_tag(token: &str) -> bool {
    matches!(
        token,
        OTSL_NL | OTSL_FCEL | OTSL_ECEL | OTSL_LCEL | OTSL_UCEL | OTSL_XCEL
    )
}

fn otsl_export_to_html(table_cells: &[TableCell], num_rows: usize, num_cols: usize) -> String {
    if table_cells.is_empty() {
        return String::new();
    }
    let mut grid = vec![vec![None; num_cols]; num_rows];
    for (idx, cell) in table_cells.iter().enumerate() {
        let end_r = cell.end_row_offset_idx.min(num_rows);
        let end_c = cell.end_col_offset_idx.min(num_cols);
        for row in grid.iter_mut().take(end_r).skip(cell.start_row_offset_idx) {
            for cell_ref in row.iter_mut().take(end_c).skip(cell.start_col_offset_idx) {
                *cell_ref = Some(idx);
            }
        }
    }

    let mut body = String::new();
    for (i, row) in grid.iter().enumerate().take(num_rows) {
        body.push_str("<tr>");
        for (j, cell_idx) in row.iter().enumerate().take(num_cols) {
            let Some(idx) = cell_idx else {
                continue;
            };
            let cell = &table_cells[*idx];
            if cell.start_row_offset_idx != i || cell.start_col_offset_idx != j {
                continue;
            }
            let mut opening = String::from("td");
            if cell.row_span > 1 {
                opening.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
            }
            if cell.col_span > 1 {
                opening.push_str(&format!(" colspan=\"{}\"", cell.col_span));
            }
            let content = html_escape::encode_text(cell.text.trim());
            body.push_str(&format!("<{}>{}</td>", opening, content));
        }
        body.push_str("</tr>");
    }
    format!("<table>{}</table>", body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_otsl_conversion() {
        let input = "a\tb\tc\nd\te\tf";
        let html = simple_otsl_conversion(input);
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>a</td>"));
    }
}
