use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

// Shared regex patterns
pub static TABLE_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"</?(table|tr|th|td|thead|tbody|tfoot)[^>]*>").expect("static regex"));
static OTSL_TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(<fcel>|<lcel>|<ucel>|<xcel>|<ecel>|<nl>)").expect("static regex"));

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

    // If it contains OTSL tokens, try to parse with UnionFind strategy
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
    let first = OTSL_TOKEN_RE.find(input)?;
    let prefix = input[..first.start()].trim();
    let tokenized = &input[first.start()..];

    let mut rows = parse_table_token_rows(tokenized);
    if rows.is_empty() {
        return None;
    }
    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 {
        return None;
    }

    // Pad rows
    for row in rows.iter_mut() {
        if row.len() < col_count {
            row.extend(std::iter::repeat_n(
                (TableCellMarker::F, String::new()),
                col_count - row.len(),
            ));
        }
    }

    let row_count = rows.len();
    let cell_count = row_count * col_count;
    let mut uf = UnionFind::new(cell_count);

    // Build merge sets
    for (r, row) in rows.iter().enumerate().take(row_count) {
        for (c, cell) in row.iter().enumerate().take(col_count) {
            let idx = r * col_count + c;
            match cell.0 {
                TableCellMarker::F => {} // No merging needed
                TableCellMarker::L => {
                    if c > 0 {
                        uf.union(idx, idx - 1);
                    }
                }
                TableCellMarker::U => {
                    if r > 0 {
                        uf.union(idx, idx - col_count);
                    }
                }
                TableCellMarker::X => {
                    if c > 0 {
                        uf.union(idx, idx - 1);
                    }
                    if r > 0 {
                        uf.union(idx, idx - col_count);
                    }
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    struct CellInfo {
        min_r: usize,
        min_c: usize,
        max_r: usize,
        max_c: usize,
        text: String,
        text_priority: u8,
        text_r: usize,
        text_c: usize,
        has_text: bool,
    }

    let mut infos: HashMap<usize, CellInfo> = HashMap::new();
    for (r, row) in rows.iter().enumerate().take(row_count) {
        for (c, cell) in row.iter().enumerate().take(col_count) {
            let idx = r * col_count + c;
            let root = uf.find(idx);
            let entry = infos.entry(root).or_insert(CellInfo {
                min_r: r,
                min_c: c,
                max_r: r,
                max_c: c,
                text: String::new(),
                text_priority: u8::MAX,
                text_r: r,
                text_c: c,
                has_text: false,
            });

            entry.min_r = entry.min_r.min(r);
            entry.min_c = entry.min_c.min(c);
            entry.max_r = entry.max_r.max(r);
            entry.max_c = entry.max_c.max(c);

            let text = cell.1.trim();
            if !text.is_empty() {
                let priority = if cell.0 == TableCellMarker::F { 0 } else { 1 };
                let better = !entry.has_text
                    || priority < entry.text_priority
                    || (priority == entry.text_priority && (r, c) < (entry.text_r, entry.text_c));
                if better {
                    entry.text = text.to_string();
                    entry.text_priority = priority;
                    entry.text_r = r;
                    entry.text_c = c;
                    entry.has_text = true;
                }
            }
        }
    }

    let mut html = String::new();
    html.push_str("<table>");
    if !prefix.is_empty() {
        html.push_str("<caption>");
        html.push_str(&html_escape::encode_text(prefix));
        html.push_str("</caption>");
    }
    html.push_str("<tbody>");

    for r in 0..row_count {
        html.push_str("<tr>");
        for c in 0..col_count {
            let idx = r * col_count + c;
            let root = uf.find(idx);
            let info = infos.get(&root)?;
            // Only emit cell if it's the top-left of the merged region
            if info.min_r == r && info.min_c == c {
                let rowspan = info.max_r - info.min_r + 1;
                let colspan = info.max_c - info.min_c + 1;
                html.push_str("<td");
                if rowspan > 1 {
                    html.push_str(&format!(" rowspan=\"{rowspan}\"",));
                }
                if colspan > 1 {
                    html.push_str(&format!(" colspan=\"{colspan}\"",));
                }
                html.push('>');
                html.push_str(&html_escape::encode_text(info.text.trim()));
                html.push_str("</td>");
            }
        }
        html.push_str("</tr>");
    }

    html.push_str("</tbody></table>");
    Some(html)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableCellMarker {
    F,
    L,
    U,
    X,
}

impl TableCellMarker {
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "<fcel>" => Some(Self::F),
            "<lcel>" => Some(Self::L),
            "<ucel>" => Some(Self::U),
            "<xcel>" => Some(Self::X),
            _ => None,
        }
    }
}

fn parse_table_token_rows(input: &str) -> Vec<Vec<(TableCellMarker, String)>> {
    let mut rows = Vec::new();
    let mut current_row: Vec<(TableCellMarker, String)> = Vec::new();
    let mut current_marker: Option<TableCellMarker> = None;
    let mut current_text = String::new();
    let mut cursor = 0usize;

    for mat in OTSL_TOKEN_RE.find_iter(input) {
        let between = &input[cursor..mat.start()];
        if current_marker.is_some() {
            current_text.push_str(between);
        }
        cursor = mat.end();

        match mat.as_str() {
            "<fcel>" | "<lcel>" | "<ucel>" | "<xcel>" => {
                if let Some(marker) = current_marker.take() {
                    current_row.push((marker, current_text.trim().to_string()));
                    current_text.clear();
                }
                current_marker = TableCellMarker::from_token(mat.as_str());
            }
            "<ecel>" | "<nl>" => {
                if let Some(marker) = current_marker.take() {
                    current_row.push((marker, current_text.trim().to_string()));
                    current_text.clear();
                }
                if !current_row.is_empty() {
                    rows.push(std::mem::take(&mut current_row));
                }
            }
            _ => {}
        }
    }

    // Handle any remaining text after the last token
    if let Some(marker) = current_marker.take() {
        current_text.push_str(&input[cursor..]);
        current_row.push((marker, current_text.trim().to_string()));
    }
    if !current_row.is_empty() {
        rows.push(current_row);
    }
    rows
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            return x;
        }
        let root = self.find(self.parent[x]);
        self.parent[x] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
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
