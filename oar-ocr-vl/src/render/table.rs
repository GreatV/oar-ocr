mod pipe;
pub(crate) use pipe::{
    malformed_table as malformed_markdown_table, markdown_table_to_html, repair_chart_table,
};
mod otsl;
pub(crate) use otsl::{
    CellMarkup, HtmlInput, NoTokens, OtslOptions, RowPadding, SpanSearch,
    convert_otsl_to_html_with_options,
};

use once_cell::sync::Lazy;
use regex::Regex;

pub(crate) const TABLE_FIGURE_LABEL: &str = "__table_figure";

pub(crate) fn table_figure_image(index: usize) -> String {
    format!("<img src=\"image_{}.png\" alt=\"Image\" />", index + 1)
}

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

#[derive(Debug, Clone, Copy)]
pub(crate) enum TableWrapper {
    Literal,
    Fragment,
    Document,
}

pub(crate) fn table_span(input: &str, wrapper: TableWrapper) -> Option<&str> {
    static FRAGMENT: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?is)<table\b[^>]*>.*?</table\s*>").unwrap());
    static DOCUMENT: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?is)\A<table\b[^>]*>.*?</table\s*>\z").unwrap());
    match wrapper {
        TableWrapper::Literal => {
            let start = input.find("<table>")?;
            let end = input[start..].find("</table>")?;
            Some(&input[start..start + end + "</table>".len()])
        }
        TableWrapper::Fragment => FRAGMENT.find(input).map(|m| m.as_str()),
        TableWrapper::Document => DOCUMENT.find(input).map(|m| m.as_str()),
    }
}

/// Convert an HTML `<table>` snippet to PaddleOCR-VL's raw OTSL token form.
///
/// PaddleOCR-VL's "Table Recognition:" prompt emits OTSL tokens
/// (`<fcel>`, `<lcel>`, `<ucel>`, `<xcel>`, `<ecel>`, `<nl>`) which the
/// model's post-process then translates into HTML. To feed an HTML-shaped
/// table (e.g. from a layout pipeline) back into PaddleOCR-VL's token form we
/// need the inverse conversion from HTML to OTSL.
///
/// The parser is intentionally tolerant — it uses regex-based extraction
/// rather than a full HTML parser and mirrors `clean_html_table`'s repair of
/// common attribute typos (`<tdcolspan=`, `<tdrowspan=`). The cell regexes
/// match case-insensitively, so tag casing in the input is preserved through
/// the precheck.
///
/// ## Algorithm
///
/// 1. Extract rows (`<tr>...</tr>`) and within each row, cells
///    (`<td>...</td>` / `<th>...</th>`) with optional `colspan` / `rowspan`.
/// 2. Lay cells onto a 2D grid, skipping positions occupied by an earlier
///    cell's row/col span.
/// 3. Emit row-by-row: each original cell anchor becomes `<fcel>content` (or
///    `<ecel>` if empty); spanned positions emit `<lcel>` (col-only),
///    `<ucel>` (row-only), or `<xcel>` (both). End each row with `<nl>`.
///
/// Returns `None` when the input is empty, contains no `<tr>` tag, or has no
/// parseable cells. Callers can then decide whether to skip the draft.
pub fn convert_html_to_otsl(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() || !TR_OPEN_RE.is_match(trimmed) {
        return None;
    }
    // Repair the common `<tdcolspan=` / `<tdrowspan=` typos before parsing.
    let repaired = trimmed
        .replace("<tdcolspan=", "<td colspan=")
        .replace("<tdrowspan=", "<td rowspan=");

    // Parsed rows: each row is a Vec of (rowspan, colspan, text).
    // Empty rows are preserved — a `<tr></tr>` can occur when the previous
    // row's `rowspan` consumes the entire row, and dropping it would break
    // the grid's row count.
    let mut rows: Vec<Vec<(usize, usize, String)>> = Vec::new();
    for tr_caps in TR_RE.captures_iter(&repaired) {
        let Some(inner) = tr_caps.get(1) else {
            continue;
        };
        let mut cells: Vec<(usize, usize, String)> = Vec::new();
        for cell_caps in CELL_RE.captures_iter(inner.as_str()) {
            let attrs = cell_caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let body = cell_caps.get(2).map(|m| m.as_str()).unwrap_or("");
            let colspan = extract_span(attrs, "colspan");
            let rowspan = extract_span(attrs, "rowspan");
            let text = clean_cell_text(body);
            cells.push((rowspan, colspan, text));
        }
        rows.push(cells);
    }
    if rows.is_empty() {
        return None;
    }

    // Determine final grid width: widest row after applying colspans.
    let mut num_cols = 0usize;
    for cells in &rows {
        let row_width: usize = cells.iter().map(|(_, cs, _)| *cs).sum();
        num_cols = num_cols.max(row_width);
    }
    let num_rows = rows.len();
    if num_cols == 0 {
        return None;
    }

    // Place cells onto the grid. `cell_grid[r][c]` is either `None` (still
    // unassigned), or `Some((anchor_r, anchor_c, rowspan, colspan, text))`.
    type Anchor = (usize, usize, usize, usize, String);
    let mut grid: Vec<Vec<Option<Anchor>>> = vec![vec![None; num_cols]; num_rows];
    for (r, cells) in rows.into_iter().enumerate() {
        let mut c = 0usize;
        for (rowspan, colspan, text) in cells {
            // Skip positions already occupied by a previous rowspan.
            while c < num_cols && grid[r][c].is_some() {
                c += 1;
            }
            if c >= num_cols {
                break;
            }
            let rs = rowspan.max(1);
            let cs = colspan.max(1);
            let rs_end = (r + rs).min(num_rows);
            let cs_end = (c + cs).min(num_cols);
            for row in grid[r..rs_end].iter_mut() {
                for slot in row[c..cs_end].iter_mut() {
                    *slot = Some((r, c, rs, cs, text.clone()));
                }
            }
            c += cs;
        }
    }

    // Emit OTSL row by row.
    let mut out = String::new();
    for (r, row) in grid.iter().enumerate() {
        for (c, slot) in row.iter().enumerate() {
            match slot {
                None => out.push_str(OTSL_ECEL),
                Some((anchor_r, anchor_c, _rs, _cs, text)) => {
                    let is_row_anchor = *anchor_r == r;
                    let is_col_anchor = *anchor_c == c;
                    match (is_row_anchor, is_col_anchor) {
                        (true, true) => {
                            if text.is_empty() {
                                out.push_str(OTSL_ECEL);
                            } else {
                                out.push_str(OTSL_FCEL);
                                out.push_str(text);
                            }
                        }
                        (true, false) => out.push_str(OTSL_LCEL),
                        (false, true) => out.push_str(OTSL_UCEL),
                        (false, false) => out.push_str(OTSL_XCEL),
                    }
                }
            }
        }
        out.push_str(OTSL_NL);
    }
    Some(out)
}

static TR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<tr[^>]*>(.*?)</tr>").expect("static regex: <tr>...</tr>"));
static TR_OPEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)<tr[\s>]").expect("static regex: <tr opening"));
static CELL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<t[dh]([^>]*)>(.*?)</t[dh]>").expect("static regex: <td|th>...</td|th>")
});
static STRIP_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<[^>]*>").expect("static regex: html tag stripper"));
// `colspan` / `rowspan` attribute scanners. The leading `(?:^|\s)` anchors
// the match so substrings like `data-colspan=` or `class="mycolspan"` don't
// trip the parser (mis-extracting a span from an unrelated attribute).
static COLSPAN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(?:^|\s)colspan\s*=\s*"?(\d+)"?"#).expect("static regex: colspan attr")
});
static ROWSPAN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?i)(?:^|\s)rowspan\s*=\s*"?(\d+)"?"#).expect("static regex: rowspan attr")
});

fn extract_span(attrs: &str, name: &str) -> usize {
    let re: &Regex = match name {
        "colspan" => &COLSPAN_RE,
        "rowspan" => &ROWSPAN_RE,
        _ => return 1,
    };
    re.captures(attrs)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1)
}

fn clean_cell_text(body: &str) -> String {
    // Strip any nested tags (rare in OCR tables but possible — e.g. <br>, <b>).
    let stripped = STRIP_TAG_RE.replace_all(body, "");
    // Decode the few HTML entities the existing post-process emits via
    // `html_escape::encode_text`. We avoid pulling a full entity decoder for
    // a hot path — these five cover what `otsl_export_to_html` produces.
    let decoded = stripped
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'");
    decoded.trim().to_string()
}

/// Convert OTSL table tokens (or TSV text) to HTML table.
pub fn convert_otsl_to_html(input: &str) -> String {
    convert_otsl_to_html_with_options(input, &OtslOptions::default())
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

    #[test]
    fn convert_html_to_otsl_simple_grid() {
        // 2x2 plain table, no merges.
        let html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>");
    }

    #[test]
    fn convert_html_to_otsl_empty_cells_become_ecel() {
        let html = "<table><tr><td>a</td><td></td></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>a<ecel><nl>");
    }

    #[test]
    fn convert_html_to_otsl_colspan_emits_lcel() {
        // Row 1 converts <td colspan="2">A</td> to <fcel>A<lcel>.
        let html = "<table><tr><td colspan=\"2\">A</td></tr><tr><td>x</td><td>y</td></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>A<lcel><nl><fcel>x<fcel>y<nl>");
    }

    #[test]
    fn convert_html_to_otsl_rowspan_emits_ucel() {
        // Column 0 spans both rows; column 1 is two separate cells.
        let html = "<table><tr><td rowspan=\"2\">A</td><td>b</td></tr><tr><td>c</td></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>A<fcel>b<nl><ucel><fcel>c<nl>");
    }

    #[test]
    fn convert_html_to_otsl_xcel_for_combined_span() {
        // 2x2 cell merged together; the bottom-right corner must be <xcel>.
        let html = "<table><tr><td colspan=\"2\" rowspan=\"2\">A</td></tr><tr></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>A<lcel><nl><ucel><xcel><nl>");
    }

    #[test]
    fn convert_html_to_otsl_handles_tdcolspan_typo() {
        // PaddleOCR-VL post-process repairs `<tdcolspan=` — accept the typo on input too.
        let html = "<table><tr><tdcolspan=\"2\">A</td></tr><tr><td>x</td><td>y</td></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>A<lcel><nl><fcel>x<fcel>y<nl>");
    }

    #[test]
    fn convert_html_to_otsl_decodes_html_entities() {
        // The forward converter html_escapes content; the inverse must
        // round-trip the most common entities.
        let html = "<table><tr><td>a &amp; b</td><td>x &lt; y</td></tr></table>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>a & b<fcel>x < y<nl>");
    }

    #[test]
    fn convert_html_to_otsl_returns_none_for_non_table_input() {
        assert!(convert_html_to_otsl("plain text").is_none());
        assert!(convert_html_to_otsl("<p>not a table</p>").is_none());
        assert!(convert_html_to_otsl("").is_none());
    }

    #[test]
    fn convert_html_to_otsl_accepts_uppercase_tags() {
        // The cell regexes are case-insensitive; the precheck must be too,
        // otherwise mixed-case HTML drops on the floor.
        let html = "<TABLE><TR><TD>a</TD><TD>b</TD></TR></TABLE>";
        let otsl = convert_html_to_otsl(html).expect("conversion");
        assert_eq!(otsl, "<fcel>a<fcel>b<nl>");
    }

    #[test]
    fn extract_span_ignores_substring_attribute_matches() {
        // `data-colspan=` / `xrowspan=` are unrelated attributes that happen
        // to end with our needle. The scanner must not pick numbers out of
        // them — a plain `<td>` should still yield colspan=1, rowspan=1.
        assert_eq!(extract_span(r#" data-colspan="7""#, "colspan"), 1);
        assert_eq!(extract_span(r#" xrowspan="9""#, "rowspan"), 1);
        assert_eq!(extract_span(r#" class="mycolspan""#, "colspan"), 1);
        // Genuine match still works, including unquoted and mixed-case forms.
        assert_eq!(extract_span(r#" colspan="3""#, "colspan"), 3);
        assert_eq!(extract_span(" COLSPAN=4", "colspan"), 4);
        assert_eq!(extract_span(r#" class="data" rowspan="2""#, "rowspan"), 2);
    }

    #[test]
    fn convert_html_to_otsl_roundtrips_through_otsl_to_html() {
        // An OTSL-to-HTML-to-OTSL round trip should reconstruct the same OTSL for a simple
        // grid (modulo whitespace handling). This guards against drift between
        // the two converters.
        let otsl_in = "<fcel>a<fcel>b<nl><fcel>c<fcel>d<nl>";
        let html = convert_otsl_to_html(otsl_in);
        let otsl_out = convert_html_to_otsl(&html).expect("round-trip");
        assert_eq!(otsl_out, otsl_in);
        let widest = OtslOptions {
            row_padding: RowPadding::Widest,
            cell_markup: CellMarkup::EscapeAll,
            no_tokens: NoTokens::Empty,
            html_input: HtmlInput::CompletePassThrough,
            empty_fcel_is_anchor: true,
            emit_unowned_cells: true,
            trim_input: false,
            ..Default::default()
        };
        assert_eq!(
            convert_otsl_to_html_with_options("<fcel>A<fcel>B<nl><fcel><lcel><fcel>C<nl>", &widest),
            "<table><tr><td>A</td><td>B</td><td></td></tr><tr><td colspan=\"2\"></td><td>C</td></tr></table>"
        );
        assert_eq!(convert_otsl_to_html_with_options("plain", &widest), "");
        let explicit = OtslOptions {
            preserve_ecel_text: true,
            emit_unowned_cells: false,
            no_tokens: NoTokens::Preserve,
            ..widest
        };
        // Synthetic PaddleX and MinerU reference outputs, selected by dialect.
        let fixtures: Vec<serde_json::Value> =
            serde_json::from_str(include_str!("table/fixtures/otsl_reference.json")).unwrap();
        for fixture in fixtures {
            let options = if fixture["dialect"] == "repair" {
                OtslOptions {
                    cell_markup: CellMarkup::EscapeAll,
                    emit_unowned_cells: true,
                    ..Default::default()
                }
            } else {
                widest
            };
            assert_eq!(
                convert_otsl_to_html_with_options(fixture["input"].as_str().unwrap(), &options),
                fixture["expected"].as_str().unwrap(),
                "{fixture}"
            );
        }
        assert_eq!(
            convert_otsl_to_html_with_options("plain", &explicit),
            "plain"
        );
        assert_eq!(
            convert_otsl_to_html_with_options("<ecel>x & y<nl>", &explicit),
            "<table><tr><td>x &amp; y</td></tr></table>"
        );
        let ragged = OtslOptions {
            row_padding: RowPadding::Ragged,
            cell_markup: CellMarkup::PreserveMarkup,
            span_search: SpanSearch::NearestAnchor,
            no_tokens: NoTokens::EmptyTable,
            html_input: HtmlInput::Tokenize,
            trim_text: false,
            trim_input: false,
            case_insensitive: true,
            newline_to_br: true,
            unwrap_otsl: true,
            empty_fcel_is_anchor: true,
            ..Default::default()
        };
        assert_eq!(
            convert_otsl_to_html_with_options("<FCEL>A<NL>", &ragged),
            "<table><tr><td>A</td></tr></table>"
        );
        assert_eq!(
            convert_otsl_to_html_with_options(
                "<otsl><fcel> A<br><sup>2</sup> &amp;\n<lcel><nl><ecel><nl></otsl>",
                &ragged
            ),
            "<table><tr><td colspan=\"2\"> A<br><sup>2</sup> &amp;<br></td></tr><tr><td></td></tr></table>"
        );
        assert_eq!(
            convert_otsl_to_html_with_options("<fcel>A<lcel><nl><ucel><xcel><nl>", &ragged),
            "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td></tr><tr></tr></table>"
        );
    }
}
