//! Shared OTSL tokenization, cell ownership, and HTML emission.

use super::{OTSL_TOKEN_RE, clean_html_table, simple_otsl_conversion, strip_table_tokens_fallback};
use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum RowPadding {
    #[default]
    Repair,
    Widest,
    Ragged,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum CellMarkup {
    #[default]
    EscapeText,
    EscapeAll,
    PreserveMarkup,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum SpanSearch {
    #[default]
    Contiguous,
    NearestAnchor,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum NoTokens {
    #[default]
    Tsv,
    Empty,
    Preserve,
    EmptyTable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum HtmlInput {
    #[default]
    Clean,
    CompletePassThrough,
    Tokenize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OtslOptions {
    pub row_padding: RowPadding,
    pub cell_markup: CellMarkup,
    pub span_search: SpanSearch,
    pub no_tokens: NoTokens,
    pub html_input: HtmlInput,
    pub empty_fcel_is_anchor: bool,
    pub preserve_ecel_text: bool,
    pub emit_unowned_cells: bool,
    pub trim_text: bool,
    pub trim_input: bool,
    pub case_insensitive: bool,
    pub newline_to_br: bool,
    pub unwrap_otsl: bool,
}

impl Default for OtslOptions {
    fn default() -> Self {
        Self {
            row_padding: RowPadding::Repair,
            cell_markup: CellMarkup::EscapeText,
            span_search: SpanSearch::Contiguous,
            no_tokens: NoTokens::Tsv,
            html_input: HtmlInput::Clean,
            empty_fcel_is_anchor: false,
            preserve_ecel_text: false,
            emit_unowned_cells: false,
            trim_text: true,
            trim_input: true,
            case_insensitive: false,
            newline_to_br: false,
            unwrap_otsl: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Full,
    Empty,
    Left,
    Up,
    Cross,
    Newline,
}

impl Kind {
    fn parse(text: &str) -> Self {
        match text.as_bytes()[1].to_ascii_lowercase() {
            b'f' => Self::Full,
            b'e' => Self::Empty,
            b'l' => Self::Left,
            b'u' => Self::Up,
            b'x' => Self::Cross,
            _ => Self::Newline,
        }
    }
}

#[derive(Clone, Copy)]
struct Token<'a> {
    kind: Kind,
    text: &'a str,
    following: &'a str,
}

impl Token<'_> {
    const EMPTY: Self = Self {
        kind: Kind::Empty,
        text: "",
        following: "",
    };
}

struct Cell<'a> {
    text: &'a str,
    row: usize,
    col: usize,
    rowspan: usize,
    colspan: usize,
}

pub(crate) fn convert_otsl_to_html_with_options(input: &str, options: &OtslOptions) -> String {
    // PaddleX (Apache-2.0), MinerU post_process/otsl2html.py (Apache-2.0),
    // MonkeyOCRv2 core_runner.py (Apache-2.0), and Diffusion run_end2end.py (MIT).
    let text = if options.trim_input {
        input.trim()
    } else {
        input
    };
    if text.trim().is_empty() {
        return if options.no_tokens == NoTokens::EmptyTable {
            "<table></table>".into()
        } else {
            String::new()
        };
    }
    match options.html_input {
        HtmlInput::Clean if text.contains("<table") => return clean_html_table(text),
        HtmlInput::CompletePassThrough
            if text.starts_with("<table") && text.ends_with("</table>") =>
        {
            return text.into();
        }
        _ => {}
    }
    static WRAPPER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<otsl>(.*?)</otsl>").unwrap());
    if options.unwrap_otsl && WRAPPER.is_match(text) {
        return WRAPPER
            .replace_all(text, |caps: &regex::Captures<'_>| {
                convert_otsl_to_html_with_options(&caps[1], options)
            })
            .into_owned();
    }
    static CASELESS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)<(?:fcel|ecel|lcel|ucel|xcel|nl)>").unwrap());
    let regex = if options.case_insensitive {
        &*CASELESS
    } else {
        &*OTSL_TOKEN_RE
    };
    let matches: Vec<_> = regex.find_iter(text).collect();
    if matches.is_empty() {
        return match options.no_tokens {
            NoTokens::Tsv => simple_otsl_conversion(text),
            NoTokens::Empty => String::new(),
            NoTokens::Preserve => text.trim().into(),
            NoTokens::EmptyTable => "<table><tr></tr></table>".into(),
        };
    }
    let mut rows = vec![Vec::new()];
    let mut has_newline = false;
    for (index, token) in matches.iter().enumerate() {
        let kind = Kind::parse(token.as_str());
        if kind == Kind::Newline {
            has_newline = true;
            rows.push(Vec::new());
        } else {
            let next = matches.get(index + 1);
            rows.last_mut().unwrap().push(Token {
                kind,
                text: &text[token.end()..next.map_or(text.len(), |m| m.start())],
                following: next.map_or("", |m| m.as_str()),
            });
        }
    }
    if options.row_padding == RowPadding::Ragged {
        if matches
            .last()
            .is_some_and(|m| Kind::parse(m.as_str()) == Kind::Newline && m.end() == text.len())
        {
            rows.pop();
        }
    } else {
        rows.retain(|row| !row.is_empty());
        let width = padded_width(&rows, options.row_padding, has_newline);
        for row in &mut rows {
            row.resize(width, Token::EMPTY);
        }
    }
    let mut owners: Vec<Vec<Option<usize>>> = rows.iter().map(|r| vec![None; r.len()]).collect();
    let mut cells: Vec<Cell<'_>> = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        for (c, token) in row.iter().enumerate() {
            if options.span_search == SpanSearch::NearestAnchor {
                let owner = match token.kind {
                    Kind::Left => owners[r][..c].iter().rev().flatten().next().copied(),
                    Kind::Up => owners[..r]
                        .iter()
                        .rev()
                        .find_map(|row| row.get(c).copied().flatten()),
                    _ => None,
                };
                if let Some(owner) = owner {
                    if token.kind == Kind::Left {
                        cells[owner].colspan += 1;
                    } else {
                        cells[owner].rowspan += 1;
                    }
                    continue;
                }
                if token.kind == Kind::Cross {
                    continue;
                }
            } else if !matches!(token.kind, Kind::Full | Kind::Empty) {
                continue;
            }
            let mut content = if token.kind == Kind::Full
                || (token.kind == Kind::Empty && options.preserve_ecel_text)
            {
                token.text
            } else {
                ""
            };
            let legacy_empty = token.kind == Kind::Full
                && content.trim().is_empty()
                && !options.empty_fcel_is_anchor;
            if legacy_empty {
                content = token.following;
            }
            if options.trim_text {
                content = content.trim();
            }
            let mut cell = Cell {
                text: content,
                row: r,
                col: c,
                rowspan: 1,
                colspan: 1,
            };
            if options.span_search == SpanSearch::Contiguous {
                let next = row.get(c + if legacy_empty { 2 } else { 1 });
                if next.is_some_and(|t| matches!(t.kind, Kind::Left | Kind::Cross)) {
                    cell.colspan += row[c + 1..]
                        .iter()
                        .take_while(|t| matches!(t.kind, Kind::Left | Kind::Cross))
                        .count();
                }
                cell.rowspan += rows[r + 1..]
                    .iter()
                    .take_while(|row| {
                        row.get(c)
                            .is_some_and(|t| matches!(t.kind, Kind::Up | Kind::Cross))
                    })
                    .count();
            }
            let index = cells.len();
            for owner_row in owners.iter_mut().skip(r).take(cell.rowspan) {
                for slot in owner_row.iter_mut().skip(c).take(cell.colspan) {
                    *slot = Some(index);
                }
            }
            cells.push(cell);
        }
    }
    if cells.is_empty() && options.no_tokens == NoTokens::Tsv {
        return strip_table_tokens_fallback(text);
    }
    if cells.is_empty() && options.no_tokens == NoTokens::Empty {
        return String::new();
    }
    let mut html = String::from("<table>");
    for (r, row) in owners.iter().enumerate() {
        html.push_str("<tr>");
        for (c, owner) in row.iter().enumerate() {
            let Some(index) = owner else {
                if options.emit_unowned_cells {
                    html.push_str("<td></td>");
                }
                continue;
            };
            let cell = &cells[*index];
            if cell.row != r || cell.col != c {
                continue;
            }
            html.push_str("<td");
            if cell.rowspan > 1 {
                html.push_str(&format!(" rowspan=\"{}\"", cell.rowspan));
            }
            if cell.colspan > 1 {
                html.push_str(&format!(" colspan=\"{}\"", cell.colspan));
            }
            html.push('>');
            html.push_str(&cell_html(cell.text, options));
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

fn padded_width(rows: &[Vec<Token<'_>>], padding: RowPadding, has_newline: bool) -> usize {
    let widest = rows.iter().map(Vec::len).max().unwrap_or_default();
    if padding != RowPadding::Repair || !has_newline {
        return widest;
    }
    let minimum = rows
        .iter()
        .filter_map(|row| {
            row.iter()
                .rposition(|t| t.kind == Kind::Full)
                .map(|c| c + 1)
        })
        .max()
        .unwrap_or_default();
    (minimum..=widest)
        .min_by_key(|&width| {
            rows.iter()
                .map(|row| row.len().abs_diff(width))
                .sum::<usize>()
        })
        .unwrap_or(widest)
}

fn cell_html(text: &str, options: &OtslOptions) -> String {
    let escape = |text: &str| {
        let mut out = html_escape::encode_text(text).into_owned();
        if options.cell_markup != CellMarkup::EscapeText {
            out = out.replace('"', "&quot;").replace('\'', "&#x27;");
        }
        if options.newline_to_br {
            out = out.replace('\n', "<br>");
        }
        out
    };
    if options.cell_markup != CellMarkup::PreserveMarkup {
        return escape(text);
    }
    static ESCAPED: LazyLock<Regex> =
        LazyLock::new(|| Regex::new("(?i)\u{e100}(fcel|ecel|lcel|ucel|xcel|nl)>").unwrap());
    static MARKUP: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?is)</?[a-z][^>]*>|&(?:#\d+|#x[0-9a-f]+|[a-z][a-z0-9]+);").unwrap()
    });
    static BR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\A<br\s*/?>\z").unwrap());
    let text = text.replace("\u{e100}\u{e100}", "\u{e100}");
    let text = ESCAPED
        .replace_all(&text, "<$1>")
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let mut html = String::new();
    let mut end = 0;
    for token in MARKUP.find_iter(&text) {
        html.push_str(&escape(&text[end..token.start()]));
        html.push_str(if BR.is_match(token.as_str()) {
            "<br>"
        } else {
            token.as_str()
        });
        end = token.end();
    }
    html.push_str(&escape(&text[end..]));
    html
}
