//! MonkeyOCRv2 table and formula formatting.

use regex::Regex;
use std::sync::LazyLock;

#[derive(Default)]
struct Cell {
    text: String,
    rowspan: usize,
    colspan: usize,
}

// Adapted from MonkeyOCRv2 parsing/core_runner.py (Apache-2.0). Unlike the
// PaddleX converter, spans search left/up, xcel only occupies a grid slot,
// rows stay ragged, and serialized cell markup/whitespace is preserved.
pub(super) fn otsl_to_html(text: &str) -> String {
    static WRAPPER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<otsl>(.*?)</otsl>").unwrap());
    static TOKEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)<(fcel|ecel|lcel|ucel|xcel|nl)>").unwrap());
    if text.trim().is_empty() {
        return "<table></table>".into();
    }
    if WRAPPER.is_match(text) {
        return WRAPPER
            .replace_all(text, |caps: &regex::Captures<'_>| otsl_to_html(&caps[1]))
            .into_owned();
    }
    let tokens: Vec<_> = TOKEN.captures_iter(text).collect();
    let mut grid: Vec<Vec<Option<usize>>> = vec![Vec::new()];
    let mut cells: Vec<Cell> = Vec::new();
    let mut row = 0;
    for (index, caps) in tokens.iter().enumerate() {
        let tag = caps[1].to_ascii_lowercase();
        if tag == "nl" {
            row += 1;
            grid.push(Vec::new());
            continue;
        }
        let col = grid[row].len();
        let owner = match tag.as_str() {
            "lcel" => grid[row]
                .iter()
                .rev()
                .flatten()
                .copied()
                .find(|&i| cells[i].rowspan > 0),
            "ucel" => grid[..row]
                .iter()
                .rev()
                .filter_map(|r| r.get(col).copied().flatten())
                .find(|&i| cells[i].rowspan > 0),
            _ => None,
        };
        if let Some(owner) = owner {
            if tag == "lcel" {
                cells[owner].colspan += 1;
            } else {
                cells[owner].rowspan += 1;
            }
            // Continuation slots are not anchors for subsequent searches.
            grid[row].push(None);
        } else if tag == "xcel" {
            grid[row].push(None);
        } else {
            let start = caps.get(0).unwrap().end();
            let end = tokens
                .get(index + 1)
                .map_or(text.len(), |next| next.get(0).unwrap().start());
            cells.push(Cell {
                text: if tag == "fcel" {
                    text[start..end].into()
                } else {
                    String::new()
                },
                rowspan: 1,
                colspan: 1,
            });
            grid[row].push(Some(cells.len() - 1));
        }
    }
    if tokens.last().is_some_and(|caps| {
        caps[1].eq_ignore_ascii_case("nl") && caps.get(0).unwrap().end() == text.len()
    }) {
        grid.pop();
    }
    let mut html = String::from("<table>");
    for row in grid {
        html.push_str("<tr>");
        for index in row.into_iter().flatten() {
            let cell = &cells[index];
            html.push_str("<td");
            if cell.rowspan > 1 {
                html.push_str(&format!(" rowspan=\"{}\"", cell.rowspan));
            }
            if cell.colspan > 1 {
                html.push_str(&format!(" colspan=\"{}\"", cell.colspan));
            }
            html.push('>');
            html.push_str(&cell_html(&cell.text));
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

fn cell_html(text: &str) -> String {
    static ESCAPED_TOKEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new("(?i)\u{e100}(fcel|ecel|lcel|ucel|xcel|nl)>").unwrap());
    static MARKUP: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?is)</?[a-z][^>]*>|&(?:#\d+|#x[0-9a-f]+|[a-z][a-z0-9]+);").unwrap()
    });
    static BR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\A<br\s*/?>\z").unwrap());
    let text = text.replace("\u{e100}\u{e100}", "\u{e100}");
    let text = ESCAPED_TOKEN
        .replace_all(&text, "<$1>")
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let escape = |s: &str| {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;")
            .replace('\n', "<br>")
    };
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

pub(super) fn format_formula(text: &str) -> String {
    static QUAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:\\quad\s*){5,}").unwrap());
    static QQUAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:\\qquad\s*){5,}").unwrap());
    static TAG: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$|\\tag\{([^{}]*)\}\s*$").unwrap()
    });
    static BEGIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\A\\begin\{([^}]+)\}").unwrap());
    let text = QUAD.replace_all(text.trim().trim_matches('$').trim(), r"\quad ");
    let mut content = QQUAD.replace_all(&text, r"\qquad ").trim().to_string();
    let mut extracted = None;
    let mut environment = None;
    for pass in 0..2 {
        if let Some(caps) = TAG.captures(&content) {
            // The reference reads only group 1 and loses a \tag{} value.
            // Preserve that value as well instead of introducing content loss.
            extracted = caps
                .get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().to_string());
            content.truncate(caps.get(0).unwrap().start());
            content = content.trim_end().into();
        }
        if pass == 0
            && let Some(caps) = BEGIN.captures(&content)
        {
            let name = caps[1].to_string();
            content = content[caps.get(0).unwrap().end()..].trim_start().into();
            if let Some(body) = content.trim_end().strip_suffix(&format!("\\end{{{name}}}")) {
                content = body.trim_end().into();
            }
            environment = Some(name);
        }
    }
    if let Some(name) = environment {
        content = format!("\\begin{{{name}}}\n{content}\n\\end{{{name}}}");
    }
    format!("$$\n{content}\n$$\n{}", extracted.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_tables_preserve_ragged_spans_and_formula_numbers() {
        assert_eq!(
            otsl_to_html("<FCEL>A<NL>"),
            "<table><tr><td>A</td></tr></table>"
        );
        assert_eq!(
            otsl_to_html("<otsl><fcel> A<br><sup>2</sup> &amp;\n<lcel><nl><ecel><nl></otsl>"),
            "<table><tr><td colspan=\"2\"> A<br><sup>2</sup> &amp;<br></td></tr><tr><td></td></tr></table>"
        );
        assert_eq!(
            otsl_to_html("<fcel>A<lcel><nl><ucel><xcel><nl>"),
            "<table><tr><td rowspan=\"2\" colspan=\"2\">A</td></tr><tr></tr></table>"
        );
        assert_eq!(
            format_formula(
                r"$$\begin{aligned}x=1\quad\quad\quad\quad\quad\quad (3)\end{aligned}$$"
            ),
            "$$\n\\begin{aligned}\nx=1\n\\end{aligned}\n$$\n3"
        );
        assert_eq!(format_formula(r"x=1\tag{4}"), "$$\nx=1\n$$\n4");
    }
}
