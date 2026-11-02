//! Protected paragraph splitting and mixed inline math.
//!
//! Structural rules follow MiniCPM-OCR's Apache-2.0 postprocess/markdown.py.

use regex::Regex;
use std::sync::LazyLock;

fn inline_spans(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if text[i..].starts_with(r"\(") {
            if let Some(end) = text[i + 2..].find(r"\)") {
                let end = i + 2 + end + 2;
                if !text[i..end].contains('\n') {
                    spans.push((i, end));
                    i = end;
                    continue;
                }
            }
        } else if bytes[i] == b'$'
            && (i == 0 || !matches!(bytes[i - 1], b'\\' | b'$'))
            && bytes.get(i + 1) != Some(&b'$')
        {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b'\n' {
                if bytes[j] == b'$' && bytes[j - 1] != b'\\' {
                    if bytes.get(j + 1) != Some(&b'$') && j > i + 1 {
                        spans.push((i, j + 1));
                        i = j + 1;
                    }
                    break;
                }
                j += 1;
            }
            if i == j + 1 {
                continue;
            }
        }
        i += text[i..].chars().next().map_or(1, char::len_utf8);
    }
    spans
}

pub(crate) fn has_mixed_inline_math(text: &str) -> bool {
    let text = text.trim();
    let spans = inline_spans(text);
    let mut position = 0;
    for (start, end) in spans {
        if !text[position..start].trim().is_empty() {
            return true;
        }
        position = end;
    }
    !text[position..].trim().is_empty()
}

fn pipe_row(line: &str) -> bool {
    let line = line.trim();
    line.matches('|').count() >= 2 && (line.starts_with('|') || line.contains("---"))
}

pub(crate) fn split_protected_paragraphs(text: &str, protect_inline: bool) -> String {
    let lines: Vec<_> = text.split('\n').collect();
    let mut output = Vec::new();
    let (mut code, mut dollar, mut bracket, mut table, mut inline_dollar, mut inline_paren) =
        (false, false, false, false, false, false);
    let mut environments = 0i64;
    for (i, &line) in lines.iter().enumerate() {
        let stripped = line.trim();
        if stripped.starts_with("```") {
            output.push(line.to_string());
            code = !code;
            continue;
        }
        if code {
            output.push(line.to_string());
            continue;
        }
        let lower = stripped.to_ascii_lowercase();
        table |= lower.contains("<table");
        if table {
            output.push(line.to_string());
            if lower.contains("</table>") {
                table = false;
            }
            continue;
        }
        if pipe_row(line) {
            output.push(line.to_string());
            if !lines.get(i + 1).is_some_and(|line| pipe_row(line)) && !line.trim().is_empty() {
                output.push(String::new());
            }
            continue;
        }
        let bytes = line.as_bytes();
        let mut doubles = 0;
        let mut singles = 0;
        let mut offset = 0;
        while offset < bytes.len() {
            if bytes[offset] == b'$' && (offset == 0 || bytes[offset - 1] != b'\\') {
                if bytes.get(offset + 1) == Some(&b'$') {
                    doubles += 1;
                    offset += 2;
                    continue;
                }
                if offset == 0 || bytes[offset - 1] != b'$' {
                    singles += 1;
                }
            }
            offset += 1;
        }
        let bracket_opens = line
            .split_once(r"\[")
            .is_some_and(|(_, after)| !after.contains(r"\]"));
        let paren_opens = line
            .split_once(r"\(")
            .is_some_and(|(_, after)| !after.contains(r"\)"));
        let net = line.matches(r"\begin{").count() as i64 - line.matches(r"\end{").count() as i64;
        if dollar || bracket || inline_dollar || inline_paren || environments > 0 {
            output.push(line.to_string());
            if dollar && doubles % 2 == 1 {
                dollar = false;
            }
            if bracket && line.contains(r"\]") {
                bracket = false;
            }
            if inline_dollar && singles % 2 == 1 {
                inline_dollar = false;
            }
            if inline_paren && line.contains(r"\)") {
                inline_paren = false;
            }
            environments = (environments + net).max(0);
        } else if doubles % 2 == 1 || bracket_opens || net > 0 {
            output.push(line.to_string());
            dollar = doubles % 2 == 1;
            bracket = bracket_opens;
            environments = net.max(0);
        } else if protect_inline && (singles % 2 == 1 || paren_opens) {
            output.push(line.to_string());
            inline_dollar = singles % 2 == 1;
            inline_paren = paren_opens;
        } else if !stripped.is_empty() {
            if output.last().is_some_and(|s| !s.trim().is_empty()) {
                output.push(String::new());
            }
            output.push(stripped.to_string());
            output.push(String::new());
        }
    }
    collapse_blank_lines(&output.join("\n"))
        .trim_matches('\n')
        .into()
}

pub(crate) fn collapse_blank_lines(text: &str) -> String {
    static BLANKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());
    BLANKS.replace_all(text, "\n\n").into_owned()
}

pub(crate) fn normalize_crop_text(text: &str, display: bool) -> String {
    static INLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\\((.+?)\\\)").unwrap());
    static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());
    let text = super::repetition::truncate_word_repetition(text.trim(), 6, 64, 3, true);
    let text = if display {
        text
    } else {
        INLINE
            .replace_all(&text, |c: &regex::Captures<'_>| {
                format!("${}$", c[1].trim())
            })
            .into_owned()
    };
    collapse_blank_lines(&SPACES.replace_all(&text, " "))
}
