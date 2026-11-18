//! Render the LaTeX tabular dialect used by Nemotron Parse, retaining cell spans and explicit headers.
use regex::{Captures, Regex};
use std::{collections::HashMap, fmt::Write, sync::LazyLock};

static ENV: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}").unwrap()
});
static BOLD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*(.+?)\*\*").unwrap());
static ITALIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*([^*]+)\*").unwrap());
static TAGS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static NUMERIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[\s\d.,%$€£()+−-]+$").unwrap());

fn argument(input: &str) -> Option<(&str, &str)> {
    let s = input.trim_start().strip_prefix('{')?;
    let mut depth = 1;
    let mut escaped = false;
    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[..i], &s[i + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
    let (mut depth, mut start, mut i) = (0usize, 0, 0);
    let mut out = Vec::new();
    while i < s.len() {
        let rest = &s[i..];
        if depth == 0 && rest.starts_with(sep) {
            out.push(&s[start..i]);
            i += sep.len();
            start = i;
            continue;
        }
        let ch = rest.chars().next().unwrap();
        i += ch.len_utf8();
        match ch {
            '\\' => i += s[i..].chars().next().map_or(0, char::len_utf8),
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Parse nested rich-table markup before unescaping literal punctuation.
fn rich_inline(text: &str) -> Option<String> {
    fn escaped(out: &mut String, ch: char) {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    fn parse<'a>(
        mut input: &'a str,
        close: Option<&str>,
        depth: usize,
    ) -> Option<(String, &'a str)> {
        if depth > 64 {
            return None;
        }
        let mut out = String::new();
        while !input.is_empty() {
            if let Some(tail) = close.and_then(|marker| input.strip_prefix(marker)) {
                return Some((out, tail));
            }
            if let Some(tail) = input
                .strip_prefix("<br>")
                .or_else(|| input.strip_prefix('\n'))
            {
                out.push_str("<br/>");
                input = tail;
                continue;
            }
            let marker = [
                ("**", "**", "b"),
                ("_", "_", "i"),
                ("~~", "~~", "s"),
                ("<u>", "</u>", "u"),
                ("<s>", "</s>", "s"),
                ("<i>", "</i>", "i"),
                ("<sup>", "</sup>", "sup"),
                ("<sub>", "</sub>", "sub"),
            ]
            .into_iter()
            .find(|(open, _, _)| input.starts_with(open));
            let mark = input
                .strip_prefix("<mark")
                .filter(|tail| tail.starts_with('>') || tail.starts_with(char::is_whitespace))
                .and_then(|tail| {
                    tail.find('>')
                        .map(|i| (&input[..5 + i + 1], "</mark>", "mark"))
                });
            if let Some((open, closing, tag)) = marker.or(mark) {
                let (body, tail) = parse(&input[open.len()..], Some(closing), depth + 1)?;
                write!(out, "<{tag}>{body}</{tag}>").ok()?;
                input = tail;
                continue;
            }
            let ch = input.chars().next()?;
            input = &input[ch.len_utf8()..];
            if ch != '\\' {
                if "%$#_*{}^~<>".contains(ch) {
                    return None;
                }
                escaped(&mut out, ch);
                continue;
            }
            let next = input.chars().next()?;
            if "&%$#_*{}".contains(next) {
                escaped(&mut out, next);
                input = &input[next.len_utf8()..];
                continue;
            }
            let n = input.bytes().take_while(u8::is_ascii_alphabetic).count();
            let (command, rest) = input.split_at(n);
            let (mut body, mut tail) = argument(rest)?;
            let tag = match command {
                "textstrong" | "textbf" => "b",
                "textem" | "textit" => "i",
                "underline" => "u",
                "sout" => "s",
                "textsuperscript" => "sup",
                "textsubscript" => "sub",
                "colorbox" => {
                    (body, tail) = argument(tail)?;
                    "mark"
                }
                "hl" => "mark",
                _ if body.is_empty() => {
                    out.push_str(match command {
                        "linebreak" => "<br/>",
                        "mbox" => "",
                        "textbackslash" => "\\",
                        "textasciitilde" => "~",
                        "textasciicircum" => "^",
                        "textless" => "&lt;",
                        "textgreater" => "&gt;",
                        _ => return None,
                    });
                    input = tail;
                    continue;
                }
                _ => return None,
            };
            let (nested, _) = parse(body, None, depth + 1)?;
            write!(out, "<{tag}>{nested}</{tag}>").ok()?;
            input = tail;
        }
        close.is_none().then_some((out, input))
    }
    parse(text, None, 0).map(|(html, _)| html)
}

fn inline(mut s: &str, depth: usize) -> String {
    if depth > 64 {
        return s.to_owned();
    }
    let mut out = String::new();
    while let Some(i) = s.find('\\') {
        out.push_str(&s[..i]);
        s = &s[i + 1..];
        let len = s.bytes().take_while(u8::is_ascii_alphabetic).count();
        let (cmd, rest) = s.split_at(len);
        if cmd.is_empty() {
            if let Some(ch) = rest.chars().next() {
                out.push(ch);
                s = &rest[ch.len_utf8()..];
            } else {
                out.push('\\');
                break;
            }
            continue;
        }
        let tag = match cmd {
            "textstrong" | "textbf" => "b",
            "textit" | "emph" => "i",
            "underline" | "textunderline" => "u",
            "sout" | "st" | "textstrike" => "s",
            "textsuperscript" => "sup",
            "textsubscript" => "sub",
            "texttt" => "code",
            "textmark" => "mark",
            _ => "",
        };
        if !tag.is_empty()
            && let Some((body, tail)) = argument(rest)
        {
            let _ = write!(out, "<{tag}>{}</{tag}>", inline(body, depth + 1));
            s = tail;
            continue;
        }
        let literal = match cmd {
            "newline" => "<br/>",
            "textasciitilde" => "~",
            "textasciicircum" => "^",
            "textbackslash" => "\\",
            "textless" => "&lt;",
            "textgreater" => "&gt;",
            "unknown" => "",
            _ => {
                out.push('\\');
                out.push_str(cmd);
                s = rest;
                continue;
            }
        };
        out.push_str(literal);
        s = rest.strip_prefix("{}").unwrap_or(rest);
    }
    out.push_str(s);
    let bold = BOLD.replace_all(&out, "<b>$1</b>");
    ITALIC.replace_all(&bold, "<i>$1</i>").into_owned()
}

/// The legacy converter accepts inline HTML in a cell, including unclosed inline tags.
fn finish_cell_html(fragment: &str) -> String {
    static TAG: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<(/?)([a-z][a-z0-9:_-]*)\b[^>]*>").unwrap());
    let text = |out: &mut String, value: &str| {
        let value = html_escape::decode_html_entities(value);
        out.push_str(&html_escape::encode_text(&value));
    };
    let mut result = String::new();
    let mut open = Vec::<String>::new();
    let mut end = 0;
    for tag in TAG.captures_iter(fragment) {
        let span = tag.get(0).unwrap();
        text(&mut result, &fragment[end..span.start()]);
        end = span.end();
        let name = tag[2].to_ascii_lowercase();
        if &tag[1] == "/" {
            if let Some(depth) = open.iter().rposition(|t| *t == name) {
                for t in open.drain(depth..).rev() {
                    let _ = write!(result, "</{t}>");
                }
            }
        } else if matches!(
            name.as_str(),
            "area"
                | "base"
                | "br"
                | "col"
                | "embed"
                | "hr"
                | "img"
                | "input"
                | "link"
                | "meta"
                | "param"
                | "source"
                | "track"
                | "wbr"
        ) {
            result.push_str(span.as_str().trim_end_matches("/>").trim_end_matches('>'));
            result.push_str("/>");
        } else {
            result.push_str(span.as_str());
            open.push(name);
        }
    }
    text(&mut result, &fragment[end..]);
    for name in open.into_iter().rev() {
        let _ = write!(result, "</{name}>");
    }
    result
}

struct Cell {
    rows: usize,
    cols: usize,
    header: bool,
    html: String,
}
impl Cell {
    fn parse(mut text: &str, native: bool) -> Option<Self> {
        let (mut rows, mut cols) = (1, 1);
        text = text.trim();
        while let Some((is_row, rest)) = text
            .strip_prefix(r"\multirow")
            .map(|v| (true, v))
            .or_else(|| text.strip_prefix(r"\multicolumn").map(|v| (false, v)))
        {
            let (count, rest) = argument(rest)?;
            let (_, rest) = argument(rest)?;
            let (body, tail) = argument(rest)?;
            if !tail.trim().is_empty() {
                return None;
            }
            let n = count
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|v| (1..=4096).contains(v))?;
            if is_row {
                rows = n;
            } else {
                cols = n;
            }
            text = body;
        }
        let header = text.starts_with(r"\thead{");
        if header {
            let (body, tail) = argument(&text[6..])?;
            if !tail.trim().is_empty() {
                return None;
            }
            text = body;
        }
        Some(Self {
            rows,
            cols,
            header,
            html: if native {
                rich_inline(text)?
            } else {
                finish_cell_html(&inline(if text == "{}" { "" } else { text }, 0))
            },
        })
    }
    fn label(&self) -> String {
        let text = TAGS.replace_all(&self.html, " ");
        html_escape::decode_html_entities(&text)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn bold(&self) -> bool {
        self.html.contains("<b>") || self.html.contains("<strong>")
    }
}

fn infer_headers(rows: &mut [Vec<Cell>]) {
    if rows.len() < 2 || rows.iter().flatten().any(|c| c.header) {
        return;
    }
    let mut start = 0;
    while start < 3
        && start + 1 < rows.len()
        && matches!(&rows[start][..], [c] if c.cols>1 && c.rows==1)
    {
        start += 1;
    }
    let row = &rows[start];
    let labels: Vec<_> = row
        .iter()
        .map(Cell::label)
        .filter(|s| !s.is_empty())
        .collect();
    let textual = |s: &str| !s.is_empty() && !NUMERIC.is_match(s);
    let mut end = start + 1;
    if !labels.is_empty() && labels.iter().filter(|s| textual(s)).count() * 2 >= labels.len() {
        let span = row.iter().map(|c| c.rows).max().unwrap_or(1);
        if (2..=4).contains(&span) {
            end = (start + span).min(rows.len());
        } else if span == 1
            && row.iter().any(|c| c.cols > 1)
            && let Some(next) = rows.get(start + 1)
        {
            let nonempty: Vec<_> = next.iter().filter(|c| !c.label().is_empty()).collect();
            if nonempty.len() >= 2
                && nonempty.iter().all(|c| c.bold())
                && nonempty.iter().filter(|c| textual(&c.label())).count() * 2 > nonempty.len()
            {
                end += 1;
            }
        }
    }
    if end > start + 1 {
        for c in rows[start..end].iter_mut().flatten() {
            c.header = true;
        }
        return;
    }
    let first = &rows[0];
    if (2..=20).contains(&first.len())
        && rows[1].len() >= 2
        && first.iter().all(|c| c.label().chars().count() <= 100)
        && first.iter().filter(|c| textual(&c.label())).count() * 5 >= first.len() * 3
        && first.iter().filter(|c| c.bold()).count() * 5 >= first.len() * 3
    {
        for c in &mut rows[0] {
            c.header = true;
        }
    }
}

fn table(body: &str, headers: bool, columns: Option<usize>) -> Option<String> {
    let native = columns.is_some();
    let clean = if native {
        std::borrow::Cow::Borrowed(body)
    } else {
        std::borrow::Cow::Owned(body.replace(r"\hline", ""))
    };
    let raw_rows = split_top(clean.trim(), r"\\");
    let raw_rows: Vec<_> = raw_rows
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect();
    let count = raw_rows.len();
    if native && (count == 0 || count > 4096 || columns? > 512) {
        return None;
    }
    let mut rows = Vec::with_capacity(count);
    let mut occupied = HashMap::new();
    for (r, row) in raw_rows.into_iter().enumerate() {
        let mut cells = Vec::new();
        let mut col = 0;
        for part in split_top(row, "&") {
            let c = Cell::parse(part, native)?;
            if columns.is_some_and(|n| col + c.cols > n || r + c.rows > count) {
                return None;
            }
            let span = col..col + c.cols;
            let covered = |x| occupied.get(&x).is_some_and(|&end| r < end);
            let placeholder = c.html.is_empty()
                && c.rows == 1
                && (!native || !c.header)
                && span.clone().all(covered);
            if !placeholder {
                if native && span.clone().any(covered) {
                    return None;
                }
                if c.rows > 1 {
                    for x in span.clone() {
                        occupied.insert(x, r + c.rows);
                    }
                }
                cells.push(c);
            }
            col = span.end;
        }
        if columns.is_some_and(|n| col != n) {
            return None;
        }
        rows.push(cells);
    }
    if headers {
        infer_headers(&mut rows);
    }
    let mut out = String::from("<table>");
    for row in rows {
        if !native {
            out.push('\n');
        }
        out.push_str("<tr>");
        for c in row {
            if !native {
                out.push('\n');
            }
            let tag = if c.header { "th" } else { "td" };
            let _ = write!(out, "<{tag}");
            if native || c.cols > 1 {
                let _ = write!(out, " colspan=\"{}\"", c.cols);
            }
            if native || c.rows > 1 {
                let _ = write!(out, " rowspan=\"{}\"", c.rows);
            }
            let _ = write!(out, ">{}</{tag}>", c.html);
        }
        if !native {
            out.push('\n');
        }
        out.push_str("</tr>");
    }
    if !native {
        out.push('\n');
    }
    out.push_str("</table>");
    Some(out)
}

/// Apply the checkpoint's header inference to literal HTML tables in the Markdown stream.
pub(crate) fn promote_html_headers(text: &str) -> std::borrow::Cow<'_, str> {
    static TABLE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?is)<table\b[^>]*>.*?</table>").unwrap());
    static HEADER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)<th\b").unwrap());
    if !text.as_bytes().contains(&b'<') {
        return std::borrow::Cow::Borrowed(text);
    }
    TABLE.replace_all(text, |table: &Captures<'_>| {
        let source = &table[0];
        if HEADER.is_match(source) {
            return source.to_owned();
        }
        let mut rows: Vec<Vec<Cell>> = super::TR_RE
            .captures_iter(source)
            .map(|tr| {
                super::CELL_RE
                    .captures_iter(&tr[1])
                    .map(|cell| Cell {
                        cols: super::extract_span(&cell[1], "colspan"),
                        rows: super::extract_span(&cell[1], "rowspan"),
                        html: cell[2].to_owned(),
                        header: false,
                    })
                    .collect()
            })
            .collect();
        infer_headers(&mut rows);
        if !rows.iter().flatten().any(|cell| cell.header) {
            return source.to_owned();
        }
        let mut rows = rows.iter();
        super::TR_RE
            .replace_all(source, |tr: &Captures<'_>| {
                let cells = rows.next().unwrap();
                if cells.first().is_none_or(|cell| !cell.header) {
                    return tr[0].to_owned();
                }
                super::CELL_RE
                    .replace_all(&tr[0], |cell: &Captures<'_>| {
                        format!("<th{}>{}</th>", &cell[1], &cell[2])
                    })
                    .into_owned()
            })
            .into_owned()
    })
}

pub(crate) fn to_html(text: &str, infer_headers: bool) -> String {
    let text = if infer_headers {
        promote_html_headers(text)
    } else {
        std::borrow::Cow::Borrowed(text)
    };
    ENV.replace_all(&text, |c: &Captures<'_>| {
        let native = c[1].bytes().all(|v| b"clrx".contains(&v))
            && c[2].starts_with('\n')
            && c[2].ends_with('\n')
            && c[2].trim_end().ends_with(r"\\");
        native
            .then(|| table(&c[2], infer_headers, Some(c[1].len())))
            .flatten()
            .or_else(|| table(&c[2], infer_headers, None))
            .unwrap_or_else(|| c[0].to_string())
    })
    .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn merged_header_skips_covered_empty_cell_and_retains_unit_spans() {
        let latex = "\\begin{tabular}{cc}\n\\multirow{2}{*}{\\thead{Region}} & A\\\\\n & B\\\\\n\\end{tabular}";
        assert_eq!(
            to_html(latex, true),
            "<table><tr><th colspan=\"1\" rowspan=\"2\">Region</th><td colspan=\"1\" rowspan=\"1\">A</td></tr><tr><td colspan=\"1\" rowspan=\"1\">B</td></tr></table>"
        );
    }
}
