//! The two MinerU table dialects; neither uses PaddleX's square-row padding.

use regex::Regex;
use std::sync::LazyLock;

pub(crate) fn mineru_otsl_to_html(text: &str) -> String {
    convert(text, false)
}
pub(crate) fn diffusion_otsl_to_html(text: &str) -> String {
    convert(text, true)
}

fn convert(text: &str, diffusion: bool) -> String {
    // MinerU-Team/mineru-vl-utils post_process/otsl2html.py (Apache-2.0).
    // Diffusion scripts/run_end2end.py (MIT) uses the same span scheme, but retains
    // no-token input and ecel text, and emits only explicit cell anchors.
    if text.starts_with("<table") && text.ends_with("</table>") {
        return text.into();
    }
    static TOKEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"<fcel>|<ecel>|<lcel>|<ucel>|<xcel>").unwrap());
    let mut rows: Vec<Vec<(&str, String)>> = Vec::new();
    for row in text.split("<nl>") {
        let tokens: Vec<_> = TOKEN.find_iter(row).collect();
        if tokens.is_empty() {
            continue;
        }
        rows.push(
            tokens
                .iter()
                .enumerate()
                .map(|(i, token)| {
                    let end = tokens.get(i + 1).map_or(row.len(), |next| next.start());
                    let content = if token.as_str() == "<fcel>"
                        || (diffusion && token.as_str() == "<ecel>")
                    {
                        row[token.end()..end].trim().into()
                    } else {
                        String::new()
                    };
                    (token.as_str(), content)
                })
                .collect(),
        );
    }
    let Some(width) = rows.iter().map(Vec::len).max() else {
        return if diffusion {
            text.trim().into()
        } else {
            String::new()
        };
    };
    for row in &mut rows {
        row.resize(width, ("<ecel>", String::new()));
    }
    let mut owners = vec![vec![None; width]; rows.len()];
    let mut spans = vec![vec![(1, 1); width]; rows.len()];
    let mut has_cells = false;
    for (r, row) in rows.iter().enumerate() {
        for (c, (tag, _)) in row.iter().enumerate() {
            if !matches!(*tag, "<fcel>" | "<ecel>") {
                continue;
            }
            has_cells = true;
            let colspan = 1 + row[c + 1..]
                .iter()
                .take_while(|(t, _)| matches!(*t, "<lcel>" | "<xcel>"))
                .count();
            let rowspan = 1 + rows[r + 1..]
                .iter()
                .take_while(|row| matches!(row[c].0, "<ucel>" | "<xcel>"))
                .count();
            spans[r][c] = (rowspan, colspan);
            for covered_row in owners.iter_mut().skip(r).take(rowspan) {
                for owner in covered_row.iter_mut().skip(c).take(colspan) {
                    *owner = Some((r, c));
                }
            }
        }
    }
    if !has_cells && !diffusion {
        return String::new();
    }
    let mut html = String::from("<table>");
    for (r, row) in owners.iter().enumerate() {
        html.push_str("<tr>");
        for (c, owner) in row.iter().enumerate() {
            if *owner != Some((r, c)) {
                if owner.is_none() && !diffusion {
                    html.push_str("<td></td>");
                }
                continue;
            }
            let (rowspan, colspan) = spans[r][c];
            html.push_str("<td");
            if rowspan > 1 {
                html.push_str(&format!(" rowspan=\"{rowspan}\""));
            }
            if colspan > 1 {
                html.push_str(&format!(" colspan=\"{colspan}\""));
            }
            html.push('>');
            html.push_str(
                &rows[r][c]
                    .1
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;")
                    .replace('\'', "&#x27;"),
            );
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_rows_and_empty_fcel_use_native_spans_without_tsv_fallback() {
        let raw = "<fcel>A<fcel>B<nl><fcel><lcel><fcel>C<nl>";
        assert_eq!(
            mineru_otsl_to_html(raw),
            "<table><tr><td>A</td><td>B</td><td></td></tr><tr><td colspan=\"2\"></td><td>C</td></tr></table>"
        );
        assert_eq!(mineru_otsl_to_html("plain"), "");
        assert_eq!(diffusion_otsl_to_html("plain"), "plain");
        assert_eq!(
            diffusion_otsl_to_html("<ecel>x & y<nl>"),
            "<table><tr><td>x &amp; y</td></tr></table>"
        );
    }
}
