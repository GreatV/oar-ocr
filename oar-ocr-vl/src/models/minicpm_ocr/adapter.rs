//! Adapt MiniCPM-OCR generation to the crate's region-recognition contract.

use super::{MiniCpmOcr, MiniCpmOcrTask};
use crate::{BackendCapabilities, Error, RecognitionBackend, RecognitionTask};
use image::RgbImage;
use once_cell::sync::Lazy;
use regex::Regex;

static OTSL_CELL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<(fcel|ecel|lcel|ucel|xcel)>").expect("static regex"));

/// Unwrap the `<otsl>...</otsl>` envelope both tables and charts come in.
fn strip_otsl_envelope(text: &str) -> &str {
    let text = text.trim();
    let text = text.strip_prefix("<otsl>").unwrap_or(text);
    text.strip_suffix("</otsl>").unwrap_or(text).trim()
}

/// Render the OTSL data table of a chart as a Markdown pipe table, the form
/// the shared text path keeps (it strips HTML table tags). Merged cells
/// become empty cells; output without OTSL cell tokens passes through.
fn chart_otsl_to_markdown(text: &str) -> String {
    let body = strip_otsl_envelope(text);
    if !OTSL_CELL_RE.is_match(body) {
        return body.to_string();
    }
    let rows: Vec<Vec<String>> = body
        .split("<nl>")
        .filter_map(|row| {
            let mut cells = Vec::new();
            let mut tokens = OTSL_CELL_RE.find_iter(row).peekable();
            while let Some(token) = tokens.next() {
                let end = tokens.peek().map_or(row.len(), |next| next.start());
                cells.push(if token.as_str() == "<fcel>" {
                    row[token.end()..end].trim().replace('|', "\\|")
                } else {
                    String::new()
                });
            }
            (!cells.is_empty()).then_some(cells)
        })
        .collect();
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let line = |cells: &[String]| {
        let padded = (0..columns).map(|i| cells.get(i).map_or("", String::as_str));
        format!("| {} |", padded.collect::<Vec<_>>().join(" | "))
    };
    let mut lines = Vec::with_capacity(rows.len() + 1);
    for (index, row) in rows.iter().enumerate() {
        lines.push(line(row));
        if index == 0 {
            lines.push(format!("|{}", " --- |".repeat(columns)));
        }
    }
    lines.join("\n")
}

impl RecognitionBackend for MiniCpmOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let model_task = match task {
            RecognitionTask::Ocr => MiniCpmOcrTask::Text,
            RecognitionTask::Table => MiniCpmOcrTask::Table,
            RecognitionTask::Formula => MiniCpmOcrTask::Formula,
            RecognitionTask::Chart => MiniCpmOcrTask::Chart,
        };
        let options = self
            .generation_options
            .clone()
            .with_max_new_tokens(max_tokens);
        let text = self.generate_with_prompt(&image, model_task.prompt(), &options)?;
        // Tables go on to DocParser's OTSL-to-HTML conversion; other task
        // formatting is owned by DocParser and the shared render helpers.
        Ok(match task {
            RecognitionTask::Table => strip_otsl_envelope(&text).to_string(),
            RecognitionTask::Chart => chart_otsl_to_markdown(&text),
            _ => text,
        })
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            truncate_repetitive_output: true,
            ..BackendCapabilities::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::table::convert_otsl_to_html;

    #[test]
    fn otsl_tables_and_charts_lose_their_envelope() {
        let table = "<otsl><fcel>Year<fcel>Share<nl><fcel>2017<lcel><nl></otsl>";
        let html = convert_otsl_to_html(strip_otsl_envelope(table));
        assert!(
            html.contains("<td>Year</td>") && !html.contains("otsl"),
            "{html}"
        );

        let chart = "<otsl><fcel>Year<fcel>A|B<nl><fcel>2017<ecel><nl></otsl>";
        assert_eq!(
            chart_otsl_to_markdown(chart),
            "| Year | A\\|B |\n| --- | --- |\n| 2017 |  |"
        );
    }
}
