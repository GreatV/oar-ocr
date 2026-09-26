//! Region-recognition adapter for WeVisDoc.

use super::WeVisDoc;
use super::model::LoopGuard;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

/// Extract the first `<table>…</table>` span. Region crops often carry a
/// heading, caption, or Markdown around the table, and `DocParser` feeds the
/// result straight into `TableResult.html`. Verbatim when no table is found.
fn extract_html_table(raw: &str) -> &str {
    let Some(start) = raw.find("<table>") else {
        return raw;
    };
    let Some(end) = raw[start..].find("</table>") else {
        return raw;
    };
    &raw[start..start + end + "</table>".len()]
}

/// Structured regions repeat legitimately; their loop guard only steps in
/// near the token budget. Text-like regions keep the eager guard.
fn loop_guard_for_task(task: RecognitionTask) -> LoopGuard {
    match task {
        RecognitionTask::Table | RecognitionTask::Formula => LoopGuard::Conservative,
        RecognitionTask::Ocr | RecognitionTask::Chart => LoopGuard::Standard,
    }
}

fn postprocess(task: RecognitionTask, raw: &str) -> String {
    match task {
        RecognitionTask::Table => extract_html_table(raw).trim().to_string(),
        _ => raw.trim().to_string(),
    }
}

impl RecognitionBackend for WeVisDoc {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        // WeVisDoc has no per-task prompts: the official instruction already
        // emits Markdown text, HTML tables, and LaTeX formulas, so every
        // region task runs the same full-page prompt.
        let tokens = self
            .generate_tokens_for_regions(
                std::slice::from_ref(&image),
                max_tokens,
                loop_guard_for_task(task),
            )?
            .pop()
            .ok_or_else(|| Error::invalid_input("WeVisDoc returned no recognition result"))??;
        Ok(postprocess(task, &self.decode_tokens(&tokens)?))
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input(format!(
                "WeVisDoc images count ({}) != tasks count ({})",
                images.len(),
                tasks.len()
            )));
        }
        // Mixed batches take the conservative guard: it never trims, so
        // table rows survive even when one element is plain text.
        let guard = tasks
            .iter()
            .copied()
            .map(loop_guard_for_task)
            .reduce(|a, b| if a == b { a } else { LoopGuard::Conservative })
            .unwrap_or(LoopGuard::Standard);
        Ok(self
            .generate_tokens_for_regions(&images, max_tokens, guard)?
            .into_iter()
            .zip(tasks.iter().copied())
            .map(|(result, task)| {
                result.and_then(|tokens| Ok(postprocess(task, &self.decode_tokens(&tokens)?)))
            })
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            // Tables come back as HTML, not OTSL.
            table_output_is_otsl: false,
            // Region decoding stops token-level loops itself; the parser's
            // text-level truncation would only add false hits on top of it.
            truncate_repetitive_output: false,
            // The official page prompt tells the model to ignore figures,
            // so chart crops would produce empty or stray-label output.
            supports_chart: false,
            ..BackendCapabilities::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_extraction_takes_the_first_table_span() {
        let raw = "Table 1: totals\n\n<table>\n<tr><td>a</td></tr>\n</table>\n\nNotes.\n\n<table>\n<tr><td>b</td></tr>\n</table>";
        assert_eq!(
            postprocess(RecognitionTask::Table, raw),
            "<table>\n<tr><td>a</td></tr>\n</table>"
        );
    }

    #[test]
    fn table_extraction_is_verbatim_without_a_table() {
        assert_eq!(
            postprocess(RecognitionTask::Table, "no table here"),
            "no table here"
        );
        // An unclosed table stays verbatim too.
        assert_eq!(
            postprocess(RecognitionTask::Table, "prefix <table><tr>"),
            "prefix <table><tr>"
        );
    }

    #[test]
    fn other_tasks_pass_through_trimmed() {
        let raw = "  some text  ";
        for task in [
            RecognitionTask::Ocr,
            RecognitionTask::Formula,
            RecognitionTask::Chart,
        ] {
            assert_eq!(postprocess(task, raw), "some text");
        }
    }
}
