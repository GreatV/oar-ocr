//! Region-recognition adapter for jina-ocr-v1.

use super::JinaOcr;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

fn postprocess(task: RecognitionTask, raw: &str) -> String {
    match task {
        RecognitionTask::Table => {
            crate::render::table::table_span(raw, crate::render::table::TableWrapper::Literal)
                .unwrap_or(raw)
                .trim()
                .to_string()
        }
        _ => raw.trim().to_string(),
    }
}

impl RecognitionBackend for JinaOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        // jina-ocr-v1 has a single OCR instruction; it already emits Markdown
        // text, HTML tables, and LaTeX formulas for any crop.
        let tokens = self
            .generate_tokens(std::slice::from_ref(&image), max_tokens)?
            .pop()
            .ok_or_else(|| Error::invalid_input("JinaOCR returned no recognition result"))??;
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
                "JinaOCR images count ({}) != tasks count ({})",
                images.len(),
                tasks.len()
            )));
        }
        Ok(self
            .generate_tokens(&images, max_tokens)?
            .into_iter()
            .zip(tasks.iter().copied())
            .map(|(result, task)| {
                result.and_then(|tokens| Ok(postprocess(task, &self.decode_tokens(&tokens)?)))
            })
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: false,
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
        assert_eq!(
            postprocess(RecognitionTask::Table, "prefix <table><tr>"),
            "prefix <table><tr>"
        );
    }

    #[test]
    fn other_tasks_pass_through_trimmed() {
        for task in [
            RecognitionTask::Ocr,
            RecognitionTask::Formula,
            RecognitionTask::Chart,
        ] {
            assert_eq!(postprocess(task, "  some text  "), "some text");
        }
    }
}
