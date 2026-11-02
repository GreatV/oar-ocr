//! Region-recognition adapter for the model-independent pipeline API.

use super::{PaddleOcrVl, PaddleOcrVlTask};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::utils::truncate_repetitive_content;
use image::RgbImage;

fn task(task: RecognitionTask) -> PaddleOcrVlTask {
    match task {
        RecognitionTask::Ocr => PaddleOcrVlTask::Ocr,
        RecognitionTask::Table => PaddleOcrVlTask::Table,
        RecognitionTask::Formula => PaddleOcrVlTask::Formula,
        RecognitionTask::Chart => PaddleOcrVlTask::Chart,
    }
}

impl RecognitionBackend for PaddleOcrVl {
    fn recognize(
        &self,
        image: RgbImage,
        requested: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let task = task(requested);
        self.generate_pipeline_raw(&[image], &[task], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("PaddleOCR-VL: no result returned"))?
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let tasks: Vec<_> = tasks.iter().copied().map(task).collect();
        self.generate_pipeline_raw(&images, &tasks, max_tokens)
    }

    fn recognition_batch_key(&self, image: &RgbImage, _task: RecognitionTask) -> u64 {
        PaddleOcrVl::recognition_batch_key(self, image).unwrap_or(0)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            preprocess_formula_margin: true,
            table_output_is_otsl: true,
            supports_chart: self.use_chart_recognition,
            truncate_repetitive_output: false,
        }
    }

    fn postprocess_region_output(
        &self,
        label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        Some(format_region(label, task, raw))
    }
}

fn format_region(label: &str, task: RecognitionTask, raw: &str) -> String {
    // PaddleX paddleocr_vl/pipeline.py (Apache-2.0), before Markdown rendering.
    let minimum = if task == RecognitionTask::Table {
        5000
    } else {
        50
    };
    let mut text = if raw.chars().count() < minimum {
        raw.into()
    } else {
        truncate_repetitive_content(raw, 10, 10, 10)
    };
    if (text.contains(r"\(") && text.contains(r"\)"))
        || (text.contains(r"\[") && text.contains(r"\]"))
    {
        text = text
            .replace('$', "")
            .replace(r"\(", " $ ")
            .replace(r"\)", " $")
            .replace(r"\[\[", r"\[")
            .replace(r"\]\]", r"\]")
            .replace(r"\[", " $$ ")
            .replace(r"\]", " $$ ");
        if label == "formula_number" {
            text = text.replace('$', "");
        }
    }
    if task == RecognitionTask::Table && crate::render::table::looks_like_table_tokens(&text) {
        let html = crate::render::table::convert_otsl_to_html(&text);
        if !html.is_empty() {
            text = html;
        }
    } else if task == RecognitionTask::Chart {
        text = format_chart(&text);
    }
    text
}

fn format_chart(text: &str) -> String {
    // PaddleX markdown_format_funcs.format_chart2html_table (Apache-2.0).
    let mut lines = text.split('\n');
    let mut html =
        String::from("<table border=1 style='margin: auto; width: max-content;'>\n  <thead><tr>");
    for cell in lines.next().unwrap_or_default().split('|') {
        html.push_str(&format!(
            "<th style='text-align: center;'>{}</th>",
            cell.trim()
        ));
    }
    html.push_str("</tr></thead>\n  <tbody>\n");
    for line in lines {
        html.push_str("    <tr>");
        for cell in line.split('|') {
            html.push_str(&format!(
                "<td style='text-align: center;'>{}</td>",
                cell.trim()
            ));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("  </tbody>\n</table>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_cleanup_preserves_newlines_and_observes_task_length_gates() {
        assert_eq!(
            format_region(
                "display_formula",
                RecognitionTask::Formula,
                "\\[a\\\\\n+b\\]"
            ),
            " $$ a\\\\\n+b $$ "
        );
        assert_eq!(
            format_region("text", RecognitionTask::Ocr, "a...._____"),
            "a...._____"
        );
        assert_eq!(
            format_region("formula_number", RecognitionTask::Ocr, r"\(1\)"),
            "  1 "
        );
        assert!(
            format_region("table", RecognitionTask::Table, "<fcel>\\(x\\)<nl>").contains("$ x $")
        );
        assert!(
            format_region("chart", RecognitionTask::Chart, "Year | Value\n2024 | 7")
                .contains("<td style='text-align: center;'>7</td>")
        );
    }
}
