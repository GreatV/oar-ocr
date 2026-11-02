//! Region-recognition adapter for TeleOCR.

use super::{TeleOcr, postprocess_formula};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::render::text::truncate_repetitive_content;
use crate::utils::image::resize_for_mineru;
use image::RgbImage;

fn task_for_label(label: &str, suggested: Option<RecognitionTask>) -> Option<RecognitionTask> {
    if label == "seal" {
        Some(RecognitionTask::Ocr)
    } else {
        suggested
    }
}

fn prompt(label: &str, task: RecognitionTask) -> &'static str {
    if label == "seal" {
        return "\nSeal Recognition:";
    }
    match task {
        RecognitionTask::Ocr => "\nPlease output the text content from the image.",
        RecognitionTask::Chart => {
            "\nThis is a scientific figure. Please extract the table implied by this figure."
        }
        RecognitionTask::Table => {
            "\nThis is the image of a table. Please output the table in OTSL format."
        }
        RecognitionTask::Formula => {
            "\nPlease write out the expression of the formula in the image using LaTeX format."
        }
    }
}

fn postprocess(task: RecognitionTask, raw: &str) -> String {
    match task {
        RecognitionTask::Formula => postprocess_formula(raw),
        RecognitionTask::Chart => crate::render::table::convert_otsl_to_html(raw),
        _ => raw.trim().to_string(),
    }
}

impl TeleOcr {
    fn recognize_regions(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        labels: &[&str],
        max_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() || labels.len() != tasks.len() {
            return Err(Error::invalid_input(
                "TeleOCR image, label and task counts differ",
            ));
        }
        let prompts: Vec<_> = labels
            .iter()
            .zip(tasks)
            .map(|(&label, &task)| prompt(label, task))
            .collect();
        let images: Vec<_> = images
            .iter()
            .map(|image| resize_for_mineru(image, 28, 50.0))
            .collect();
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .zip(tasks.iter().copied())
            .map(|(result, task)| {
                result.map(|output| {
                    postprocess(task, &truncate_repetitive_content(&output, 10, 10, 10))
                })
            })
            .collect())
    }
}

impl RecognitionBackend for TeleOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.recognize_regions(vec![image], &[task], &[""], max_tokens)?
            .pop()
            .ok_or_else(|| Error::invalid_input("TeleOCR returned no result"))?
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        self.recognize_regions(images, tasks, &vec![""; tasks.len()], max_tokens)
    }

    fn recognition_task_for_label(
        &self,
        label: &str,
        suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        task_for_label(label, suggested)
    }

    fn recognize_labeled_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        labels: &[&str],
        options: &crate::GenerationOptions,
    ) -> BatchResult<String> {
        self.recognize_regions(images, tasks, labels, options.max_new_tokens)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            ..BackendCapabilities::default()
        }
    }

    fn postprocess_region_output(
        &self,
        _label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        (task == RecognitionTask::Chart).then(|| raw.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_prompts_and_chart_output_use_the_region_recipe() {
        assert!(prompt("text", RecognitionTask::Ocr).starts_with('\n'));
        assert!(prompt("chart", RecognitionTask::Chart).contains("scientific figure"));
        assert_eq!(task_for_label("seal", None), Some(RecognitionTask::Ocr));
        assert_eq!(prompt("seal", RecognitionTask::Ocr), "\nSeal Recognition:");
        assert!(postprocess(RecognitionTask::Chart, "<fcel>A<nl>").contains("<td>A</td>"));
    }
}
