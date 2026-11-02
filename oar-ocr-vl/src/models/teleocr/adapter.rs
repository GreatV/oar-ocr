//! Region-recognition adapter for TeleOCR.

use super::{TeleOcr, postprocess_formula};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::render::text::truncate_repetitive_content;
use crate::utils::image::resize_for_mineru;
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
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

impl RecognitionBackend for TeleOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let image = resize_for_mineru(&image, 28, 50.0);
        let output = self
            .generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("TeleOCR returned no result"))??;
        Ok(postprocess(
            task,
            &truncate_repetitive_content(&output, 10, 10, 10),
        ))
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let prompts: Vec<_> = tasks.iter().copied().map(prompt).collect();
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
        assert!(prompt(RecognitionTask::Ocr).starts_with('\n'));
        assert!(prompt(RecognitionTask::Chart).contains("scientific figure"));
        assert!(postprocess(RecognitionTask::Chart, "<fcel>A<nl>").contains("<td>A</td>"));
    }
}
