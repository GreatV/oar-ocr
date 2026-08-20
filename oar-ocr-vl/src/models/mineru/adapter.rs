//! Region-recognition adapter for MinerU.

use super::MinerU;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::utils::image::resize_for_mineru;
use crate::utils::truncate_repetitive_content;
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr => "\nText Recognition:",
        RecognitionTask::Table => "\nTable Recognition:",
        RecognitionTask::Formula => "\nFormula Recognition:",
        RecognitionTask::Chart => "\nDocument Parsing:",
    }
}

fn clean(output: String) -> String {
    truncate_repetitive_content(&output, 10, 10, 10)
        .trim()
        .to_string()
}

impl RecognitionBackend for MinerU {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let image = resize_for_mineru(&image, 28, 50.0);
        self.generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("MinerU2.5: no result returned"))?
            .map(clean)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let images: Vec<_> = images
            .iter()
            .map(|image| resize_for_mineru(image, 28, 50.0))
            .collect();
        let prompts: Vec<_> = tasks.iter().copied().map(prompt).collect();
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .map(|result| result.map(clean))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            ..BackendCapabilities::default()
        }
    }
}
