//! Region-recognition adapter for GLM-OCR.

use super::GlmOcr;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{RecognitionBackend, RecognitionTask};
use crate::utils::truncate_repetitive_content;
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr | RecognitionTask::Chart => "Text Recognition:",
        RecognitionTask::Table => "Table Recognition:",
        RecognitionTask::Formula => "Formula Recognition:",
    }
}

fn clean(output: String) -> String {
    truncate_repetitive_content(&output, 10, 10, 10)
        .trim()
        .to_string()
}

impl RecognitionBackend for GlmOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("GLM-OCR: no result returned"))?
            .map(clean)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let prompts: Vec<_> = tasks.iter().copied().map(prompt).collect();
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .map(|result| result.map(clean))
            .collect())
    }
}
