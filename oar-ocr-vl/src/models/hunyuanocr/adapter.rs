//! Region-recognition adapter for HunyuanOCR.

use super::HunyuanOcr;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{RecognitionBackend, RecognitionTask};
use crate::utils::truncate_repetitive_content;
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr => {
            "Detect and recognize text in the image, and output the text coordinates in a formatted manner."
        }
        RecognitionTask::Table => "Parse the table in the image into HTML.",
        RecognitionTask::Formula => {
            "Identify the formula in the image and represent it using LaTeX format."
        }
        RecognitionTask::Chart => {
            "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts."
        }
    }
}

fn clean(output: String) -> String {
    truncate_repetitive_content(&output, 10, 10, 10)
        .trim()
        .to_string()
}

impl RecognitionBackend for HunyuanOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("HunyuanOCR: no result returned"))?
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

    fn recognition_batch_key(&self, image: &RgbImage, _task: RecognitionTask) -> u64 {
        HunyuanOcr::recognition_batch_key(self, image).unwrap_or(0)
    }
}
