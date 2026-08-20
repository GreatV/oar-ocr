//! Region-recognition adapter for NaviDC-OCR.

use super::{NaviDcOcr, NaviDcTask, postprocess_formula};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::render::text::truncate_repetitive_content;
use crate::utils::image::resize_for_mineru;
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr | RecognitionTask::Chart => NaviDcTask::Text.prompt(),
        RecognitionTask::Table => NaviDcTask::Table.prompt(),
        RecognitionTask::Formula => NaviDcTask::Formula.prompt(),
    }
}

fn postprocess(task: RecognitionTask, raw: &str) -> String {
    match task {
        RecognitionTask::Formula => postprocess_formula(raw),
        _ => raw.trim().to_string(),
    }
}

impl RecognitionBackend for NaviDcOcr {
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
            .ok_or_else(|| Error::invalid_input("NaviDC-OCR returned no result"))??;
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
}
