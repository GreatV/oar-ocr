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
        let (raw, _) = self
            .generate_with_raw(&[image], &[task], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("PaddleOCR-VL: no result returned"))??;
        Ok(task.postprocess(truncate_repetitive_content(&raw, 10, 10, 10)))
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let tasks: Vec<_> = tasks.iter().copied().map(task).collect();
        Ok(self
            .generate_with_raw(&images, &tasks, max_tokens)?
            .into_iter()
            .zip(tasks)
            .map(|(result, task)| {
                result
                    .map(|(raw, _)| task.postprocess(truncate_repetitive_content(&raw, 10, 10, 10)))
            })
            .collect())
    }

    fn recognition_batch_key(&self, image: &RgbImage, _task: RecognitionTask) -> u64 {
        PaddleOcrVl::recognition_batch_key(self, image).unwrap_or(0)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            preprocess_formula_margin: true,
            ..BackendCapabilities::default()
        }
    }
}
