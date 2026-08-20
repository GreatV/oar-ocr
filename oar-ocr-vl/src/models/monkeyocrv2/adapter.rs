//! Region-recognition adapter for MonkeyOCRv2.

use super::{MonkeyOcrV2, MonkeyOcrV2Task};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

fn map_task(task: RecognitionTask) -> MonkeyOcrV2Task {
    match task {
        RecognitionTask::Ocr | RecognitionTask::Chart => MonkeyOcrV2Task::Text,
        RecognitionTask::Table => MonkeyOcrV2Task::Table,
        RecognitionTask::Formula => MonkeyOcrV2Task::Formula,
    }
}

impl RecognitionBackend for MonkeyOcrV2 {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.generate(&[image], &[map_task(task)], max_tokens)?
            .pop()
            .ok_or_else(|| Error::invalid_input("MonkeyOCRv2 returned no recognition result"))?
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let tasks: Vec<_> = tasks.iter().copied().map(map_task).collect();
        self.generate(&images, &tasks, max_tokens)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            truncate_repetitive_output: true,
            ..BackendCapabilities::default()
        }
    }
}
