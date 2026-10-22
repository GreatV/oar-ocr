//! Adapt OXR generation to the crate's region-recognition contract.

use super::{Oxr, OxrTask};
use crate::{BackendCapabilities, Error, RecognitionBackend, RecognitionTask};
use image::RgbImage;

impl RecognitionBackend for Oxr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let task = match task {
            RecognitionTask::Ocr => OxrTask::Text,
            RecognitionTask::Formula => OxrTask::Formula,
            RecognitionTask::Table => OxrTask::Table,
            RecognitionTask::Chart => {
                return Err(Error::invalid_input("OXR has no chart-recognition prompt"));
            }
        };
        let options = self
            .generation_options
            .clone()
            .with_max_new_tokens(max_tokens);
        let trace = self.generate_one(&image, task.prompt(), &options, false)?;
        // Task formatting is owned by DocParser and the shared render helpers.
        self.decode_tokens(&trace.tokens)
            .map(|text| text.trim().to_string())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_chart: false,
            truncate_repetitive_output: true,
            ..BackendCapabilities::default()
        }
    }
}
