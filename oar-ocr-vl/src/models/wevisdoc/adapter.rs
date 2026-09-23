//! Region-recognition adapter for WeVisDoc.

use super::WeVisDoc;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

impl RecognitionBackend for WeVisDoc {
    fn recognize(
        &self,
        image: RgbImage,
        _task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        // WeVisDoc has no per-task prompts: the official instruction already
        // emits Markdown text, HTML tables, and LaTeX formulas, so every
        // region task runs the same full-page prompt.
        let tokens = self
            .generate_tokens(std::slice::from_ref(&image), max_tokens)?
            .pop()
            .ok_or_else(|| Error::invalid_input("WeVisDoc returned no recognition result"))??;
        self.decode_tokens(&tokens)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input(format!(
                "WeVisDoc images count ({}) != tasks count ({})",
                images.len(),
                tasks.len()
            )));
        }
        Ok(self
            .generate_tokens(&images, max_tokens)?
            .into_iter()
            .map(|result| result.and_then(|tokens| self.decode_tokens(&tokens)))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            // Tables come back as HTML, not OTSL.
            table_output_is_otsl: false,
            ..BackendCapabilities::default()
        }
    }
}
