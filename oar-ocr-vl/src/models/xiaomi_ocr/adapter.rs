//! Region-recognition adapter for Xiaomi-OCR-0.

use super::model::{
    DEFAULT_PROMPT, FORMULA_REGION_PROMPT, TABLE_REGION_PROMPT, TEXT_REGION_PROMPT, XiaomiOcr,
};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

/// Official per-task instruction for a region crop.
///
/// Xiaomi defines no chart prompt: whole-page parsing ignores figures, so the
/// backend declares `supports_chart: false` and the parser leaves chart
/// regions unrecognized. Direct [`RecognitionBackend::recognize`] calls with
/// [`RecognitionTask::Chart`] fall back to the whole-page instruction.
fn prompt_for_task(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr => TEXT_REGION_PROMPT,
        RecognitionTask::Table => TABLE_REGION_PROMPT,
        RecognitionTask::Formula => FORMULA_REGION_PROMPT,
        RecognitionTask::Chart => DEFAULT_PROMPT,
    }
}

/// Region-recognition capabilities of the backend: tables come back as OTSL
/// (the pipeline converts them), and there is no chart prompt.
const CAPABILITIES: BackendCapabilities = BackendCapabilities {
    table_output_is_otsl: true,
    preprocess_formula_margin: false,
    truncate_repetitive_output: false,
    supports_chart: false,
};

impl RecognitionBackend for XiaomiOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let tokens = self.generate_tokens_with_prompt(&image, prompt_for_task(task), max_tokens)?;
        let text = self.decode_tokens_raw(&tokens)?;
        // Table output stays OTSL: the pipeline converts it to HTML via the
        // `table_output_is_otsl` capability.
        Ok(text)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input(format!(
                "Xiaomi-OCR-0 images count ({}) != tasks count ({})",
                images.len(),
                tasks.len()
            )));
        }
        Ok(images
            .iter()
            .zip(tasks.iter().copied())
            .map(|(image, task)| self.recognize(image.clone(), task, max_tokens))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        CAPABILITIES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_map_to_the_official_prompts() {
        assert_eq!(prompt_for_task(RecognitionTask::Ocr), TEXT_REGION_PROMPT);
        assert_eq!(prompt_for_task(RecognitionTask::Table), TABLE_REGION_PROMPT);
        assert_eq!(
            prompt_for_task(RecognitionTask::Formula),
            FORMULA_REGION_PROMPT
        );
        assert_eq!(prompt_for_task(RecognitionTask::Chart), DEFAULT_PROMPT);
    }

    // Pins the contract the DocParser pipeline relies on: table output must
    // stay OTSL (the pipeline converts it), and chart regions must be left
    // unrecognized.
    const _: () = assert!(CAPABILITIES.table_output_is_otsl);
    const _: () = assert!(!CAPABILITIES.supports_chart);
    const _: () = assert!(!CAPABILITIES.truncate_repetitive_output);
    const _: () = assert!(!CAPABILITIES.preprocess_formula_margin);
}
