//! Task-specific prompts for externally detected crops.

use super::{OvisOcr3, prompts};
use crate::ovisocr2::postprocess_recognition_text;
use crate::{Error, RecognitionBackend, RecognitionTask};
use image::RgbImage;

impl RecognitionBackend for OvisOcr3 {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let prompt = match task {
            RecognitionTask::Ocr => prompts::PLAIN_OCR,
            RecognitionTask::Formula => prompts::FORMULA_RECOGNITION,
            RecognitionTask::Chart => prompts::CHART_PARSING,
            // There is no table-only prompt; document Markdown requests HTML tables.
            RecognitionTask::Table => prompts::DOCUMENT_MARKDOWN,
        };
        let text = self.generate_one(&image, prompt, max_tokens)?;
        Ok(postprocess_recognition_text(text, task))
    }
}
