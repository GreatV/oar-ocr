//! Region-recognition adapter for OvisOCR2.

use super::OvisOcr2;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{RecognitionBackend, RecognitionTask};
use image::RgbImage;

impl RecognitionBackend for OvisOcr2 {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let tokens = self
            .generate_tokens(&[image], max_tokens)?
            .pop()
            .ok_or_else(|| Error::invalid_input("OvisOCR2 returned no recognition result"))??;
        let text = self.decode_tokens_raw(&tokens)?;
        Ok(super::model::postprocess_recognition_text(text, task))
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input(format!(
                "OvisOCR2 images count ({}) != tasks count ({})",
                images.len(),
                tasks.len()
            )));
        }
        Ok(self
            .generate_tokens(&images, max_tokens)?
            .into_iter()
            .zip(tasks.iter().copied())
            .map(|(result, task)| {
                result.and_then(|tokens| {
                    let text = self.decode_tokens_raw(&tokens)?;
                    Ok(super::model::postprocess_recognition_text(text, task))
                })
            })
            .collect())
    }
}
