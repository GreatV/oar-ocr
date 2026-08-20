//! Model-independent contracts for cropped-region recognition.

use crate::api::error::{BatchResult, Error};
use crate::api::generation::GenerationOptions;
use image::RgbImage;

/// Region-recognition behavior consumed by the pipeline scheduler.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub table_output_is_otsl: bool,
    pub preprocess_formula_margin: bool,
    pub truncate_repetitive_output: bool,
}

/// Semantic task requested for a detected document region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionTask {
    /// General OCR/text recognition.
    Ocr,
    /// Table structure recognition.
    Table,
    /// Formula recognition.
    Formula,
    /// Chart recognition.
    Chart,
}

/// A model adapter that recognizes cropped document regions.
///
/// This contract deliberately lives below the document parser so model
/// implementations never need to depend on pipeline orchestration.
pub trait RecognitionBackend {
    /// Generate text/content for one cropped image region.
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error>;

    /// Recognize one region using the extensible generation options contract.
    fn recognize_with_options(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        options: &GenerationOptions,
    ) -> Result<String, Error> {
        self.recognize(image, task, options.max_new_tokens)
    }

    /// Generate content for multiple cropped regions.
    ///
    /// Backends with native batching should override this method. The outer
    /// result represents a batch-level failure; inner results retain per-item
    /// decode or post-processing failures.
    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input(format!(
                "region batch: images count ({}) != tasks count ({})",
                images.len(),
                tasks.len()
            )));
        }
        Ok(images
            .into_iter()
            .zip(tasks.iter().copied())
            .map(|(image, task)| self.recognize(image, task, max_tokens))
            .collect())
    }

    /// Recognize a region batch using the extensible generation options contract.
    fn recognize_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        options: &GenerationOptions,
    ) -> BatchResult<String> {
        self.recognize_batch(images, tasks, options.max_new_tokens)
    }

    /// Stable grouping key for requests with compatible prepared shapes.
    fn recognition_batch_key(&self, _image: &RgbImage, _task: RecognitionTask) -> u64 {
        0
    }

    /// Describe preprocessing and output-format requirements declaratively.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
}
