//! Model-independent contracts for cropped-region recognition.

use crate::api::error::{BatchResult, Error};
use crate::api::generation::GenerationOptions;
use image::RgbImage;

/// Region-recognition behavior consumed by the pipeline scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub table_output_is_otsl: bool,
    pub preprocess_formula_margin: bool,
    pub truncate_repetitive_output: bool,
    /// Whether the backend can recognize chart regions. Backends whose
    /// prompts ignore figures set this to `false`; the parser then leaves
    /// chart regions unrecognized instead of collecting empty output.
    pub supports_chart: bool,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            table_output_is_otsl: false,
            preprocess_formula_margin: false,
            truncate_repetitive_output: false,
            supports_chart: true,
        }
    }
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

    /// Refine the scheduled task using the layout's original semantic label.
    /// Returning `None` leaves the region unrecognized.
    fn recognition_task_for_label(
        &self,
        _label: &str,
        suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        suggested
    }

    /// Recognize labeled regions, allowing model-native prompts and token
    /// budgets without adding every model task to the shared task enum.
    fn recognize_labeled_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        labels: &[&str],
        options: &GenerationOptions,
    ) -> BatchResult<String> {
        if labels.len() != tasks.len() {
            return Err(Error::invalid_input("region label and task counts differ"));
        }
        self.recognize_batch_with_options(images, tasks, options)
    }

    /// Optionally predict a reading-order permutation for detected regions.
    /// Boxes use source-image pixels. `None` preserves the detector's order.
    fn predict_reading_order(
        &self,
        _image: &RgbImage,
        _regions: &[crate::document::structure::LayoutElement],
        _max_tokens: usize,
    ) -> Result<Option<Vec<usize>>, Error> {
        Ok(None)
    }

    /// Whether adjacent text crops may be merged into one recognition request.
    fn merge_adjacent_text_regions(&self) -> bool {
        true
    }

    /// Supply model-native final formatting for one region. `None` uses the
    /// shared task formatter; `Some` preserves the backend's returned markup.
    fn postprocess_region_output(
        &self,
        _label: &str,
        _task: RecognitionTask,
        _raw: &str,
    ) -> Option<String> {
        None
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

impl<T: RecognitionBackend + ?Sized> RecognitionBackend for &T {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        (**self).recognize(image, task, max_tokens)
    }

    fn recognize_with_options(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        options: &GenerationOptions,
    ) -> Result<String, Error> {
        (**self).recognize_with_options(image, task, options)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        (**self).recognize_batch(images, tasks, max_tokens)
    }

    fn recognize_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        options: &GenerationOptions,
    ) -> BatchResult<String> {
        (**self).recognize_batch_with_options(images, tasks, options)
    }

    fn recognition_task_for_label(
        &self,
        label: &str,
        suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        (**self).recognition_task_for_label(label, suggested)
    }

    fn recognize_labeled_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        labels: &[&str],
        options: &GenerationOptions,
    ) -> BatchResult<String> {
        (**self).recognize_labeled_batch_with_options(images, tasks, labels, options)
    }

    fn predict_reading_order(
        &self,
        image: &RgbImage,
        regions: &[crate::document::structure::LayoutElement],
        max_tokens: usize,
    ) -> Result<Option<Vec<usize>>, Error> {
        (**self).predict_reading_order(image, regions, max_tokens)
    }

    fn merge_adjacent_text_regions(&self) -> bool {
        (**self).merge_adjacent_text_regions()
    }

    fn postprocess_region_output(
        &self,
        label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        (**self).postprocess_region_output(label, task, raw)
    }

    fn recognition_batch_key(&self, image: &RgbImage, task: RecognitionTask) -> u64 {
        (**self).recognition_batch_key(image, task)
    }

    fn capabilities(&self) -> BackendCapabilities {
        (**self).capabilities()
    }
}
