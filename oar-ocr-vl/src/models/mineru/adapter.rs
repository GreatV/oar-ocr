//! Region-recognition adapter for MinerU.

use super::MinerU;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::render::table::{
    CellMarkup, HtmlInput, NoTokens, OtslOptions, RowPadding, convert_otsl_to_html_with_options,
};
use crate::utils::image::resize_for_mineru;
use crate::utils::truncate_repetitive_content;
use image::RgbImage;

pub(super) fn table_options() -> OtslOptions {
    OtslOptions {
        row_padding: RowPadding::Widest,
        cell_markup: CellMarkup::EscapeAll,
        no_tokens: NoTokens::Empty,
        html_input: HtmlInput::CompletePassThrough,
        empty_fcel_is_anchor: true,
        emit_unowned_cells: true,
        trim_input: false,
        ..Default::default()
    }
}

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr => "\nText Recognition:",
        RecognitionTask::Table => "\nTable Recognition:",
        RecognitionTask::Formula => "\nFormula Recognition:",
        RecognitionTask::Chart => "\nText Recognition:",
    }
}

fn clean(output: String) -> String {
    truncate_repetitive_content(&output, 10, 10, 10)
        .trim()
        .to_string()
}

impl RecognitionBackend for MinerU {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let image = resize_for_mineru(&image, 28, 50.0);
        self.generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("MinerU2.5: no result returned"))?
            .map(clean)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let images: Vec<_> = images
            .iter()
            .map(|image| resize_for_mineru(image, 28, 50.0))
            .collect();
        let prompts: Vec<_> = tasks.iter().copied().map(prompt).collect();
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .map(|result| result.map(clean))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            supports_chart: false,
            ..BackendCapabilities::default()
        }
    }

    fn postprocess_region_output(
        &self,
        label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        if task == RecognitionTask::Table {
            return Some(convert_otsl_to_html_with_options(raw, &table_options()));
        }
        let block = crate::DocumentBlock {
            block_type: if task == RecognitionTask::Formula {
                "equation"
            } else {
                label
            }
            .into(),
            bbox: [0., 0., 1., 1.],
            angle: None,
            content: Some(raw.into()),
        };
        Some(
            crate::pipeline::mineru_postprocess::postprocess_blocks(vec![block])
                .into_iter()
                .next()
                .and_then(|b| b.content)
                .unwrap_or_default(),
        )
    }
}
