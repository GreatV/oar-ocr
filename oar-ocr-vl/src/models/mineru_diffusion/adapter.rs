//! Region-recognition adapter for MinerU-Diffusion.

use super::{DiffusionGenerationConfig, MinerUDiffusion};
use crate::api::error::Error;
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::render::table::{CellMarkup, HtmlInput, NoTokens, OtslOptions, RowPadding};
use image::RgbImage;

pub(super) fn table_options() -> OtslOptions {
    OtslOptions {
        row_padding: RowPadding::Widest,
        cell_markup: CellMarkup::EscapeAll,
        no_tokens: NoTokens::Preserve,
        html_input: HtmlInput::CompletePassThrough,
        empty_fcel_is_anchor: true,
        preserve_ecel_text: true,
        trim_input: false,
        ..Default::default()
    }
}

impl RecognitionBackend for MinerUDiffusion {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        if max_tokens == 0 {
            return Ok(String::new());
        }
        let prompt = match task {
            RecognitionTask::Ocr => "\nText Recognition:",
            RecognitionTask::Table => "\nTable Recognition:",
            RecognitionTask::Formula => "\nFormula Recognition:",
            RecognitionTask::Chart => "\nText Recognition:",
        };
        let mut generation = DiffusionGenerationConfig::default();
        let budget = super::layout::task_budget(
            if task == RecognitionTask::Table {
                "table"
            } else {
                "text"
            },
            max_tokens,
        );
        generation.gen_length = budget
            .checked_next_multiple_of(generation.block_length)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "MinerU-Diffusion max_tokens {max_tokens} is too large"
                ))
            })?;
        let image = super::layout::resize_crop(&image, 28, 50.)?;
        let mut tokens = self.generate_token_ids(&image, prompt, &generation)?;
        tokens.truncate(budget);
        self.decode_tokens(&tokens, true)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            truncate_repetitive_output: true,
            ..BackendCapabilities::default()
        }
    }

    fn postprocess_region_output(
        &self,
        _label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        let kind = match task {
            RecognitionTask::Table => "table",
            RecognitionTask::Formula => "equation",
            _ => "text",
        };
        Some(super::layout::format_content(kind, raw))
    }
}
