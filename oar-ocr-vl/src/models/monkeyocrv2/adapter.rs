//! Region-recognition adapter for MonkeyOCRv2.

use super::{MonkeyOcrV2, MonkeyOcrV2Task};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use crate::render::table::{
    CellMarkup, HtmlInput, NoTokens, OtslOptions, RowPadding, SpanSearch,
    convert_otsl_to_html_with_options,
};
use crate::render::text::{
    FormulaOptions, FormulaRepair, MathStripping, format_formula_with_options,
};
use image::RgbImage;

pub(super) fn table_options() -> OtslOptions {
    OtslOptions {
        row_padding: RowPadding::Ragged,
        cell_markup: CellMarkup::PreserveMarkup,
        span_search: SpanSearch::NearestAnchor,
        no_tokens: NoTokens::EmptyTable,
        html_input: HtmlInput::Tokenize,
        empty_fcel_is_anchor: true,
        preserve_ecel_text: false,
        emit_unowned_cells: false,
        trim_text: false,
        trim_input: false,
        case_insensitive: true,
        newline_to_br: true,
        unwrap_otsl: true,
    }
}

pub(super) fn formula_options() -> FormulaOptions {
    FormulaOptions {
        opening: "$$\n",
        closing: "\n$$\n",
        stripping: MathStripping::Dollars,
        repair: FormulaRepair::EquationNumber,
        normalize_mu: false,
        newline_to_break: false,
        clean_tokens: false,
    }
}

fn map_task(task: RecognitionTask) -> MonkeyOcrV2Task {
    match task {
        RecognitionTask::Ocr | RecognitionTask::Chart => MonkeyOcrV2Task::Text,
        RecognitionTask::Table => MonkeyOcrV2Task::Table,
        RecognitionTask::Formula => MonkeyOcrV2Task::Formula,
    }
}

pub(super) fn task_for_label(label: &str) -> Option<RecognitionTask> {
    match label {
        "text" | "figure_title" | "doc_title" | "paragraph_title" | "header" | "footer" => {
            Some(RecognitionTask::Ocr)
        }
        "formula" => Some(RecognitionTask::Formula),
        "table" => Some(RecognitionTask::Table),
        _ => None,
    }
}

impl RecognitionBackend for MonkeyOcrV2 {
    fn page_preprocessor(&self) -> Option<&dyn crate::PagePreprocessor> {
        self.dewarp
            .as_ref()
            .map(|dewarp| dewarp as &dyn crate::PagePreprocessor)
    }

    fn doc_parser_profile(&self) -> crate::DocParserProfile {
        parsing_profile()
    }

    fn region_token_budget(&self, _label: &str, _task: RecognitionTask) -> Option<usize> {
        Some(5000)
    }

    fn recognition_task_for_label(
        &self,
        label: &str,
        _suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        task_for_label(label)
    }

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
            truncate_repetitive_output: false,
            supports_chart: false,
            ..BackendCapabilities::default()
        }
    }

    fn postprocess_region_output(
        &self,
        _label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        match task {
            RecognitionTask::Table => {
                Some(convert_otsl_to_html_with_options(raw, &table_options()))
            }
            RecognitionTask::Formula => Some(format_formula_with_options(raw, &formula_options())),
            _ => Some(raw.trim().to_string()),
        }
    }
}

pub(super) fn parsing_profile() -> crate::DocParserProfile {
    use crate::api::recognition::{
        AuxiliaryLabelAction, AuxiliaryLabelRule, EmptyLayoutBehavior, RegionErrorBehavior,
    };
    let ignored: Vec<String> = ["header", "footer"].map(str::to_string).into();
    crate::DocParserProfile {
        max_tokens: 5000,
        skip_auxiliary_regions: false,
        skip_region_blocks: false,
        merge_labels: Vec::new(),
        markdown_pretty: false,
        layout: Some(
            crate::LayoutPostprocessOptions::default()
                .with_overlap_threshold(None)
                .with_drop_labels(Vec::new()),
        ),
        auxiliary_labels: ignored
            .iter()
            .map(|label| AuxiliaryLabelRule {
                label: label.clone(),
                action: AuxiliaryLabelAction::HideAtRender,
            })
            .collect(),
        markdown_ignore_labels: ignored,
        empty_layout: EmptyLayoutBehavior::Empty,
        region_errors: RegionErrorBehavior::KeepRegion,
        crop: crate::pipeline::crop::CropOptions::default()
            .with_rounding(crate::pipeline::crop::CropRounding::Nearest)
            .with_minimum_one_pixel(true),
        render_style: crate::render::markdown::RenderStyle::default()
            .with_skip_empty_chunks(true)
            .with_table_leading_newline(false)
            .with_strip_table_envelope(false)
            .with_section_templates(false)
            .with_section_heading_level(Some(2))
            .with_per_line_titles(true)
            .with_normalize_text_newlines(false),
        formula: formula_options(),
        table: table_options(),
        ..Default::default()
    }
}
