//! NaviDC-OCR (Vision-Language) model.
//!
//! Native Rust inference for the ~1.2B StarDoc-AI NaviDC-OCR checkpoint
//! (Qwen2.5-VL backbone with a windowed vision tower). Construct with
//! [`NaviDcOcr::from_dir`] pointing at the model directory (for example
//! `models/StarDoc-AI/NaviDC-OCR`). Supported tasks (see [`NaviDcTask`]):
//! - OCR text recognition
//! - Table recognition (outputs OTSL; convert with
//!   [`crate::utils::convert_otsl_to_html`])
//! - Formula recognition (outputs LaTeX)
//! - Code recognition
//! - Layout analysis (full and multi-point/distorted variants)
//! - Scientific-figure table extraction (outputs OTSL)
//!
//! Layout tasks expect the input resized to 1036×1036 (bicubic), as the
//! model card's quickstart does; [`NaviDcTask::resize_square`] reports that
//! expectation.

mod adapter;
mod config;
mod model;
mod text;
use crate::backbones::qwen25_vl as vision;

pub use config::{NaviDcConfig, NaviDcRopeScaling, NaviDcTextConfig, NaviDcVisionConfig};
pub use model::NaviDcOcr;

/// NaviDC-OCR task with the official prompt from the model card's quickstart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NaviDcTask {
    /// Plain text recognition.
    Text,
    /// Table structure recognition; the model emits OTSL.
    Table,
    /// Formula recognition; the model emits LaTeX.
    Formula,
    /// Code snippet parsing.
    Code,
    /// Full-page layout analysis (resize the input to 1036×1036 first, see
    /// [`NaviDcTask::resize_square`]).
    Layout,
    /// Multi-point layout segmentation for distorted/camera-captured pages.
    LayoutDistorted,
    /// Table extraction from scientific figures; the model emits OTSL.
    ScientificFigure,
}

impl NaviDcTask {
    /// Official per-task prompt (verbatim from the model card quickstart).
    /// `LayoutDistorted` keeps its leading `\n`.
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Text => "Please output the text content from the image.",
            Self::Table => "This is the image of a table. Please output the table in OTSL format.",
            Self::Formula => {
                "Please write out the expression of the formula in the image using LaTeX format."
            }
            Self::Code => "The image contains a code snippet, please output the parsing result.",
            Self::Layout => "Analyze the image layout.",
            Self::LayoutDistorted => "\nMulti-point Layout Segmentation Analysis.",
            Self::ScientificFigure => {
                "This is a scientific figure. Please extract the table implied by this figure."
            }
        }
    }

    /// Side length the model card's quickstart resizes layout inputs to
    /// (bicubic); `None` for tasks fed the raw crop.
    pub fn resize_square(self) -> Option<u32> {
        match self {
            Self::Layout | Self::LayoutDistorted => Some(1036),
            _ => None,
        }
    }

    /// Whether this task's raw output is OTSL (convert with
    /// [`crate::utils::convert_otsl_to_html`]).
    pub fn outputs_otsl(self) -> bool {
        matches!(self, Self::Table | Self::ScientificFigure)
    }

    /// Task-specific post-processing: formulas are unwrapped from `\[..\]`
    /// and re-wrapped as `$$..$$` unless already `$`-delimited (the model
    /// card's `post_process`); other tasks pass through trimmed.
    pub fn postprocess(self, text: &str) -> String {
        match self {
            Self::Formula => postprocess_formula(text),
            _ => text.trim().to_string(),
        }
    }
}

pub(crate) fn postprocess_formula(text: &str) -> String {
    let trimmed = text.trim();
    let unwrapped = trimmed.strip_prefix("\\[").unwrap_or(trimmed);
    let unwrapped = unwrapped.strip_suffix("\\]").unwrap_or(unwrapped).trim();
    if unwrapped.starts_with('$') && unwrapped.ends_with('$') {
        unwrapped.to_string()
    } else {
        format!("$${unwrapped}$$")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_prompts_match_official_quickstart() {
        assert_eq!(
            NaviDcTask::Text.prompt(),
            "Please output the text content from the image."
        );
        assert_eq!(
            NaviDcTask::Table.prompt(),
            "This is the image of a table. Please output the table in OTSL format."
        );
        assert_eq!(
            NaviDcTask::Formula.prompt(),
            "Please write out the expression of the formula in the image using LaTeX format."
        );
        assert_eq!(
            NaviDcTask::Code.prompt(),
            "The image contains a code snippet, please output the parsing result."
        );
        assert_eq!(NaviDcTask::Layout.prompt(), "Analyze the image layout.");
        assert_eq!(
            NaviDcTask::LayoutDistorted.prompt(),
            "\nMulti-point Layout Segmentation Analysis."
        );
        assert_eq!(
            NaviDcTask::ScientificFigure.prompt(),
            "This is a scientific figure. Please extract the table implied by this figure."
        );
    }

    #[test]
    fn layout_tasks_resize_to_1036() {
        assert_eq!(NaviDcTask::Layout.resize_square(), Some(1036));
        assert_eq!(NaviDcTask::LayoutDistorted.resize_square(), Some(1036));
        assert_eq!(NaviDcTask::Text.resize_square(), None);
    }

    #[test]
    fn otsl_tasks_are_table_like() {
        assert!(NaviDcTask::Table.outputs_otsl());
        assert!(NaviDcTask::ScientificFigure.outputs_otsl());
        assert!(!NaviDcTask::Text.outputs_otsl());
    }

    #[test]
    fn formula_postprocess_unwraps_brackets_and_wraps_dollars() {
        assert_eq!(postprocess_formula("\\[E = mc^2\\]"), "$$E = mc^2$$");
        assert_eq!(postprocess_formula("  \\[E = mc^2\\]  "), "$$E = mc^2$$");
        assert_eq!(postprocess_formula("E = mc^2"), "$$E = mc^2$$");
    }

    #[test]
    fn formula_postprocess_keeps_dollar_delimited_content() {
        assert_eq!(postprocess_formula("$E = mc^2$"), "$E = mc^2$");
        assert_eq!(postprocess_formula("$$E = mc^2$$"), "$$E = mc^2$$");
    }

    #[test]
    fn formula_postprocess_strips_only_one_bracket_side() {
        // removeprefix/removesuffix act independently, mirroring the model card.
        assert_eq!(postprocess_formula("\\[E = mc^2"), "$$E = mc^2$$");
        assert_eq!(postprocess_formula("E = mc^2\\]"), "$$E = mc^2$$");
    }
}
