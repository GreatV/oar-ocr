//! Verbatim OvisOCR3 task prompts.
//!
//! Fill placeholders with literal `str::replace`, since JSON and LaTeX braces
//! are part of the instructions. For example:
//! ```
//! use oar_ocr_vl::ovisocr3::prompts;
//! let prompt = prompts::REGION_OCR.replace("{target_region}", "[0,0,999,999]");
//! ```
//! Constants for other tasks do not add specialized pipelines or multi-image input.

// Source: ATH-MaaS/OvisOCR3 (https://huggingface.co/ATH-MaaS/OvisOCR3), Apache-2.0.
pub const CHART_PARSING: &str = include_str!("prompts/chart_parsing.txt");
pub const CROSSPAGE_TABLE: &str = include_str!("prompts/crosspage_table.txt");
pub const DOCUMENT_LAYOUT: &str = include_str!("prompts/document_layout.txt");
pub const DOCUMENT_MARKDOWN: &str = include_str!("prompts/document_markdown.txt");
pub const DOCUMENT_STRUCTURED: &str = include_str!("prompts/document_structured.txt");
pub const FIGURE_CAPTIONING: &str = include_str!("prompts/figure_captioning.txt");
pub const FORMULA_RECOGNITION: &str = include_str!("prompts/formula_recognition.txt");
pub const IMAGE_CLASSIFICATION: &str = include_str!("prompts/image_classification.txt");
pub const KIE_FIELDS: &str = include_str!("prompts/kie_fields.txt");
pub const KIE_OPEN: &str = include_str!("prompts/kie_open.txt");
pub const KIE_SCHEMA: &str = include_str!("prompts/kie_schema.txt");
pub const PLAIN_OCR: &str = include_str!("prompts/plain_ocr.txt");
pub const POSITION_AWARE_VQA: &str = include_str!("prompts/position_aware_vqa.txt");
pub const REFERENCE_GROUNDING: &str = include_str!("prompts/reference_grounding.txt");
pub const REGION_OCR: &str = include_str!("prompts/region_ocr.txt");
pub const TEXT_TRANSLATION: &str = include_str!("prompts/text_translation.txt");
pub const VISUAL_QUESTION_ANSWERING: &str = include_str!("prompts/visual_question_answering.txt");
pub const WORD_SPOTTING: &str = include_str!("prompts/word_spotting.txt");
