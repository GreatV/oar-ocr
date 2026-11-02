//! Adapt MiniCPM-OCR generation to the crate's region-recognition contract.

use super::{MiniCpmOcr, MiniCpmOcrTask};
use crate::render::{
    table::convert_otsl_to_html_with_options,
    text::{
        self, FormulaOptions, FormulaRepair, MathStripping, RepeatAlgorithm, RepetitionOptions,
    },
};
use crate::{BackendCapabilities, Error, RecognitionBackend, RecognitionTask};
use image::RgbImage;

fn table_options() -> crate::render::table::OtslOptions {
    crate::render::table::OtslOptions {
        cell_markup: crate::render::table::CellMarkup::EscapeAll,
        emit_unowned_cells: true,
        ..Default::default()
    }
}

fn strip_otsl_envelope(text: &str) -> &str {
    let text = text.trim();
    let text = text.strip_prefix("<otsl>").unwrap_or(text);
    text.strip_suffix("</otsl>").unwrap_or(text).trim()
}

fn format_region(label: &str, task: RecognitionTask, raw: &str) -> String {
    let mixed = label == "inline_formula" && text::has_mixed_inline_math(raw);
    let mut content = text::clean_repetition(
        raw.trim(),
        label == "table",
        &RepetitionOptions {
            algorithm: RepeatAlgorithm::Paragraph,
            minimum_chars: 50,
            table_minimum_chars: Some(5000),
            min_period: 8,
            min_repeats: 5,
            ..Default::default()
        },
    );
    let display = matches!(label, "display_formula" | "formula")
        || task == RecognitionTask::Formula && label != "inline_formula";
    if display || label == "inline_formula" {
        if display {
            content = text::repair_display_math(&content);
        }
        for (left, right) in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")] {
            if content.len() >= left.len() + right.len()
                && content.starts_with(left)
                && content.ends_with(right)
            {
                content = content[left.len()..content.len() - right.len()]
                    .trim()
                    .into();
            }
        }
        if display {
            static SEPARATOR: std::sync::LazyLock<regex::Regex> =
                std::sync::LazyLock::new(|| regex::Regex::new(r"\\\]\s*\\\[").unwrap());
            content = SEPARATOR
                .replace_all(&content, regex::NoExpand("$$   $$"))
                .replace(r"\[", "")
                .replace(r"\]", "");
        }
        content = text::format_formula_with_options(
            &content,
            &FormulaOptions {
                opening: "",
                closing: "",
                stripping: MathStripping::OuterDollars,
                repair: FormulaRepair::Balanced {
                    fix_delimiters: display,
                },
                normalize_mu: false,
                newline_to_break: false,
                clean_tokens: false,
            },
        );
        content = text::normalize_crop_text(&content, display);
        if content.trim().is_empty() {
            return String::new();
        }
        if display {
            return format!("$$\n{content}\n$$");
        }
        if mixed {
            return text::split_protected_paragraphs(&content, true);
        }
        return format!("${content}$");
    }
    if matches!(task, RecognitionTask::Table | RecognitionTask::Chart) {
        let body = content
            .trim()
            .strip_prefix("```html")
            .unwrap_or(content.trim());
        let body = body.strip_prefix("```").unwrap_or(body);
        content = body.strip_suffix("```").unwrap_or(body).trim().into();
        if label == "table" {
            let missing = content
                .matches("<table")
                .count()
                .saturating_sub(content.matches("</table>").count());
            content.push_str(&"</table>".repeat(missing));
        }
        content = text::normalize_crop_text(&content, false);
        let body = strip_otsl_envelope(&content);
        if crate::render::table::looks_like_table_tokens(body) {
            return convert_otsl_to_html_with_options(body, &table_options());
        }
        return body.to_string();
    }
    content = text::normalize_crop_text(&content, false);
    if matches!(
        label,
        "text"
            | "content"
            | "abstract"
            | "reference"
            | "reference_content"
            | "vertical_text"
            | "vision_footnote"
            | "algorithm"
            | "header"
            | "footer"
            | "footnote"
            | "aside_text"
    ) {
        text::split_protected_paragraphs(&content, false)
    } else {
        content
    }
}

impl MiniCpmOcr {
    fn recognize_region(
        &self,
        image: &RgbImage,
        model_task: MiniCpmOcrTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let options = self
            .generation_options
            .clone()
            .with_max_new_tokens(max_tokens);
        let text = self.generate_with_prompt(image, model_task.prompt(), &options)?;
        Ok(text)
    }
}

impl RecognitionBackend for MiniCpmOcr {
    fn doc_parser_profile(&self) -> crate::DocParserProfile {
        use crate::api::recognition::{
            AuxiliaryLabelAction, AuxiliaryLabelRule, EmptyLayoutBehavior, FormulaNumberComposition,
        };
        let ignored: Vec<String> = [
            "number",
            "footnote",
            "header",
            "header_image",
            "footer",
            "footer_image",
            "aside_text",
        ]
        .map(str::to_string)
        .into();
        crate::DocParserProfile {
            max_tokens: 8192,
            skip_auxiliary_regions: false,
            markdown_pretty: false,
            merge_labels: vec!["text".into(), "vertical_text".into()],
            merge_blocker_labels: Some(
                [
                    "image",
                    "header_image",
                    "footer_image",
                    "chart",
                    "seal",
                    "table",
                ]
                .map(str::to_string)
                .into(),
            ),
            auxiliary_labels: ignored
                .iter()
                .map(|label| AuxiliaryLabelRule {
                    label: label.clone(),
                    action: AuxiliaryLabelAction::HideAtRender,
                })
                .collect(),
            markdown_ignore_labels: ignored,
            formula_numbers: FormulaNumberComposition::SeparateParagraph,
            empty_layout: EmptyLayoutBehavior::Empty,
            crop: crate::pipeline::crop::CropOptions::default()
                .with_shape_mode(crate::pipeline::layout::LayoutShapeMode::Auto),
            layout: Some(
                crate::pipeline::layout::LayoutPostprocessOptions::paddlex()
                    .with_polygon_overlap_threshold(Some(0.9))
                    .with_polygon_overlap_strict(true)
                    .with_visual_labels(Vec::new())
                    .with_table_visual_exemption(false),
            ),
            render_style: crate::render::markdown::RenderStyle::default()
                .with_table_leading_newline(false)
                .with_strip_table_envelope(false)
                .with_section_templates(false)
                .with_collapse_title_newlines(false)
                .with_section_heading_level(Some(2))
                .with_caption_prefix("*".into())
                .with_caption_suffix("*".into())
                .with_normalize_text_newlines(false)
                .with_skip_empty_chunks(true),
            ..Default::default()
        }
    }

    fn region_token_budget(&self, label: &str, task: RecognitionTask) -> Option<usize> {
        Some(region_recipe(label, task).1)
    }

    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.recognize_region(&image, region_recipe("", task).0, max_tokens)
    }

    fn recognition_task_for_label(
        &self,
        label: &str,
        suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        match label {
            "seal" | "inline_formula" => Some(RecognitionTask::Ocr),
            "image" | "header_image" | "footer_image" => None,
            _ => suggested,
        }
    }

    fn recognize_labeled_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        labels: &[&str],
        options: &crate::GenerationOptions,
    ) -> crate::BatchResult<String> {
        if images.len() != tasks.len() || labels.len() != tasks.len() {
            return Err(Error::invalid_input(
                "MiniCPM-OCR image, label and task counts differ",
            ));
        }
        Ok(images
            .into_iter()
            .zip(tasks)
            .zip(labels)
            .map(|((image, &task), &label)| {
                let (model_task, budget) = region_recipe(label, task);
                self.recognize_region(&image, model_task, options.max_new_tokens.min(budget))
            })
            .collect())
    }

    fn postprocess_region_output(
        &self,
        label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        Some(format_region(label, task, raw))
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            truncate_repetitive_output: false,
            ..BackendCapabilities::default()
        }
    }
}

// Official minicpmvocr prompt-map budgets; the caller's cap always wins.
fn region_recipe(label: &str, task: RecognitionTask) -> (MiniCpmOcrTask, usize) {
    let model_task = match label {
        "seal" => MiniCpmOcrTask::Seal,
        "inline_formula" => MiniCpmOcrTask::Text,
        _ => match task {
            RecognitionTask::Ocr => MiniCpmOcrTask::Text,
            RecognitionTask::Formula => MiniCpmOcrTask::Formula,
            RecognitionTask::Table => MiniCpmOcrTask::Table,
            RecognitionTask::Chart => MiniCpmOcrTask::Chart,
        },
    };
    let budget = match label {
        "table" => 8192,
        "seal" | "formula_number" | "number" => 256,
        "doc_title" | "paragraph_title" | "figure_title" | "header" | "footer" => 512,
        "inline_formula" | "footnote" | "vision_footnote" | "aside_text" => 1024,
        "reference" | "reference_content" | "vertical_text" | "display_formula" => 2048,
        "text" | "content" | "abstract" | "algorithm" | "chart" => 4096,
        _ => 8192,
    };
    (model_task, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::table::convert_otsl_to_html;

    #[test]
    fn layout_labels_select_native_prompts_and_budgets() {
        let (task, budget) = region_recipe("seal", RecognitionTask::Ocr);
        assert_eq!((task.prompt(), budget), ("Seal Recognition:", 256));
        let (task, budget) = region_recipe("inline_formula", RecognitionTask::Formula);
        assert_eq!((task.prompt(), budget), ("Text Recognition:", 1024));
        assert_eq!(region_recipe("table", RecognitionTask::Table).1, 8192);
        assert_eq!(region_recipe("number", RecognitionTask::Ocr).1, 256);
    }

    #[test]
    fn otsl_tables_and_charts_lose_their_envelope() {
        let table = "<otsl><fcel>Year<fcel>Share<nl><fcel>2017<lcel><nl></otsl>";
        let html = convert_otsl_to_html(strip_otsl_envelope(table));
        assert!(
            html.contains("<td>Year</td>") && !html.contains("otsl"),
            "{html}"
        );

        let cases: serde_json::Value = serde_json::from_str(include_str!(
            "../../render/text/fixtures/document_reference.json"
        ))
        .unwrap();
        for case in cases.as_array().unwrap() {
            let label = case["label"].as_str().unwrap();
            let raw = case["input"].as_str().unwrap();
            let task = if label == "display_formula" {
                RecognitionTask::Formula
            } else {
                RecognitionTask::Ocr
            };
            assert_eq!(
                format_region(label, task, raw),
                case["output"].as_str().unwrap(),
                "{label}: {raw}"
            );
        }
        let chart = "<otsl><fcel>Year<fcel>A|B<nl><fcel>2017<ecel><nl></otsl>";
        assert_eq!(
            format_region("chart", RecognitionTask::Chart, chart),
            "<table><tr><td>Year</td><td>A|B</td></tr><tr><td>2017</td><td></td></tr></table>"
        );
    }
}
