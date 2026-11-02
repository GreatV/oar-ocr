//! Region-recognition adapter for GLM-OCR.

use super::GlmOcr;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr | RecognitionTask::Chart => "Text Recognition:",
        RecognitionTask::Table => "Table Recognition:",
        RecognitionTask::Formula => "Formula Recognition:",
    }
}

fn clean(output: String) -> String {
    crate::render::text::clean_repetition(
        &output,
        false,
        &crate::render::text::RepetitionOptions {
            algorithm: crate::render::text::RepeatAlgorithm::Content,
            minimum_chars: 2048,
            preserve_complete_tables: true,
            min_period: 10,
            trim: true,
            ..Default::default()
        },
    )
}

fn prepare_region(image: &RgbImage) -> Result<RgbImage, Error> {
    let (h, w) = (image.height() as f64, image.width() as f64);
    if h == 0. || w == 0. {
        return Err(Error::invalid_input("GLM-OCR cannot process an empty crop"));
    }
    // SDK utils/image_utils.py: t=t_factor=2, factor=28; no preliminary
    // small-side scaling. The model processor runs after this bicubic resize.
    let (mut rh, mut rw) = (
        (h / 28.).round_ties_even() * 28.,
        (w / 28.).round_ties_even() * 28.,
    );
    if 2. * rh * rw > 71_372_800. {
        let beta = (2. * h * w / 71_372_800.).sqrt();
        rh = (h / beta / 28.).floor() * 28.;
        rw = (w / beta / 28.).floor() * 28.;
    } else if 2. * rh * rw < 12_544. {
        let beta = (12_544. / (2. * h * w)).sqrt();
        rh = (h * beta / 28.).ceil() * 28.;
        rw = (w * beta / 28.).ceil() * 28.;
    }
    crate::runtime::image::pil_bicubic_resize(image, rw.max(28.) as u32, rh.max(28.) as u32)
}

impl RecognitionBackend for GlmOcr {
    fn doc_parser_profile(&self) -> crate::DocParserProfile {
        use crate::api::recognition::{
            AuxiliaryLabelAction, AuxiliaryLabelRule, EmptyLayoutBehavior, FormulaNumberComposition,
        };
        use crate::pipeline::layout::{
            ContainmentMode, ContainmentRule, LayoutPostprocessOptions, LayoutShapeMode,
        };
        let layout = LayoutPostprocessOptions::paddlex()
            .with_containment_ratio(0.8)
            .with_containment_rules(
                (0..25)
                    .map(|class_id| ContainmentRule {
                        class_id,
                        mode: if class_id == 18 {
                            ContainmentMode::Small
                        } else {
                            ContainmentMode::Large
                        },
                    })
                    .collect(),
            )
            .with_preserved_labels(["image", "seal", "chart"].map(str::to_string).into())
            .with_min_size(0.)
            .with_overlap_threshold(None)
            .with_inline_formula_overlap(None)
            .with_polygon_overlap_threshold(None)
            .with_drop_labels(Vec::new())
            .with_shape_mode(LayoutShapeMode::Rect)
            .with_round_boxes(false);
        crate::DocParserProfile {
            max_tokens: 8192,
            skip_auxiliary_regions: false,
            merge_labels: Vec::new(),
            markdown_pretty: false,
            auxiliary_labels: [
                "header",
                "header_image",
                "footer",
                "footer_image",
                "number",
                "footnote",
                "aside_text",
                "reference",
            ]
            .into_iter()
            .map(|label| AuxiliaryLabelRule {
                label: label.into(),
                action: AuxiliaryLabelAction::Drop,
            })
            .collect(),
            formula_numbers: FormulaNumberComposition::AdjacentTag,
            empty_layout: EmptyLayoutBehavior::Empty,
            crop: crate::pipeline::crop::CropOptions::default().with_coordinate_grid(Some(1000)),
            layout: Some(layout),
            render_style: crate::render::markdown::RenderStyle::default()
                .with_table_leading_newline(false)
                .with_strip_table_envelope(false)
                .with_section_templates(false)
                .with_collapse_title_newlines(false)
                .with_section_heading_level(Some(2))
                .with_skip_empty_chunks(true)
                .with_strip_title_markers(true)
                .with_normalize_bullets(true)
                .with_merge_hyphenated_text(true),
            ..Default::default()
        }
    }

    fn recognition_task_for_label(
        &self,
        label: &str,
        suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        match label {
            "chart" | "image" => None,
            "seal" | "vision_footnote" => Some(RecognitionTask::Ocr),
            _ => suggested,
        }
    }

    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let image = prepare_region(&image)?;
        self.generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("GLM-OCR: no result returned"))?
            .map(clean)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let prompts: Vec<_> = tasks.iter().copied().map(prompt).collect();
        let images = images
            .iter()
            .map(prepare_region)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .map(|result| result.map(clean))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            truncate_repetitive_output: false,
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
        Some(if task == RecognitionTask::Formula {
            crate::render::text::format_formula_with_options(
                raw,
                &crate::render::text::FormulaOptions {
                    opening: "$$\n",
                    closing: "\n$$",
                    stripping: crate::render::text::MathStripping::OuterDollars,
                    repair: crate::render::text::FormulaRepair::None,
                    normalize_mu: false,
                    newline_to_break: false,
                    clean_tokens: false,
                },
            )
        } else {
            if matches!(label, "doc_title" | "paragraph_title") {
                raw.trim().to_string()
            } else {
                crate::render::markdown::assembly::normalize_bullet(raw)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_crops_and_cleanup_preserve_thin_regions_and_legitimate_tables() {
        assert_eq!(
            prepare_region(&RgbImage::new(400, 20))
                .unwrap()
                .dimensions(),
            (392, 28)
        );
        assert_eq!(
            prepare_region(&RgbImage::new(175, 10))
                .unwrap()
                .dimensions(),
            (336, 28)
        );
        let table = format!("<table>{}</table>", "<tr><td>x</td></tr>".repeat(120));
        assert_eq!(clean(table.clone()), table);
        assert_eq!(
            clean("a\na\na\na\na\na\na\na\na\na".into()),
            "a\na\na\na\na\na\na\na\na\na"
        );
        assert_eq!(
            clean(format!("prefix {}", "abcdefghij".repeat(210))),
            "prefix abcdefghij"
        );
    }
}
