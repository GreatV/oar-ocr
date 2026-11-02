//! Complete-page parser adapter for MonkeyOCRv2.

use super::adapter::{formula_options, table_options};
use super::{DEFAULT_MAX_NEW_TOKENS, MonkeyOcrV2, MonkeyOcrV2Task};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::decoding::normalize_python_literals as normalize_literals;
use crate::document::geometry::BoundingBox;
use crate::document::page::{DocumentBlock, PageDocument, ParseDiagnostic};
use crate::pipeline::doc_parser::DocParserOptions;
use crate::pipeline::layout::{
    LayoutDetectionElement, LayoutDetections, LayoutPrediction, LayoutSource,
};
use crate::pipeline::page_parser::{LayoutPageParser, LayoutPageParserOptions};
use crate::render::table::convert_otsl_to_html_with_options;
use crate::render::text::format_formula_with_options;
use image::RgbImage;

/// MonkeyOCRv2 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct MonkeyOcrV2ParseOptions {
    /// Page task. `EndToEnd` uses the model's selected page pipeline; disable
    /// two-stage parsing on the model to request the one-pass protocol here.
    pub task: MonkeyOcrV2Task,
    /// Per-request cap, further bounded by layout=4096 and recognition=5000
    /// when the native two-stage pipeline is selected.
    pub max_new_tokens: usize,
}

impl Default for MonkeyOcrV2ParseOptions {
    fn default() -> Self {
        Self {
            task: MonkeyOcrV2Task::EndToEnd,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

impl PageParser for MonkeyOcrV2 {
    type Options = MonkeyOcrV2ParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        if self.two_stage_parsing && options.task == MonkeyOcrV2Task::EndToEnd {
            return self.parse_two_stage(image, options.max_new_tokens);
        }
        let preprocessor = if matches!(
            options.task,
            MonkeyOcrV2Task::EndToEnd | MonkeyOcrV2Task::Layout
        ) {
            crate::RecognitionBackend::page_preprocessor(self)
        } else {
            None
        };
        let image = crate::pipeline::page_preprocessing::prepare_page(
            std::borrow::Cow::Borrowed(image),
            preprocessor,
        )?;
        let raw = self
            .generate(
                std::slice::from_ref(image.as_ref()),
                &[options.task],
                options.max_new_tokens,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("MonkeyOCRv2 returned no page result"))??;
        Ok(parse_output(raw, options.task))
    }
}

impl MonkeyOcrV2 {
    /// Parse a page using its native layout pass and the shared region pipeline.
    /// The caller cap applies up to 4096 layout tokens and 5000 region tokens.
    pub fn parse_two_stage(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<PageDocument, Error> {
        let layout = MonkeyLayout {
            model: self,
            max_tokens: max_new_tokens.min(4096),
        };
        LayoutPageParser::new(layout, self)
            .with_options(DocParserOptions::default().with_max_tokens(max_new_tokens.min(5000)))
            .parse_page(image, &LayoutPageParserOptions::default())
    }
}

struct MonkeyLayout<'a> {
    model: &'a MonkeyOcrV2,
    max_tokens: usize,
}
impl LayoutSource for MonkeyLayout<'_> {
    fn detect(&self, image: &RgbImage) -> Result<LayoutDetections, Error> {
        self.detect_with_options(image, &Default::default())
            .map(|prediction| prediction.detections)
    }
    fn detect_with_options(
        &self,
        image: &RgbImage,
        _: &crate::LayoutPostprocessOptions,
    ) -> Result<LayoutPrediction, Error> {
        let raw = self
            .model
            .generate(
                std::slice::from_ref(image),
                &[MonkeyOcrV2Task::Layout],
                self.max_tokens,
            )?
            .pop()
            .ok_or_else(|| Error::invalid_input("MonkeyOCRv2 returned no layout result"))??;
        Ok(layout_prediction(raw, image))
    }
}

fn layout_prediction(raw: String, image: &RgbImage) -> LayoutPrediction {
    let page = parse_output(raw, MonkeyOcrV2Task::Layout);
    let elements = page
        .blocks
        .into_iter()
        .map(|block| {
            let [x1, y1, x2, y2] = block.bbox;
            let start = crate::document::geometry::point_from_grid(
                [x1, y1],
                1.,
                image.width(),
                image.height(),
            );
            let end = crate::document::geometry::point_from_grid(
                [x2, y2],
                1.,
                image.width(),
                image.height(),
            );
            LayoutDetectionElement {
                bbox: BoundingBox::from_coords(start.x, start.y, end.x, end.y),
                element_type: block.block_type,
                score: 1.,
            }
        })
        .collect();
    LayoutPrediction {
        detections: LayoutDetections::new(elements),
        raw_output: page.raw_output,
        diagnostics: page.diagnostics,
        ..Default::default()
    }
}

fn parse_output(raw: String, task: MonkeyOcrV2Task) -> PageDocument {
    let mut page = PageDocument {
        raw_output: Some(raw.clone()),
        ..PageDocument::default()
    };
    if matches!(
        task,
        MonkeyOcrV2Task::Text | MonkeyOcrV2Task::Formula | MonkeyOcrV2Task::Table
    ) {
        page.markdown = Some(match task {
            MonkeyOcrV2Task::Table => convert_otsl_to_html_with_options(&raw, &table_options()),
            MonkeyOcrV2Task::Formula => format_formula_with_options(&raw, &formula_options()),
            _ => raw.trim().to_string(),
        });
        return page;
    }
    // The official protocol uses Python literal dictionaries as well as JSON.
    // Decode strings without evaluating model output, then normalize the literals.
    let normalized = normalize_literals(&raw);
    let objects = crate::document::decoding::complete_objects(&normalized);
    for object in objects {
        let Some(label) = object.get("label").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(coords) = object
            .get("bbox")
            .and_then(|v| v.as_array())
            .filter(|v| v.len() == 4)
        else {
            continue;
        };
        let Some(coords) = coords
            .iter()
            .map(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if coords.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let bbox = [
            coords[0].min(coords[2]),
            coords[1].min(coords[3]),
            coords[0].max(coords[2]),
            coords[1].max(coords[3]),
        ]
        .map(|v| (v as f32 / 1000.0).clamp(0.0, 1.0));
        let label = match label {
            "Title" => "doc_title",
            "Section-header" => "paragraph_title",
            "Page-header" => "header",
            "Page-footer" => "footer",
            "List-item" => "text",
            "Caption" => "figure_title",
            other => other,
        }
        .to_lowercase();
        let content = object
            .get("content")
            .and_then(|v| v.as_str())
            .map(|content| {
                if label == "table" {
                    convert_otsl_to_html_with_options(content, &table_options())
                } else {
                    content.trim().to_string()
                }
            });
        page.blocks.push(DocumentBlock {
            block_type: label,
            bbox,
            angle: None,
            content,
        });
    }
    if page.blocks.is_empty() {
        page.diagnostics.push(ParseDiagnostic {
            block_index: None,
            stage: "postprocess".into(),
            message: "No valid layout records in MonkeyOCRv2 output".into(),
        });
    } else if task == MonkeyOcrV2Task::EndToEnd {
        page.markdown = Some(page.render_markdown());
    }
    page
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_stage_routes_rounded_crops_and_retains_region_failures() {
        let raw = "[{'label':'Title','bbox':[350,0,1000,200]}, {'label':'Formula','bbox':[0,200,1000,500]}, {'label':'Page-footer','bbox':[0,800,1000,900]}, {'label':'Text','bbox':[0,900,1000,1000]}, {'label':'Region','bbox':[0,0,1000,1000]}]";
        struct Layout(String);
        impl LayoutSource for Layout {
            fn detect(&self, image: &RgbImage) -> Result<LayoutDetections, Error> {
                Ok(layout_prediction(self.0.clone(), image).detections)
            }
            fn detect_with_options(
                &self,
                image: &RgbImage,
                _: &crate::LayoutPostprocessOptions,
            ) -> Result<LayoutPrediction, Error> {
                Ok(layout_prediction(self.0.clone(), image))
            }
        }
        struct Backend {
            calls: std::cell::Cell<usize>,
        }
        impl crate::RecognitionBackend for Backend {
            fn doc_parser_profile(&self) -> crate::DocParserProfile {
                super::super::adapter::parsing_profile()
            }
            fn recognition_task_for_label(
                &self,
                label: &str,
                _: Option<crate::RecognitionTask>,
            ) -> Option<crate::RecognitionTask> {
                super::super::adapter::task_for_label(label)
            }
            fn recognize(
                &self,
                _: RgbImage,
                _: crate::RecognitionTask,
                _: usize,
            ) -> Result<String, Error> {
                panic!("labeled dispatch")
            }
            fn recognize_labeled_batch_with_options(
                &self,
                images: Vec<RgbImage>,
                tasks: &[crate::RecognitionTask],
                labels: &[&str],
                options: &crate::GenerationOptions,
            ) -> crate::BatchResult<String> {
                assert_eq!(options.max_new_tokens, 777);
                Ok(images
                    .into_iter()
                    .zip(tasks)
                    .zip(labels)
                    .map(|((crop, task), label)| {
                        self.calls.set(self.calls.get() + 1);
                        match *label {
                            "doc_title" => {
                                assert_eq!(crop.dimensions(), (6, 2));
                                Ok("Heading".into())
                            }
                            "formula" => {
                                assert_eq!(*task, crate::RecognitionTask::Formula);
                                Ok(r"x\eqno (3)".into())
                            }
                            "footer" => Ok("hidden margin".into()),
                            _ => Err(Error::invalid_input("region failed")),
                        }
                    })
                    .collect())
            }
            fn postprocess_region_output(
                &self,
                _: &str,
                task: crate::RecognitionTask,
                raw: &str,
            ) -> Option<String> {
                Some(if task == crate::RecognitionTask::Formula {
                    format_formula_with_options(raw, &formula_options())
                } else {
                    raw.trim().into()
                })
            }
        }
        let backend = Backend {
            calls: std::cell::Cell::new(0),
        };
        let parser = LayoutPageParser::new(Layout(raw.into()), &backend)
            .with_options(DocParserOptions::default().with_max_tokens(777));
        let page = parser
            .parse_page(&RgbImage::new(10, 10), &Default::default())
            .unwrap();
        assert_eq!(backend.calls.get(), 4);
        assert_eq!(page.blocks.len(), 5);
        assert_eq!(page.blocks[4].block_type, "region");
        assert_eq!(page.blocks[4].content, None);
        assert_eq!(page.blocks[2].content.as_deref(), Some("hidden margin"));
        assert_eq!(page.markdown.as_deref(), Some("# Heading\n\n$$\nx\n$$\n3"));
        assert_eq!(page.raw_output.as_deref(), Some(raw));
        assert_eq!(page.diagnostics[0].block_index, Some(3));
    }
    #[test]
    fn parses_literal_records_and_preserves_raw_output() {
        let raw = r#"[{'label':'Title','bbox':[0,0,1000,100],'content':"A title"}, {'label':'Text','bbox':[0,100,1000,200],'content':'a \\times b'}]"#;
        let page = parse_output(raw.into(), MonkeyOcrV2Task::EndToEnd);
        assert_eq!(page.blocks.len(), 2);
        assert_eq!(page.blocks[1].content.as_deref(), Some(r"a \times b"));
        assert!(page.markdown.unwrap().contains("# A title"));
        assert_eq!(page.raw_output.as_deref(), Some(raw));
        let unicode = parse_output(
            r#"[{"label":"Text","bbox":[0,0,1000,1000],"content":"\u4e2d\u6587"}]"#.into(),
            MonkeyOcrV2Task::EndToEnd,
        );
        assert_eq!(unicode.blocks[0].content.as_deref(), Some("中文"));
        let literals = parse_output(
            "[{'label':'Text','bbox':[0,0,1000,100], 'content':None, 'flags':[True,False]}, {'label':'Text','bbox':[0,100,1000,200],'content':'None True False TrueName'}]".into(),
            MonkeyOcrV2Task::EndToEnd,
        );
        assert_eq!(literals.blocks.len(), 2);
        assert_eq!(literals.blocks[0].content, None);
        assert_eq!(
            literals.blocks[1].content.as_deref(),
            Some("None True False TrueName")
        );
        assert_eq!(
            normalize_literals("TrueName Falsehood NoneValue"),
            "TrueName Falsehood NoneValue"
        );
    }
}
