//! Complete-page Markdown or structured JSON parsing.

use super::{DEFAULT_MAX_NEW_TOKENS, OvisOcr3, postprocess_text, prompts};
use crate::document::structure::LayoutElementType;
use crate::ovisocr2::filter_visual_image_tags;
use crate::render::markdown::default_markdown_ignore_labels;
use crate::{DocumentBlock, Error, PageDocument, PageParser, ParseDiagnostic};
use image::RgbImage;
use serde_json::Value;

/// The page protocol requested from OvisOCR3.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OvisOcr3ParseMode {
    /// One Markdown document, with optional image-tag filtering.
    #[default]
    Markdown,
    /// A JSON array of regions in natural reading order.
    ///
    /// Recover complete prefix objects on truncation. Invalid boxes fall back
    /// to the full page with diagnostics; raw output is always retained.
    Structured,
}

/// OvisOCR3 page parsing options.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OvisOcr3ParseOptions {
    /// Maximum number of generated tokens per page; defaults to 16,384.
    pub max_new_tokens: usize,
    /// Retain image-tag blocks in Markdown, including structured-mode Markdown
    /// and invalid-JSON fallback. Structured block contents remain unchanged.
    pub keep_image_tags: bool,
    /// Page protocol to request; defaults to Markdown.
    pub mode: OvisOcr3ParseMode,
}

impl Default for OvisOcr3ParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            keep_image_tags: false,
            mode: OvisOcr3ParseMode::Markdown,
        }
    }
}

impl OvisOcr3ParseOptions {
    /// Set the per-page generation limit.
    pub fn with_max_new_tokens(mut self, value: usize) -> Self {
        self.max_new_tokens = value;
        self
    }

    /// Keep image-tag blocks in rendered Markdown.
    pub fn with_keep_image_tags(mut self, value: bool) -> Self {
        self.keep_image_tags = value;
        self
    }

    /// Select the Markdown or structured JSON page protocol.
    pub fn with_mode(mut self, value: OvisOcr3ParseMode) -> Self {
        self.mode = value;
        self
    }
}

impl PageParser for OvisOcr3 {
    type Options = OvisOcr3ParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let prompt = match options.mode {
            OvisOcr3ParseMode::Markdown => prompts::DOCUMENT_MARKDOWN,
            OvisOcr3ParseMode::Structured => prompts::DOCUMENT_STRUCTURED,
        };
        let text = self.generate_one(image, prompt, options.max_new_tokens)?;
        Ok(parse_output(text, options))
    }
}

fn structured_regions(text: &str) -> (Vec<Value>, Option<String>) {
    // OvisOCR3 document_crosspage.py structured_to_markdown (Apache-2.0).
    // Recover only complete values.
    let mut text = text.trim();
    if let Some(body) = text.strip_prefix("```") {
        text = body
            .strip_prefix("json")
            .or_else(|| body.strip_prefix("JSON"))
            .unwrap_or(body)
            .trim_start();
        text = text.strip_suffix("```").unwrap_or(text).trim_end();
    }
    let error = match serde_json::from_str::<Vec<Value>>(text) {
        Ok(regions) => return (regions, None),
        Err(error) => error,
    };
    let mut regions = Vec::new();
    if let Some(mut tail) = text.strip_prefix('[') {
        loop {
            tail = tail.trim_start();
            if tail.is_empty() || tail.starts_with(']') {
                break;
            }
            let mut stream = serde_json::Deserializer::from_str(tail).into_iter::<Value>();
            let Some(Ok(value)) = stream.next() else {
                break;
            };
            regions.push(value);
            tail = tail[stream.byte_offset()..].trim_start();
            if let Some(next) = tail.strip_prefix(',') {
                tail = next;
            } else {
                break;
            }
        }
    }
    (
        regions,
        Some(format!(
            "Structured JSON is invalid or truncated: {error}; retained complete prefix objects"
        )),
    )
}

fn parse_output(text: String, options: &OvisOcr3ParseOptions) -> PageDocument {
    let mut diagnostics = Vec::new();
    if options.mode == OvisOcr3ParseMode::Structured {
        let (regions, problem) = structured_regions(&text);
        if let Some(message) = &problem {
            diagnostics.push(ParseDiagnostic {
                block_index: None,
                stage: "structured_json".into(),
                message: message.clone(),
            });
        }
        if problem.is_none() || !regions.is_empty() {
            let mut blocks = Vec::new();
            for (index, region) in regions.iter().enumerate() {
                let Some(content) = region.get("content").and_then(Value::as_str) else {
                    diagnostics.push(ParseDiagnostic {
                        block_index: Some(index),
                        stage: "structured_json".into(),
                        message: "Structured object has no string content".into(),
                    });
                    continue;
                };
                let category = region
                    .get("category")
                    .and_then(Value::as_str)
                    .unwrap_or("text");
                if region.get("category").and_then(Value::as_str).is_none()
                    || LayoutElementType::from_label(category) == LayoutElementType::Other
                {
                    diagnostics.push(ParseDiagnostic {
                        block_index: Some(index),
                        stage: "structured_json".into(),
                        message: "Structured category is missing or unknown; retaining content"
                            .into(),
                    });
                }
                let coordinates = region
                    .get("bbox")
                    .and_then(Value::as_array)
                    .filter(|v| v.len() == 4)
                    .and_then(|v| v.iter().map(Value::as_u64).collect::<Option<Vec<_>>>());
                let bbox = coordinates
                    .filter(|v| v.iter().all(|&n| n < 1000) && v[0] <= v[2] && v[1] <= v[3])
                    .map(|v| [v[0], v[1], v[2], v[3]].map(|n| n as f32 / 1000.))
                    .unwrap_or_else(|| {
                        diagnostics.push(ParseDiagnostic {
                            block_index: Some(index),
                            stage: "structured_json".into(),
                            message:
                                "Invalid structured bbox; using full page and retaining content"
                                    .into(),
                        });
                        [0., 0., 1., 1.]
                    });
                blocks.push(DocumentBlock {
                    block_type: category.into(),
                    bbox,
                    angle: None,
                    content: Some(content.into()),
                });
            }
            // The model card builds Markdown by concatenating each
            // region's content in reading order; the content is already
            // Markdown, LaTeX, or HTML. Default exclusions still apply.
            let ignored = default_markdown_ignore_labels();
            let markdown = blocks
                .iter()
                .filter(|region| {
                    let label = LayoutElementType::from_label(&region.block_type);
                    !ignored.iter().any(|ignored| ignored == label.as_str())
                })
                .filter_map(|region| region.content.as_deref().map(str::trim))
                .filter(|content| !content.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            let markdown = if options.keep_image_tags {
                markdown
            } else {
                filter_visual_image_tags(&markdown)
            };
            return PageDocument {
                blocks,
                markdown: Some(markdown),
                raw_output: Some(text),
                diagnostics,
            };
        }
    }
    let markdown = postprocess_text(text.trim().to_string(), options.keep_image_tags);
    let raw_output = match options.mode {
        OvisOcr3ParseMode::Markdown => markdown.clone(),
        OvisOcr3ParseMode::Structured => text,
    };
    PageDocument {
        markdown: Some(markdown),
        raw_output: Some(raw_output),
        diagnostics,
        ..PageDocument::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn structured_regions_scale_map_categories_and_fall_back_on_truncation() {
        let options = OvisOcr3ParseOptions::default().with_mode(OvisOcr3ParseMode::Structured);
        let text = r###"[
            {"bbox":[0,0,999,100],"category":"doc_title","content":"# Title"},
            {"bbox":[125,250,875,500],"category":"display_formula","content":"x = 1"},
            {"bbox":[0,500,500,750],"category":"vertical_text","content":"Hello"},
            {"bbox":[500,750,875,999],"category":"vision_footnote","content":"Source"},
            {"bbox":[0,750,500,800],"category":"paragraph_title","content":"## Section"},
            {"bbox":[0,0,999,50],"category":"header","content":"Header"},
            {"bbox":[0,950,999,999],"category":"footer","content":"Footer"},
            {"bbox":[0,950,100,999],"category":"number","content":"99"},
            {"bbox":[0,800,100,900],"category":"image","content":"<img src=\"images/bbox_0_800_100_900.jpg\" />"}
        ]"###;
        let image_size = (1600, 1200);
        let page = parse_output(text.to_string(), &options);
        assert_eq!(page.raw_output.as_deref(), Some(text));
        let expected_markdown = "# Title\n\nx = 1\n\nHello\n\n## Section";
        assert_eq!(page.markdown.as_deref(), Some(expected_markdown));
        assert_eq!(page.blocks.len(), 9);
        assert!(page.diagnostics.is_empty());
        let exported = page.to_json(image_size.0, image_size.1);
        assert_eq!(
            exported["blocks"][1]["bbox"],
            json!([200.0, 300.0, 1400.0, 600.0])
        );
        assert_eq!(exported["blocks"][1]["type"], "formula");
        assert_eq!(exported["blocks"][1]["label"], "display_formula");
        assert_eq!(exported["blocks"][2]["type"], "text");
        assert_eq!(exported["blocks"][3]["type"], "footnote");
        assert_eq!(exported["blocks"][3]["order"], 4);
        let with_images = parse_output(
            text.to_string(),
            &options.clone().with_keep_image_tags(true),
        );
        assert_eq!(
            with_images.markdown.unwrap(),
            format!("{expected_markdown}\n\n<img src=\"images/bbox_0_800_100_900.jpg\" />")
        );

        let truncated =
            "[{\"bbox\":[125,250,875,500],\"category\":\"text\",\"content\":\"unfinished";
        let raw = format!("  {truncated}\n\n<img src=\"images/bbox_1_2_3_4.jpg\" />\n ");
        let fallback = parse_output(raw.clone(), &options);
        assert!(fallback.blocks.is_empty());
        assert_eq!(fallback.markdown.as_deref(), Some(truncated));
        assert_eq!(fallback.raw_output.as_deref(), Some(raw.as_str()));
        assert_eq!(fallback.diagnostics.len(), 1);
        assert_eq!(fallback.diagnostics[0].block_index, None);
        assert_eq!(fallback.diagnostics[0].stage, "structured_json");
        assert!(fallback.diagnostics[0].message.contains("truncated"));

        let recovered = parse_output("```json\n[{\"bbox\":[5,0,1,999],\"category\":\"text\",\"content\":\"keep\"},{\"content\":\"unfinished".into(), &options);
        assert_eq!(recovered.blocks.len(), 1);
        assert_eq!(recovered.blocks[0].bbox, [0., 0., 1., 1.]);
        assert_eq!(recovered.markdown.as_deref(), Some("keep"));
        assert_eq!(recovered.diagnostics.len(), 2);

        let markdown = parse_output(raw, &OvisOcr3ParseOptions::default());
        assert!(markdown.blocks.is_empty());
        assert!(markdown.diagnostics.is_empty());
        assert_eq!(markdown.markdown.as_deref(), Some(truncated));
        assert_eq!(markdown.raw_output, markdown.markdown);
    }
}
