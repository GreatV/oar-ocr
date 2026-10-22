//! Complete-page Markdown or structured JSON parsing.

use super::{DEFAULT_MAX_NEW_TOKENS, OvisOcr3, postprocess_text, prompts};
use crate::document::structure::LayoutElementType;
use crate::ovisocr2::filter_visual_image_tags;
use crate::render::markdown::default_markdown_ignore_labels;
use crate::{DocumentBlock, Error, PageDocument, PageParser, ParseDiagnostic};
use image::RgbImage;
use serde::Deserialize;

/// The page protocol requested from OvisOCR3.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OvisOcr3ParseMode {
    /// One Markdown document, with optional image-tag filtering.
    #[default]
    Markdown,
    /// A JSON array of regions in natural reading order.
    ///
    /// Invalid or truncated JSON is logged and returned as Markdown without
    /// blocks, retaining the raw response and a diagnostic. JSON is never repaired.
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

#[derive(Deserialize)]
struct Region {
    bbox: [u16; 4],
    category: String,
    content: String,
}

fn parse_output(text: String, options: &OvisOcr3ParseOptions) -> PageDocument {
    let mut diagnostics = Vec::new();
    if options.mode == OvisOcr3ParseMode::Structured {
        let message = match serde_json::from_str::<Vec<Region>>(&text) {
            Ok(regions)
                if regions.iter().all(|region| {
                    let [left, top, right, bottom] = region.bbox;
                    region.bbox.iter().all(|&coordinate| coordinate < 1000)
                        && left <= right
                        && top <= bottom
                }) =>
            {
                // The model card builds Markdown by concatenating each
                // region's content in reading order; the content is already
                // Markdown, LaTeX, or HTML. Default exclusions still apply.
                let ignored = default_markdown_ignore_labels();
                let markdown = regions
                    .iter()
                    .filter(|region| {
                        let label = LayoutElementType::from_label(&region.category);
                        !ignored.iter().any(|ignored| ignored == label.as_str())
                    })
                    .map(|region| region.content.trim())
                    .filter(|content| !content.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let markdown = if options.keep_image_tags {
                    markdown
                } else {
                    filter_visual_image_tags(&markdown)
                };
                let blocks = regions
                    .into_iter()
                    .map(|region| DocumentBlock {
                        // PageDocument::to_json maps these labels through
                        // LayoutElementType while retaining native aliases.
                        block_type: region.category,
                        // PageDocument stores normalized boxes. Exporting with
                        // the input image's dimensions gives x*w/1000, y*h/1000;
                        // coordinates refer to that input (including a crop).
                        bbox: region.bbox.map(|coordinate| coordinate as f32 / 1000.0),
                        angle: None,
                        content: Some(region.content),
                    })
                    .collect();
                return PageDocument {
                    blocks,
                    markdown: Some(markdown),
                    raw_output: Some(text),
                    ..PageDocument::default()
                };
            }
            Ok(_) => "Structured output has invalid boxes; using Markdown cleanup".to_string(),
            Err(error) => {
                format!("Structured JSON is invalid or truncated: {error}; using Markdown cleanup")
            }
        };
        tracing::warn!(%message, "OvisOCR3 structured parsing failed");
        diagnostics.push(ParseDiagnostic {
            block_index: None,
            stage: "structured_json".to_string(),
            message,
        });
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

        let markdown = parse_output(raw, &OvisOcr3ParseOptions::default());
        assert!(markdown.blocks.is_empty());
        assert!(markdown.diagnostics.is_empty());
        assert_eq!(markdown.markdown.as_deref(), Some(truncated));
        assert_eq!(markdown.raw_output, markdown.markdown);
    }
}
