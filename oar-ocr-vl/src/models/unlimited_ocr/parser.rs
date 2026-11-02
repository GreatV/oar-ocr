//! Complete-page parser adapter for Unlimited-OCR-2.

use super::markdown::{PAGE_SPLIT_RE, base_type, parse_blocks, to_markdown};
use super::{UnlimitedOcr, UnlimitedOcrGenerationOptions};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{DocumentBlock, DocumentOutput, PageDocument};
use image::RgbImage;

/// Single-page parsing options.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct UnlimitedOcrParseOptions {
    pub generation: UnlimitedOcrGenerationOptions,
    /// Keep figures as the official converter's
    /// `![](page1_x1_y1_x2_y2.jpg)` references (0-1000 grid); dropped by
    /// default, with their boxes still in the page blocks.
    pub keep_image_refs: bool,
}

impl UnlimitedOcrParseOptions {
    pub fn with_generation(mut self, value: UnlimitedOcrGenerationOptions) -> Self {
        self.generation = value;
        self
    }

    pub fn with_keep_image_refs(mut self, value: bool) -> Self {
        self.keep_image_refs = value;
        self
    }
}

impl PageParser for UnlimitedOcr {
    type Options = UnlimitedOcrParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let raw = self.generate(std::slice::from_ref(image), &options.generation)?;
        Ok(page_from_raw(raw, options))
    }

    fn parse_document(
        &self,
        images: &[RgbImage],
        options: &Self::Options,
    ) -> Result<DocumentOutput, Error> {
        if images.is_empty() {
            return Ok(DocumentOutput::default());
        }
        let raw = self.generate(images, &options.generation)?;
        document_from_raw(raw, images.len(), options)
    }
}

fn document_from_raw(
    raw: String,
    expected_pages: usize,
    options: &UnlimitedOcrParseOptions,
) -> Result<DocumentOutput, Error> {
    let markdown = Some(to_markdown(&raw, options.keep_image_refs));
    let mut segments: Vec<_> = PAGE_SPLIT_RE.split(&raw).map(str::to_string).collect();
    if segments.first().is_some_and(|page| page.trim().is_empty()) {
        segments.remove(0);
    }
    let surplus = segments.len().saturating_sub(expected_pages);
    if surplus > 0 && expected_pages > 0 {
        let tail = segments.split_off(expected_pages - 1).join("<PAGE>");
        segments.push(tail);
    }
    let pages = (0..expected_pages)
        .map(|index| {
            let mut page = page_from_raw(segments.get(index).cloned().unwrap_or_default(), options);
            if let Some(markdown) = &mut page.markdown {
                *markdown = markdown.replace("![](page1_", &format!("![](page{}_", index + 1));
            }
            if index >= segments.len() || segments[index].trim().is_empty() {
                page.diagnostics.push(crate::ParseDiagnostic {
                    block_index: None,
                    stage: "generation".into(),
                    message: "Document generation stopped before this page".into(),
                });
            }
            if surplus > 0 && index + 1 == expected_pages {
                page.diagnostics.push(crate::ParseDiagnostic {
                    block_index: None,
                    stage: "generation".into(),
                    message: format!(
                        "Folded {surplus} surplus page segments into the last supplied page"
                    ),
                });
            }
            page
        })
        .collect();
    Ok(DocumentOutput {
        pages,
        markdown,
        raw_output: Some(raw),
    })
}

fn page_from_raw(raw: String, options: &UnlimitedOcrParseOptions) -> PageDocument {
    // Blocks keep the model's labels without the cross-page prefixes;
    // text outside any block declaration covers the whole page.
    let blocks = parse_blocks(&raw)
        .into_iter()
        .filter(|block| !block.content.is_empty() || block.bbox.is_some())
        .map(|block| DocumentBlock {
            block_type: match base_type(&block.label) {
                "__raw__" => "text".to_string(),
                base => base.to_string(),
            },
            bbox: block.bbox.map_or([0.0, 0.0, 1.0, 1.0], |bbox| {
                bbox.map(|v| v.clamp(0, 1000) as f32 / 1000.0)
            }),
            angle: None,
            content: (!block.content.is_empty()).then_some(block.content),
        })
        .collect();
    PageDocument {
        blocks,
        markdown: Some(to_markdown(&raw, options.keep_image_refs)),
        raw_output: Some(raw),
        ..PageDocument::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn document_keeps_joint_markdown_and_reports_missing_pages() {
        let raw = "<PAGE><BLOCK>cross_text[0,0,1000,100]<FORK>Hello<PAGE><BLOCK>append_text[0,0,1000,100]<FORK> world<BLOCK>image[0,100,1000,900]<FORK>";
        let options = UnlimitedOcrParseOptions::default().with_keep_image_refs(true);
        let document = document_from_raw(raw.into(), 3, &options).unwrap();
        assert_eq!(document.pages.len(), 3);
        assert!(document.markdown.as_deref().unwrap().contains("Helloworld"));
        assert!(
            document.pages[1]
                .markdown
                .as_deref()
                .unwrap()
                .contains("page2_")
        );
        assert_eq!(document.pages[2].diagnostics.len(), 1);
        assert_eq!(document.raw_output.as_deref(), Some(raw));
        let folded = document_from_raw(raw.into(), 1, &options).unwrap();
        assert_eq!(folded.pages.len(), 1);
        assert_eq!(folded.pages[0].diagnostics.len(), 1);
        assert_eq!(folded.pages[0].blocks.len(), 3);
        assert!(
            folded.pages[0]
                .markdown
                .as_deref()
                .unwrap()
                .contains("Helloworld")
        );
        assert_eq!(folded.markdown, document.markdown);
        assert_eq!(folded.raw_output.as_deref(), Some(raw));

        let variant = raw
            .replacen("<PAGE>", "< page 1 >", 1)
            .replace("<PAGE>", "</PAGE2>");
        let variant = document_from_raw(variant, 3, &options).unwrap();
        assert_eq!(
            variant.pages[0].to_json(100, 100),
            document.pages[0].to_json(100, 100)
        );
        assert_eq!(
            variant.pages[1].to_json(100, 100),
            document.pages[1].to_json(100, 100)
        );
        assert_eq!(variant.pages[2].diagnostics.len(), 1);
    }
}
