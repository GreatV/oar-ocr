//! Complete-page parser adapter for Unlimited-OCR-2.

use super::markdown::{base_type, parse_blocks, to_markdown};
use super::{UnlimitedOcr, UnlimitedOcrGenerationOptions};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{DocumentBlock, PageDocument};
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
        Ok(PageDocument {
            blocks,
            markdown: Some(to_markdown(&raw, options.keep_image_refs)),
            raw_output: Some(raw),
            ..PageDocument::default()
        })
    }
}
