//! Complete-page parser adapter for OvisOCR2.

use super::{DEFAULT_MAX_NEW_TOKENS, OvisOcr2};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// OvisOCR2 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct OvisOcr2ParseOptions {
    pub max_new_tokens: usize,
    pub keep_image_tags: bool,
}

impl Default for OvisOcr2ParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            keep_image_tags: false,
        }
    }
}

impl PageParser for OvisOcr2 {
    type Options = OvisOcr2ParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let markdown = self
            .parse_with_image_tags(
                std::slice::from_ref(image),
                options.max_new_tokens,
                options.keep_image_tags,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("OvisOCR2 returned no page result"))??;
        Ok(PageDocument {
            markdown: Some(markdown.clone()),
            raw_output: Some(markdown),
            ..PageDocument::default()
        })
    }
}
