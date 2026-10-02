//! Complete-page parser adapter for Xiaomi-OCR-0.

use super::model::{DEFAULT_MAX_NEW_TOKENS, XiaomiOcr, finalize_markdown};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// Xiaomi-OCR-0 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct XiaomiOcrParseOptions {
    pub max_new_tokens: usize,
}

impl Default for XiaomiOcrParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

impl PageParser for XiaomiOcr {
    type Options = XiaomiOcrParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let tokens = self
            .generate_tokens(std::slice::from_ref(image), options.max_new_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("Xiaomi-OCR-0 returned no page result"))??;
        // `raw_output` keeps the official tail-cleanup view of the model
        // output; `markdown` additionally converts OTSL table blocks to HTML
        // (the official whole-page post-processing).
        let raw_output = self.decode_tokens(&tokens)?;
        let markdown = finalize_markdown(&raw_output);
        Ok(PageDocument {
            markdown: Some(markdown),
            raw_output: Some(raw_output),
            ..PageDocument::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_official_generation_limit() {
        let options = XiaomiOcrParseOptions::default();
        assert_eq!(options.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(DEFAULT_MAX_NEW_TOKENS, 4_096);
    }
}
