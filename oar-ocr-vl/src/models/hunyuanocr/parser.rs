//! Complete-page parser adapter for HunyuanOCR.

use super::HunyuanOcr;
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// HunyuanOCR prompt-driven complete-page parsing options.
#[derive(Debug, Clone)]
pub struct HunyuanOcrParseOptions {
    pub prompt: String,
    pub max_new_tokens: usize,
}

impl Default for HunyuanOcrParseOptions {
    fn default() -> Self {
        Self {
            prompt: "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.".to_string(),
            max_new_tokens: 16_384,
        }
    }
}

impl PageParser for HunyuanOcr {
    type Options = HunyuanOcrParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let raw = self
            .generate(
                std::slice::from_ref(image),
                &[options.prompt.as_str()],
                options.max_new_tokens,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("HunyuanOCR returned no page result"))??;
        Ok(PageDocument {
            markdown: Some(raw.clone()),
            raw_output: Some(raw),
            ..PageDocument::default()
        })
    }
}
