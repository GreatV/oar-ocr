//! Complete-page parser adapter for WeVisDoc.

use super::{DEFAULT_MAX_NEW_TOKENS, WeVisDoc};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// WeVisDoc complete-page parsing options.
#[derive(Debug, Clone)]
pub struct WeVisDocParseOptions {
    pub max_new_tokens: usize,
}

impl Default for WeVisDocParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

impl PageParser for WeVisDoc {
    type Options = WeVisDocParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let markdown = self
            .generate(std::slice::from_ref(image), options.max_new_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("WeVisDoc returned no page result"))??;
        Ok(PageDocument {
            markdown: Some(markdown.clone()),
            raw_output: Some(markdown),
            ..PageDocument::default()
        })
    }
}
