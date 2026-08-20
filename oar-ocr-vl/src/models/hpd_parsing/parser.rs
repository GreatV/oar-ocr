//! Complete-page parser adapter for HPD-Parsing.

use super::{HpdGenerationConfig, HpdParsing};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

impl PageParser for HpdParsing {
    type Options = HpdGenerationConfig;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let raw = self
            .parse(std::slice::from_ref(image), options)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("HPD-Parsing returned no page result"))??;
        Ok(PageDocument {
            raw_output: Some(raw),
            ..PageDocument::default()
        })
    }
}
