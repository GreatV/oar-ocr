//! Complete-page parser adapter for MonkeyOCRv2.

use super::{DEFAULT_MAX_NEW_TOKENS, MonkeyOcrV2, MonkeyOcrV2Task};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// MonkeyOCRv2 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct MonkeyOcrV2ParseOptions {
    pub task: MonkeyOcrV2Task,
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
        let raw = self
            .generate(
                std::slice::from_ref(image),
                &[options.task],
                options.max_new_tokens,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("MonkeyOCRv2 returned no page result"))??;
        Ok(PageDocument {
            raw_output: Some(raw),
            ..PageDocument::default()
        })
    }
}
