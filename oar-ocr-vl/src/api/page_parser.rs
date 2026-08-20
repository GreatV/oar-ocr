//! High-level complete-page parsing contract.

use crate::api::error::Error;
use crate::document::page::PageDocument;
use image::RgbImage;

/// A model or composed pipeline capable of parsing a complete page.
pub trait PageParser {
    /// Parser-specific generation and scheduling options.
    type Options;

    /// Parse a complete page into the common VL document representation.
    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error>;
}
