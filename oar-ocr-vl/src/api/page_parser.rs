//! High-level complete-page parsing contract.

use crate::api::error::Error;
use crate::document::page::{DocumentOutput, PageDocument};
use image::RgbImage;

/// A model or composed pipeline capable of parsing a complete page.
pub trait PageParser {
    /// Parser-specific generation and scheduling options.
    type Options;

    /// Parse a complete page into the common VL document representation.
    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error>;

    /// Parse ordered pages. Models with a joint document protocol override
    /// this method; other parsers process each page independently.
    fn parse_document(
        &self,
        images: &[RgbImage],
        options: &Self::Options,
    ) -> Result<DocumentOutput, Error> {
        images
            .iter()
            .map(|image| self.parse_page(image, options))
            .collect::<Result<Vec<_>, _>>()
            .map(DocumentOutput::from_pages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_document_default_preserves_order_and_errors() {
        struct Parser;
        impl PageParser for Parser {
            type Options = ();
            fn parse_page(&self, image: &RgbImage, _: &()) -> Result<PageDocument, Error> {
                if image.width() == 3 {
                    return Err(Error::invalid_input("failed page"));
                }
                Ok(PageDocument {
                    markdown: Some(image.width().to_string()),
                    ..PageDocument::default()
                })
            }
        }
        let output = Parser
            .parse_document(&[RgbImage::new(1, 1), RgbImage::new(2, 1)], &())
            .unwrap();
        assert_eq!(output.markdown.as_deref(), Some("1\n\n2"));
        assert!(Parser.parse_document(&[RgbImage::new(3, 1)], &()).is_err());
        assert!(Parser.parse_document(&[], &()).unwrap().pages.is_empty());
    }
}
