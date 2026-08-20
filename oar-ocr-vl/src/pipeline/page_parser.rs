//! Complete-page adapter for the external-layout document pipeline.

use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::api::recognition::RecognitionBackend;
use crate::document::page::{DocumentBlock, PageDocument};
use crate::pipeline::doc_parser::{DocParser, DocParserConfig};
use crate::pipeline::layout::LayoutSource;
use image::RgbImage;

fn normalize_bbox(
    bbox: &crate::document::geometry::BoundingBox,
    width: f32,
    height: f32,
) -> [f32; 4] {
    [
        (bbox.x_min() / width).clamp(0.0, 1.0),
        (bbox.y_min() / height).clamp(0.0, 1.0),
        (bbox.x_max() / width).clamp(0.0, 1.0),
        (bbox.y_max() / height).clamp(0.0, 1.0),
    ]
}

/// A complete-page parser composed from a layout source and region recognizer.
pub struct LayoutFirstPageParser<'a, L: LayoutSource + ?Sized, B: RecognitionBackend + ?Sized> {
    layout: &'a L,
    parser: DocParser<'a, B>,
}

impl<'a, L: LayoutSource + ?Sized, B: RecognitionBackend + ?Sized> LayoutFirstPageParser<'a, L, B> {
    pub fn new(layout: &'a L, backend: &'a B) -> Self {
        Self {
            layout,
            parser: DocParser::new(backend),
        }
    }

    pub fn with_config(layout: &'a L, backend: &'a B, config: DocParserConfig) -> Self {
        Self {
            layout,
            parser: DocParser::with_config(backend, config),
        }
    }

    pub fn with_region_batch_size(mut self, size: usize) -> Self {
        self.parser = self.parser.with_region_batch_size(size);
        self
    }
}

impl<L: LayoutSource + ?Sized, B: RecognitionBackend + ?Sized> PageParser
    for LayoutFirstPageParser<'_, L, B>
{
    type Options = ();

    fn parse_page(
        &self,
        image: &RgbImage,
        _options: &Self::Options,
    ) -> Result<PageDocument, Error> {
        let width = image.width().max(1) as f32;
        let height = image.height().max(1) as f32;
        let result = self.parser.parse(self.layout, image.clone())?;
        let markdown = crate::render::markdown::to_markdown(
            &result.layout_elements,
            &self.parser.config().markdown_ignore_labels,
            self.parser.config().markdown_pretty,
        );
        let blocks = result
            .layout_elements
            .into_iter()
            .map(|element| DocumentBlock {
                block_type: element
                    .label
                    .unwrap_or_else(|| element.element_type.as_str().to_string()),
                bbox: normalize_bbox(&element.bbox, width, height),
                angle: None,
                content: element.text,
            })
            .collect();
        Ok(PageDocument {
            blocks,
            markdown: Some(markdown),
            ..PageDocument::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_bbox;
    use crate::document::geometry::BoundingBox;

    #[test]
    fn normalized_bbox_clamps_custom_layout_coordinates_to_the_page() {
        let bbox = BoundingBox::from_coords(-20.0, -10.0, 120.0, 110.0);

        assert_eq!(normalize_bbox(&bbox, 100.0, 100.0), [0.0, 0.0, 1.0, 1.0]);
    }
}
