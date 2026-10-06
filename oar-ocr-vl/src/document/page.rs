//! Model-independent page document returned by high-level parsing pipelines.

use crate::document::structure::StructureResult;
use serde::{Deserialize, Serialize};

/// One normalized page block and its optional recognized content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBlock {
    /// Model-native semantic label.
    #[serde(rename = "type")]
    pub block_type: String,
    /// Normalized `[x1, y1, x2, y2]` coordinates in `[0, 1]`.
    pub bbox: [f32; 4],
    /// Clockwise rotation applied before recognition.
    pub angle: Option<u16>,
    /// Recognized block content.
    pub content: Option<String>,
}

/// Non-fatal issue produced while parsing a page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseDiagnostic {
    /// Block index when the issue is block-specific.
    pub block_index: Option<usize>,
    /// Pipeline stage that produced the issue.
    pub stage: String,
    /// Human-readable diagnostic.
    pub message: String,
}

/// Unified high-level output for a parsed page.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageDocument {
    /// Structured page blocks, when the model exposes layout.
    pub blocks: Vec<DocumentBlock>,
    /// Ready-to-render Markdown, when produced natively or by a renderer.
    pub markdown: Option<String>,
    /// Raw model protocol output retained for lossless diagnostics.
    pub raw_output: Option<String>,
    /// Non-fatal block or post-processing failures.
    pub diagnostics: Vec<ParseDiagnostic>,
}

impl PageDocument {
    /// Converts a layout-first result using the default Markdown renderer.
    pub fn from_structure(result: StructureResult, image_width: u32, image_height: u32) -> Self {
        let markdown = result.to_markdown();
        Self::from_structure_with_markdown(result, image_width, image_height, markdown)
    }

    /// Converts a layout-first result with Markdown rendered by the caller.
    ///
    /// Blocks follow the result's reading order and retain original labels.
    /// Coordinates are normalized to the image dimensions and clamped to `[0, 1]`.
    /// For pixel coordinates and full structure metadata, use
    /// [`LayoutPageParser::parse_structure`](crate::LayoutPageParser::parse_structure).
    /// No single raw model output exists for this pipeline.
    pub fn from_structure_with_markdown(
        result: StructureResult,
        image_width: u32,
        image_height: u32,
        markdown: String,
    ) -> Self {
        let width = image_width.max(1) as f32;
        let height = image_height.max(1) as f32;
        let blocks = result
            .layout_elements
            .into_iter()
            .map(|element| DocumentBlock {
                block_type: element
                    .label
                    .unwrap_or_else(|| element.element_type.as_str().to_string()),
                bbox: [
                    (element.bbox.x_min() / width).clamp(0.0, 1.0),
                    (element.bbox.y_min() / height).clamp(0.0, 1.0),
                    (element.bbox.x_max() / width).clamp(0.0, 1.0),
                    (element.bbox.y_max() / height).clamp(0.0, 1.0),
                ],
                angle: None,
                content: element.text,
            })
            .collect();
        Self {
            blocks,
            markdown: Some(markdown),
            raw_output: None,
            diagnostics: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::geometry::BoundingBox;
    use crate::document::structure::{LayoutElement, LayoutElementType};

    #[test]
    fn structure_conversion_preserves_blocks_and_markdown() {
        let bbox = BoundingBox::from_coords(10.0, 20.0, 90.0, 60.0);
        let mut text = LayoutElement::new(bbox.clone(), LayoutElementType::Text, 0.75)
            .with_label("paragraph")
            .with_text("hello");
        text.order_index = Some(1);
        let table = LayoutElement::new(bbox.clone(), LayoutElementType::Table, 0.8)
            .with_text("<table><tr><td>cell</td></tr></table>");
        let result = StructureResult::new("page.png", 3).with_layout_elements(vec![text, table]);
        let expected_markdown = result.to_markdown();
        let page = PageDocument::from_structure(result, 100, 80);

        assert_eq!(page.markdown.as_deref(), Some(expected_markdown.as_str()));
        assert_eq!(page.blocks.len(), 2);
        assert_eq!(page.blocks[0].block_type, "paragraph");
        assert_eq!(page.blocks[0].bbox, [0.1, 0.25, 0.9, 0.75]);
        assert_eq!(page.blocks[0].content.as_deref(), Some("hello"));
        assert_eq!(page.blocks[1].block_type, "table");
        assert_eq!(page.blocks[0].angle, None);
        assert!(page.raw_output.is_none());
        assert_eq!(
            page.blocks[1].content.as_deref(),
            Some("<table><tr><td>cell</td></tr></table>")
        );
        assert!(page.diagnostics.is_empty());
        let expected_page = serde_json::to_value(&page).unwrap();
        let roundtrip: PageDocument = serde_json::from_value(expected_page.clone()).unwrap();
        assert_eq!(serde_json::to_value(roundtrip).unwrap(), expected_page);
    }

    #[test]
    fn normalized_coordinates_clamp_to_page_and_handle_empty_images() {
        let result =
            StructureResult::new("<memory>", 0).with_layout_elements(vec![LayoutElement::new(
                BoundingBox::from_coords(-20.0, -10.0, 120.0, 110.0),
                LayoutElementType::Other,
                1.0,
            )]);
        let page = PageDocument::from_structure(result.clone(), 100, 100);
        assert_eq!(page.blocks[0].bbox, [0.0, 0.0, 1.0, 1.0]);
        let page = PageDocument::from_structure(result, 0, 0);
        assert_eq!(page.blocks[0].bbox, [0.0, 0.0, 1.0, 1.0]);
    }
}
