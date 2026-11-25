//! Structure analysis from text: OCR lines grouped into layout blocks and tables by LayoutNet.
use super::OAROCR;
use super::reading::FAINT_LABEL;
use oar_ocr_core::core::OCRError;
use oar_ocr_core::domain::structure::{
    LayoutElement, LayoutElementType, StructureResult, TableCell, TableResult, TableType,
};
use oar_ocr_core::models::layout_net::{BLOCK_CLASSES, LayoutNetModel, LayoutUnit};
use oar_ocr_core::processors::BoundingBox;
use std::sync::Arc;

/// OCR followed by a layout model that reads the recognized lines.
#[derive(Debug)]
pub(super) struct TextLayoutStructure {
    pub(super) ocr: OAROCR,
    pub(super) layout_net: LayoutNetModel,
}

/// The element type of a LayoutNet block class.
///
/// A "Table" block that was not decoded into a grid, a formula read as plain text, and text inside figures or key-value forms are ordinary text here.
fn element_type(class: &str) -> LayoutElementType {
    match class {
        "Heading" => LayoutElementType::ParagraphTitle,
        "Title" => LayoutElementType::DocTitle,
        "PageHeader" => LayoutElementType::Header,
        "PageFooter" => LayoutElementType::Footer,
        "List" => LayoutElementType::List,
        "Caption" => LayoutElementType::FigureTableChartTitle,
        "Footnote" => LayoutElementType::Footnote,
        "Code" => LayoutElementType::Algorithm,
        _ => LayoutElementType::Text,
    }
}

fn rect(b: [f32; 4]) -> BoundingBox {
    BoundingBox::from_coords(b[0], b[1], b[2], b[3])
}

impl TextLayoutStructure {
    pub(super) fn predict_image(
        &self,
        image: image::RgbImage,
    ) -> Result<StructureResult, OCRError> {
        let page = self
            .ocr
            .predict(vec![image])?
            .pop()
            .ok_or_else(|| OCRError::invalid_input("OCR returned no page"))?;
        let lines: Vec<LayoutUnit> = page
            .text_regions
            .iter()
            .filter_map(|r| {
                let text = r.text.as_deref().filter(|t| !t.trim().is_empty())?;
                let b = &r.bounding_box;
                let line = match b.points.as_slice() {
                    [p0, p1, p2, p3] => LayoutUnit::from_polygon(
                        [[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]],
                        text,
                    ),
                    _ => LayoutUnit::new([b.x_min(), b.y_min(), b.x_max(), b.y_max()], text),
                };
                // A faint token's reading is hidden from the model: a lone value between two cells makes it fuse the table's rows.
                Some(
                    line.confidence(r.confidence.unwrap_or(1.0))
                        .opaque(r.label.as_deref() == Some(FAINT_LABEL)),
                )
            })
            .collect();
        let found = self.layout_net.predict(&page.input_img, &lines)?;
        let units = found.units;

        let mut in_table = vec![false; units.len()];
        let mut elements = Vec::new();
        let mut tables = Vec::new();
        for table in found.tables {
            for &i in &table.units {
                in_table[i] = true;
            }
            let bbox = rect(table.bbox);
            elements.push(
                LayoutElement::new(bbox.clone(), LayoutElementType::Table, table.score)
                    .with_label("Table"),
            );
            let cells = table
                .cells
                .iter()
                .map(|c| {
                    TableCell::new(rect(c.bbox), table.score)
                        .with_position(c.row, c.col)
                        .with_span(c.row_span, c.col_span)
                        .with_text(c.text.clone())
                })
                .collect();
            tables.push(
                TableResult::new(bbox, TableType::Unknown)
                    .with_structure_confidence(table.score)
                    .with_cells(cells)
                    .with_html_structure(table.html)
                    .with_e2e(true),
            );
        }
        if found.blocks.is_empty() {
            // The model skips pages with almost no text, whose units are then the blocks.
            for unit in &units {
                let element = LayoutElement::new(rect(unit.bbox), LayoutElementType::Text, 1.0);
                elements.push(element.with_text(unit.text.clone()));
            }
        }
        for block in found.blocks {
            // Blocks cover every unit, so a block mostly inside a decoded table repeats it.
            let inside = block.units.iter().filter(|&&i| in_table[i]).count();
            if inside * 2 > block.units.len() {
                continue;
            }
            let class = BLOCK_CLASSES[block.class];
            let kind = element_type(class);
            let text = if kind == LayoutElementType::List {
                // One item per unit.
                let items: Vec<&str> = block
                    .units
                    .iter()
                    .map(|&i| units[i].text.as_str())
                    .collect();
                items.join("\n")
            } else {
                block.text
            };
            elements.push(
                LayoutElement::new(rect(block.bbox), kind, block.score)
                    .with_label(class)
                    .with_text(text),
            );
        }
        let (width, height) = page.input_img.dimensions();
        super::structure::sort_layout_elements(&mut elements, width as f32, height as f32);
        for (i, element) in elements.iter_mut().enumerate() {
            element.order_index = Some(i as u32 + 1);
        }
        Ok(StructureResult {
            input_path: Arc::from("memory"),
            index: 0,
            layout_elements: elements,
            tables,
            formulas: Vec::new(),
            text_regions: (!page.text_regions.is_empty()).then_some(page.text_regions),
            orientation_angle: page.orientation_angle,
            region_blocks: None,
            rectified_img: Some(page.input_img),
            page_continuation_flags: None,
        })
    }
}
