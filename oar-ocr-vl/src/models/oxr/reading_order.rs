//! OXR's grounding-box reading-order request.

use super::Oxr;
use crate::{
    Error,
    document::structure::{LayoutElement, LayoutElementType},
};
use image::RgbImage;

impl Oxr {
    /// Return the model's reading-order permutation for pixel-coordinate boxes. Labels are mapped onto the official OXR layout vocabulary in the prompt. Malformed model output returns an empty vector so callers can retain detector order.
    pub fn reading_order(
        &self,
        image: &RgbImage,
        regions: &[LayoutElement],
        max_tokens: usize,
    ) -> Result<Vec<usize>, Error> {
        if regions.len() <= 1 {
            return Ok((0..regions.len()).collect());
        }
        let prompt = build_prompt(regions, image.width(), image.height())?;
        let budget = max_tokens.min(regions.len().saturating_mul(4).max(16));
        let options = self.generation_options.clone().with_max_new_tokens(budget);
        let trace = self.generate_one(image, &prompt, &options, false)?;
        Ok(parse_order(
            &self.decode_tokens(&trace.tokens)?,
            regions.len(),
        ))
    }
}

fn official_label(region: &LayoutElement) -> &str {
    match region
        .label
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "chemical_structure" | "chemical structure" => {
            return "Chemical Structure";
        }
        "code" | "code_block" | "code block" | "algorithm" => return "Code Block",
        "stamp" => return "Stamp",
        _ => {}
    }
    match region.element_type {
        LayoutElementType::DocTitle | LayoutElementType::ParagraphTitle => "Title",
        LayoutElementType::Formula => "Formula",
        LayoutElementType::Table => "Table",
        LayoutElementType::Image
        | LayoutElementType::Chart
        | LayoutElementType::HeaderImage
        | LayoutElementType::FooterImage => "Picture",
        LayoutElementType::Header => "Page-header",
        LayoutElementType::Footer => "Page-footer",
        LayoutElementType::Seal => "Stamp",
        LayoutElementType::Footnote => "Footnote",
        LayoutElementType::FigureTitle
        | LayoutElementType::TableTitle
        | LayoutElementType::ChartTitle
        | LayoutElementType::FigureTableChartTitle => "Caption",
        _ => "Text",
    }
}

fn build_prompt(regions: &[LayoutElement], width: u32, height: u32) -> Result<String, Error> {
    if width == 0 || height == 0 {
        return Err(Error::invalid_input(
            "OXR reading order needs nonzero image dimensions",
        ));
    }
    let mut lines = Vec::with_capacity(regions.len());
    for (index, region) in regions.iter().enumerate() {
        let coords = [
            region.bbox.x_min(),
            region.bbox.y_min(),
            region.bbox.x_max(),
            region.bbox.y_max(),
        ];
        if coords.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid_input(
                "OXR reading-order boxes must be finite",
            ));
        }
        let scales = [
            1000.0 / f64::from(width),
            1000.0 / f64::from(height),
            1000.0 / f64::from(width),
            1000.0 / f64::from(height),
        ];
        let [x1, y1, x2, y2] =
            std::array::from_fn(|i| (f64::from(coords[i]) * scales[i]).round_ties_even() as i64);
        lines.push(format!("{index} | <|object_ref_start|>{}<|object_ref_end|> | <|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>", official_label(region)));
    }
    Ok(format!(
        "\nGiven the detected layout boxes below, output the reading order as a whitespace-separated permutation of 0-based box IDs only.\n\nBoxes:\n{}\n\nOutput only the box IDs.",
        lines.join("\n")
    ))
}

fn parse_order(text: &str, count: usize) -> Vec<usize> {
    let mut order = Vec::new();
    for value in text.split_whitespace() {
        if !value.bytes().all(|ch| ch.is_ascii_digit()) {
            return Vec::new();
        }
        let Ok(value) = value.parse::<usize>() else {
            return Vec::new();
        };
        order.push(value);
    }
    let mut sorted = order.clone();
    sorted.sort_unstable();
    if !sorted.iter().copied().eq(0..count) {
        return Vec::new();
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::geometry::BoundingBox;
    #[test]
    fn grounding_prompt_matches_official_rounding_and_vocabulary() {
        let region = LayoutElement::new(
            BoundingBox::from_coords(0.1, 0.3, 100., 200.),
            LayoutElementType::Other,
            1.,
        )
        .with_label("chemical_structure");
        let prompt = build_prompt(&[region], 200, 400).unwrap();
        assert!(prompt.contains("0 | <|object_ref_start|>Chemical Structure<|object_ref_end|> | <|box_start|>(1,1),(500,500)<|box_end|>"));
        assert!(prompt.starts_with("\nGiven the detected layout boxes below,"));
        assert_eq!(parse_order("2 0 1", 3), [2, 0, 1]);
        for invalid in [
            "0 0 1",
            "0 1",
            "0,1,2",
            "0 1 3",
            "0 1 -2",
            "9999999999999999999999999999999999999999",
        ] {
            assert!(parse_order(invalid, 3).is_empty());
        }
    }
}
