//! Bridge between the layout drafter pipeline and HSD's region/page drafts.
//!
//! The HSD algorithm itself ([`super::matching`], [`super::prefix_tree`],
//! [`super::verify`]) is intentionally tokenizer-agnostic; this module is the
//! one place that knows about [`LayoutElement`] and accepts an injected
//! tokenizer closure. Backends call into here to turn the drafter's
//! recognition output into the [`RegionDraft`] / [`Draft`] values consumed by
//! [`super::verify::spec_decode`].
//!
//! ## Tokenizer requirement
//!
//! The closure passed to [`build_region_drafts`] / [`build_page_draft`] /
//! [`page_draft_from_region_outputs`] **must be the target VLM's tokenizer**.
//! HSD matches drafts against the verifier's accepted-token tail at token
//! granularity; using a different tokenizer (even one that's "close enough")
//! will quietly destroy the acceptance length.

use image::RgbImage;
use oar_ocr_core::core::OCRError;
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType};
use oar_ocr_core::processors::BoundingBox;
use oar_ocr_core::utils::BBoxCrop;

use super::types::{Draft, RegionDraft, RegionKind};
use crate::utils::to_markdown;

/// Map a layout element type to the coarse HSD region kind.
pub fn map_layout_kind(t: LayoutElementType) -> RegionKind {
    use LayoutElementType::*;
    match t {
        DocTitle
        | ParagraphTitle
        | FigureTitle
        | TableTitle
        | ChartTitle
        | FigureTableChartTitle => RegionKind::Title,
        Text | Content | Abstract | AsideText | Reference | ReferenceContent | Footnote
        | Number => RegionKind::Text,
        List => RegionKind::List,
        Table => RegionKind::Table,
        Formula | FormulaNumber => RegionKind::Formula,
        Image | Chart | Seal | HeaderImage | FooterImage => RegionKind::Figure,
        Header => RegionKind::Header,
        Footer => RegionKind::Footer,
        Algorithm | Region | Other => RegionKind::Other,
    }
}

/// Extract axis-aligned `[x_min, y_min, x_max, y_max]` from a bounding box.
pub fn bbox_xyxy(bbox: &BoundingBox) -> [f32; 4] {
    [bbox.x_min(), bbox.y_min(), bbox.x_max(), bbox.y_max()]
}

/// Crop an image to an HSD `[x_min, y_min, x_max, y_max]` bounding box.
pub fn crop_region_image(image: &RgbImage, bbox: &[f32; 4]) -> Result<RgbImage, OCRError> {
    let bb = BoundingBox::from_coords(bbox[0], bbox[1], bbox[2], bbox[3]);
    BBoxCrop::crop_bounding_box(image, &bb)
}

/// Serialize a single layout element to the markdown the target VLM would
/// emit for that region in isolation. Falls back to plain text on unknown
/// element types.
pub fn region_markdown(elem: &LayoutElement) -> String {
    // Reuse the existing converter on a singleton slice; it trims trailing
    // `\n\n` so the output is suitable for direct tokenization.
    to_markdown(std::slice::from_ref(elem), &[])
}

/// Wrap verified region text in the coarse markdown shell used when Stage-1
/// outputs are reassembled into a Stage-2 page draft.
pub fn format_verified_region(text: &str, kind: RegionKind) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    match kind {
        RegionKind::Title => format!("# {trimmed}"),
        RegionKind::Formula => {
            if trimmed.starts_with("$$") {
                trimmed.to_string()
            } else {
                format!("$$\n{trimmed}\n$$")
            }
        }
        _ => trimmed.to_string(),
    }
}

/// Aggregate elements into the full-page markdown draft (reading order
/// already applied by the drafter pipeline).
pub fn page_markdown(elements: &[LayoutElement], ignore_labels: &[String]) -> String {
    to_markdown(elements, ignore_labels)
}

/// Build per-region drafts using the supplied target-VLM tokenizer.
///
/// Regions are skipped when:
/// - their `label` matches one of `ignore_labels`,
/// - their text is empty / whitespace-only after markdown serialization, or
/// - the tokenizer yields zero tokens.
pub fn build_region_drafts<F>(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    tokenize: F,
) -> Vec<RegionDraft>
where
    F: Fn(&str) -> Vec<u32>,
{
    let mut out: Vec<RegionDraft> = Vec::with_capacity(elements.len());
    for elem in elements {
        if let Some(label) = &elem.label
            && ignore_labels.iter().any(|l| l == label)
        {
            continue;
        }
        let md = region_markdown(elem);
        if md.trim().is_empty() {
            continue;
        }
        let toks = tokenize(&md);
        if toks.is_empty() {
            continue;
        }
        out.push(RegionDraft {
            bbox: bbox_xyxy(&elem.bbox),
            draft: Draft::new(toks),
            reading_order: elem.order_index.map(|x| x as usize),
            kind: map_layout_kind(elem.element_type),
        });
    }
    out
}

/// Build the Stage-2 page-level draft by tokenizing the full-page markdown
/// (as derived by [`page_markdown`]).
pub fn build_page_draft<F>(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    tokenize: F,
) -> Draft
where
    F: Fn(&str) -> Vec<u32>,
{
    let md = page_markdown(elements, ignore_labels);
    Draft::new(tokenize(&md))
}

/// Build the Stage-2 page-level draft by joining already-verified Stage-1
/// region outputs in reading order. Regions are separated with a blank line so
/// they look like distinct paragraphs / blocks to the target VLM.
pub fn page_draft_from_region_outputs<F>(region_outputs_in_order: &[String], tokenize: F) -> Draft
where
    F: Fn(&str) -> Vec<u32>,
{
    let mut joined = String::new();
    for m in region_outputs_in_order
        .iter()
        .map(|m| m.trim())
        .filter(|m| !m.is_empty())
    {
        if !joined.is_empty() {
            joined.push_str("\n\n");
        }
        joined.push_str(m);
    }
    Draft::new(tokenize(&joined))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oar_ocr_core::processors::Point;

    /// Deterministic byte-tokenizer for tests: each UTF-8 byte → one token id.
    fn byte_tok(s: &str) -> Vec<u32> {
        s.bytes().map(|b| b as u32).collect()
    }

    fn elem(
        ty: LayoutElementType,
        text: &str,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        order: Option<u32>,
    ) -> LayoutElement {
        let bbox = BoundingBox::new(vec![
            Point::new(x1, y1),
            Point::new(x2, y1),
            Point::new(x2, y2),
            Point::new(x1, y2),
        ]);
        let mut e = LayoutElement::new(bbox, ty, 0.99);
        e.text = Some(text.to_string());
        e.order_index = order;
        e
    }

    #[test]
    fn map_layout_kind_covers_main_types() {
        assert_eq!(
            map_layout_kind(LayoutElementType::DocTitle),
            RegionKind::Title
        );
        assert_eq!(map_layout_kind(LayoutElementType::Text), RegionKind::Text);
        assert_eq!(map_layout_kind(LayoutElementType::Table), RegionKind::Table);
        assert_eq!(
            map_layout_kind(LayoutElementType::Formula),
            RegionKind::Formula
        );
        assert_eq!(
            map_layout_kind(LayoutElementType::Image),
            RegionKind::Figure
        );
        assert_eq!(
            map_layout_kind(LayoutElementType::Header),
            RegionKind::Header
        );
        assert_eq!(map_layout_kind(LayoutElementType::List), RegionKind::List);
    }

    #[test]
    fn bbox_xyxy_returns_axis_aligned() {
        let b = BoundingBox::new(vec![
            Point::new(10.0, 20.0),
            Point::new(110.0, 25.0),
            Point::new(115.0, 80.0),
            Point::new(15.0, 75.0),
        ]);
        let xy = bbox_xyxy(&b);
        assert!((xy[0] - 10.0).abs() < 1e-3);
        assert!((xy[1] - 20.0).abs() < 1e-3);
        assert!((xy[2] - 115.0).abs() < 1e-3);
        assert!((xy[3] - 80.0).abs() < 1e-3);
    }

    #[test]
    fn region_markdown_title_gets_heading_prefix() {
        let e = elem(
            LayoutElementType::DocTitle,
            "Hello world",
            0.0,
            0.0,
            100.0,
            20.0,
            Some(1),
        );
        let md = region_markdown(&e);
        assert!(md.starts_with("# "), "expected H1 prefix, got: {md:?}");
        assert!(md.contains("Hello world"));
    }

    #[test]
    fn region_markdown_formula_wrapped_in_dollars() {
        let e = elem(
            LayoutElementType::Formula,
            "x^2 + y^2 = 1",
            0.0,
            0.0,
            100.0,
            20.0,
            None,
        );
        let md = region_markdown(&e);
        assert!(md.contains("$$"), "expected $$ wrapping, got: {md:?}");
    }

    #[test]
    fn region_markdown_text_passes_through() {
        let e = elem(
            LayoutElementType::Text,
            "plain paragraph",
            0.0,
            0.0,
            100.0,
            20.0,
            None,
        );
        let md = region_markdown(&e);
        assert!(md.contains("plain paragraph"));
    }

    #[test]
    fn build_region_drafts_skips_empty_and_filters_labels() {
        let mut e1 = elem(
            LayoutElementType::Text,
            "first",
            0.0,
            0.0,
            10.0,
            10.0,
            Some(1),
        );
        let mut e2 = elem(
            LayoutElementType::Text,
            "   ",
            0.0,
            0.0,
            10.0,
            10.0,
            Some(2),
        );
        let mut e3 = elem(
            LayoutElementType::Header,
            "skip-me",
            0.0,
            0.0,
            10.0,
            10.0,
            Some(3),
        );
        let mut e4 = elem(
            LayoutElementType::Text,
            "second",
            0.0,
            0.0,
            10.0,
            10.0,
            Some(4),
        );
        e1.label = Some("Text".into());
        e2.label = Some("Text".into());
        e3.label = Some("Header".into());
        e4.label = Some("Text".into());

        let drafts = build_region_drafts(&[e1, e2, e3, e4], &["Header".to_string()], byte_tok);
        assert_eq!(drafts.len(), 2);
        assert_eq!(drafts[0].kind, RegionKind::Text);
        assert_eq!(drafts[0].reading_order, Some(1));
        assert_eq!(drafts[1].reading_order, Some(4));
        // Tokens are non-empty.
        assert!(!drafts[0].draft.is_empty());
        assert!(!drafts[1].draft.is_empty());
    }

    #[test]
    fn build_region_drafts_keeps_kind_and_bbox() {
        let e = elem(
            LayoutElementType::Table,
            "<table><tr><td>a</td></tr></table>",
            1.0,
            2.0,
            3.0,
            4.0,
            Some(7),
        );
        let drafts = build_region_drafts(&[e], &[], byte_tok);
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].kind, RegionKind::Table);
        assert_eq!(drafts[0].reading_order, Some(7));
        assert_eq!(drafts[0].bbox, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn build_page_draft_concatenates() {
        let e1 = elem(
            LayoutElementType::DocTitle,
            "T",
            0.0,
            0.0,
            10.0,
            10.0,
            Some(1),
        );
        let e2 = elem(
            LayoutElementType::Text,
            "body",
            0.0,
            0.0,
            10.0,
            10.0,
            Some(2),
        );
        let pd = build_page_draft(&[e1, e2], &[], byte_tok);
        assert!(!pd.is_empty());
        // Page draft should be at least as long as title alone (sanity).
        let title_only = byte_tok("# T");
        assert!(pd.tokens.len() >= title_only.len());
    }

    #[test]
    fn page_draft_from_region_outputs_uses_blank_line_separator() {
        let outputs = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let d = page_draft_from_region_outputs(&outputs, byte_tok);
        let s = String::from_utf8(d.tokens.iter().map(|&t| t as u8).collect()).unwrap();
        assert_eq!(s, "alpha\n\nbeta\n\ngamma");
    }

    #[test]
    fn page_draft_from_region_outputs_handles_empty_input() {
        let d = page_draft_from_region_outputs(&[], byte_tok);
        assert!(d.is_empty());
    }

    #[test]
    fn page_draft_from_region_outputs_trims_per_region_padding() {
        let outputs = vec!["  spaced  ".to_string(), "next  ".to_string()];
        let d = page_draft_from_region_outputs(&outputs, byte_tok);
        let s = String::from_utf8(d.tokens.iter().map(|&t| t as u8).collect()).unwrap();
        assert_eq!(s, "spaced\n\nnext");
    }

    #[test]
    fn page_draft_from_region_outputs_skips_empty_regions() {
        let outputs = vec![
            "alpha".to_string(),
            "   ".to_string(),
            String::new(),
            "beta".to_string(),
        ];
        let d = page_draft_from_region_outputs(&outputs, byte_tok);
        let s = String::from_utf8(d.tokens.iter().map(|&t| t as u8).collect()).unwrap();
        assert_eq!(s, "alpha\n\nbeta");
    }
}
