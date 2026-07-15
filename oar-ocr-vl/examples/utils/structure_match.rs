//! Source-aware matching from `StructureResult` candidates to OmniDocBench
//! target regions.
//!
//! Two-pass policy:
//!
//! 1. **Same-category pass**: only candidates whose `LayoutElementType`
//!    shares the target's `semantic_category()` are eligible, using a
//!    relaxed IoU floor (`same_category_iou`). The category pre-filter
//!    bounds poisoning risk so a lower IoU is safe.
//! 2. **Cross-category fallback**: any candidate at the strict IoU floor
//!    (`cross_category_iou`). Preserves the previous "max IoU wins" safety
//!    net for cases where the structure pipeline assigns an unexpected
//!    type to the matching region.
//!
//! Tables and formulas are pre-typed by the structure pipeline (they live
//! on `StructureResult::tables` / `::formulas`), so they always use the
//! same-category threshold. They optionally fall back to generic layout
//! text if `allow_generic_fallback` is set.
//!
//! For target types whose `semantic_category()` is `"region"` or
//! `"other"`, the same-category pass is skipped (the category carries no
//! useful signal) and we go straight to the cross-category fallback.

use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType, StructureResult};
use oar_ocr_core::processors::BoundingBox;

#[derive(Debug, Clone, Copy)]
pub struct MatchThresholds {
    pub same_category_iou: f32,
    pub cross_category_iou: f32,
    pub allow_generic_fallback: bool,
}

impl MatchThresholds {
    pub fn new(
        same_category_iou: f32,
        cross_category_iou: f32,
        allow_generic_fallback: bool,
    ) -> Self {
        Self {
            same_category_iou,
            cross_category_iou,
            allow_generic_fallback,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StructureMatch {
    pub source: &'static str,
    pub text: String,
    pub iou: f32,
    pub same_category: bool,
}

pub fn match_region(
    result: &StructureResult,
    elem: &LayoutElement,
    th: MatchThresholds,
) -> Option<StructureMatch> {
    match elem.element_type {
        LayoutElementType::Table => best_table(result, &elem.bbox, th),
        LayoutElementType::Chart => None,
        LayoutElementType::Formula => best_formula(result, &elem.bbox, th),
        LayoutElementType::Image
        | LayoutElementType::HeaderImage
        | LayoutElementType::FooterImage => None,
        other => best_layout(result, &elem.bbox, other, th),
    }
}

fn best_layout(
    result: &StructureResult,
    target: &BoundingBox,
    target_type: LayoutElementType,
    th: MatchThresholds,
) -> Option<StructureMatch> {
    let target_cat = target_type.semantic_category();
    let same_cat_useful = !matches!(target_cat, "region" | "other");

    if same_cat_useful {
        let same = result
            .layout_elements
            .iter()
            .filter_map(|c| {
                let text = c.text.as_ref()?.trim();
                if text.is_empty() {
                    return None;
                }
                if c.element_type.semantic_category() != target_cat {
                    return None;
                }
                let iou = target.iou(&c.bbox);
                (iou >= th.same_category_iou).then(|| (iou, text.to_string()))
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((iou, text)) = same {
            return Some(StructureMatch {
                source: "layout",
                text,
                iou,
                same_category: true,
            });
        }
    }

    result
        .layout_elements
        .iter()
        .filter_map(|c| {
            let text = c.text.as_ref()?.trim();
            if text.is_empty() {
                return None;
            }
            let iou = target.iou(&c.bbox);
            (iou >= th.cross_category_iou).then(|| (iou, text.to_string()))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(iou, text)| StructureMatch {
            source: "layout",
            text,
            iou,
            same_category: false,
        })
}

fn best_table(
    result: &StructureResult,
    target: &BoundingBox,
    th: MatchThresholds,
) -> Option<StructureMatch> {
    let direct = result
        .tables
        .iter()
        .filter_map(|table| {
            let html = table.html_structure.as_ref()?.trim();
            if html.is_empty() {
                return None;
            }
            let iou = target.iou(&table.bbox);
            (iou >= th.same_category_iou).then(|| (iou, html.to_string()))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(iou, text)| StructureMatch {
            source: "table",
            text,
            iou,
            same_category: true,
        });

    direct.or_else(|| {
        if !th.allow_generic_fallback {
            return None;
        }
        best_layout(result, target, LayoutElementType::Table, th)
    })
}

fn best_formula(
    result: &StructureResult,
    target: &BoundingBox,
    th: MatchThresholds,
) -> Option<StructureMatch> {
    let direct = result
        .formulas
        .iter()
        .filter_map(|formula| {
            let latex = formula.latex.trim();
            if latex.is_empty() {
                return None;
            }
            let iou = target.iou(&formula.bbox);
            (iou >= th.same_category_iou).then(|| (iou, latex.to_string()))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(iou, text)| StructureMatch {
            source: "formula",
            text,
            iou,
            same_category: true,
        });

    direct.or_else(|| {
        if !th.allow_generic_fallback {
            return None;
        }
        best_layout(result, target, LayoutElementType::Formula, th)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oar_ocr_core::domain::structure::{
        FormulaResult, LayoutElement, LayoutElementType, StructureResult, TableResult, TableType,
    };
    use oar_ocr_core::processors::BoundingBox;

    fn bb(x1: f32, y1: f32, x2: f32, y2: f32) -> BoundingBox {
        BoundingBox::from_coords(x1, y1, x2, y2)
    }

    fn cand(t: LayoutElementType, b: BoundingBox, text: &str) -> LayoutElement {
        LayoutElement::new(b, t, 1.0).with_text(text)
    }

    fn target(t: LayoutElementType, b: BoundingBox) -> LayoutElement {
        LayoutElement::new(b, t, 1.0)
    }

    fn empty_result() -> StructureResult {
        StructureResult::new("test.jpg", 0)
    }

    fn th_default() -> MatchThresholds {
        MatchThresholds::new(0.5, 0.8, false)
    }

    /// Same-category at relaxed IoU wins over cross-category that would
    /// otherwise be ineligible (below the strict floor) — exactly the
    /// poisoning case the policy is designed to avoid.
    #[test]
    fn same_category_beats_lower_iou_cross_category() {
        let mut r = empty_result();
        // Cross-category candidate at IoU below strict 0.8 (would poison
        // under a pure max-IoU policy if relaxed):
        r.layout_elements.push(cand(
            LayoutElementType::Text,
            bb(0.0, 30.0, 100.0, 100.0),
            "BODY TEXT (wrong type)",
        ));
        // Same-category candidate at IoU ~0.66 (passes 0.5 floor):
        r.layout_elements.push(cand(
            LayoutElementType::DocTitle,
            bb(10.0, 10.0, 100.0, 60.0),
            "TITLE TEXT",
        ));

        let t = target(LayoutElementType::DocTitle, bb(0.0, 0.0, 100.0, 50.0));
        let m = match_region(&r, &t, th_default()).unwrap();
        assert_eq!(m.text, "TITLE TEXT");
        assert!(m.same_category);
    }

    /// Cross-category falls through at strict 0.8 floor when no same-cat
    /// candidate is eligible.
    #[test]
    fn cross_category_only_at_strict_threshold() {
        let mut r = empty_result();
        // No same-category candidate; only a Text candidate at IoU = 1.0.
        r.layout_elements.push(cand(
            LayoutElementType::Text,
            bb(0.0, 0.0, 100.0, 50.0),
            "FALLBACK BODY",
        ));

        let t = target(LayoutElementType::DocTitle, bb(0.0, 0.0, 100.0, 50.0));
        let m = match_region(&r, &t, th_default()).unwrap();
        assert_eq!(m.text, "FALLBACK BODY");
        assert!(!m.same_category);
    }

    /// Cross-category candidate below strict threshold yields no match —
    /// don't poison with a partial overlap of the wrong type.
    #[test]
    fn cross_category_below_strict_returns_none() {
        let mut r = empty_result();
        r.layout_elements.push(cand(
            LayoutElementType::Text,
            bb(40.0, 0.0, 100.0, 50.0),
            "PARTIAL OVERLAP",
        ));
        let t = target(LayoutElementType::DocTitle, bb(0.0, 0.0, 100.0, 50.0));
        assert!(match_region(&r, &t, th_default()).is_none());
    }

    /// "region" / "other" semantic categories skip the same-cat pass and go
    /// straight to the cross-category fallback.
    #[test]
    fn region_target_skips_same_category_pass() {
        let mut r = empty_result();
        r.layout_elements.push(cand(
            LayoutElementType::Text,
            bb(0.0, 0.0, 100.0, 50.0),
            "ANY TEXT",
        ));
        let t = target(LayoutElementType::Region, bb(0.0, 0.0, 100.0, 50.0));
        let m = match_region(&r, &t, th_default()).unwrap();
        assert_eq!(m.text, "ANY TEXT");
        assert!(!m.same_category);
    }

    /// A table target matches a table candidate at a relaxed threshold (the candidate set
    /// is already type-restricted, so a low IoU floor is safe).
    #[test]
    fn table_target_uses_relaxed_threshold() {
        let mut r = empty_result();
        r.tables.push(
            TableResult::new(bb(0.0, 0.0, 100.0, 60.0), TableType::Wired)
                .with_html_structure("<table>x</table>"),
        );
        let t = target(LayoutElementType::Table, bb(0.0, 0.0, 100.0, 50.0));
        let m = match_region(&r, &t, th_default()).unwrap();
        assert_eq!(m.source, "table");
        assert!(m.same_category);
    }

    /// A formula target matches a formula candidate at a relaxed threshold.
    #[test]
    fn formula_target_uses_relaxed_threshold() {
        let mut r = empty_result();
        r.formulas.push(FormulaResult::new(
            bb(0.0, 0.0, 100.0, 60.0),
            r"\sum x",
            1.0,
        ));
        let t = target(LayoutElementType::Formula, bb(0.0, 0.0, 100.0, 50.0));
        let m = match_region(&r, &t, th_default()).unwrap();
        assert_eq!(m.source, "formula");
        assert!(m.same_category);
    }

    /// Without `allow_generic_fallback`, a missing table candidate yields
    /// None even if a generic layout candidate would have matched.
    #[test]
    fn table_no_generic_fallback_by_default() {
        let mut r = empty_result();
        r.layout_elements.push(cand(
            LayoutElementType::Table,
            bb(0.0, 0.0, 100.0, 50.0),
            "table-as-text",
        ));
        let t = target(LayoutElementType::Table, bb(0.0, 0.0, 100.0, 50.0));
        assert!(match_region(&r, &t, th_default()).is_none());
    }

    /// With `allow_generic_fallback`, the same case finds a draft via
    /// the generic layout pass.
    #[test]
    fn table_generic_fallback_when_enabled() {
        let mut r = empty_result();
        r.layout_elements.push(cand(
            LayoutElementType::Table,
            bb(0.0, 0.0, 100.0, 50.0),
            "table-as-text",
        ));
        let t = target(LayoutElementType::Table, bb(0.0, 0.0, 100.0, 50.0));
        let th = MatchThresholds::new(0.5, 0.8, true);
        let m = match_region(&r, &t, th).unwrap();
        assert_eq!(m.source, "layout");
    }

    /// Image / Chart targets are intentionally non-drafted.
    #[test]
    fn image_and_chart_targets_return_none() {
        let mut r = empty_result();
        r.layout_elements.push(cand(
            LayoutElementType::Image,
            bb(0.0, 0.0, 100.0, 50.0),
            "alt text",
        ));
        let img = target(LayoutElementType::Image, bb(0.0, 0.0, 100.0, 50.0));
        let chart = target(LayoutElementType::Chart, bb(0.0, 0.0, 100.0, 50.0));
        assert!(match_region(&r, &img, th_default()).is_none());
        assert!(match_region(&r, &chart, th_default()).is_none());
    }
}
