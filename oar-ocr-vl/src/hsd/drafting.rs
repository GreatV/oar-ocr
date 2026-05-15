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
//!
//! ### Tokenizer parity with the paper's HF Transformers stack
//!
//! Both this crate and HF Transformers ultimately call the same
//! [`tokenizers`](https://docs.rs/tokenizers) Rust crate, so there is **no
//! algorithmic divergence** between the paper's tokenization and this stack's
//! tokenization as long as the same `tokenizer.json` is loaded. All four VLM
//! HSD paths consistently use `tokenizer.encode(text, false)` (i.e.
//! `add_special_tokens = False`), which matches HF Transformers
//! `tokenizer(text, add_special_tokens=False)`.
//!
//! Any remaining byte-level AAL loss is therefore **adapter-level**, not
//! tokenizer-level — i.e. the drafter's serialized string doesn't exactly
//! match the target VLM's natural output convention. That is what
//! [`TargetDraftAdapter`] exists to address; new long-tail divergences
//! (heading prefix style, HTML attribute order, math-wrapper spacing) belong
//! in a new adapter branch with a unit test, not in tokenizer wrapping.

use image::RgbImage;
use oar_ocr_core::core::OCRError;
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType, StructureResult};
use oar_ocr_core::processors::BoundingBox;
use oar_ocr_core::utils::BBoxCrop;

use super::types::{Draft, RegionDraft, RegionKind};
use crate::utils::table::{convert_html_to_otsl, convert_otsl_to_html, looks_like_table_tokens};
use crate::utils::to_markdown;

/// Target-side text surface used to serialize drafter regions before tokenizing
/// them for DSV. Keeping this explicit prevents benchmark and backend code from
/// silently feeding one model another model's natural output convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetDraftAdapter {
    /// Generic markdown conversion used by the original structure pipeline.
    Markdown,
    /// HunyuanOCR parsing style: headings are plain text, isolated formulas are
    /// inline `$$ ... $$`, and page numbers keep the separator HunyuanOCR tends to emit.
    HunyuanOcr,
    /// Plain region text with no markdown shell. This matches the current
    /// GLM-OCR / MinerU page benchmark prompt better than HunyuanOCR markdown.
    PlainText,
    /// PaddleOCR-VL element-level raw output form (pre-postprocess):
    /// - Formula: `$$ ... $$` wrapped (post-process strips the wrapper).
    /// - Table: OTSL tokens pass through; HTML is converted to OTSL when the
    ///   table parser recognizes the structure, otherwise it passes through.
    /// - Headings / text / list / page-number: plain text, no `#` shell.
    PaddleOcrVl,
    /// GLM-OCR Recognition-prompt style: model emits LaTeX without `$$`
    /// wrappers, HTML tables pass through, headings are plain text. JSON
    /// outputs (information extraction prompts) are not handled here — they
    /// require a separate JSON-shaped draft path.
    GlmOcr,
    /// MinerU2.5 two-step-extract per-element style:
    /// - Formula: `$$\n ... \n$$` block form.
    /// - Table: HTML pass through.
    /// - Headings: plain text (no `#` — the two-step extractor emits the
    ///   layout label separately and per-element text is bare).
    /// - Page numbers: plain text without the HunyuanOCR-style `---` separator.
    MinerU,
}

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
    region_markdown_for(elem, TargetDraftAdapter::Markdown)
}

fn raw_region_text(elem: &LayoutElement) -> Option<&str> {
    elem.text
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn is_visual_only(elem: &LayoutElement) -> bool {
    matches!(
        elem.element_type,
        LayoutElementType::Image
            | LayoutElementType::HeaderImage
            | LayoutElementType::FooterImage
            | LayoutElementType::Seal
    )
}

fn align_hunyuanocr_heading(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("SEC.")
        && rest.chars().next().is_some_and(|ch| ch.is_ascii_digit())
    {
        return format!("SEC. {rest}");
    }
    text.to_string()
}

/// Normalize a table draft into the HTML form HunyuanOCR / GLM-OCR / MinerU
/// naturally emit. OTSL token streams (`<fcel> ... <nl>`) are converted via
/// `convert_otsl_to_html`; HTML pass-through is preserved; other text is
/// returned unchanged so the per-target adapter caller can decide.
fn table_text_as_html(text: &str) -> String {
    if looks_like_table_tokens(text) {
        convert_otsl_to_html(text)
    } else {
        text.to_string()
    }
}

fn hunyuanocr_region_markdown(elem: &LayoutElement) -> String {
    if is_visual_only(elem) {
        return String::new();
    }
    let Some(text) = raw_region_text(elem) else {
        return String::new();
    };
    match elem.element_type {
        LayoutElementType::Header
        | LayoutElementType::DocTitle
        | LayoutElementType::ParagraphTitle => align_hunyuanocr_heading(text),
        LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
            if text.starts_with("$$") || text.starts_with("\\[") {
                text.to_string()
            } else {
                format!("$$ {text} $$")
            }
        }
        LayoutElementType::Number => format!("{text}\n\n---"),
        // HunyuanOCR emits HTML tables; auto-convert OTSL drafts so a
        // PaddleOCR-VL drafter's OTSL output can be matched.
        LayoutElementType::Table => table_text_as_html(text),
        _ => text.to_string(),
    }
}

fn plain_region_markdown(elem: &LayoutElement) -> String {
    if is_visual_only(elem) {
        return String::new();
    }
    raw_region_text(elem).unwrap_or("").to_string()
}

fn paddleocr_vl_region_markdown(elem: &LayoutElement) -> String {
    if is_visual_only(elem) {
        return String::new();
    }
    let Some(text) = raw_region_text(elem) else {
        return String::new();
    };
    match elem.element_type {
        LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
            if text.starts_with("$$") || text.starts_with("\\[") || text.starts_with('$') {
                text.to_string()
            } else {
                format!("$${text}$$")
            }
        }
        // PaddleOCR-VL emits raw OTSL tokens for tables. If the drafter
        // supplied OTSL, pass it through. If the drafter supplied HTML,
        // attempt the inverse `convert_html_to_otsl` so the draft matches
        // what PaddleOCR-VL actually emits before its post-process step.
        // Fall back to the original text only when the input is neither
        // recognizable form.
        LayoutElementType::Table => {
            if looks_like_table_tokens(text) {
                text.to_string()
            } else if text.contains("<tr") {
                convert_html_to_otsl(text).unwrap_or_else(|| text.to_string())
            } else {
                text.to_string()
            }
        }
        // Everything else: no markdown shell, no `# ` prefix, no `---`
        // separator. PaddleOCR-VL element prompts never emit those.
        _ => text.to_string(),
    }
}

fn glmocr_region_markdown(elem: &LayoutElement) -> String {
    if is_visual_only(elem) {
        return String::new();
    }
    let Some(text) = raw_region_text(elem) else {
        return String::new();
    };
    match elem.element_type {
        LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
            // GLM-OCR's "Formula Recognition:" prompt emits bare LaTeX
            // without `$$` delimiters. Strip them if the drafter wrapped.
            strip_math_wrappers_str(text).to_string()
        }
        // GLM-OCR's "Table Recognition:" prompt emits HTML — convert OTSL
        // drafts to HTML so source-PaddleOCR-VL → target-GLM-OCR matches.
        LayoutElementType::Table => table_text_as_html(text),
        // Headings / page numbers / lists: plain text. GLM-OCR's
        // Recognition prompts never emit `#` headings or the `---` separator.
        _ => text.to_string(),
    }
}

fn mineru_region_markdown(elem: &LayoutElement) -> String {
    if is_visual_only(elem) {
        return String::new();
    }
    let Some(text) = raw_region_text(elem) else {
        return String::new();
    };
    match elem.element_type {
        LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
            // MinerU2.5 two_step_extract emits display formulas in block form
            // `$$\n ... \n$$`. Strip any existing wrapper before re-wrapping
            // so callers can pass either form.
            let core = strip_math_wrappers_str(text);
            format!("$$\n{core}\n$$")
        }
        // MinerU's per-element table prompt emits HTML; convert OTSL drafts
        // (e.g. from a PaddleOCR-VL drafter) so the byte form matches.
        LayoutElementType::Table => table_text_as_html(text),
        // Other elements stay plain — MinerU's per-element step emits
        // content without surrounding markdown shell; the layout step adds
        // heading levels separately.
        _ => text.to_string(),
    }
}

/// Strip `$$ ... $$` or `$ ... $` wrappers from a math snippet. Mirrors
/// `oar_ocr_vl::utils::text::strip_math_wrappers` but kept here to avoid a
/// crate-internal dependency cycle when running the HSD unit tests in
/// isolation.
fn strip_math_wrappers_str(input: &str) -> &str {
    let mut trimmed = input.trim();
    trimmed = trimmed
        .strip_prefix("$$")
        .and_then(|s| s.strip_suffix("$$"))
        .unwrap_or(trimmed);
    trimmed = trimmed
        .strip_prefix('$')
        .and_then(|s| s.strip_suffix('$'))
        .unwrap_or(trimmed);
    trimmed.trim()
}

/// Serialize a single layout element using the target VLM's draft adapter.
pub fn region_markdown_for(elem: &LayoutElement, adapter: TargetDraftAdapter) -> String {
    match adapter {
        TargetDraftAdapter::Markdown => {
            // Reuse the existing converter on a singleton slice; it trims
            // trailing `\n\n` so the output is suitable for direct tokenization.
            to_markdown(std::slice::from_ref(elem), &[])
        }
        TargetDraftAdapter::HunyuanOcr => hunyuanocr_region_markdown(elem),
        TargetDraftAdapter::PlainText => plain_region_markdown(elem),
        TargetDraftAdapter::PaddleOcrVl => paddleocr_vl_region_markdown(elem),
        TargetDraftAdapter::GlmOcr => glmocr_region_markdown(elem),
        TargetDraftAdapter::MinerU => mineru_region_markdown(elem),
    }
}

/// Cross-VLM un-postprocess + re-adapt for a single piece of raw region text.
///
/// DSV matches drafts against the target VLM's natural output at token
/// granularity, so when a different VLM (source) supplies the draft we must:
///
/// 1. Obtain the source's **pre-postprocess** decoded string (use the
///    source backend's `decode_tokens_raw` rather than `decode_tokens`).
/// 2. Re-shape that raw string into the target's natural surface via the
///    target's [`TargetDraftAdapter`].
///
/// Step 2 is what this helper does. Step 1 must be performed by the caller
/// using the source backend's API — the source's post-process is per-backend
/// and not encoded here.
///
/// ## Table surface handling
///
/// PaddleOCR-VL emits tables as raw OTSL tokens; HunyuanOCR / GLM-OCR / MinerU
/// emit tables as HTML. The adapters now bridge the two with
/// [`crate::utils::table::convert_otsl_to_html`] /
/// [`crate::utils::table::convert_html_to_otsl`] so cross-VLM table drafts
/// land on the target's natural surface (PaddleOCR-VL -> HTML-emitting backends
/// and vice versa). Cells with structure that neither parser recognizes still
/// fall back to pass-through.
///
/// `element_type` and `bbox` are used to drive the adapter's per-kind
/// dispatch; only `text` is actually consumed beyond that.
pub fn convert_raw_to_target_adapter(
    raw_text: &str,
    element_type: LayoutElementType,
    target_adapter: TargetDraftAdapter,
) -> String {
    let stub_bbox = BoundingBox::from_coords(0.0, 0.0, 1.0, 1.0);
    let mut elem = LayoutElement::new(stub_bbox, element_type, 1.0);
    elem.text = Some(raw_text.to_string());
    region_markdown_for(&elem, target_adapter)
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
///
/// Use [`region_markdowns`] instead when feeding Stage 2: the paper's Eq. 3
/// formulates `Ỹ^pg` as an unordered *set* of per-region drafts, not a single
/// concatenated string. Concatenating breaks the sliding-window matcher when
/// inter-region transitions don't appear in the target VLM's natural output.
pub fn page_markdown(elements: &[LayoutElement], ignore_labels: &[String]) -> String {
    to_markdown(elements, ignore_labels)
}

/// Aggregate target-adapted region drafts into a human-readable page draft.
/// DSV Stage 2 should still prefer [`region_markdowns_for`] so each region
/// remains an independent draft candidate.
pub fn page_markdown_for(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    adapter: TargetDraftAdapter,
) -> String {
    region_markdowns_for(elements, ignore_labels, adapter).join("\n\n")
}

/// One markdown draft per layout element, in input (reading) order.
///
/// This is the Stage-2 draft set when Stage 1 is disabled and the layout
/// drafter pipeline is the sole draft source (paper §3.1 Eq. 3 with the
/// pipeline outputs in place of `ŷ^(i)`). Each element's markdown becomes a
/// separate entry in `Ỹ^pg`, which the DSV matcher scans independently —
/// this preserves per-region n-gram locality even when the target VLM's
/// full-page output format differs significantly from the drafter's
/// markdown style.
///
/// Elements are skipped when their `label` matches one of `ignore_labels`,
/// when their serialized markdown is empty / whitespace-only, or when they
/// represent purely visual regions (`Image` / `HeaderImage` / `FooterImage`
/// / `Seal`) that have no recognized text.
pub fn region_markdowns(elements: &[LayoutElement], ignore_labels: &[String]) -> Vec<String> {
    region_markdowns_for(elements, ignore_labels, TargetDraftAdapter::Markdown)
}

/// One target-adapted draft per layout element, in input (reading) order.
pub fn region_markdowns_for(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    adapter: TargetDraftAdapter,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(elements.len());
    for elem in elements {
        if let Some(label) = &elem.label
            && ignore_labels.iter().any(|l| l == label)
        {
            continue;
        }
        let md = region_markdown_for(elem, adapter);
        let trimmed = md.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_string());
        }
    }
    out
}

/// Serialize multiple text candidates per layout element through a target
/// adapter. The returned Vec is flat because Stage 2 consumes an unordered
/// draft set; Stage 1 grouping is handled by
/// [`build_region_draft_candidates_with_adapter`].
pub fn region_markdown_candidates_for<C>(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    adapter: TargetDraftAdapter,
    text_candidates: C,
) -> Vec<String>
where
    C: Fn(&LayoutElement) -> Vec<String>,
{
    let mut out: Vec<String> = Vec::new();
    for elem in elements {
        if let Some(label) = &elem.label
            && ignore_labels.iter().any(|l| l == label)
        {
            continue;
        }
        for candidate in text_candidates(elem) {
            let mut candidate_elem = elem.clone();
            candidate_elem.text = Some(candidate);
            let md = region_markdown_for(&candidate_elem, adapter);
            let trimmed = md.trim();
            if trimmed.is_empty() || out.iter().any(|prev| prev == trimmed) {
                continue;
            }
            out.push(trimmed.to_string());
        }
    }
    out
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
    build_region_drafts_with_adapter(
        elements,
        ignore_labels,
        TargetDraftAdapter::Markdown,
        tokenize,
    )
}

/// Build per-region drafts using the supplied target-VLM tokenizer and text adapter.
pub fn build_region_drafts_with_adapter<F>(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    adapter: TargetDraftAdapter,
    tokenize: F,
) -> Vec<RegionDraft>
where
    F: Fn(&str) -> Vec<u32>,
{
    build_region_draft_candidates_with_adapter(
        elements,
        ignore_labels,
        adapter,
        |elem| elem.text.iter().cloned().collect(),
        tokenize,
    )
}

/// Build per-region drafts from multiple raw text candidates per layout element.
///
/// This is the Stage-1 multi-draft plumbing point: OCR top-k candidates or
/// outputs from multiple drafters can be supplied through `text_candidates`.
/// Each candidate is serialized through the target adapter, tokenized, and
/// deduplicated before being packed into one [`RegionDraft`].
pub fn build_region_draft_candidates_with_adapter<C, T>(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    adapter: TargetDraftAdapter,
    text_candidates: C,
    tokenize: T,
) -> Vec<RegionDraft>
where
    C: Fn(&LayoutElement) -> Vec<String>,
    T: Fn(&str) -> Vec<u32>,
{
    let mut out: Vec<RegionDraft> = Vec::with_capacity(elements.len());
    for elem in elements {
        if let Some(label) = &elem.label
            && ignore_labels.iter().any(|l| l == label)
        {
            continue;
        }
        let mut drafts: Vec<Draft> = Vec::new();
        for candidate in text_candidates(elem) {
            let mut candidate_elem = elem.clone();
            candidate_elem.text = Some(candidate);
            let md = region_markdown_for(&candidate_elem, adapter);
            if md.trim().is_empty() {
                continue;
            }
            let toks = tokenize(&md);
            if toks.is_empty() || drafts.iter().any(|draft| draft.tokens == toks) {
                continue;
            }
            drafts.push(Draft::new(toks));
        }
        if drafts.is_empty() {
            continue;
        }
        out.push(RegionDraft {
            bbox: bbox_xyxy(&elem.bbox),
            drafts,
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

/// Adapt an [`OARStructure`](https://docs.rs/oar-ocr)-style [`StructureResult`]
/// into the `Vec<LayoutElement>` shape every model's `generate_hsd_full`
/// expects.
///
/// The OAR structure pipeline (PP-DocLayout + PP-OCRv5 + table / formula /
/// seal predictors) populates per-element fields slightly differently than
/// the HSD drafting code expects:
///
/// - Text-bearing regions (paragraphs, titles, captions, etc.) already have
///   their recognized text in `LayoutElement.text`.
/// - **Tables** keep their HTML in a separate [`TableResult`] keyed by bbox;
///   `LayoutElement.text` on the `Table` element is usually empty.
/// - **Formulas** keep their LaTeX in a separate [`FormulaResult`] keyed by
///   bbox; `LayoutElement.text` on the `Formula` element is usually empty.
///
/// This helper backfills those two cases by IoU-matching (>0.5) the table /
/// formula side records onto their corresponding layout elements, then
/// dedups elements whose `(element_type, bbox)` IoU > 0.98 (the structure
/// pipeline occasionally double-emits a region as both layout and table /
/// formula). The result is ready to feed to
/// `HunyuanOcr::generate_hsd_full` / `GlmOcr::generate_hsd_full` /
/// `MinerU::generate_hsd_full` / `PaddleOcrVl::generate_hsd_full` without
/// further glue.
///
/// [`OARStructure`]: https://docs.rs/oar-ocr/latest/oar_ocr/oarocr/structure/struct.OARStructure.html
/// [`TableResult`]: oar_ocr_core::domain::structure::TableResult
/// [`FormulaResult`]: oar_ocr_core::domain::structure::FormulaResult
pub fn structure_result_to_layout_elements(result: &StructureResult) -> Vec<LayoutElement> {
    let mut elements = result.layout_elements.clone();
    for elem in &mut elements {
        match elem.element_type {
            LayoutElementType::Table => {
                if elem.text.as_deref().is_none_or(|s| s.trim().is_empty())
                    && let Some(html) = result
                        .tables
                        .iter()
                        .filter_map(|table| {
                            let html = table.html_structure.as_deref()?.trim();
                            (!html.is_empty()).then(|| (table.bbox.iou(&elem.bbox), html))
                        })
                        .filter(|(iou, _)| *iou > 0.5)
                        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(_, html)| html.to_string())
                {
                    elem.text = Some(html);
                }
            }
            LayoutElementType::Formula => {
                if elem.text.as_deref().is_none_or(|s| s.trim().is_empty())
                    && let Some(latex) = result
                        .formulas
                        .iter()
                        .filter_map(|formula| {
                            let latex = formula.latex.trim();
                            (!latex.is_empty()).then(|| (formula.bbox.iou(&elem.bbox), latex))
                        })
                        .filter(|(iou, _)| *iou > 0.5)
                        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(_, latex)| latex.to_string())
                {
                    elem.text = Some(latex);
                }
            }
            _ => {}
        }
    }
    let mut unique: Vec<LayoutElement> = Vec::with_capacity(elements.len());
    for elem in elements {
        let duplicate = unique
            .iter()
            .any(|prev| prev.element_type == elem.element_type && prev.bbox.iou(&elem.bbox) > 0.98);
        if !duplicate {
            unique.push(elem);
        }
    }
    unique
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
    fn region_markdowns_skips_visual_only_and_ignored_and_empty() {
        let elements = vec![
            elem(
                LayoutElementType::Text,
                "hello world",
                0.0,
                0.0,
                10.0,
                10.0,
                Some(0),
            ),
            // Visual-only — must be skipped (no text to verify).
            elem(LayoutElementType::Image, "", 10.0, 0.0, 20.0, 10.0, Some(1)),
            // Empty text — must be skipped.
            elem(
                LayoutElementType::Text,
                "   ",
                0.0,
                10.0,
                10.0,
                20.0,
                Some(2),
            ),
            elem(
                LayoutElementType::Formula,
                "x = 1",
                10.0,
                10.0,
                20.0,
                20.0,
                Some(3),
            ),
            // Ignored by label.
            {
                let mut e = elem(
                    LayoutElementType::Text,
                    "skip me",
                    0.0,
                    20.0,
                    10.0,
                    30.0,
                    Some(4),
                );
                e.label = Some("footer".to_string());
                e
            },
        ];
        let got = region_markdowns(&elements, &["footer".to_string()]);
        // Two surviving drafts: the text element and the formula element.
        // Order matches input order (reading order), not sorted by anything.
        assert_eq!(got.len(), 2);
        assert!(got[0].contains("hello world"));
        assert!(got[1].contains("x = 1"));
    }

    #[test]
    fn region_markdowns_per_region_independence() {
        // Two text regions: each becomes a separate draft. The output is NOT
        // a joined string — that's the paper Eq. 3 alignment.
        let elements = vec![
            elem(
                LayoutElementType::Text,
                "first region",
                0.0,
                0.0,
                10.0,
                10.0,
                Some(0),
            ),
            elem(
                LayoutElementType::Text,
                "second region",
                0.0,
                10.0,
                10.0,
                20.0,
                Some(1),
            ),
        ];
        let got = region_markdowns(&elements, &[]);
        assert_eq!(got.len(), 2);
        // Neither draft contains the *other* draft's content — there's no
        // pre-joining via "\n\n" inside the helper.
        assert!(!got[0].contains("second region"));
        assert!(!got[1].contains("first region"));
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
    fn hunyuanocr_adapter_uses_hunyuanocr_surface() {
        let title = elem(
            LayoutElementType::DocTitle,
            "SEC.1 Introduction",
            0.0,
            0.0,
            100.0,
            20.0,
            Some(0),
        );
        let formula = elem(
            LayoutElementType::Formula,
            "x^2 + y^2 = 1",
            0.0,
            20.0,
            100.0,
            40.0,
            Some(1),
        );
        let page_number = elem(
            LayoutElementType::Number,
            "12",
            0.0,
            40.0,
            100.0,
            60.0,
            Some(2),
        );

        assert_eq!(
            region_markdown_for(&title, TargetDraftAdapter::HunyuanOcr),
            "SEC. 1 Introduction"
        );
        assert_eq!(
            region_markdown_for(&formula, TargetDraftAdapter::HunyuanOcr),
            "$$ x^2 + y^2 = 1 $$"
        );
        assert_eq!(
            region_markdown_for(&page_number, TargetDraftAdapter::HunyuanOcr),
            "12\n\n---"
        );
    }

    #[test]
    fn paddleocr_vl_adapter_uses_raw_form() {
        let title = elem(
            LayoutElementType::DocTitle,
            "A heading",
            0.0,
            0.0,
            100.0,
            20.0,
            Some(0),
        );
        let formula = elem(
            LayoutElementType::Formula,
            "x = 1",
            0.0,
            20.0,
            100.0,
            40.0,
            Some(1),
        );
        let formula_wrapped = elem(
            LayoutElementType::Formula,
            "$$y = 2$$",
            0.0,
            40.0,
            100.0,
            60.0,
            Some(2),
        );
        let table_otsl = elem(
            LayoutElementType::Table,
            "<fcel>a<fcel>b<nl>",
            0.0,
            60.0,
            100.0,
            80.0,
            Some(3),
        );
        let page_number = elem(
            LayoutElementType::Number,
            "12",
            0.0,
            80.0,
            100.0,
            100.0,
            Some(4),
        );

        // Headings stay plain — PaddleOCR-VL has no `# ` shell.
        assert_eq!(
            region_markdown_for(&title, TargetDraftAdapter::PaddleOcrVl),
            "A heading"
        );
        // Bare LaTeX gets wrapped into `$$..$$` raw form (post-process strips it).
        assert_eq!(
            region_markdown_for(&formula, TargetDraftAdapter::PaddleOcrVl),
            "$$x = 1$$"
        );
        // Already wrapped formulas pass through unchanged.
        assert_eq!(
            region_markdown_for(&formula_wrapped, TargetDraftAdapter::PaddleOcrVl),
            "$$y = 2$$"
        );
        // OTSL passes through as-is (PaddleOCR-VL emits OTSL natively).
        assert_eq!(
            region_markdown_for(&table_otsl, TargetDraftAdapter::PaddleOcrVl),
            "<fcel>a<fcel>b<nl>"
        );
        // Page numbers stay plain — no HunyuanOCR-style `---` separator.
        assert_eq!(
            region_markdown_for(&page_number, TargetDraftAdapter::PaddleOcrVl),
            "12"
        );
    }

    #[test]
    fn glmocr_adapter_strips_formula_wrappers() {
        let title = elem(
            LayoutElementType::DocTitle,
            "A heading",
            0.0,
            0.0,
            100.0,
            20.0,
            Some(0),
        );
        let formula_wrapped = elem(
            LayoutElementType::Formula,
            "$$x = 1$$",
            0.0,
            20.0,
            100.0,
            40.0,
            Some(1),
        );
        let formula_bare = elem(
            LayoutElementType::Formula,
            "y = 2",
            0.0,
            40.0,
            100.0,
            60.0,
            Some(2),
        );
        let table = elem(
            LayoutElementType::Table,
            "<table><tr><td>a</td></tr></table>",
            0.0,
            60.0,
            100.0,
            80.0,
            Some(3),
        );

        assert_eq!(
            region_markdown_for(&title, TargetDraftAdapter::GlmOcr),
            "A heading"
        );
        // `$$..$$` stripped to bare LaTeX (GLM-OCR emits unwrapped LaTeX).
        assert_eq!(
            region_markdown_for(&formula_wrapped, TargetDraftAdapter::GlmOcr),
            "x = 1"
        );
        // Already bare LaTeX passes through.
        assert_eq!(
            region_markdown_for(&formula_bare, TargetDraftAdapter::GlmOcr),
            "y = 2"
        );
        // Tables pass through as-is.
        assert_eq!(
            region_markdown_for(&table, TargetDraftAdapter::GlmOcr),
            "<table><tr><td>a</td></tr></table>"
        );
    }

    #[test]
    fn convert_raw_to_target_adapter_pipes_paddleocr_vl_raw_into_hunyuanocr() {
        // Scenario: PaddleOCR-VL's `decode_tokens_raw` returns `$$x = 1$$` for
        // a formula region. Feeding that as a draft for HunyuanOCR should pass
        // through unchanged (HunyuanOCR also emits `$$ ... $$`).
        let out = convert_raw_to_target_adapter(
            "$$x = 1$$",
            LayoutElementType::Formula,
            TargetDraftAdapter::HunyuanOcr,
        );
        assert_eq!(out, "$$x = 1$$");
    }

    #[test]
    fn convert_raw_to_target_adapter_strips_wrapper_for_glmocr() {
        // PaddleOCR-VL raw `$$x = 1$$` → GLM-OCR adapter strips wrapper to
        // bare LaTeX since GLM-OCR's Formula Recognition prompt emits bare.
        let out = convert_raw_to_target_adapter(
            "$$x = 1$$",
            LayoutElementType::Formula,
            TargetDraftAdapter::GlmOcr,
        );
        assert_eq!(out, "x = 1");
    }

    #[test]
    fn convert_raw_to_target_adapter_rewraps_for_mineru() {
        // PaddleOCR-VL raw `$$x = 1$$` → MinerU adapter rewraps to block form.
        let out = convert_raw_to_target_adapter(
            "$$x = 1$$",
            LayoutElementType::Formula,
            TargetDraftAdapter::MinerU,
        );
        assert_eq!(out, "$$\nx = 1\n$$");
    }

    #[test]
    fn convert_raw_to_target_adapter_table_passthrough_for_non_paddleocr_vl() {
        // HunyuanOCR / GLM-OCR / MinerU adapters keep HTML tables as-is — these
        // targets emit HTML natively.
        let html = "<table><tr><td>a</td></tr></table>";
        assert_eq!(
            convert_raw_to_target_adapter(
                html,
                LayoutElementType::Table,
                TargetDraftAdapter::HunyuanOcr,
            ),
            html,
        );
        assert_eq!(
            convert_raw_to_target_adapter(
                html,
                LayoutElementType::Table,
                TargetDraftAdapter::MinerU,
            ),
            html,
        );
    }

    #[test]
    fn paddleocr_vl_adapter_converts_html_table_to_otsl() {
        // PaddleOCR-VL emits OTSL natively; feeding an HTML table draft now
        // gets converted to OTSL via convert_html_to_otsl, closing the major
        // byte-mismatch source that previously tanked AAL on table regions.
        let html = "<table><tr><td>a</td><td>b</td></tr></table>";
        assert_eq!(
            convert_raw_to_target_adapter(
                html,
                LayoutElementType::Table,
                TargetDraftAdapter::PaddleOcrVl,
            ),
            "<fcel>a<fcel>b<nl>",
        );
    }

    #[test]
    fn hunyuanocr_adapter_converts_otsl_to_html() {
        // PaddleOCR-VL → HunyuanOCR: source emits OTSL, target wants HTML.
        // The adapter must run convert_otsl_to_html so the table draft
        // matches HunyuanOCR's natural output form.
        let out = convert_raw_to_target_adapter(
            "<fcel>a<fcel>b<nl>",
            LayoutElementType::Table,
            TargetDraftAdapter::HunyuanOcr,
        );
        assert!(out.contains("<table>"), "expected HTML, got: {out:?}");
        assert!(out.contains("<td>a</td>"));
        assert!(out.contains("<td>b</td>"));
    }

    #[test]
    fn glmocr_adapter_converts_otsl_to_html() {
        let out = convert_raw_to_target_adapter(
            "<fcel>a<fcel>b<nl>",
            LayoutElementType::Table,
            TargetDraftAdapter::GlmOcr,
        );
        assert!(out.contains("<table>"));
        assert!(out.contains("<td>a</td>"));
    }

    #[test]
    fn mineru_adapter_converts_otsl_to_html() {
        let out = convert_raw_to_target_adapter(
            "<fcel>a<fcel>b<nl>",
            LayoutElementType::Table,
            TargetDraftAdapter::MinerU,
        );
        assert!(out.contains("<table>"));
        assert!(out.contains("<td>a</td>"));
    }

    #[test]
    fn html_target_adapters_passthrough_existing_html() {
        // Already-HTML drafts pass through untouched for HTML targets.
        let html = "<table><tr><td>x</td></tr></table>";
        for adapter in [
            TargetDraftAdapter::HunyuanOcr,
            TargetDraftAdapter::GlmOcr,
            TargetDraftAdapter::MinerU,
        ] {
            assert_eq!(
                convert_raw_to_target_adapter(html, LayoutElementType::Table, adapter),
                html,
            );
        }
    }

    #[test]
    fn paddleocr_vl_adapter_passes_through_otsl_unchanged() {
        // Already-OTSL drafts pass through (no double conversion).
        let otsl = "<fcel>a<nl>";
        assert_eq!(
            convert_raw_to_target_adapter(
                otsl,
                LayoutElementType::Table,
                TargetDraftAdapter::PaddleOcrVl,
            ),
            otsl,
        );
    }

    #[test]
    fn mineru_adapter_uses_block_formula_and_plain_headings() {
        let title = elem(
            LayoutElementType::DocTitle,
            "A heading",
            0.0,
            0.0,
            100.0,
            20.0,
            Some(0),
        );
        let formula = elem(
            LayoutElementType::Formula,
            "x = 1",
            0.0,
            20.0,
            100.0,
            40.0,
            Some(1),
        );
        let formula_wrapped = elem(
            LayoutElementType::Formula,
            "$$y = 2$$",
            0.0,
            40.0,
            100.0,
            60.0,
            Some(2),
        );
        let page_number = elem(
            LayoutElementType::Number,
            "12",
            0.0,
            60.0,
            100.0,
            80.0,
            Some(3),
        );

        // Headings plain — MinerU's per-element step emits text only.
        assert_eq!(
            region_markdown_for(&title, TargetDraftAdapter::MinerU),
            "A heading"
        );
        // Block-form display formula `$$\n...\n$$`.
        assert_eq!(
            region_markdown_for(&formula, TargetDraftAdapter::MinerU),
            "$$\nx = 1\n$$"
        );
        // Re-wraps already-wrapped formulas to canonical block form.
        assert_eq!(
            region_markdown_for(&formula_wrapped, TargetDraftAdapter::MinerU),
            "$$\ny = 2\n$$"
        );
        // No `---` separator for page numbers.
        assert_eq!(
            region_markdown_for(&page_number, TargetDraftAdapter::MinerU),
            "12"
        );
    }

    #[test]
    fn all_vlm_adapters_skip_visual_only_elements_regardless_of_text() {
        // Image / HeaderImage / FooterImage / Seal carry no text the verifier
        // can match against the target VLM's natural output. Every per-VLM
        // adapter must return the empty string for these element types even
        // when the drafter populated `text` (e.g. a caption transcribed under
        // the image). Otherwise a stray draft like "*Figure 1: ...*" leaks
        // into the matcher and tanks AAL.
        let visual_types = [
            LayoutElementType::Image,
            LayoutElementType::HeaderImage,
            LayoutElementType::FooterImage,
            LayoutElementType::Seal,
        ];
        let vlm_adapters = [
            TargetDraftAdapter::HunyuanOcr,
            TargetDraftAdapter::GlmOcr,
            TargetDraftAdapter::MinerU,
            TargetDraftAdapter::PaddleOcrVl,
        ];
        for ty in visual_types {
            let e = elem(ty, "nonempty caption", 0.0, 0.0, 100.0, 50.0, Some(0));
            for adapter in vlm_adapters {
                let out = region_markdown_for(&e, adapter);
                assert!(
                    out.is_empty(),
                    "{:?} adapter must skip visual-only {:?}, got: {:?}",
                    adapter,
                    ty,
                    out
                );
            }
        }
    }

    #[test]
    fn plain_adapter_does_not_add_markdown_shell() {
        let title = elem(
            LayoutElementType::DocTitle,
            "A heading",
            0.0,
            0.0,
            100.0,
            20.0,
            Some(0),
        );
        let formula = elem(
            LayoutElementType::Formula,
            "x = 1",
            0.0,
            20.0,
            100.0,
            40.0,
            Some(1),
        );

        assert_eq!(
            region_markdown_for(&title, TargetDraftAdapter::PlainText),
            "A heading"
        );
        assert_eq!(
            region_markdown_for(&formula, TargetDraftAdapter::PlainText),
            "x = 1"
        );
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
        assert_eq!(drafts[0].drafts.len(), 1);
        assert_eq!(drafts[1].drafts.len(), 1);
        assert!(!drafts[0].drafts[0].is_empty());
        assert!(!drafts[1].drafts[0].is_empty());
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
        assert_eq!(drafts[0].drafts.len(), 1);
    }

    #[test]
    fn build_region_draft_candidates_packs_and_dedups_multi_drafts() {
        let e = elem(
            LayoutElementType::Text,
            "unused",
            1.0,
            2.0,
            3.0,
            4.0,
            Some(7),
        );
        let drafts = build_region_draft_candidates_with_adapter(
            &[e],
            &[],
            TargetDraftAdapter::PlainText,
            |_| vec!["alpha".to_string(), "alpha".to_string(), "beta".to_string()],
            byte_tok,
        );
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].drafts.len(), 2);
        assert_eq!(
            drafts[0].drafts[0].tokens,
            b"alpha".iter().map(|&b| b as u32).collect::<Vec<_>>()
        );
        assert_eq!(
            drafts[0].drafts[1].tokens,
            b"beta".iter().map(|&b| b as u32).collect::<Vec<_>>()
        );
        assert_eq!(drafts[0].kind, RegionKind::Text);
        assert_eq!(drafts[0].reading_order, Some(7));
    }

    #[test]
    fn region_markdown_candidates_serializes_and_dedups_multi_drafts() {
        let e = elem(
            LayoutElementType::Formula,
            "unused",
            1.0,
            2.0,
            3.0,
            4.0,
            Some(7),
        );
        let drafts =
            region_markdown_candidates_for(&[e], &[], TargetDraftAdapter::PlainText, |_| {
                vec!["x+y".to_string(), "x+y".to_string(), "a=b".to_string()]
            });
        assert_eq!(drafts, vec!["x+y".to_string(), "a=b".to_string()]);
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
