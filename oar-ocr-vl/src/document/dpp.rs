//! Document Parsing Protocol (DPP) export, pinned to working draft `a9c49e2`.
//!
//! One pass produces the Markdown, the region sidecar and a findings report. A table caption
//! is rendered inside its table and also listed as its own Caption item.

use super::page::{DocumentBlock, DocumentOutput, PageDocument, canonical_alias};
use super::structure::LayoutElementType;
use crate::api::recognition::FormulaNumberComposition;
use crate::render::markdown::assembly::formula_number_pair;
use crate::render::table::{
    HtmlInput, NoTokens, OtslOptions, RowPadding, TABLE_FIGURE_LABEL, TableWrapper,
    convert_otsl_to_html_with_options, markdown_table_to_html, table_figure_image, table_span,
};
use crate::render::text::{
    BracketMatching, DelimiterOptions, FormulaOptions, FormulaRepair, MathStripping,
    format_formula_with_options, map_html_equations, normalize_math_delimiters,
    strip_math_wrappers,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

const PROTOCOL_REVISION: &str = "a9c49e24f6db653500cf7eab4b08019f23b74ea6";

/// Requested box resolution. Region (`none`) boxes are emitted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DppGranularity {
    /// One box per element, without finer boxes.
    #[default]
    None,
    /// Table-cell boxes.
    Cell,
    /// Line boxes.
    Line,
    /// Word boxes.
    Word,
}

/// Units of the page dimensions and exported boxes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum DppCoordinateSpace {
    /// Pixels of the input image.
    #[default]
    InputImagePixels,
    /// PDF points.
    PdfPoints,
}

/// DPP's seventeen semantic element classes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DppClass {
    /// A coherent prose block.
    #[default]
    Text,
    /// The document title on its cover/title page.
    Title,
    /// A section heading.
    #[serde(rename = "Section-header")]
    SectionHeader,
    /// One list item, including continuation lines.
    #[serde(rename = "List-item")]
    ListItem,
    /// A caption associated with an object.
    Caption,
    /// A footnote or source note.
    Footnote,
    /// Content in the page's header band.
    #[serde(rename = "Page-header")]
    PageHeader,
    /// Content in the page's footer band.
    #[serde(rename = "Page-footer")]
    PageFooter,
    /// A picture, diagram, chart or meaningful mark.
    Picture,
    /// A relational grid rendered as HTML.
    Table,
    /// A displayed expression including its printed equation number.
    Formula,
    /// A selected checkbox mark, excluding its label.
    #[serde(rename = "Checkbox-Selected")]
    CheckboxSelected,
    /// An unselected checkbox mark, excluding its label.
    #[serde(rename = "Checkbox-Unselected")]
    CheckboxUnselected,
    /// Source code or other preformatted content.
    Code,
    /// A table of contents, index or directory.
    #[serde(rename = "Document Index")]
    DocumentIndex,
    /// A form's irregular fields and controls.
    Form,
    /// A label/value region without a regular grid.
    #[serde(rename = "Key-Value Region")]
    KeyValueRegion,
}

/// Export options; region boxes and input-image pixels are the defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppOptions {
    /// Requested box resolution.
    pub granularity: DppGranularity,
    /// Units of the supplied dimensions and exported boxes.
    pub coordinate_space: DppCoordinateSpace,
}

impl DppOptions {
    /// Set the requested box resolution.
    pub fn with_granularity(mut self, value: DppGranularity) -> Self {
        self.granularity = value;
        self
    }

    /// Set the units of the supplied dimensions.
    pub fn with_coordinate_space(mut self, value: DppCoordinateSpace) -> Self {
        self.coordinate_space = value;
        self
    }
}

/// Page dimensions and annotations supplied by the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppPageContext {
    /// One-based page number for a page export; document export numbers pages by position.
    pub page: usize,
    /// Page width in the selected coordinate space, or unknown.
    pub width: Option<f64>,
    /// Page height in the selected coordinate space, or unknown.
    pub height: Option<f64>,
    /// Marks the page as blank.
    pub blank: bool,
    /// Whether this is the document's cover/title page, when known.
    pub is_cover: Option<bool>,
    /// Overrides indexed by the original page's block indices.
    pub blocks: Vec<DppBlockOptions>,
}

impl Default for DppPageContext {
    fn default() -> Self {
        Self {
            page: 1,
            width: None,
            height: None,
            blank: false,
            is_cover: None,
            blocks: Vec::new(),
        }
    }
}

impl DppPageContext {
    /// Set the standalone page number.
    pub fn with_page(mut self, value: usize) -> Self {
        self.page = value;
        self
    }

    /// Set width and height in the coordinate space selected by the export options.
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Declare whether the page is confirmed blank.
    pub fn with_blank(mut self, value: bool) -> Self {
        self.blank = value;
        self
    }

    /// Supply or clear the cover/title-page annotation.
    pub fn with_is_cover(mut self, value: Option<bool>) -> Self {
        self.is_cover = value;
        self
    }

    /// Replace the block overrides.
    pub fn with_blocks(mut self, value: Vec<DppBlockOptions>) -> Self {
        self.blocks = value;
        self
    }
}

/// Annotations for one parsed block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppBlockOptions {
    /// Zero-based index in the original `PageDocument::blocks` vector.
    pub block_index: usize,
    /// Semantic class override.
    pub class: Option<DppClass>,
    /// Printed heading's document depth, from 1 to 6.
    pub heading_level: Option<u8>,
    /// Original index of the Table/chart owning this caption.
    pub caption_of: Option<usize>,
    /// Whether this table continues from the previous page, when known.
    #[serde(rename = "continuesFromPreviousPage")]
    pub continues_from_previous_page: Option<bool>,
    /// Whether this table continues onto the next page, when known.
    #[serde(rename = "continuesOnNextPage")]
    pub continues_on_next_page: Option<bool>,
}

impl DppBlockOptions {
    /// Select the original block index.
    pub fn with_block_index(mut self, value: usize) -> Self {
        self.block_index = value;
        self
    }

    /// Set or clear the class override.
    pub fn with_class(mut self, value: Option<DppClass>) -> Self {
        self.class = value;
        self
    }

    /// Set or clear the heading depth.
    pub fn with_heading_level(mut self, value: Option<u8>) -> Self {
        self.heading_level = value;
        self
    }

    /// Set or clear the caption's owner index.
    pub fn with_caption_of(mut self, value: Option<usize>) -> Self {
        self.caption_of = value;
        self
    }

    /// Supply or clear continuation from the previous page.
    pub fn with_continues_from_previous_page(mut self, value: Option<bool>) -> Self {
        self.continues_from_previous_page = value;
        self
    }

    /// Supply or clear continuation onto the next page.
    pub fn with_continues_on_next_page(mut self, value: Option<bool>) -> Self {
        self.continues_on_next_page = value;
        self
    }
}

/// Axis-aligned top-left box in the caller's declared page coordinate space.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppBox {
    /// Left coordinate.
    pub x: f64,
    /// Top coordinate.
    pub y: f64,
    /// Width.
    pub w: f64,
    /// Height.
    pub h: f64,
}

/// One region sidecar item.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppItem {
    /// Semantic role, serialized as `type`.
    #[serde(rename = "type")]
    pub item_type: DppClass,
    /// Rendered Markdown of the element.
    pub md: String,
    /// Region box; omitted when the page dimensions or the block box are unavailable.
    #[serde(rename = "bBox", skip_serializing_if = "Option::is_none")]
    pub b_box: Option<DppBox>,
    /// Table continuation flag; omitted for other classes.
    #[serde(
        rename = "continuesFromPreviousPage",
        skip_serializing_if = "Option::is_none"
    )]
    pub continues_from_previous_page: Option<bool>,
    /// Table continuation flag; omitted for other classes.
    #[serde(
        rename = "continuesOnNextPage",
        skip_serializing_if = "Option::is_none"
    )]
    pub continues_on_next_page: Option<bool>,
}

/// A DPP page object; document exports serialize a vector of these objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppPage {
    /// One-based page number.
    pub page: usize,
    /// Page width, or null when unknown.
    pub width: Option<f64>,
    /// Page height, or null when unknown.
    pub height: Option<f64>,
    /// Regions in exported reading order.
    pub items: Vec<DppItem>,
}

impl Default for DppPage {
    fn default() -> Self {
        Self {
            page: 1,
            width: None,
            height: None,
            items: Vec::new(),
        }
    }
}

/// One MUST violation, tied to a DPP rule and optionally to a parsed block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppFinding {
    /// One-based page number.
    pub page: usize,
    /// Original zero-based block index, or null for a page-level finding.
    pub block_index: Option<usize>,
    /// Stable rule identifier from the pinned draft.
    pub rule_id: String,
    /// Human-readable explanation.
    pub message: String,
}

impl Default for DppFinding {
    fn default() -> Self {
        Self {
            page: 1,
            block_index: None,
            rule_id: String::new(),
            message: String::new(),
        }
    }
}

/// Export settings and MUST violations in the serialized output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppReport {
    /// Exact upstream Git revision of the draft.
    pub protocol_revision: String,
    /// Units of the exported boxes.
    pub coordinate_space: DppCoordinateSpace,
    /// Requested box resolution.
    pub requested_granularity: DppGranularity,
    /// Emitted box resolution.
    pub emitted_granularity: DppGranularity,
    /// Findings, sorted by page and block.
    pub findings: Vec<DppFinding>,
}

impl Default for DppReport {
    fn default() -> Self {
        Self {
            protocol_revision: PROTOCOL_REVISION.into(),
            coordinate_space: DppCoordinateSpace::default(),
            requested_granularity: DppGranularity::default(),
            emitted_granularity: DppGranularity::None,
            findings: Vec::new(),
        }
    }
}

/// DPP export result. Serialize `sidecar` on its own to get DPP JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DppOutput<T> {
    /// DPP Markdown with region anchors.
    pub markdown: String,
    /// A page object for page export, or an array for document export.
    pub sidecar: T,
    /// Draft revision, units and findings.
    pub report: DppReport,
}

impl PageDocument {
    /// Export this page as DPP Markdown, a sidecar page object and findings. Box units follow
    /// `options` (input-image pixels by default).
    pub fn to_dpp(&self, context: &DppPageContext, options: &DppOptions) -> DppOutput<DppPage> {
        let mut report = new_report(options);
        let page = context.page.max(1);
        let lowered = lower_page(self, context, page, options, &mut report);
        report.finish();
        DppOutput {
            markdown: lowered.markdown,
            sidecar: lowered.page,
            report,
        }
    }
}

impl DocumentOutput {
    /// Export all pages as DPP Markdown, a sidecar array and findings. `contexts[i]` describes
    /// page `i + 1`.
    pub fn to_dpp(
        &self,
        contexts: &[DppPageContext],
        options: &DppOptions,
    ) -> DppOutput<Vec<DppPage>> {
        let mut report = new_report(options);
        let default_context = DppPageContext::default();
        let lowered: Vec<_> = self
            .pages
            .iter()
            .enumerate()
            .map(|(index, source)| {
                lower_page(
                    source,
                    contexts.get(index).unwrap_or(&default_context),
                    index + 1,
                    options,
                    &mut report,
                )
            })
            .collect();
        check_continuation(&lowered, &mut report);
        let mut markdown = lowered
            .iter()
            .map(|page| page.markdown.as_str())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");
        if self.pages.is_empty() {
            let transcript = self
                .markdown
                .as_deref()
                .or(self.raw_output.as_deref())
                .unwrap_or("");
            if !transcript.is_empty() {
                markdown = transcript.to_string();
                report.add(
                    1,
                    None,
                    "output-the-sidecar",
                    "Document has only a joint transcript; sidecar is empty",
                );
            }
        }
        report.finish();
        DppOutput {
            markdown,
            sidecar: lowered.into_iter().map(|page| page.page).collect(),
            report,
        }
    }
}

fn new_report(options: &DppOptions) -> DppReport {
    DppReport {
        coordinate_space: options.coordinate_space,
        requested_granularity: options.granularity,
        ..DppReport::default()
    }
}

impl DppReport {
    fn add(
        &mut self,
        page: usize,
        block_index: Option<usize>,
        rule: &str,
        message: impl Into<String>,
    ) {
        self.findings.push(DppFinding {
            page,
            block_index,
            rule_id: rule.into(),
            message: message.into(),
        });
    }

    fn finish(&mut self) {
        self.findings.sort_by(|a, b| {
            (a.page, a.block_index, &a.rule_id, &a.message).cmp(&(
                b.page,
                b.block_index,
                &b.rule_id,
                &b.message,
            ))
        });
        self.findings.dedup_by(|a, b| {
            a.page == b.page
                && a.block_index == b.block_index
                && a.rule_id == b.rule_id
                && a.message == b.message
        });
    }
}

struct Region {
    index: usize,
    label: String,
    item: DppItem,
    options: DppBlockOptions,
}

struct TableEdge {
    index: usize,
    from: Option<bool>,
    to: Option<bool>,
}

struct LoweredPage {
    page: DppPage,
    markdown: String,
    tables: Vec<TableEdge>,
}

fn lower_page(
    source: &PageDocument,
    context: &DppPageContext,
    page: usize,
    options: &DppOptions,
    report: &mut DppReport,
) -> LoweredPage {
    let valid_dimension = |value: Option<f64>| value.filter(|v| v.is_finite() && *v > 0.0);
    let mut output = DppPage {
        page,
        width: valid_dimension(context.width),
        height: valid_dimension(context.height),
        items: Vec::new(),
    };
    if output.width.is_none() || output.height.is_none() {
        report.add(
            page,
            None,
            "output-the-sidecar",
            "Page dimensions missing; boxes and anchors omitted",
        );
    }
    if options.granularity != DppGranularity::None {
        report.add(
            page,
            None,
            "output-boxes-and-granularity",
            "Only region boxes are available",
        );
    }
    let transcript = source
        .markdown
        .as_deref()
        .or(source.raw_output.as_deref())
        .unwrap_or("");
    if context.blank
        && source.blocks.is_empty()
        && source
            .markdown
            .as_deref()
            .is_none_or(|s| s.trim().is_empty())
        && source
            .raw_output
            .as_deref()
            .is_none_or(|s| s.trim().is_empty())
    {
        return LoweredPage {
            page: output,
            markdown: "<!-- blank page -->".into(),
            tables: Vec::new(),
        };
    }
    if source.blocks.is_empty() {
        for rule in ["output-the-sidecar", "principles-traceability"] {
            report.add(
                page,
                None,
                rule,
                "Page has no layout blocks; sidecar items are empty",
            );
        }
        separator_findings(transcript, page, None, report);
        return LoweredPage {
            page: output,
            markdown: transcript.into(),
            tables: Vec::new(),
        };
    }
    let mut overrides = vec![None; source.blocks.len()];
    for block in &context.blocks {
        if let Some(slot) = overrides.get_mut(block.block_index) {
            *slot = Some(block);
        }
    }
    let mut regions = Vec::with_capacity(source.blocks.len());
    for (index, block) in source.blocks.iter().enumerate() {
        let settings = overrides[index].cloned().unwrap_or_default();
        let label = block.block_type.to_lowercase();
        let class = settings
            .class
            .unwrap_or_else(|| classify(&label, block, context));
        let mut item = DppItem {
            item_type: class,
            b_box: region_box(block.bbox, output.width, output.height),
            ..DppItem::default()
        };
        if item.b_box.is_none() {
            for rule in ["output-the-sidecar", "principles-traceability"] {
                report.add(
                    page,
                    Some(index),
                    rule,
                    "Block box missing or invalid; bBox and anchor omitted",
                );
            }
        }
        if class == DppClass::Table {
            item.continues_from_previous_page =
                Some(settings.continues_from_previous_page.unwrap_or(false));
            item.continues_on_next_page = Some(settings.continues_on_next_page.unwrap_or(false));
        }
        let content = block.content.as_deref().unwrap_or("");
        if !content.trim().is_empty()
            || matches!(
                class,
                DppClass::CheckboxSelected | DppClass::CheckboxUnselected
            )
        {
            item.md = format_content(content, &label, class, &settings, page, index, report);
        }
        regions.push(Region {
            index,
            label,
            item,
            options: settings,
        });
    }
    let mut embedded = embed_captions(&mut regions, page, report);
    embed_figures(&mut regions, source, page, &mut embedded, report);
    embed_formula_numbers(&mut regions, source, &output, &mut embedded, report);
    regions.sort_by_key(|region| match region.item.item_type {
        DppClass::PageHeader => 0,
        DppClass::PageFooter => 2,
        _ => 1,
    });
    let mut chunks = Vec::new();
    let mut tables = Vec::new();
    for region in regions {
        if region.item.md.trim().is_empty() {
            report.add(
                page,
                Some(region.index),
                if region.item.item_type == DppClass::Picture {
                    "images-images-and-figures"
                } else {
                    "output-the-sidecar"
                },
                "Region has no recognized content; Markdown element is missing",
            );
        } else if !embedded[region.index] {
            separator_findings(&region.item.md, page, Some(region.index), report);
            chunks.push(anchored(&region.item, page));
        }
        if region.item.item_type == DppClass::Table {
            tables.push(TableEdge {
                index: region.index,
                from: region.options.continues_from_previous_page,
                to: region.options.continues_on_next_page,
            });
        }
        output.items.push(region.item);
    }
    LoweredPage {
        page: output,
        markdown: chunks.join("\n\n"),
        tables,
    }
}

fn region_box(bbox: [f32; 4], width: Option<f64>, height: Option<f64>) -> Option<DppBox> {
    let (width, height) = (width?, height?);
    if bbox
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || bbox[0] > bbox[2]
        || bbox[1] > bbox[3]
    {
        return None;
    }
    let [x1, y1, x2, y2] = bbox.map(f64::from);
    let values = [
        x1 * width,
        y1 * height,
        (x2 - x1) * width,
        (y2 - y1) * height,
    ];
    let values = values.map(|v| ((v * 100.0).round() / 100.0) + 0.0);
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let [x, y, w, h] = values;
    Some(DppBox { x, y, w, h })
}

fn anchored(item: &DppItem, page: usize) -> String {
    match &item.b_box {
        Some(b) => format!(
            "<!-- el p={page} x={} y={} w={} h={} -->\n{}",
            b.x, b.y, b.w, b.h, item.md
        ),
        None => item.md.clone(),
    }
}

fn separator_findings(md: &str, page: usize, index: Option<usize>, report: &mut DppReport) {
    let mut fenced = false;
    if md.lines().any(|line| {
        let line = line.trim();
        if line.starts_with("```") || line.starts_with("~~~") {
            fenced = !fenced;
        }
        !fenced && line == "---"
    }) {
        report.add(
            page,
            index,
            "output-markdown-rendering",
            "Content contains a `---` line, which DPP reserves for page breaks",
        );
    }
}

fn check_continuation(pages: &[LoweredPage], report: &mut DppReport) {
    for pair in pages.windows(2) {
        let left = pair[0].tables.last();
        let right = pair[1].tables.first();
        // Unset flags are written to the sidecar as `false`, so compare them as such.
        let continues = left.is_some_and(|a| a.to == Some(true));
        let continued = right.is_some_and(|b| b.from == Some(true));
        if continues != continued {
            for (page, edge) in [(pair[0].page.page, left), (pair[1].page.page, right)] {
                report.add(
                    page,
                    edge.map(|e| e.index),
                    "tables-cross-page-tables",
                    "Continuation flags disagree across the page break",
                );
            }
        }
    }
}

fn classify(label: &str, block: &DocumentBlock, context: &DppPageContext) -> DppClass {
    use DppClass::*;
    let label = label
        .strip_prefix("cross_")
        .or_else(|| label.strip_prefix("append_"))
        .unwrap_or(label);
    let canonical = LayoutElementType::from_label(canonical_alias(label));
    match label {
        "doc_title" => {
            if context.is_cover == Some(false) {
                SectionHeader
            } else {
                Title
            }
        }
        "section-header" | "abstract_title" | "reference_title" | "content_title" => SectionHeader,
        "document index" => DocumentIndex,
        "list-item" => ListItem,
        "image_block" | "img" | "chemical_structure" | "chemical structure"
        | TABLE_FIGURE_LABEL => Picture,
        "figure_caption" | "code_caption" => Caption,
        "number" | "page_number" => {
            if block.bbox[1] < 0.5 {
                PageHeader
            } else {
                PageFooter
            }
        }
        "code" | "code_block" | "code block" => Code,
        "checkbox-selected" => CheckboxSelected,
        "checkbox-unselected" => CheckboxUnselected,
        "form" => Form,
        "key-value region" | "key_value_region" => KeyValueRegion,
        "ocr" | "phonetic" | "inline_formula" | "__raw__" => Text,
        _ => match canonical {
            LayoutElementType::ParagraphTitle => SectionHeader,
            LayoutElementType::Content => DocumentIndex,
            LayoutElementType::List => ListItem,
            LayoutElementType::Table => Table,
            LayoutElementType::Image | LayoutElementType::Chart | LayoutElementType::Seal => {
                Picture
            }
            LayoutElementType::Formula => Formula,
            LayoutElementType::FigureTitle
            | LayoutElementType::TableTitle
            | LayoutElementType::ChartTitle
            | LayoutElementType::FigureTableChartTitle => Caption,
            LayoutElementType::Footnote => Footnote,
            LayoutElementType::Header | LayoutElementType::HeaderImage => PageHeader,
            LayoutElementType::Footer | LayoutElementType::FooterImage => PageFooter,
            LayoutElementType::Algorithm => Code,
            LayoutElementType::Text
            | LayoutElementType::Abstract
            | LayoutElementType::Reference
            | LayoutElementType::ReferenceContent
            | LayoutElementType::AsideText => Text,
            _ => Text,
        },
    }
}

fn format_content(
    content: &str,
    label: &str,
    class: DppClass,
    settings: &DppBlockOptions,
    page: usize,
    index: usize,
    report: &mut DppReport,
) -> String {
    use DppClass::*;
    let text = content.trim();
    if class == Code {
        let body = content.trim_matches(['\r', '\n']).trim_end();
        return if body.trim_start().starts_with("```") || body.trim_start().starts_with("~~~") {
            body.into()
        } else {
            let longest = body.split(|ch| ch != '`').map(str::len).max().unwrap_or(0);
            let fence = "`".repeat((longest + 1).max(3));
            format!("{fence}\n{body}\n{fence}")
        };
    }
    let base_label = label
        .strip_prefix("cross_")
        .or_else(|| label.strip_prefix("append_"))
        .unwrap_or(label);
    let diagram = text.starts_with("```mermaid\n") || text.starts_with("```diagram/");
    let chart = base_label == "chart" && !diagram;
    if class == Picture && matches!(base_label, "seal" | "stamp") {
        let text = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        return if (text.starts_with(&format!("[{base_label}:")) && text.ends_with(']'))
            || text == format!("[{base_label}]")
        {
            text
        } else {
            format!("[{base_label}: {text}]")
        };
    }
    let text = if class == Table || (class == Picture && chart) {
        format_table(text, chart, page, index, report)
    } else {
        text.to_string()
    };
    let formula_options = FormulaOptions {
        stripping: MathStripping::OuterDollars,
        repair: FormulaRepair::None,
        normalize_mu: false,
        newline_to_break: false,
        clean_tokens: false,
        ..Default::default()
    };
    let options = DelimiterOptions {
        matching: BracketMatching::Spans,
        inline: ("$$", "$$"),
        display: ("$$", "$$"),
        ..Default::default()
    };
    let mut unpaired = false;
    let mut unmatched = false;
    let mut normalize = |prose: &str| {
        let text = map_html_equations(
            prose,
            |part| {
                let (part, retained) = inline_dollars(part);
                unpaired |= retained;
                normalize_math_delimiters(&part, &options)
            },
            |body| format_formula_with_options(body, &formula_options),
        );
        unmatched |= [r"\(", r"\)", r"\[", r"\]"]
            .iter()
            .any(|s| text.contains(s))
            || !text.matches("$$").count().is_multiple_of(2);
        text
    };
    let text = map_math_prose(&text, |prose| {
        if matches!(class, Form | KeyValueRegion | DocumentIndex) {
            static FIELD_ID: LazyLock<Regex> =
                LazyLock::new(|| Regex::new(r"^\s*(?:[-*+]|\d+\.)\s+\\\[[^\r\n]*?\\\]").unwrap());
            prose
                .split_inclusive('\n')
                .map(|line| {
                    let end = FIELD_ID.find(line).map_or(0, |field| field.end());
                    format!("{}{}", &line[..end], normalize(&line[end..]))
                })
                .collect()
        } else {
            normalize(prose)
        }
    });
    if unpaired {
        report.add(
            page,
            Some(index),
            "text-inline-equations",
            "Unpaired single-dollar delimiter",
        );
    }
    if unmatched {
        report.add(
            page,
            Some(index),
            "text-inline-equations",
            "Unmatched math delimiter",
        );
        if class == Formula {
            return text;
        }
    }
    match class {
        Formula => {
            let body = strip_math_wrappers(&text);
            if body.replace(r"\$", "").contains('$') {
                return text;
            }
            static TAG: LazyLock<Regex> =
                LazyLock::new(|| Regex::new(r"\\tag(\*)?\{([^{}]+)\}").unwrap());
            let mut number = String::new();
            let body = TAG.replace(body, |caps: &regex::Captures<'_>| {
                number = if caps.get(1).is_some() {
                    format!(" {}", caps[2].trim())
                } else {
                    format!(" ({})", caps[2].trim())
                };
                ""
            });
            format!(
                "{}{number}",
                format_formula_with_options(&body, &formula_options)
            )
        }
        Title | SectionHeader => {
            let default_level = if class == Title { 1 } else { 2 };
            let level = settings.heading_level.unwrap_or(default_level).clamp(1, 6);
            let title = text
                .lines()
                .map(|line| unheading(line).trim())
                .collect::<Vec<_>>()
                .join(" ");
            format!("{} {title}", "#".repeat(usize::from(level)))
        }
        PageHeader | PageFooter => text.lines().map(unheading).collect::<Vec<_>>().join("\n"),
        ListItem => {
            let mut first = true;
            text.lines()
                .map(|line| {
                    let body = line.trim_start();
                    let indentation = &line[..line.len() - body.len()];
                    if body.is_empty() {
                        return line.to_string();
                    }
                    let initial = std::mem::replace(&mut first, false);
                    if let Some(body) = body.strip_prefix("- ").or_else(|| body.strip_prefix("+ "))
                    {
                        format!("{indentation}* {body}")
                    } else if !initial
                        || body.starts_with("* ")
                        || body.split_once(". ").is_some_and(|(n, _)| {
                            !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
                        })
                    {
                        line.to_string()
                    } else {
                        format!("{indentation}* {body}")
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        CheckboxSelected | CheckboxUnselected => {
            let marker = if class == CheckboxSelected {
                "[x]"
            } else {
                "[ ]"
            };
            if text.is_empty() {
                marker.into()
            } else if text.starts_with(marker) {
                text
            } else {
                format!("{marker} {text}")
            }
        }
        Form | KeyValueRegion | DocumentIndex => {
            if !text.starts_with("- ") && !text.starts_with("* ") {
                let rule = if class == DocumentIndex {
                    "lists-table-of-contents"
                } else {
                    "forms-the-key-value-list"
                };
                report.add(page, Some(index), rule, "Content is not a structured list");
            }
            text
        }
        Picture => {
            if text.contains("![") || text.contains("<img") || text.starts_with("[Non-Text]") {
                report.add(
                    page,
                    Some(index),
                    "images-images-and-figures",
                    "Picture uses an image reference or placeholder",
                );
            }
            if let Some(body) = text.strip_prefix("```mermaid\n") {
                format!("```diagram/mermaid\n{body}")
            } else {
                text
            }
        }
        _ => text,
    }
}

fn unheading(line: &str) -> &str {
    let trimmed = line.trim_start();
    let body = trimmed.trim_start_matches('#');
    if body.len() != trimmed.len() && body.starts_with(' ') {
        body.trim_start()
    } else {
        line
    }
}

fn format_table(
    text: &str,
    chart: bool,
    page: usize,
    index: usize,
    report: &mut DppReport,
) -> String {
    let lower = text.to_ascii_lowercase();
    let mut html = if ["<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>"]
        .iter()
        .any(|token| lower.contains(token))
    {
        convert_otsl_to_html_with_options(
            text,
            &OtslOptions {
                row_padding: RowPadding::Ragged,
                html_input: HtmlInput::CompletePassThrough,
                no_tokens: NoTokens::Preserve,
                empty_fcel_is_anchor: true,
                preserve_ecel_text: true,
                case_insensitive: true,
                unwrap_otsl: true,
                ..Default::default()
            },
        )
    } else if let Some(html) = markdown_table_to_html(text) {
        cell_markup(&html)
    } else {
        text.into()
    };
    if let Some((start, closing)) = table_bounds(&html) {
        if html[start..closing].to_ascii_lowercase().contains("<table") {
            report.add(
                page,
                Some(index),
                "tables-nested-tables",
                "Nested table kept as is",
            );
        }
        let opening = &html[..start];
        if chart && !opening.contains("data-source=") {
            html.insert_str(start - 1, " data-source=\"chart\"");
        } else if chart
            && !opening.contains("data-source=\"chart\"")
            && !opening.contains("data-source='chart'")
        {
            report.add(
                page,
                Some(index),
                "output-markdown-rendering",
                "Chart table has an unexpected data-source attribute",
            );
        } else if !chart && opening.contains("data-source=") {
            report.add(
                page,
                Some(index),
                "output-markdown-rendering",
                "Table has a data-source attribute",
            );
        }
    } else {
        report.add(
            page,
            Some(index),
            "output-markdown-rendering",
            "Table is not complete HTML or a simple pipe grid; kept as is",
        );
    }
    if html.to_ascii_lowercase().contains("<img") {
        report.add(
            page,
            Some(index),
            "images-images-and-figures",
            "Table contains image references in place of picture descriptions",
        );
    }
    html
}

fn table_bounds(html: &str) -> Option<(usize, usize)> {
    static OPENING: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"(?is)\A<table\b(?:[^"'<>]|"[^"]*"|'[^']*')*>"#).unwrap());
    table_span(html, TableWrapper::Document)?;
    Some((
        OPENING.find(html)?.end(),
        html.to_ascii_lowercase().rfind("</table")?,
    ))
}

fn embed_figures(
    regions: &mut [Region],
    source: &PageDocument,
    page: usize,
    embedded: &mut [bool],
    report: &mut DppReport,
) {
    for index in 0..regions.len() {
        if regions[index].item.item_type != DppClass::Picture {
            continue;
        }
        let image = table_figure_image(index);
        let owner = regions.iter().position(|candidate| {
            candidate.item.item_type == DppClass::Table
                && candidate.item.md.contains(&image)
                && crate::pipeline::geometry::cover_ratio(
                    source.blocks[candidate.index].bbox,
                    source.blocks[index].bbox,
                ) >= 0.99
        });
        let Some(owner) = owner else {
            continue;
        };
        let start = regions[owner].item.md.find(&image).unwrap();
        let body = &regions[index].item.md;
        let fragment = if body.is_empty() || body == &image {
            image.clone()
        } else {
            let fragment = format!("{image}\n\n{body}");
            if !regions[owner].item.md[start..].starts_with(&fragment) {
                continue;
            }
            fragment
        };
        regions[index].item.md = fragment;
        let anchored = anchored(&regions[index].item, page);
        let end = start + regions[index].item.md.len();
        regions[owner].item.md.replace_range(start..end, &anchored);
        embedded[index] = true;
        report.add(
            page,
            Some(index),
            "output-the-sidecar",
            format!("Picture is also part of block {owner}'s table md"),
        );
    }
}

fn embed_formula_numbers(
    regions: &mut [Region],
    source: &PageDocument,
    page: &DppPage,
    embedded: &mut [bool],
    report: &mut DppReport,
) {
    let pairing_label = |region: &Region| {
        if region.item.item_type == DppClass::Formula {
            "formula"
        } else if region.label == "formula_number" {
            "formula_number"
        } else {
            ""
        }
    };
    let mut i = 0;
    while i + 1 < regions.len() {
        let pair = formula_number_pair(
            pairing_label(&regions[i]),
            pairing_label(&regions[i + 1]),
            FormulaNumberComposition::AdjacentTag,
        );
        let Some((formula, number)) = pair else {
            i += 1;
            continue;
        };
        let (formula, number) = (i + formula, i + number);
        if !regions[formula].item.md.is_empty() && !regions[number].item.md.is_empty() {
            regions[number].item.item_type = DppClass::Formula;
            regions[number].item.md = strip_math_wrappers(&regions[number].item.md).trim().into();
            let fragment = anchored(&regions[number].item, page.page);
            regions[formula].item.md.push_str(&format!(" {fragment}"));
            let a = source.blocks[formula].bbox;
            let b = source.blocks[number].bbox;
            if regions[formula].item.b_box.is_some() && regions[number].item.b_box.is_some() {
                regions[formula].item.b_box = region_box(
                    [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ],
                    page.width,
                    page.height,
                );
            }
            embedded[number] = true;
            report.add(
                page.page,
                Some(number),
                "output-the-sidecar",
                format!("Equation number is also part of block {formula}'s formula md"),
            );
        }
        i += 2;
    }
    for region in regions {
        if region.label == "formula_number" && !embedded[region.index] {
            report.add(
                page.page,
                Some(region.index),
                "text-displayed-equations",
                "Equation number could not be composed with an adjacent formula",
            );
        }
    }
}

fn embed_captions(regions: &mut [Region], page: usize, report: &mut DppReport) -> Vec<bool> {
    let mut embedded = vec![false; regions.len()];
    for index in 0..regions.len() {
        if regions[index].item.item_type != DppClass::Caption || regions[index].item.md.is_empty() {
            continue;
        }
        let explicit = regions[index].options.caption_of;
        let candidates: Vec<_> = [
            index.checked_sub(1),
            index.checked_add(1).filter(|i| *i < regions.len()),
        ]
        .into_iter()
        .flatten()
        .filter(|&i| {
            let compatible =
                regions[i].item.item_type == DppClass::Table || regions[i].label == "chart";
            compatible
                && table_bounds(&regions[i].item.md).is_some()
                && (matches!(
                    regions[index].label.as_str(),
                    "table_title"
                        | "table_caption"
                        | "chart_title"
                        | "caption"
                        | "figure_table_chart_title"
                ) || (regions[i].item.item_type == DppClass::Table
                    && regions[index].label == "figure_title"))
        })
        .collect();
        let owner = explicit.or_else(|| (candidates.len() == 1).then(|| candidates[0]));
        let Some(owner) = owner.filter(|&i| {
            i < regions.len()
                && i != index
                && matches!(
                    regions[i].item.item_type,
                    DppClass::Table | DppClass::Picture
                )
        }) else {
            continue;
        };
        let Some((opening, closing)) = table_bounds(&regions[owner].item.md) else {
            continue;
        };
        let md = &regions[index].item.md;
        if regions[owner]
            .item
            .md
            .to_ascii_lowercase()
            .contains("<caption")
        {
            continue;
        }
        let caption = if md.starts_with("<caption>") && md.ends_with("</caption>") {
            md.clone()
        } else {
            format!("<caption>{}</caption>", inline_markup(md, true))
        };
        regions[index].item.md = caption;
        let fragment = anchored(&regions[index].item, page);
        let position = if index < owner { opening } else { closing };
        regions[owner]
            .item
            .md
            .insert_str(position, &format!("\n{fragment}\n"));
        embedded[index] = true;
        report.add(
            page,
            Some(index),
            "output-the-sidecar",
            format!("Caption is also part of block {owner}'s table md"),
        );
    }
    embedded
}

/// Pipe-table cells keep their Markdown emphasis, code and links as text, but DPP writes them as
/// HTML inside table cells (`tables-bold-and-italic-inside-tables`). Math is left for the math pass.
fn cell_markup(html: &str) -> String {
    static CELL: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(<t[dh]>)(.*?)(</t[dh]>)").unwrap());
    CELL.replace_all(html, |cell: &regex::Captures<'_>| {
        format!("{}{}{}", &cell[1], inline_markup(&cell[2], false), &cell[3])
    })
    .into_owned()
}

/// Converts Markdown emphasis, code spans and links to HTML, leaving math untouched. Pipe-table
/// cells arrive escaped; raw text needs `escape_code` so code such as `Vec<T>` stays literal.
fn inline_markup(text: &str, escape_code: bool) -> String {
    static MARKUP: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(concat!(
            r"(?P<math>\$\$.*?\$\$|\$[^$]+\$|\\\(.*?\\\))|",
            r"\[(?P<text>[^\]]*)\]\((?P<href>[^)\s]*)\)|",
            r"`(?P<code>[^`]+)`|",
            r"\*\*(?P<strong>[^*\s](?:[^*]*[^*\s])?)\*\*|",
            r"\*(?P<em>[^*\s](?:[^*]*[^*\s])?)\*",
        ))
        .unwrap()
    });
    MARKUP
        .replace_all(text, |caps: &regex::Captures<'_>| {
            if let Some(code) = caps.name("code") {
                let code = code.as_str();
                if escape_code {
                    format!("<code>{}</code>", html_escape::encode_text(code))
                } else {
                    format!("<code>{code}</code>")
                }
            } else if let Some(text) = caps.name("strong") {
                format!("<strong>{}</strong>", text.as_str())
            } else if let Some(text) = caps.name("em") {
                format!("<em>{}</em>", text.as_str())
            } else if let Some(href) = caps.name("href") {
                format!(r#"<a href="{}">{}</a>"#, href.as_str(), &caps["text"])
            } else {
                caps[0].to_string()
            }
        })
        .into_owned()
}

fn map_math_prose(text: &str, mut prose: impl FnMut(&str) -> String) -> String {
    // Code, link destinations and HTML tags retain their literal contents.
    static MARKUP: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?is)<code\b[^>]*>.*?</code>|`+|\]\(|<(?:[^"'<>]|"[^"]*"|'[^']*')*>"#)
            .unwrap()
    });
    static TICKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`+").unwrap());
    let mut output = String::new();
    let (mut cursor, mut search) = (0, 0);
    while let Some(marker) = MARKUP.find_at(text, search) {
        search = marker.end();
        if matches!(marker.as_str(), "<eq>" | "</eq>") {
            continue;
        }
        let end = if marker.as_str().starts_with('`') {
            TICKS
                .find_iter(&text[marker.end()..])
                .find(|close| close.len() == marker.len())
                .map(|close| marker.end() + close.end())
        } else if marker.as_str() == "](" {
            let (mut depth, mut escaped) = (1, false);
            text[marker.end()..]
                .char_indices()
                .find_map(|(offset, ch)| {
                    if escaped {
                        escaped = false;
                        return None;
                    }
                    match ch {
                        '\\' => escaped = true,
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                    (depth == 0).then_some(marker.end() + offset + ch.len_utf8())
                })
        } else {
            Some(marker.end())
        };
        let Some(end) = end else { continue };
        output.push_str(&prose(&text[cursor..marker.start()]));
        output.push_str(&text[marker.start()..end]);
        cursor = end;
        search = end;
    }
    output.push_str(&prose(&text[cursor..]));
    output
}

fn inline_dollars(text: &str) -> (String, bool) {
    static TOKENS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\.|(?s:\$\$.*?\$\$)|\$\$|\$").unwrap());
    let tokens: Vec<_> = TOKENS.find_iter(text).collect();
    let mut output = String::new();
    let (mut cursor, mut i) = (0, 0);
    let mut unpaired = false;
    while i < tokens.len() {
        let open = tokens[i];
        output.push_str(&text[cursor..open.start()]);
        if open.as_str() == "$"
            && let Some(next) = text[open.end()..]
                .chars()
                .next()
                .filter(|ch| !ch.is_whitespace())
        {
            let close = tokens[i + 1..]
                .iter()
                .position(|token| token.as_str().starts_with('$'));
            if let Some(offset) = close {
                let close = tokens[i + 1 + offset];
                let body = &text[open.end()..close.start()];
                // Pandoc tex_math_dollars: tight inner boundaries and no digit after the closer.
                if close.as_str() == "$"
                    && !body.contains(['\n', '\r'])
                    && !body.ends_with(char::is_whitespace)
                    && !text
                        .as_bytes()
                        .get(close.end())
                        .is_some_and(u8::is_ascii_digit)
                {
                    output.push_str(&format!("$${body}$$"));
                    cursor = close.end();
                    i += offset + 2;
                    continue;
                }
            }
            unpaired |= !next.is_ascii_digit();
        }
        output.push_str(open.as_str());
        cursor = open.end();
        i += 1;
    }
    output.push_str(&text[cursor..]);
    (output, unpaired)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(label: &str, content: &str) -> DocumentBlock {
        DocumentBlock {
            block_type: label.into(),
            bbox: [0.25, 0.25, 0.75, 0.75],
            content: Some(content.into()),
            angle: None,
        }
    }

    fn table_options(index: usize, from: bool, to: bool) -> DppBlockOptions {
        DppBlockOptions::default()
            .with_block_index(index)
            .with_continues_from_previous_page(Some(from))
            .with_continues_on_next_page(Some(to))
    }

    #[test]
    fn paired_structured_export() {
        let page = PageDocument {
            blocks: vec![
                block(
                    "text",
                    r"Cost $25; \(x\); inline $\alpha_i^{2}$ and $x^2$; display $$q$$; range $5 to $10; simple $x$; $F = kx$, $k$, $N/m$; Let $2x$ and $y$ be positive; $5 and $10; `echo $HOME $PATH`; ``\(literal\) \[open $$ $missing ` tick``",
                ),
                block("code", r"    return x  # \(literal\)"),
                block("header", "# Report"),
                block(
                    "table",
                    r#"<table><tr><td title="cell"><eq>$x$</eq></td></tr></table>"#,
                ),
                block("figure_title", "*Results* `Vec<T>`"),
                block("footer", "End"),
                block("formula", r"\[x=y\tag{1}\]"),
                block("formula", r"$$\frac{x}{y} \tag*{(2)}$$"),
                block("text", "Display \\[\nx^2\n\\] and inline \\(x^2\\)"),
                block("form", r"- \[1a\] **Label:** $w h$"),
                block("seal", "Authority\nAPPROVED"),
                block("stamp", "PAID\n2026"),
                block(
                    "List-item",
                    "first\n  continuation\n- second\n+ third\n* fourth\n2. fifth",
                ),
                block("table", "<OTSL><FCEL>a<FCEL>b<NL><FCEL><FCEL>c<NL></OTSL>"),
                block("section-header", "# Part-\nHeading"),
                block("formula", "$x$"),
                block("text", "Unpaired $value"),
                block("code", "a: 1\n---\nb: 2"),
                block("formula", r"P = \$5"),
            ],
            markdown: Some("Cached profile Markdown".into()),
            ..PageDocument::default()
        };
        let context = DppPageContext::default().with_dimensions(600.0, 800.0);
        let output = page.to_dpp(&context, &DppOptions::default());
        assert_eq!(output.sidecar.items[0].item_type, DppClass::PageHeader);
        assert_eq!(output.sidecar.items[0].md, "Report");
        assert_eq!(
            output.sidecar.items.last().unwrap().item_type,
            DppClass::PageFooter
        );
        for fragment in [
            r"Cost $25; $$x$$; inline $$\alpha_i^{2}$$ and $$x^2$$; display $$q$$; range $5 to $10; simple $$x$$; $$F = kx$$, $$k$$, $$N/m$$; Let $$2x$$ and $$y$$ be positive; $5 and $10; `echo $HOME $PATH`; ``\(literal\) \[open $$ $missing ` tick``",
            r#"<td title="cell">$$x$$</td>"#,
            "```\n    return x  # \\(literal\\)\n```",
            "$$x=y$$ (1)",
            r"$$\frac{x}{y}$$ (2)",
            "Display $$x^2$$ and inline $$x^2$$",
            r"- \[1a\] **Label:** $$w h$$",
            "[seal: Authority APPROVED]",
            "[stamp: PAID 2026]",
            "* first\n  continuation\n* second\n* third\n* fourth\n2. fifth",
            "## Part- Heading",
            "<tr><td></td><td>c</td></tr>",
            "```\na: 1\n---\nb: 2\n```",
            r"$$P = \$5$$",
        ] {
            assert!(output.markdown.contains(fragment), "{fragment}");
        }
        assert!(!output.markdown.contains("Cached profile Markdown"));
        assert_eq!(
            output
                .markdown
                .matches("<caption><em>Results</em> <code>Vec&lt;T&gt;</code></caption>")
                .count(),
            1
        );
        assert_eq!(output.report.findings.len(), 2);
        assert_eq!(output.report.findings[0].block_index, Some(4));
        assert_eq!(output.report.findings[0].rule_id, "output-the-sidecar");
        assert_eq!(output.report.findings[1].block_index, Some(16));
        assert_eq!(output.report.findings[1].rule_id, "text-inline-equations");
        assert_eq!(output.sidecar.items[14].md, "$$x$$");
        let json = serde_json::to_value(&output.sidecar).unwrap();
        let table = json["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "Table")
            .unwrap();
        assert_eq!(table["continuesFromPreviousPage"], false);
        assert_eq!(table["continuesOnNextPage"], false);
        assert_eq!(table["bBox"]["x"], 150.0);
        assert!(
            json["items"]
                .as_array()
                .unwrap()
                .iter()
                .all(|item| item.get("layoutAwareBbox").is_none())
        );
        let points = page.to_dpp(
            &context.clone().with_dimensions(612.0, 792.0),
            &DppOptions::default().with_coordinate_space(DppCoordinateSpace::PdfPoints),
        );
        assert_eq!(points.sidecar.items[1].b_box.as_ref().unwrap().x, 153.0);
        assert_eq!(points.report.protocol_revision, PROTOCOL_REVISION);

        let mut rounded = PageDocument {
            blocks: vec![block("text", "Rounded")],
            ..PageDocument::default()
        };
        rounded.blocks[0].bbox = [0.127281, 0.25, 0.8, 0.75];
        let rounded = rounded.to_dpp(
            &context.clone().with_dimensions(1000.0, 800.0),
            &DppOptions::default(),
        );
        let bbox = rounded.sidecar.items[0].b_box.as_ref().unwrap();
        assert_eq!((bbox.x, bbox.w), (127.28, 672.72));
        assert!(rounded.markdown.contains("x=127.28 y=200 w=672.72 h=400"));

        let figure = PageDocument {
            blocks: vec![
                block(
                    "table",
                    r#"<table><tr><td><img src="image_2.png" alt="Image" /></td></tr></table>"#,
                ),
                block(TABLE_FIGURE_LABEL, ""),
            ],
            ..PageDocument::default()
        }
        .to_dpp(&context, &DppOptions::default());
        assert_eq!(figure.markdown.matches("image_2.png").count(), 1);
        assert!(
            figure
                .markdown
                .contains(&anchored(&figure.sidecar.items[1], 1))
        );
        assert_eq!(
            figure
                .report
                .findings
                .iter()
                .map(|f| f.rule_id.as_str())
                .collect::<Vec<_>>(),
            ["images-images-and-figures", "output-the-sidecar"]
        );

        let mut equation_number = block("formula_number", "(7)");
        equation_number.bbox = [0.8, 0.25, 0.9, 0.35];
        let linked = PageDocument {
            blocks: vec![
                block(
                    "text",
                    r#"[API $x$](https://example.com/items(foo)?$filter=x&$select=y) <a href="https://example.com/?$filter=x&$select=y" title="> ` \[ $literal">\(y\)</a> `<a href="$z">\(literal\)</a>`"#,
                ),
                block("formula", "$$E=mc^2$$"),
                equation_number,
                block("formula_number", "(99)"),
                block("chart", "```mermaid\ngraph TD; A-->B\n```"),
            ],
            ..PageDocument::default()
        }
        .to_dpp(&context, &DppOptions::default());
        assert_eq!(
            linked.sidecar.items[0].md,
            r#"[API $$x$$](https://example.com/items(foo)?$filter=x&$select=y) <a href="https://example.com/?$filter=x&$select=y" title="> ` \[ $literal">$$y$$</a> `<a href="$z">\(literal\)</a>`"#
        );
        let number = &linked.sidecar.items[2];
        assert_eq!(number.item_type, DppClass::Formula);
        assert_eq!(
            linked.sidecar.items[1].md,
            format!("$$E=mc^2$$ {}", anchored(number, 1))
        );
        assert_eq!(linked.markdown.matches("(7)").count(), 1);
        assert_eq!(linked.sidecar.items[1].b_box.as_ref().unwrap().w, 390.0);
        assert_eq!(linked.sidecar.items[4].item_type, DppClass::Picture);
        assert_eq!(
            linked.sidecar.items[4].md,
            "```diagram/mermaid\ngraph TD; A-->B\n```"
        );
        assert_eq!(
            linked
                .report
                .findings
                .iter()
                .map(|f| (f.block_index, f.rule_id.as_str()))
                .collect::<Vec<_>>(),
            [
                (Some(2), "output-the-sidecar"),
                (Some(3), "text-displayed-equations"),
            ]
        );

        let clean = PageDocument {
            blocks: vec![block("doc_title", "Title"), block("number", "1")],
            diagnostics: vec![crate::ParseDiagnostic {
                block_index: None,
                stage: "layout".into(),
                message: "Source diagnostic".into(),
            }],
            ..PageDocument::default()
        };
        let context = context.with_page(0).with_blank(true).with_blocks(vec![
            DppBlockOptions::default(),
            DppBlockOptions::default(),
            DppBlockOptions::default().with_block_index(99),
        ]);
        assert!(
            clean
                .to_dpp(&context, &DppOptions::default())
                .report
                .findings
                .is_empty()
        );
    }

    #[test]
    fn document_preservation() {
        let mut document = DocumentOutput::from_pages(vec![
            PageDocument {
                blocks: vec![block("table", "<table><tr><td>A</td></tr></table>")],
                ..PageDocument::default()
            },
            PageDocument {
                blocks: vec![block("table", "<table><tr><td>B</td></tr></table>")],
                ..PageDocument::default()
            },
            PageDocument::default(),
            PageDocument {
                markdown: Some("Native transcript".into()),
                ..PageDocument::default()
            },
        ]);
        document.markdown = Some("Joint merged transcript".into());
        let mut contexts = vec![
            DppPageContext::default()
                .with_dimensions(600.0, 800.0)
                .with_blocks(vec![table_options(0, false, true)]),
            DppPageContext::default()
                .with_dimensions(600.0, 800.0)
                .with_blocks(vec![table_options(0, true, false)]),
            DppPageContext::default()
                .with_dimensions(600.0, 800.0)
                .with_blank(true),
            DppPageContext::default().with_dimensions(600.0, 800.0),
            DppPageContext::default(),
        ];
        let output = document.to_dpp(&contexts, &DppOptions::default());
        assert_eq!(output.sidecar.len(), 4);
        assert_eq!(output.markdown.matches("\n\n---\n\n").count(), 3);
        assert!(
            output
                .markdown
                .contains("<!-- blank page -->\n\n---\n\nNative transcript")
        );
        assert!(!output.markdown.contains("Joint merged transcript"));
        assert_eq!(output.markdown.matches("<table>").count(), 2);
        assert!(output.sidecar[3].items.is_empty());
        assert!(
            output
                .report
                .findings
                .iter()
                .all(|finding| finding.page == 4)
        );
        contexts[0].blocks.clear();
        let unknown = document.to_dpp(&contexts, &DppOptions::default());
        assert_eq!(
            unknown.sidecar[0].items[0].continues_on_next_page,
            Some(false)
        );
        assert_eq!(
            unknown
                .report
                .findings
                .iter()
                .filter(|f| f.rule_id == "tables-cross-page-tables")
                .count(),
            2
        );
    }

    #[test]
    fn best_effort_degradation() {
        let page = PageDocument {
            blocks: vec![
                block("chart", "| Year | `A` |\n| --- | --- |\n| 2024 | 7 |"),
                block("table", "unparsed table"),
                block("text", r"Unclosed \[x"),
                block("form", "Unstructured fields"),
                block("text", "Printed\n---\nseparator"),
                block("table", "<table><tr><td><table></table></td></tr></table>"),
                DocumentBlock {
                    content: None,
                    ..block("image", "")
                },
                block("chart", ""),
                block("text", ""),
                block(
                    "table",
                    "| Source | Value |\n| --- | --- |\n| [Report](https://example.com/?$filter=x&$select=y) **1** *2* `$x$` | 1 |",
                ),
                block(
                    "table",
                    "| Formula | Value |\n| --- | --- |\n| $$a*b*c$$ | $$w h$$ |",
                ),
            ],
            ..PageDocument::default()
        };
        let output = page.to_dpp(
            &DppPageContext::default(),
            &DppOptions::default().with_granularity(DppGranularity::Word),
        );
        assert!(
            output
                .markdown
                .contains("<table data-source=\"chart\"><tr><th>Year</th><th><code>A</code></th>")
        );
        assert!(output.markdown.contains(r"Unclosed \[x"));
        assert!(output.sidecar.items.iter().all(|item| item.b_box.is_none()));
        assert_eq!(output.report.emitted_granularity, DppGranularity::None);
        for index in [6, 7] {
            assert!(
                output
                    .report
                    .findings
                    .iter()
                    .any(|f| f.block_index == Some(index)
                        && f.rule_id == "images-images-and-figures")
            );
            assert_eq!(output.sidecar.items[index].item_type, DppClass::Picture);
        }
        assert!(
            output
                .report
                .findings
                .iter()
                .any(|f| f.block_index == Some(8) && f.rule_id == "output-the-sidecar")
        );
        assert!(!output.markdown.contains("image_7.png"));
        for cell in [
            r#"<td><a href="https://example.com/?$filter=x&amp;$select=y">Report</a> <strong>1</strong> <em>2</em> <code>$x$</code></td>"#,
            "<td>$$a*b*c$$</td>",
        ] {
            assert!(output.markdown.contains(cell), "{cell}");
        }

        for rule in [
            "output-the-sidecar",
            "output-boxes-and-granularity",
            "output-markdown-rendering",
            "text-inline-equations",
            "forms-the-key-value-list",
            "tables-nested-tables",
        ] {
            assert!(
                output.report.findings.iter().any(|f| f.rule_id == rule),
                "{rule}"
            );
        }
        assert_eq!(
            serde_json::to_value(&output.sidecar).unwrap()["width"],
            serde_json::Value::Null
        );
    }
}
