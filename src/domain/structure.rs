//! Document structure analysis result types.
//!
//! This module defines the result types for document structure analysis,
//! including layout detection, table recognition, and formula recognition.

use crate::oarocr::TextRegion;
use crate::processors::BoundingBox;
use image::RgbImage;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Title numbering pattern for detecting section numbers like 1, 1.2, 1.2.3, (1), 一、etc.
/// This follows standard title numbering pattern.
static TITLE_NUMBERING_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (
            # Arabic numerals: 1, 1.2, 1.2.3, etc.
            [1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?
            |
            # Parenthesized Arabic numerals: (1), (1.2), etc.
            [(（][1-9][0-9]*(?:\.[1-9][0-9]*)*[)）]
            |
            # Chinese numerals with punctuation: 一、 二、
            [一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾][、.]?
            |
            # Parenthesized Chinese numerals: （一）
            [(（][一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[)）]
            |
            # Roman numerals with delimiter (period or followed by space)
            (?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\b)
        )
        (\s+)
        (.*)
        $
    ",
    )
    .expect("Invalid title numbering regex")
});

/// Format a paragraph title with automatic level detection based on numbering.
///
/// Following standard title formatting logic:
/// - Extracts numbering prefix (1.2.3, etc.)
/// - Determines heading level from number of dots (1.2.3 -> level 3)
/// - Returns (level, formatted_title) where level is 1-based
///
/// # Examples
///
/// - "1 Introduction" -> (1, "1 Introduction")
/// - "1.2 Methods" -> (2, "1.2 Methods")
/// - "1.2.3 Details" -> (3, "1.2.3 Details")
/// - "一、绪论" -> (1, "一、绪论")
/// - "Just text" -> (2, "Just text") (default level 2 for no numbering)
fn format_title_with_level(title: &str) -> (usize, String) {
    // Clean up line breaks
    let cleaned = title.replace("-\n", "").replace('\n', " ");

    if let Some(captures) = TITLE_NUMBERING_REGEX.captures(&cleaned) {
        let numbering = captures.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let title_content = captures.get(3).map(|m| m.as_str()).unwrap_or("");

        // Determine level from dots in numbering
        // 1 -> level 1, 1.2 -> level 2, 1.2.3 -> level 3
        let level = if numbering.contains('.') {
            numbering.matches('.').count() + 1
        } else {
            1
        };

        // Reconstruct title: numbering + space + content
        let formatted = if title_content.is_empty() {
            numbering.trim_end_matches('.').to_string()
        } else {
            format!(
                "{} {}",
                numbering.trim_end_matches('.'),
                title_content.trim_start()
            )
        };

        // Clamp level to reasonable range (1-6 for markdown)
        let level = level.clamp(1, 6);

        (level, formatted)
    } else {
        // No numbering detected, default to level 2 (## heading)
        (2, cleaned)
    }
}

/// A detected document region block (from PP-DocBlockLayout).
///
/// Region blocks represent hierarchical groupings of layout elements,
/// typically columns or logical sections of a document. They are used
/// for hierarchical reading order determination.
///
/// # PP-StructureV3 Alignment
///
/// PP-DocBlockLayout detects "region" type blocks that group related
/// layout elements together. Elements within the same region should
/// be read together before moving to the next region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionBlock {
    /// Bounding box of the region
    pub bbox: BoundingBox,
    /// Confidence score of the detection
    pub confidence: f32,
    /// Index of this region in the reading order
    pub order_index: Option<u32>,
    /// Indices of layout elements that belong to this region
    pub element_indices: Vec<usize>,
}

/// Result of document structure analysis.
///
/// This struct contains all the results from analyzing a document's structure,
/// including layout elements, tables, formulas, and OCR results.
///
/// # Coordinate System
///
/// The coordinate system of bounding boxes depends on which preprocessing was applied:
///
/// - **No preprocessing**: Boxes are in the original input image's coordinate system.
///
/// - **Orientation correction only** (`orientation_angle` set, `rectified_img` is None):
///   Boxes are transformed back to the original input image's coordinate system.
///
/// - **Rectification applied** (`rectified_img` is Some):
///   Boxes remain in the **rectified image's coordinate system**. Neural network-based
///   rectification (UVDoc) warps cannot be precisely inverted, so use `rectified_img`
///   for visualization instead of the original image.
///
/// - **Both orientation and rectification**: Boxes are in the rectified coordinate system
///   (rectification takes precedence since it's applied after orientation correction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureResult {
    /// Path to the input image file
    pub input_path: Arc<str>,
    /// Index of the image in a batch (0 for single image processing)
    pub index: usize,
    /// Detected layout elements (text regions, tables, figures, etc.)
    pub layout_elements: Vec<LayoutElement>,
    /// Recognized tables with their structure and content
    pub tables: Vec<TableResult>,
    /// Recognized mathematical formulas
    pub formulas: Vec<FormulaResult>,
    /// OCR text regions (if OCR was integrated)
    pub text_regions: Option<Vec<TextRegion>>,
    /// Document orientation angle (if orientation correction was used)
    pub orientation_angle: Option<f32>,
    /// Detected region blocks for hierarchical ordering (PP-DocBlockLayout)
    /// When present, layout_elements are already sorted by region hierarchy
    pub region_blocks: Option<Vec<RegionBlock>>,
    /// Rectified image (if document rectification was used)
    /// Note: Bounding boxes are already transformed back to original coordinates for rotation,
    /// but for rectification (UVDoc), boxes are in the rectified image's coordinate system.
    /// Use this image for visualization when rectification was applied.
    #[serde(skip)]
    pub rectified_img: Option<Arc<RgbImage>>,
}

impl StructureResult {
    /// Creates a new structure result.
    pub fn new(input_path: impl Into<Arc<str>>, index: usize) -> Self {
        Self {
            input_path: input_path.into(),
            index,
            layout_elements: Vec::new(),
            tables: Vec::new(),
            formulas: Vec::new(),
            text_regions: None,
            orientation_angle: None,
            region_blocks: None,
            rectified_img: None,
        }
    }

    /// Adds layout elements to the result.
    pub fn with_layout_elements(mut self, elements: Vec<LayoutElement>) -> Self {
        self.layout_elements = elements;
        self
    }

    /// Adds tables to the result.
    pub fn with_tables(mut self, tables: Vec<TableResult>) -> Self {
        self.tables = tables;
        self
    }

    /// Adds formulas to the result.
    pub fn with_formulas(mut self, formulas: Vec<FormulaResult>) -> Self {
        self.formulas = formulas;
        self
    }

    /// Adds OCR text regions to the result.
    pub fn with_text_regions(mut self, regions: Vec<TextRegion>) -> Self {
        self.text_regions = Some(regions);
        self
    }

    /// Adds region blocks to the result (PP-DocBlockLayout).
    ///
    /// Region blocks represent hierarchical groupings of layout elements.
    /// When set, layout_elements should already be sorted by region hierarchy.
    pub fn with_region_blocks(mut self, blocks: Vec<RegionBlock>) -> Self {
        self.region_blocks = Some(blocks);
        self
    }

    /// Converts the result to a Markdown string.
    ///
    /// Follows PP-StructureV3's formatting rules:
    /// - DocTitle: `# title`
    /// - ParagraphTitle: Auto-detect numbering (1.2.3 -> ###)
    /// - Formula: `$$latex$$`
    /// - Table: HTML with border
    /// - Images: `![Figure](caption)`
    ///
    /// Note: Low-confidence text elements that overlap with table regions are filtered out
    /// to avoid duplicate content from table OCR.
    pub fn to_markdown(&self) -> String {
        // Collect table bboxes for overlap filtering
        let table_bboxes: Vec<&BoundingBox> = self
            .layout_elements
            .iter()
            .filter(|e| e.element_type == LayoutElementType::Table)
            .map(|e| &e.bbox)
            .collect();

        let mut md = String::new();
        for element in &self.layout_elements {
            // PP-StructureV3 markdown ignores auxiliary labels.
            if matches!(
                element.element_type,
                LayoutElementType::Number
                    | LayoutElementType::Footnote
                    | LayoutElementType::Header
                    | LayoutElementType::HeaderImage
                    | LayoutElementType::Footer
                    | LayoutElementType::FooterImage
                    | LayoutElementType::AsideText
            ) {
                continue;
            }

            // Filter out low-confidence text elements that overlap with tables
            // These are typically OCR artifacts from table cell text that shouldn't be
            // output separately in markdown
            if element.element_type == LayoutElementType::Text {
                let overlaps_table = table_bboxes.iter().any(|table_bbox| {
                    element.bbox.ioa(table_bbox) > 0.3 // >30% of text is inside table
                });

                // Skip low-confidence text that overlaps with table regions
                // Standard logic filters these in the stitching phase
                if overlaps_table && element.confidence < 0.7 {
                    continue;
                }
            }

            match element.element_type {
                // Document title
                LayoutElementType::DocTitle => {
                    md.push_str("\n# ");
                    if let Some(text) = &element.text {
                        md.push_str(&text.replace("-\n", "").replace('\n', " "));
                    }
                    md.push_str("\n\n");
                }
                // Paragraph/section title - auto-detect numbering for level
                LayoutElementType::ParagraphTitle => {
                    if let Some(text) = &element.text {
                        let (level, formatted_title) = format_title_with_level(text);
                        md.push('\n');
                        for _ in 0..level {
                            md.push('#');
                        }
                        md.push(' ');
                        md.push_str(&formatted_title);
                        md.push_str("\n\n");
                    } else {
                        md.push_str("\n## \n\n");
                    }
                }
                // Table - preserve HTML structure with border
                LayoutElementType::Table => {
                    if let Some(table) =
                        self.tables.iter().find(|t| t.bbox.iou(&element.bbox) > 0.5)
                    {
                        if let Some(html) = &table.html_structure {
                            // Add border to table for better visibility
                            let table_with_border = html.replace("<table>", "<table border=\"1\">");
                            md.push('\n');
                            md.push_str(&table_with_border);
                            md.push_str("\n\n");
                        } else {
                            md.push_str("\n[Table]\n\n");
                        }
                    } else {
                        md.push_str("\n[Table]\n\n");
                    }
                }
                // Formula - wrap with $$
                LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
                    md.push_str("\n$$");
                    if let Some(latex) = &element.text {
                        md.push_str(latex);
                    }
                    md.push_str("$$\n\n");
                }
                // Image/Chart - figure format
                LayoutElementType::Image | LayoutElementType::Chart => {
                    md.push_str("\n![Figure]");
                    if let Some(caption) = &element.text {
                        md.push('(');
                        md.push_str(caption);
                        md.push(')');
                    }
                    md.push_str("\n\n");
                }
                // Seal - show as image with text
                LayoutElementType::Seal => {
                    md.push_str("\n![Seal]");
                    if let Some(text) = &element.text {
                        md.push_str("\n> ");
                        md.push_str(text);
                    }
                    md.push_str("\n\n");
                }
                // Captions
                _ if element.element_type.is_caption() => {
                    if let Some(text) = &element.text {
                        md.push('*');
                        md.push_str(text);
                        md.push_str("*\n\n");
                    }
                }
                // Abstract
                LayoutElementType::Abstract => {
                    md.push_str("\n**Abstract**\n\n");
                    if let Some(text) = &element.text {
                        md.push_str(text);
                        md.push_str("\n\n");
                    }
                }
                // Reference
                LayoutElementType::Reference => {
                    md.push_str("\n**References**\n\n");
                    if let Some(text) = &element.text {
                        md.push_str(text);
                        md.push_str("\n\n");
                    }
                }
                // List
                LayoutElementType::List => {
                    if let Some(text) = &element.text {
                        // Split by newlines and format as list items
                        for line in text.lines() {
                            md.push_str("- ");
                            md.push_str(line);
                            md.push('\n');
                        }
                        md.push('\n');
                    }
                }
                // Header/Footer - smaller text
                _ if element.element_type.is_header() || element.element_type.is_footer() => {
                    if let Some(text) = &element.text {
                        md.push_str("<small>");
                        md.push_str(text);
                        md.push_str("</small>\n\n");
                    }
                }
                // Default text elements
                _ => {
                    if let Some(text) = &element.text {
                        // Convert double newlines to paragraph breaks
                        let formatted = text.replace("\n\n", "\n").replace('\n', "\n\n");
                        md.push_str(&formatted);
                        md.push_str("\n\n");
                    }
                }
            }
        }
        md.trim().to_string()
    }

    /// Converts the result to an HTML string.
    ///
    /// Follows PP-StructureV3's formatting rules with semantic HTML tags.
    pub fn to_html(&self) -> String {
        let mut html = String::from(
            "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n</head>\n<body>\n",
        );

        for element in &self.layout_elements {
            match element.element_type {
                // Document title
                LayoutElementType::DocTitle => {
                    html.push_str("<h1>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</h1>\n");
                }
                // Paragraph/section title
                LayoutElementType::ParagraphTitle => {
                    html.push_str("<h2>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</h2>\n");
                }
                // Table - embed HTML structure
                LayoutElementType::Table => {
                    if let Some(table) =
                        self.tables.iter().find(|t| t.bbox.iou(&element.bbox) > 0.5)
                    {
                        if let Some(table_html) = &table.html_structure {
                            // Add border styling
                            let styled = table_html.replace(
                                "<table>",
                                "<table border=\"1\" style=\"border-collapse: collapse;\">",
                            );
                            html.push_str(&styled);
                            html.push('\n');
                        } else {
                            html.push_str("<p>[Table]</p>\n");
                        }
                    } else {
                        html.push_str("<p>[Table]</p>\n");
                    }
                }
                // Formula - use math tags
                LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
                    html.push_str("<p class=\"formula\">$$");
                    if let Some(latex) = &element.text {
                        html.push_str(&Self::escape_html(latex));
                    }
                    html.push_str("$$</p>\n");
                }
                // Image/Chart
                LayoutElementType::Image | LayoutElementType::Chart => {
                    html.push_str("<figure>\n<img alt=\"Figure\" />\n");
                    if let Some(caption) = &element.text {
                        html.push_str("<figcaption>");
                        html.push_str(&Self::escape_html(caption));
                        html.push_str("</figcaption>\n");
                    }
                    html.push_str("</figure>\n");
                }
                // Seal
                LayoutElementType::Seal => {
                    html.push_str("<figure class=\"seal\">\n<img alt=\"Seal\" />\n");
                    if let Some(text) = &element.text {
                        html.push_str("<figcaption>");
                        html.push_str(&Self::escape_html(text));
                        html.push_str("</figcaption>\n");
                    }
                    html.push_str("</figure>\n");
                }
                // Captions
                _ if element.element_type.is_caption() => {
                    if let Some(text) = &element.text {
                        html.push_str("<figcaption>");
                        html.push_str(&Self::escape_html(text));
                        html.push_str("</figcaption>\n");
                    }
                }
                // Abstract
                LayoutElementType::Abstract => {
                    html.push_str("<section class=\"abstract\">\n<h3>Abstract</h3>\n<p>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</p>\n</section>\n");
                }
                // Reference
                LayoutElementType::Reference | LayoutElementType::ReferenceContent => {
                    html.push_str("<section class=\"references\">\n<p>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</p>\n</section>\n");
                }
                // List
                LayoutElementType::List => {
                    html.push_str("<ul>\n");
                    if let Some(text) = &element.text {
                        for line in text.lines() {
                            html.push_str("<li>");
                            html.push_str(&Self::escape_html(line));
                            html.push_str("</li>\n");
                        }
                    }
                    html.push_str("</ul>\n");
                }
                // Header
                _ if element.element_type.is_header() => {
                    html.push_str("<header>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</header>\n");
                }
                // Footer
                _ if element.element_type.is_footer() => {
                    html.push_str("<footer>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</footer>\n");
                }
                // Default text
                _ => {
                    if let Some(text) = &element.text {
                        html.push_str("<p>");
                        html.push_str(&Self::escape_html(text));
                        html.push_str("</p>\n");
                    }
                }
            }
        }
        html.push_str("</body>\n</html>");
        html
    }

    /// Escapes HTML special characters.
    fn escape_html(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }

    /// Converts the result to a JSON Value.
    pub fn to_json_value(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self)
    }

    /// Saves the analysis results to the specified directory.
    ///
    /// This generates:
    /// - `*_res.json`: The full structured result
    /// - `*_res.md`: A Markdown representation
    /// - `*_res.html`: An HTML representation
    ///
    /// # Arguments
    ///
    /// * `to_html` - If true, save an HTML representation.
    pub fn save_results(
        &self,
        output_dir: impl AsRef<Path>,
        to_json: bool,
        to_markdown: bool,
        to_html: bool,
    ) -> std::io::Result<()> {
        let output_dir = output_dir.as_ref();
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        let input_path = Path::new(self.input_path.as_ref());
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("result");

        // Save JSON
        if to_json {
            let json_path = output_dir.join(format!("{}.json", stem));
            let json_file = std::fs::File::create(json_path)?;
            serde_json::to_writer_pretty(json_file, self)?;
        }

        // Save Markdown
        if to_markdown {
            let md_path = output_dir.join(format!("{}.md", stem));
            std::fs::write(md_path, self.to_markdown())?;
        }

        // Save HTML
        if to_html {
            let html_path = output_dir.join(format!("{}.html", stem));
            std::fs::write(html_path, self.to_html())?;
        }

        Ok(())
    }
}

/// A layout element detected in the document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutElement {
    /// Bounding box of the element
    pub bbox: BoundingBox,
    /// Type of the layout element
    pub element_type: LayoutElementType,
    /// Confidence score for the detection
    pub confidence: f32,
    /// Optional label for the element (original model label)
    pub label: Option<String>,
    /// Optional text content for the element
    pub text: Option<String>,
    /// Reading order index (1-based, assigned during stitching)
    ///
    /// This index represents the element's position in the reading order.
    /// Only elements that should be included in reading flow (text, tables,
    /// formulas, images, etc.) will have an order index assigned.
    /// Headers, footers, and other auxiliary elements may have `None`.
    pub order_index: Option<u32>,
}

impl LayoutElement {
    /// Creates a new layout element.
    pub fn new(bbox: BoundingBox, element_type: LayoutElementType, confidence: f32) -> Self {
        Self {
            bbox,
            element_type,
            confidence,
            label: None,
            text: None,
            order_index: None,
        }
    }

    /// Sets the label for the element.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the text content for the element.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }
}

/// Layout element type supporting PP-StructureV3's full label set.
///
/// This enum represents both **semantic categories** and **fine-grained labels** for layout elements.
/// PP-StructureV3 models output 20 or 23 class labels depending on the model variant.
///
/// The original model-specific label is preserved in `LayoutElement.label` field.
///
/// # PP-StructureV3 Label Categories
///
/// **Document structure:**
/// - `DocTitle` - Document title (doc_title)
/// - `ParagraphTitle` - Section/paragraph title (paragraph_title)
/// - `Text` - General text content
/// - `Content` - Table of contents (content)
/// - `Abstract` - Abstract section
///
/// **Visual elements:**
/// - `Image` - Images/figures (image, figure)
/// - `Table` - Tables
/// - `Chart` - Charts/graphs
/// - `Formula` - Mathematical formulas
///
/// **Captions and titles:**
/// - `FigureTitle` - Figure caption (figure_title)
/// - `TableTitle` - Table caption (table_title)
/// - `ChartTitle` - Chart caption (chart_title)
/// - `FigureTableChartTitle` - Combined caption type
///
/// **Page structure:**
/// - `Header` - Page header
/// - `HeaderImage` - Header image
/// - `Footer` - Page footer
/// - `FooterImage` - Footer image
/// - `Footnote` - Footnotes
///
/// **Special elements:**
/// - `Seal` - Stamps/official seals
/// - `Number` - Page numbers
/// - `Reference` - References section
/// - `ReferenceContent` - Reference content
/// - `Algorithm` - Algorithm blocks
/// - `FormulaNumber` - Formula numbers
/// - `AsideText` - Marginal/aside text
/// - `List` - List items
///
/// - `Other` - Unknown/unmapped labels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutElementType {
    /// Document title
    DocTitle,
    /// Paragraph/section title
    ParagraphTitle,
    /// General text content
    Text,
    /// Table of contents
    Content,
    /// Abstract section
    Abstract,

    /// Image or figure
    Image,
    /// Table
    Table,
    /// Chart or graph
    Chart,
    /// Mathematical formula
    Formula,

    /// Figure caption/title
    FigureTitle,
    /// Table caption/title
    TableTitle,
    /// Chart caption/title
    ChartTitle,
    /// Combined figure/table/chart title (PP-DocLayout)
    FigureTableChartTitle,

    /// Page header
    Header,
    /// Header image
    HeaderImage,
    /// Page footer
    Footer,
    /// Footer image
    FooterImage,
    /// Footnote
    Footnote,

    /// Stamp or official seal
    Seal,
    /// Page number
    Number,
    /// Reference section
    Reference,
    /// Reference content (PP-DocLayout_plus-L)
    ReferenceContent,
    /// Algorithm block
    Algorithm,
    /// Formula number
    FormulaNumber,
    /// Marginal/aside text
    AsideText,
    /// List items
    List,

    /// Generic document region block (PP-DocBlockLayout)
    /// Used for hierarchical layout ordering and block grouping
    Region,

    /// Other/unknown (original label preserved in LayoutElement.label)
    Other,
}

impl LayoutElementType {
    /// Returns the string representation of the element type.
    ///
    /// This returns the PP-StructureV3 compatible label string.
    pub fn as_str(&self) -> &'static str {
        match self {
            // Document Structure
            LayoutElementType::DocTitle => "doc_title",
            LayoutElementType::ParagraphTitle => "paragraph_title",
            LayoutElementType::Text => "text",
            LayoutElementType::Content => "content",
            LayoutElementType::Abstract => "abstract",

            // Visual Elements
            LayoutElementType::Image => "image",
            LayoutElementType::Table => "table",
            LayoutElementType::Chart => "chart",
            LayoutElementType::Formula => "formula",

            // Captions
            LayoutElementType::FigureTitle => "figure_title",
            LayoutElementType::TableTitle => "table_title",
            LayoutElementType::ChartTitle => "chart_title",
            LayoutElementType::FigureTableChartTitle => "figure_table_chart_title",

            // Page Structure
            LayoutElementType::Header => "header",
            LayoutElementType::HeaderImage => "header_image",
            LayoutElementType::Footer => "footer",
            LayoutElementType::FooterImage => "footer_image",
            LayoutElementType::Footnote => "footnote",

            // Special Elements
            LayoutElementType::Seal => "seal",
            LayoutElementType::Number => "number",
            LayoutElementType::Reference => "reference",
            LayoutElementType::ReferenceContent => "reference_content",
            LayoutElementType::Algorithm => "algorithm",
            LayoutElementType::FormulaNumber => "formula_number",
            LayoutElementType::AsideText => "aside_text",
            LayoutElementType::List => "list",

            // Region (PP-DocBlockLayout)
            LayoutElementType::Region => "region",

            // Fallback
            LayoutElementType::Other => "other",
        }
    }

    /// Creates a LayoutElementType from a string label with fine-grained mapping.
    ///
    /// This method maps model output labels to their corresponding fine-grained types,
    /// preserving the full PP-StructureV3 label set (20/23 classes).
    pub fn from_label(label: &str) -> Self {
        match label.to_lowercase().as_str() {
            // Document Structure
            "doc_title" => LayoutElementType::DocTitle,
            "paragraph_title" | "title" => LayoutElementType::ParagraphTitle,
            "text" | "paragraph" => LayoutElementType::Text,
            "content" => LayoutElementType::Content,
            "abstract" => LayoutElementType::Abstract,

            // Visual Elements
            "image" | "figure" => LayoutElementType::Image,
            "table" => LayoutElementType::Table,
            "chart" | "flowchart" => LayoutElementType::Chart,
            "formula" | "equation" => LayoutElementType::Formula,

            // Captions
            "figure_title" => LayoutElementType::FigureTitle,
            "table_title" => LayoutElementType::TableTitle,
            "chart_title" => LayoutElementType::ChartTitle,
            "figure_table_chart_title" | "caption" => LayoutElementType::FigureTableChartTitle,

            // Page Structure
            "header" => LayoutElementType::Header,
            "header_image" => LayoutElementType::HeaderImage,
            "footer" => LayoutElementType::Footer,
            "footer_image" => LayoutElementType::FooterImage,
            "footnote" => LayoutElementType::Footnote,

            // Special Elements
            "seal" => LayoutElementType::Seal,
            "number" => LayoutElementType::Number,
            "reference" => LayoutElementType::Reference,
            "reference_content" => LayoutElementType::ReferenceContent,
            "algorithm" => LayoutElementType::Algorithm,
            "formula_number" => LayoutElementType::FormulaNumber,
            "aside_text" => LayoutElementType::AsideText,
            "list" => LayoutElementType::List,

            // Region (PP-DocBlockLayout)
            "region" => LayoutElementType::Region,

            // Everything else maps to Other
            // The original label is preserved in LayoutElement.label
            _ => LayoutElementType::Other,
        }
    }

    /// Returns the semantic category for this element type.
    ///
    /// This method groups fine-grained types into broader semantic categories,
    /// useful for processing logic that doesn't need fine-grained distinctions.
    ///
    /// # Categories
    ///
    /// - **Title**: DocTitle, ParagraphTitle
    /// - **Text**: Text, Content, Abstract
    /// - **Visual**: Image, Chart
    /// - **Table**: Table
    /// - **Caption**: FigureTitle, TableTitle, ChartTitle, FigureTableChartTitle
    /// - **Header**: Header, HeaderImage
    /// - **Footer**: Footer, FooterImage, Footnote
    /// - **Formula**: Formula, FormulaNumber
    /// - **Special**: Seal, Number, Reference, ReferenceContent, Algorithm, AsideText
    /// - **List**: List
    /// - **Other**: Other
    pub fn semantic_category(&self) -> &'static str {
        match self {
            // Title category
            LayoutElementType::DocTitle | LayoutElementType::ParagraphTitle => "title",

            // Text category
            LayoutElementType::Text | LayoutElementType::Content | LayoutElementType::Abstract => {
                "text"
            }

            // Visual category
            LayoutElementType::Image | LayoutElementType::Chart => "visual",

            // Table category
            LayoutElementType::Table => "table",

            // Caption category
            LayoutElementType::FigureTitle
            | LayoutElementType::TableTitle
            | LayoutElementType::ChartTitle
            | LayoutElementType::FigureTableChartTitle => "caption",

            // Header category
            LayoutElementType::Header | LayoutElementType::HeaderImage => "header",

            // Footer category
            LayoutElementType::Footer
            | LayoutElementType::FooterImage
            | LayoutElementType::Footnote => "footer",

            // Formula category
            LayoutElementType::Formula | LayoutElementType::FormulaNumber => "formula",

            // Special category
            LayoutElementType::Seal
            | LayoutElementType::Number
            | LayoutElementType::Reference
            | LayoutElementType::ReferenceContent
            | LayoutElementType::Algorithm
            | LayoutElementType::AsideText => "special",

            // List category
            LayoutElementType::List => "list",

            // Region category (PP-DocBlockLayout)
            LayoutElementType::Region => "region",

            // Other
            LayoutElementType::Other => "other",
        }
    }

    /// Returns whether this element type is a title variant.
    pub fn is_title(&self) -> bool {
        matches!(
            self,
            LayoutElementType::DocTitle | LayoutElementType::ParagraphTitle
        )
    }

    /// Returns whether this element type is a visual element (image, chart, figure).
    pub fn is_visual(&self) -> bool {
        matches!(self, LayoutElementType::Image | LayoutElementType::Chart)
    }

    /// Returns whether this element type is a caption variant.
    pub fn is_caption(&self) -> bool {
        matches!(
            self,
            LayoutElementType::FigureTitle
                | LayoutElementType::TableTitle
                | LayoutElementType::ChartTitle
                | LayoutElementType::FigureTableChartTitle
        )
    }

    /// Returns whether this element type is a header variant.
    pub fn is_header(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Header | LayoutElementType::HeaderImage
        )
    }

    /// Returns whether this element type is a footer variant.
    pub fn is_footer(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Footer
                | LayoutElementType::FooterImage
                | LayoutElementType::Footnote
        )
    }

    /// Returns whether this element type is a formula variant.
    pub fn is_formula(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Formula | LayoutElementType::FormulaNumber
        )
    }

    /// Returns whether this element type contains text content that should be OCR'd.
    pub fn should_ocr(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Text
                | LayoutElementType::Content
                | LayoutElementType::Abstract
                | LayoutElementType::DocTitle
                | LayoutElementType::ParagraphTitle
                | LayoutElementType::FigureTitle
                | LayoutElementType::TableTitle
                | LayoutElementType::ChartTitle
                | LayoutElementType::FigureTableChartTitle
                | LayoutElementType::Header
                | LayoutElementType::HeaderImage
                | LayoutElementType::Footer
                | LayoutElementType::FooterImage
                | LayoutElementType::Footnote
                | LayoutElementType::Reference
                | LayoutElementType::ReferenceContent
                | LayoutElementType::Algorithm
                | LayoutElementType::AsideText
                | LayoutElementType::List
                | LayoutElementType::Number
        )
    }
}

/// Removes heavily-overlapping layout elements in-place.
///
/// This mirrors PP-Structure-style overlap suppression where text takes priority over images.
/// Returns the number of elements removed.
pub fn remove_overlapping_layout_elements(
    layout_elements: &mut Vec<LayoutElement>,
    overlap_threshold: f32,
) -> usize {
    use std::collections::HashSet;

    if layout_elements.len() <= 1 {
        return 0;
    }

    let bboxes: Vec<_> = layout_elements.iter().map(|e| e.bbox.clone()).collect();
    let labels: Vec<&str> = layout_elements
        .iter()
        .map(|e| e.element_type.as_str())
        .collect();

    let remove_indices =
        crate::processors::get_overlap_removal_indices(&bboxes, &labels, overlap_threshold);
    if remove_indices.is_empty() {
        return 0;
    }

    let remove_set: HashSet<usize> = remove_indices.into_iter().collect();
    let before = layout_elements.len();

    let mut idx = 0;
    layout_elements.retain(|_| {
        let keep = !remove_set.contains(&idx);
        idx += 1;
        keep
    });

    before.saturating_sub(layout_elements.len())
}

/// Applies small, PP-Structure-style label fixes to layout elements.
///
/// This is intended to capture lightweight "glue" heuristics that shouldn't live in `predict`.
pub fn apply_standardized_layout_label_fixes(layout_elements: &mut [LayoutElement]) {
    if layout_elements.is_empty() {
        return;
    }

    let mut footnote_indices: Vec<usize> = Vec::new();
    let mut paragraph_title_indices: Vec<usize> = Vec::new();
    let mut bottom_text_y_max: f32 = 0.0;
    let mut max_block_area: f32 = 0.0;
    let mut doc_title_num: usize = 0;

    for (idx, elem) in layout_elements.iter().enumerate() {
        let area =
            (elem.bbox.x_max() - elem.bbox.x_min()) * (elem.bbox.y_max() - elem.bbox.y_min());
        max_block_area = max_block_area.max(area);

        match elem.element_type {
            LayoutElementType::Footnote => footnote_indices.push(idx),
            LayoutElementType::ParagraphTitle => paragraph_title_indices.push(idx),
            LayoutElementType::Text => {
                bottom_text_y_max = bottom_text_y_max.max(elem.bbox.y_max());
            }
            LayoutElementType::DocTitle => doc_title_num += 1,
            _ => {}
        }
    }

    for idx in footnote_indices {
        if layout_elements[idx].bbox.y_max() < bottom_text_y_max {
            layout_elements[idx].element_type = LayoutElementType::Text;
            layout_elements[idx].label = Some("text".to_string());
        }
    }

    let only_one_paragraph_title = paragraph_title_indices.len() == 1 && doc_title_num == 0;
    if only_one_paragraph_title {
        let idx = paragraph_title_indices[0];
        let area = (layout_elements[idx].bbox.x_max() - layout_elements[idx].bbox.x_min())
            * (layout_elements[idx].bbox.y_max() - layout_elements[idx].bbox.y_min());

        let title_area_ratio_threshold = 0.3f32;
        if area > max_block_area * title_area_ratio_threshold {
            layout_elements[idx].element_type = LayoutElementType::DocTitle;
            layout_elements[idx].label = Some("doc_title".to_string());
        }
    }
}

/// Result of table recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableResult {
    /// Bounding box of the table in the original image
    pub bbox: BoundingBox,
    /// Table type (wired or wireless)
    pub table_type: TableType,
    /// Confidence score for table type classification (None if classifier wasn't configured/run)
    pub classification_confidence: Option<f32>,
    /// Confidence score for table structure recognition (None if structure recognition failed)
    pub structure_confidence: Option<f32>,
    /// Detected table cells
    pub cells: Vec<TableCell>,
    /// HTML structure of the table (if available)
    pub html_structure: Option<String>,
    /// OCR text content for each cell (if OCR was integrated)
    pub cell_texts: Option<Vec<Option<String>>>,
    /// Structure tokens from table structure recognition (used for HTML generation after stitching)
    #[serde(skip)]
    pub structure_tokens: Option<Vec<String>>,
}

impl TableResult {
    /// Creates a new table result.
    pub fn new(bbox: BoundingBox, table_type: TableType) -> Self {
        Self {
            bbox,
            table_type,
            classification_confidence: None,
            structure_confidence: None,
            cells: Vec::new(),
            html_structure: None,
            cell_texts: None,
            structure_tokens: None,
        }
    }

    /// Sets the classification confidence.
    pub fn with_classification_confidence(mut self, confidence: f32) -> Self {
        self.classification_confidence = Some(confidence);
        self
    }

    /// Sets the structure recognition confidence.
    pub fn with_structure_confidence(mut self, confidence: f32) -> Self {
        self.structure_confidence = Some(confidence);
        self
    }

    /// Sets the table cells.
    pub fn with_cells(mut self, cells: Vec<TableCell>) -> Self {
        self.cells = cells;
        self
    }

    /// Sets the HTML structure.
    pub fn with_html_structure(mut self, html: impl Into<String>) -> Self {
        self.html_structure = Some(html.into());
        self
    }

    /// Sets the cell texts from OCR.
    pub fn with_cell_texts(mut self, texts: Vec<Option<String>>) -> Self {
        self.cell_texts = Some(texts);
        self
    }

    /// Sets the structure tokens for later HTML generation.
    pub fn with_structure_tokens(mut self, tokens: Vec<String>) -> Self {
        self.structure_tokens = Some(tokens);
        self
    }

    /// Returns the best available confidence score for this table.
    ///
    /// This method provides a unified confidence API for callers who want to filter
    /// tables by confidence without caring whether classification or structure
    /// recognition was used. Priority:
    /// 1. If both classification and structure confidence are available, returns
    ///    the minimum (most conservative estimate)
    /// 2. If only structure confidence is available (common when classifier isn't
    ///    configured), returns that
    /// 3. If only classification confidence is available, returns that
    /// 4. Returns `None` only if neither confidence is available (stub result)
    pub fn confidence(&self) -> Option<f32> {
        match (self.classification_confidence, self.structure_confidence) {
            (Some(cls), Some(str)) => Some(cls.min(str)),
            (None, Some(str)) => Some(str),
            (Some(cls), None) => Some(cls),
            (None, None) => None,
        }
    }

    /// Returns true if this table has valid structure data.
    ///
    /// A table is considered valid if it has either cells or an HTML structure.
    /// Stub results (created when structure recognition fails) will return false.
    pub fn has_structure(&self) -> bool {
        !self.cells.is_empty() || self.html_structure.is_some()
    }
}

/// Type of table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableType {
    /// Table with visible borders
    Wired,
    /// Table without visible borders
    Wireless,
    /// Unknown table type
    Unknown,
}

/// A cell in a table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    /// Bounding box of the cell
    pub bbox: BoundingBox,
    /// Row index (0-based)
    pub row: Option<usize>,
    /// Column index (0-based)
    pub col: Option<usize>,
    /// Row span
    pub row_span: Option<usize>,
    /// Column span
    pub col_span: Option<usize>,
    /// Confidence score for the cell detection
    pub confidence: f32,
    /// Text content of the cell (if available)
    pub text: Option<String>,
}

impl TableCell {
    /// Creates a new table cell.
    pub fn new(bbox: BoundingBox, confidence: f32) -> Self {
        Self {
            bbox,
            row: None,
            col: None,
            row_span: None,
            col_span: None,
            confidence,
            text: None,
        }
    }

    /// Sets the row and column indices.
    pub fn with_position(mut self, row: usize, col: usize) -> Self {
        self.row = Some(row);
        self.col = Some(col);
        self
    }

    /// Sets the row and column spans.
    pub fn with_span(mut self, row_span: usize, col_span: usize) -> Self {
        self.row_span = Some(row_span);
        self.col_span = Some(col_span);
        self
    }

    /// Sets the text content.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }
}

/// Result of formula recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaResult {
    /// Bounding box of the formula in the original image
    pub bbox: BoundingBox,
    /// LaTeX representation of the formula
    pub latex: String,
    /// Confidence score for the recognition
    pub confidence: f32,
}

impl FormulaResult {
    /// Creates a new formula result.
    pub fn new(bbox: BoundingBox, latex: impl Into<String>, confidence: f32) -> Self {
        Self {
            bbox,
            latex: latex.into(),
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_result_creation() {
        let result = StructureResult::new("test.jpg", 0);
        assert_eq!(result.input_path.as_ref(), "test.jpg");
        assert_eq!(result.index, 0);
        assert!(result.layout_elements.is_empty());
        assert!(result.tables.is_empty());
        assert!(result.formulas.is_empty());
        assert!(result.text_regions.is_none());
    }

    #[test]
    fn test_layout_element_type_as_str() {
        assert_eq!(LayoutElementType::Text.as_str(), "text");
        assert_eq!(LayoutElementType::Table.as_str(), "table");
        assert_eq!(LayoutElementType::Formula.as_str(), "formula");
    }

    #[test]
    fn test_table_result_creation() {
        let bbox = BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0);
        let table = TableResult::new(bbox, TableType::Wired);
        assert_eq!(table.table_type, TableType::Wired);
        assert!(table.cells.is_empty());
        assert!(table.html_structure.is_none());
    }

    #[test]
    fn test_structure_result_export() {
        let bbox = BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0);
        let mut result = StructureResult::new("test.jpg", 0);

        let title = LayoutElement::new(bbox.clone(), LayoutElementType::DocTitle, 1.0)
            .with_text("Test Document");

        let text =
            LayoutElement::new(bbox.clone(), LayoutElementType::Text, 1.0).with_text("Hello world");

        result = result.with_layout_elements(vec![title, text]);

        let md = result.to_markdown();
        assert!(md.contains("# Test Document"));
        assert!(md.contains("Hello world"));

        let html = result.to_html();
        assert!(html.contains("<h1>Test Document</h1>"));
        assert!(html.contains("<p>Hello world</p>"));
    }
}
