//! Parameters used by the common Markdown renderer.

/// Formatting choices for recognized document regions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RenderStyle {
    /// Fixed section-heading depth; None derives depth from numbering.
    pub section_heading_level: Option<usize>,
    /// Text placed before a caption.
    pub caption_prefix: String,
    /// Text placed after a caption.
    pub caption_suffix: String,
    /// Apply title prefixes to each title line.
    pub per_line_titles: bool,
    /// Collapse soft title line breaks into spaces.
    pub collapse_title_newlines: bool,
    /// Separate prose lines with blank lines.
    pub normalize_text_newlines: bool,
    /// Normalize vision-footnote paragraph spacing.
    pub normalize_vision_footnotes: bool,
    /// Normalize leading bullet and number markers.
    pub normalize_bullets: bool,
    /// Join eligible hyphenated text across regions.
    pub merge_hyphenated_text: bool,
    /// Render footnotes as numbered Markdown definitions.
    pub footnote_definitions: bool,
    /// Image placeholder containing optional {index} and {label} fields.
    pub image_template: Option<String>,
    /// Skip regions whose formatted content is empty.
    pub skip_empty_chunks: bool,
    /// Remove existing heading and list markers before adding title prefixes.
    pub strip_title_markers: bool,
    /// Add a newline before table HTML.
    pub table_leading_newline: bool,
    /// Remove outer html/body tags from plain table output.
    pub strip_table_envelope: bool,
    /// Derive abstract and reference headings from section templates.
    pub section_templates: bool,
}

impl Default for RenderStyle {
    fn default() -> Self {
        Self {
            section_heading_level: None,
            caption_prefix: String::new(),
            caption_suffix: String::new(),
            per_line_titles: false,
            collapse_title_newlines: true,
            normalize_text_newlines: true,
            normalize_vision_footnotes: false,
            normalize_bullets: false,
            merge_hyphenated_text: false,
            footnote_definitions: false,
            image_template: None,
            skip_empty_chunks: false,
            strip_title_markers: false,
            table_leading_newline: true,
            strip_table_envelope: true,
            section_templates: true,
        }
    }
}

impl RenderStyle {
    /// Set whether to collapse soft title line breaks.
    pub fn with_collapse_title_newlines(mut self, value: bool) -> Self {
        self.collapse_title_newlines = value;
        self
    }

    /// Set whether to remove existing title markers.
    pub fn with_strip_title_markers(mut self, value: bool) -> Self {
        self.strip_title_markers = value;
        self
    }

    /// Set whether to remove outer html/body tags from plain tables.
    pub fn with_strip_table_envelope(mut self, value: bool) -> Self {
        self.strip_table_envelope = value;
        self
    }

    /// Set the newline before table HTML.
    pub fn with_table_leading_newline(mut self, value: bool) -> Self {
        self.table_leading_newline = value;
        self
    }

    /// Set the abstract and reference section templates.
    pub fn with_section_templates(mut self, value: bool) -> Self {
        self.section_templates = value;
        self
    }

    /// Set whether to skip empty formatted regions.
    pub fn with_skip_empty_chunks(mut self, value: bool) -> Self {
        self.skip_empty_chunks = value;
        self
    }

    /// Set the fixed section-heading depth; None derives depth from numbering.
    pub fn with_section_heading_level(mut self, value: Option<usize>) -> Self {
        self.section_heading_level = value;
        self
    }

    /// Set the text placed before a caption.
    pub fn with_caption_prefix(mut self, value: String) -> Self {
        self.caption_prefix = value;
        self
    }

    /// Set the text placed after a caption.
    pub fn with_caption_suffix(mut self, value: String) -> Self {
        self.caption_suffix = value;
        self
    }

    /// Set whether to apply title prefixes to each title line.
    pub fn with_per_line_titles(mut self, value: bool) -> Self {
        self.per_line_titles = value;
        self
    }

    /// Set whether to separate prose lines with blank lines.
    pub fn with_normalize_text_newlines(mut self, value: bool) -> Self {
        self.normalize_text_newlines = value;
        self
    }

    /// Set whether to normalize vision-footnote paragraph spacing.
    pub fn with_normalize_vision_footnotes(mut self, value: bool) -> Self {
        self.normalize_vision_footnotes = value;
        self
    }

    /// Set whether to normalize leading bullet and number markers.
    pub fn with_normalize_bullets(mut self, value: bool) -> Self {
        self.normalize_bullets = value;
        self
    }

    /// Set whether to join eligible hyphenated text across regions.
    pub fn with_merge_hyphenated_text(mut self, value: bool) -> Self {
        self.merge_hyphenated_text = value;
        self
    }

    /// Set whether to render footnotes as numbered Markdown definitions.
    pub fn with_footnote_definitions(mut self, value: bool) -> Self {
        self.footnote_definitions = value;
        self
    }

    /// Set the image placeholder containing optional {index} and {label} fields.
    pub fn with_image_template(mut self, value: Option<String>) -> Self {
        self.image_template = value;
        self
    }
}
