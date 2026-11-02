//! Markdown rendering for the standalone VL document model.

use crate::document::structure::{LayoutElement, LayoutElementType};
use once_cell::sync::Lazy;
use regex::Regex;

pub(crate) mod assembly;
mod style;
pub use style::RenderStyle;

/// Labels omitted from the default Markdown representation.
pub const DEFAULT_MARKDOWN_IGNORE_LABELS: [&str; 8] = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
    "formula_number",
];

/// Return the default ignored labels as owned strings.
pub fn default_markdown_ignore_labels() -> Vec<String> {
    DEFAULT_MARKDOWN_IGNORE_LABELS
        .iter()
        .map(|label| (*label).to_string())
        .collect()
}

static TITLE_NUMBERING_PATTERN: Lazy<Result<Regex, regex::Error>> = Lazy::new(|| {
    Regex::new(
        r"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[.、]?|[(（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[)）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)$",
    )
});

pub(crate) fn format_title(text: &str) -> String {
    let mut title = text.to_string();
    if let Ok(re) = TITLE_NUMBERING_PATTERN.as_ref()
        && let Some(caps) = re.captures(&title)
    {
        let numbering = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
        let content = caps.get(3).map(|m| m.as_str()).unwrap_or("").trim_start();
        if !numbering.is_empty() {
            title = format!("{numbering} {content}");
        }
    }

    title = title.trim_end_matches('.').to_string();
    let level = if title.contains('.') {
        title.chars().filter(|&c| c == '.').count() + 1
    } else {
        1
    };
    format!("{} {}", "#".repeat(level + 1), title)
        .replace("-\n", "")
        .replace('\n', " ")
}

fn centered_html(text: &str) -> String {
    let content = text.replace("-\n", "").replace('\n', " ");
    format!("<div style=\"text-align: center;\">{content}</div>\n")
}

fn centered_table(html: &str) -> String {
    html.replace(
        "<table>",
        "<table border=1 style='margin: auto; word-wrap: break-word;'>",
    )
    .replace(
        "<th>",
        "<th style='text-align: center; word-wrap: break-word;'>",
    )
    .replace(
        "<td>",
        "<td style='text-align: center; word-wrap: break-word;'>",
    )
}

fn text_block(text: &str) -> String {
    text.replace("\n\n", "\n").replace('\n', "\n\n")
}

fn content_block(text: &str) -> String {
    text.replace("-\n", "  \n").replace('\n', "  \n")
}

fn format_first_line(
    text: &str,
    templates_lower: &[&str],
    format: impl Fn(&str) -> String,
    splitter: &str,
) -> String {
    let mut parts: Vec<String> = text.split(splitter).map(str::to_string).collect();
    for part in &mut parts {
        if part.trim().is_empty() {
            continue;
        }
        if templates_lower
            .iter()
            .any(|template| *template == part.to_lowercase())
        {
            *part = format(part);
        }
        break;
    }
    parts.join(splitter)
}

/// Render layout elements as Markdown using PaddleX-compatible formatting.
pub fn to_markdown(elements: &[LayoutElement], ignore_labels: &[String], pretty: bool) -> String {
    to_markdown_with_style(elements, ignore_labels, pretty, &RenderStyle::default())
}

pub(crate) fn to_markdown_with_style(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    pretty: bool,
    style: &RenderStyle,
) -> String {
    let mut markdown = String::new();
    let elements = assembly::prepare_text(elements, style.merge_hyphenated_text, None);
    let mut footnote = 0;
    for (index, element) in elements.iter().enumerate() {
        let label = element.label.as_deref().unwrap_or("");
        if ignore_labels.iter().any(|ignored| ignored == label) {
            continue;
        }
        let content = element.text.as_deref().unwrap_or("");
        let normalized;
        let content = if style.normalize_bullets
            && !matches!(
                element.element_type,
                LayoutElementType::Table
                    | LayoutElementType::Formula
                    | LayoutElementType::Image
                    | LayoutElementType::Chart
                    | LayoutElementType::DocTitle
                    | LayoutElementType::ParagraphTitle
            ) {
            normalized = assembly::normalize_bullet(content);
            normalized.as_str()
        } else {
            content
        };
        let image = matches!(
            label,
            "image" | "header_image" | "footer_image" | "chart" | "seal"
        );
        if style.skip_empty_chunks
            && content.trim().is_empty()
            && !(image && style.image_template.is_some())
        {
            continue;
        }
        let heading = |content: &str, document: bool| {
            let level = if document {
                Some(1)
            } else {
                style.section_heading_level
            };
            let clean = if style.strip_title_markers {
                let clean = if document {
                    content
                } else {
                    content
                        .strip_prefix("- ")
                        .or_else(|| content.strip_prefix("* "))
                        .unwrap_or(content)
                };
                clean.trim_start_matches('#').trim_start()
            } else {
                content
            };
            if let Some(level) = level {
                let prefix = "#".repeat(level.max(1));
                if style.per_line_titles {
                    clean
                        .lines()
                        .map(|line| format!("{prefix} {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else if style.collapse_title_newlines {
                    format!("{prefix} {clean}")
                        .replace("-\n", "")
                        .replace('\n', " ")
                } else {
                    format!("{prefix} {clean}")
                }
            } else {
                format_title(clean)
            }
        };
        let table = |content: &str| {
            let html = if pretty {
                centered_table(content)
            } else if style.strip_table_envelope {
                content
                    .replace("<html>", "")
                    .replace("</html>", "")
                    .replace("<body>", "")
                    .replace("</body>", "")
            } else {
                content.to_string()
            };
            if style.table_leading_newline {
                format!("\n{html}")
            } else {
                html
            }
        };
        let mut formatted = if style.footnote_definitions && label == "footnote" {
            footnote += 1;
            format!("[^{footnote}]: {}", content.replace('\n', "\n    "))
        } else if image && content.trim().is_empty() && style.image_template.is_some() {
            style
                .image_template
                .as_deref()
                .unwrap_or("")
                .replace("{index}", &(index + 1).to_string())
                .replace("{label}", label)
        } else {
            match label {
                "paragraph_title" | "abstract_title" | "reference_title" | "content_title" => {
                    heading(content, false)
                }
                "doc_title" => heading(content, true),
                "table_title" | "figure_title" | "chart_title" => {
                    if pretty {
                        centered_html(content)
                    } else {
                        format!(
                            "{}{}{}",
                            style.caption_prefix, content, style.caption_suffix
                        )
                    }
                }
                "text" | "ocr" | "vertical_text" | "reference_content" => {
                    if style.normalize_text_newlines && style.section_templates {
                        if style.normalize_bullets {
                            single_newlines(content)
                        } else {
                            text_block(content)
                        }
                    } else {
                        content.into()
                    }
                }
                "vision_footnote" if style.normalize_vision_footnotes => text_block(content),
                "abstract" | "reference" | "content" if !style.section_templates => content.into(),
                "abstract" => format_first_line(
                    content,
                    &["摘要", "abstract"],
                    |line| format!("## {line}\n"),
                    " ",
                ),
                "reference" => format_first_line(
                    content,
                    &["参考文献", "references"],
                    |line| format!("## {line}"),
                    "\n",
                ),
                "content" => content_block(content),
                "table" => table(content),
                "formula" | "display_formula" | "inline_formula" => content.to_string(),
                "algorithm" => content.trim_matches('\n').to_string(),
                _ => match element.element_type {
                    LayoutElementType::ParagraphTitle => heading(content, false),
                    LayoutElementType::DocTitle => heading(content, true),
                    LayoutElementType::Table if pretty => table(content),
                    _ => content.to_string(),
                },
            }
        };
        if !style.section_templates
            && style.normalize_text_newlines
            && !matches!(
                element.element_type,
                LayoutElementType::Table
                    | LayoutElementType::Formula
                    | LayoutElementType::Image
                    | LayoutElementType::HeaderImage
                    | LayoutElementType::FooterImage
                    | LayoutElementType::Chart
                    | LayoutElementType::FormulaNumber
            )
        {
            formatted = single_newlines(&formatted);
        }
        if style.skip_empty_chunks && formatted.trim().is_empty() {
            continue;
        }
        if !markdown.is_empty() {
            markdown.push_str("\n\n");
        }
        markdown.push_str(&formatted);
    }
    markdown
}

fn single_newlines(text: &str) -> String {
    let mut output = String::new();
    for (i, c) in text.char_indices() {
        output.push(c);
        if c == '\n'
            && (i == 0 || text.as_bytes()[i - 1] != b'\n')
            && text.as_bytes().get(i + 1) != Some(&b'\n')
        {
            output.push('\n');
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_numbering_controls_heading_depth() {
        let title = LayoutElement::new(
            crate::BoundingBox::from_coords(0., 0., 10., 10.),
            LayoutElementType::DocTitle,
            1.,
        )
        .with_label("doc_title")
        .with_text("First\nSecond");
        let per_line = RenderStyle::default()
            .with_per_line_titles(true)
            .with_skip_empty_chunks(true);
        assert_eq!(
            to_markdown_with_style(&[title], &[], false, &per_line),
            "# First\n# Second"
        );
        let footnote = LayoutElement::new(
            crate::BoundingBox::from_coords(0., 0., 10., 10.),
            LayoutElementType::Footnote,
            1.,
        )
        .with_label("footnote")
        .with_text("Body\ncontinued");
        assert_eq!(
            to_markdown_with_style(
                &[footnote],
                &[],
                false,
                &RenderStyle::default().with_footnote_definitions(true)
            ),
            "[^1]: Body\n    continued"
        );
        let caption = LayoutElement::new(
            crate::BoundingBox::from_coords(0., 0., 10., 10.),
            LayoutElementType::FigureTitle,
            1.,
        )
        .with_label("figure_title")
        .with_text("Caption");
        assert_eq!(
            to_markdown_with_style(
                &[caption],
                &[],
                false,
                &RenderStyle::default()
                    .with_caption_prefix("*".into())
                    .with_caption_suffix("*".into())
            ),
            "*Caption*"
        );
        let parts = ["Interoper-", "ability matters."].map(|text| {
            LayoutElement::new(
                crate::BoundingBox::from_coords(0., 0., 10., 10.),
                LayoutElementType::Text,
                1.,
            )
            .with_label("text")
            .with_text(text)
        });
        assert_eq!(
            to_markdown_with_style(
                &parts,
                &[],
                false,
                &RenderStyle::default().with_merge_hyphenated_text(true)
            ),
            "Interoperability matters."
        );
        assert_eq!(format_title("Learning Curve"), "## Learning Curve");
        assert_eq!(format_title("1.2Title"), "### 1.2 Title");
        assert_eq!(format_title("1.2.3 Section"), "#### 1.2.3 Section");
        assert_eq!(format_title("I. Introduction"), "### I. Introduction");
    }
}
