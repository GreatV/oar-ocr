//! Markdown rendering for the standalone VL document model.

use crate::document::structure::{LayoutElement, LayoutElementType};
use once_cell::sync::Lazy;
use regex::Regex;

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
    let mut markdown = String::new();
    for element in elements {
        let label = element.label.as_deref().unwrap_or("");
        if ignore_labels.iter().any(|ignored| ignored == label) {
            continue;
        }
        let content = element.text.as_deref().unwrap_or("");
        let formatted = match label {
            "paragraph_title" | "abstract_title" | "reference_title" | "content_title" => {
                format_title(content)
            }
            "doc_title" => format!("# {content}").replace("-\n", "").replace('\n', " "),
            "table_title" | "figure_title" | "chart_title" => {
                if pretty {
                    centered_html(content)
                } else {
                    content.to_string()
                }
            }
            "text" | "ocr" | "vertical_text" | "reference_content" => text_block(content),
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
            "table" => {
                if pretty {
                    format!("\n{}", centered_table(content))
                } else {
                    format!("\n{content}")
                        .replace("<html>", "")
                        .replace("</html>", "")
                        .replace("<body>", "")
                        .replace("</body>", "")
                }
            }
            "formula" | "display_formula" | "inline_formula" => content.to_string(),
            "algorithm" => content.trim_matches('\n').to_string(),
            _ => match element.element_type {
                LayoutElementType::ParagraphTitle => format_title(content),
                LayoutElementType::DocTitle => {
                    format!("# {content}").replace("-\n", "").replace('\n', " ")
                }
                LayoutElementType::Table if pretty => format!("\n{}", centered_table(content)),
                _ => content.to_string(),
            },
        };
        if !markdown.is_empty() {
            markdown.push_str("\n\n");
        }
        markdown.push_str(&formatted);
    }
    markdown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_numbering_controls_heading_depth() {
        assert_eq!(format_title("Learning Curve"), "## Learning Curve");
        assert_eq!(format_title("1.2Title"), "### 1.2 Title");
        assert_eq!(format_title("1.2.3 Section"), "#### 1.2.3 Section");
        assert_eq!(format_title("I. Introduction"), "### I. Introduction");
    }
}
