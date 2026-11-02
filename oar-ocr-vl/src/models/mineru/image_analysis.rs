//! Tagged replies from the MinerU Image Analysis prompt.
// Adapted from mineru-vl-utils post_process/{__init__,image_analysis_postprocess}.py,
// Apache-2.0, commit c9797cde8f54cc48306196fc6aebe73ea31dd4a3.

use crate::render::table::{OtslOptions, convert_otsl_to_html_with_options};

/// Returns the block type and content. Like the reference, a reply without a
/// class becomes an image with its caption, which is usually empty.
pub(super) fn parse_reply(raw: &str, table_options: &OtslOptions) -> (String, String) {
    let class = field(raw, "class").to_lowercase();
    let caption = field(raw, "caption");
    let mut content = field(raw, "content").to_string();
    if content.is_empty()
        && raw.matches("<|content_start|>").count() == 1
        && !raw.contains("<|content_end|>")
    {
        content = raw
            .split_once("<|content_start|>")
            .map_or("", |(_, tail)| tail)
            .into();
    }
    match class.as_str() {
        "pure_table" => {
            let trimmed = content.trim();
            let html = if trimmed.to_lowercase().starts_with("<table")
                && trimmed.to_lowercase().ends_with("</table>")
            {
                trimmed.into()
            } else if let Some(html) = crate::render::table::markdown_table_to_html(&content) {
                html
            } else if ["<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>"]
                .iter()
                .any(|token| content.contains(token))
            {
                convert_otsl_to_html_with_options(&content, table_options)
            } else {
                String::new()
            };
            ("table".into(), wrap_table_formulas(&html))
        }
        "pure_formula" => ("equation".into(), content),
        "chart" => {
            content = crate::render::table::repair_chart_table(&content);
            if crate::render::table::malformed_markdown_table(&content) {
                content.clear();
            }
            ("chart".into(), content)
        }
        _ => {
            if class == "chemical" {
                content.clear();
            } else if class == "flowchart" {
                content = crate::render::text::format_mermaid(&content);
            }
            if class == "natural_image" || content.is_empty() {
                content = caption.into();
            }
            ("image".into(), content)
        }
    }
}

fn field<'a>(raw: &'a str, name: &str) -> &'a str {
    let start = format!("<|{name}_start|>");
    let end = format!("<|{name}_end|>");
    raw.split_once(&start)
        .and_then(|(_, tail)| tail.split_once(&end))
        .map_or("", |(text, _)| text.trim())
}

pub(super) fn wrap_table_formulas(content: &str) -> String {
    use crate::render::text::{BracketMatching, DelimiterOptions, normalize_math_delimiters};
    use regex::Regex;
    use std::sync::LazyLock;
    static EQ: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<eq>.*?</eq>").unwrap());
    let options = DelimiterOptions {
        matching: BracketMatching::NonemptySpans,
        inline: ("<eq>", "</eq>"),
        display: ("<eq>", "</eq>"),
        ..Default::default()
    };
    let mut output = String::new();
    let mut start = 0;
    for protected in EQ.find_iter(content) {
        output.push_str(&normalize_math_delimiters(
            &content[start..protected.start()],
            &options,
        ));
        output.push_str(protected.as_str());
        start = protected.end();
    }
    output.push_str(&normalize_math_delimiters(&content[start..], &options));
    output
}
