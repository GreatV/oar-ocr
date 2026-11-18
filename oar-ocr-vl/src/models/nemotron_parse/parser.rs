//! Coordinate pairs enclose literal element bodies; the class follows the closing pair.
use crate::{
    DocumentBlock, PageDocument, ParseDiagnostic,
    render::{markdown::native_chart, table::latex, text::normalize_printed_text},
};
use regex::{Captures, Regex};
use std::{collections::HashSet, sync::LazyLock};

static ELEMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<x_([0-9]+(?:\.[0-9]+)?)><y_([0-9]+(?:\.[0-9]+)?)>(.*?)<x_([0-9]+(?:\.[0-9]+)?)><y_([0-9]+(?:\.[0-9]+)?)><class_([^>]+)>").unwrap()
});
static PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*<x_[0-9]+(?:\.[0-9]+)?><y_[0-9]+(?:\.[0-9]+)?><class_[A-Za-z][A-Za-z_-]*>\s*(<x_[0-9]+(?:\.[0-9]+)?><y_[0-9]+(?:\.[0-9]+)?>)").unwrap()
});
static MARKERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<(?:x|y|class)_[^>]+>").unwrap());

fn coordinates(fields: &Captures<'_>) -> [f64; 4] {
    [1, 2, 4, 5].map(|i| fields[i].parse::<f64>().expect("numeric coordinate token"))
}

fn prepare_completion(completion: &str) -> String {
    let text = completion.replace("<s>", "").replace("</s>", "");
    if text.contains("```") || text.contains("<class_Code>") || text.contains("<class_Formula>") {
        text
    } else {
        PREFIX.replace_all(&text, "$1").into_owned()
    }
}

fn elements(completion: &str) -> impl Iterator<Item = ([f64; 4], &str, &str)> {
    let mut seen = HashSet::new();
    ELEMENT.captures_iter(completion).filter_map(move |fields| {
        let coords = coordinates(&fields);
        let body = fields.get(3).unwrap().as_str();
        let class = fields.get(6).unwrap().as_str();
        seen.insert((coords.map(f64::to_bits), body, class))
            .then_some((coords, body, class))
    })
}

/// Recover the checkpoint's decimal coordinates for pixel export. The public
/// block stores normalized f32 values while the raw stream preserves every digit.
pub(crate) fn source_boxes(raw: &str) -> Option<impl Iterator<Item = (String, [f64; 4])>> {
    let completion = prepare_completion(raw.strip_prefix(super::PROMPT)?);
    let boxes: Vec<_> = elements(&completion)
        .map(|(coords, _, class)| (class.to_owned(), coords))
        .collect();
    Some(boxes.into_iter())
}

pub(super) fn parse(raw: &str) -> PageDocument {
    let completion = prepare_completion(raw.strip_prefix(super::PROMPT).unwrap_or(raw));
    let mut page = PageDocument::default();
    let mut fragments = Vec::new();
    let mut context = Vec::new();
    for (coords, body, class) in elements(&completion) {
        // The block export strips unknown-token markers before typography cleanup.
        let plain = body
            .replace("<tbc>", "")
            .replace(r"\<|unk|\>", "")
            .replace(r"\unknown", "");
        let content = normalize_printed_text(class, &plain);
        let content = match class {
            "Table" => latex::to_html(&content, true),
            "Chart" => native_chart::render(&content, None),
            _ => content,
        };
        if class == "Table" && content.contains(r"\begin{tabular}") {
            page.diagnostics.push(ParseDiagnostic {
                block_index: Some(page.blocks.len()),
                stage: "table".into(),
                message: "Table markup could not be fully converted; original content preserved."
                    .into(),
            });
        }
        let literal = normalize_printed_text(class, body).replace("<tbc>", "");
        let value = literal.trim();
        let markdown = match class {
            "Title" if !value.is_empty() && !value.starts_with('#') => format!("# {value}"),
            "Section-header" if !value.is_empty() && !value.starts_with('#') => {
                format!("## {value}")
            }
            "Table" => latex::to_html(value, true),
            "Chart" => native_chart::render(value, None),
            _ => value.to_owned(),
        };
        fragments.push(markdown);
        context.push(native_chart::Region {
            class,
            coords,
            body,
        });
        page.blocks.push(DocumentBlock {
            block_type: class.to_owned(),
            bbox: coords.map(|v| v as f32),
            angle: None,
            content: Some(content),
        });
    }
    native_chart::relocate(&context, &mut fragments);
    let markdown = if page.blocks.is_empty() {
        latex::to_html(
            MARKERS
                .replace_all(&completion, "")
                .replace("<tbc>", "")
                .trim(),
            true,
        )
    } else {
        fragments
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    page.markdown = Some(latex::promote_html_headers(&markdown).into_owned());
    page.raw_output = Some(raw.to_owned());
    page
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron_parse::PROMPT;

    #[test]
    fn empty_completion_renders_empty_markdown() {
        let raw = format!("{PROMPT}</s>");
        let page = parse(&raw);
        assert_eq!(page.markdown.as_deref(), Some(""));
        assert!(page.blocks.is_empty());
        assert_eq!(page.raw_output.as_deref(), Some(raw.as_str()));
    }

    #[test]
    fn pixel_export_uses_decimal_source_and_honors_edited_blocks() {
        let raw = format!(
            "{}<x_0.9746><y_1.2>caption<x_0.0234><y_0.0123><class_Caption>",
            super::super::PROMPT
        );
        let mut page = parse(&raw);
        let bbox = |doc: &PageDocument| doc.to_json(1931, 1073)["blocks"][0]["bbox"].clone();
        assert_eq!(
            bbox(&page),
            serde_json::json!([0.0234 * 1931., 0.0123 * 1073., 0.9746 * 1931., 1073.])
        );
        page.blocks[0].bbox = [0., 0., 1., 1.];
        assert_eq!(bbox(&page), serde_json::json!([0., 0., 1931., 1073.]));
    }

    #[test]
    fn repair_empty_prefix_and_deduplicate_exact_elements() {
        let block = "<x_0.1><y_0.2>Hello<x_0.9><y_0.4><class_Title>";
        let raw = format!("{PROMPT}<x_0><y_0><class_Title>{block}{block}</s>");
        let page = parse(&raw);
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.blocks[0].bbox, [0.1, 0.2, 0.9, 0.4]);
        assert_eq!(page.markdown.as_deref(), Some("# Hello"));
        assert_eq!(page.raw_output.as_deref(), Some(raw.as_str()));
        assert_eq!(source_boxes(&raw).unwrap().count(), 1);
    }
}
