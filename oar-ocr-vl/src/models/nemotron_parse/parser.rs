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

/// Recover the checkpoint's decimal coordinates for pixel export. The public
/// block stores normalized f32 values while the raw stream preserves every digit.
pub(crate) fn source_boxes(raw: &str) -> Option<impl Iterator<Item = (&str, [f64; 4])>> {
    let elements = ELEMENT.captures_iter(raw.strip_prefix(super::PROMPT)?);
    Some(elements.map(|fields| (fields.get(6).unwrap().as_str(), coordinates(&fields))))
}

pub(super) fn parse(raw: &str) -> PageDocument {
    let repaired =
        if raw.contains("```") || raw.contains("<class_Code>") || raw.contains("<class_Formula>") {
            std::borrow::Cow::Borrowed(raw)
        } else {
            PREFIX.replace_all(raw, "$1")
        };
    let mut page = PageDocument::default();
    let mut seen = HashSet::new();
    let mut fragments = Vec::new();
    let mut context = Vec::new();
    let mut cleaned = String::with_capacity(repaired.len());
    let mut consumed = 0;
    for fields in ELEMENT.captures_iter(&repaired) {
        let field = |index| fields.get(index).unwrap().as_str();
        let coords = coordinates(&fields);
        let (body, class) = (field(3), field(6));
        let element = fields.get(0).unwrap();
        cleaned.push_str(&repaired[consumed..element.start()]);
        consumed = element.end();
        if !seen.insert((coords.map(f64::to_bits), body, class)) {
            continue;
        }
        cleaned.push_str(element.as_str());
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
    cleaned.push_str(&repaired[consumed..]);
    native_chart::relocate(&context, &mut fragments);
    let markdown = if page.blocks.is_empty() {
        latex::to_html(
            MARKERS
                .replace_all(&cleaned, "")
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
    page.raw_output = Some(cleaned);
    page
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let page = parse(&format!("<x_0><y_0><class_Title>{block}{block}"));
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.blocks[0].bbox, [0.1, 0.2, 0.9, 0.4]);
        assert_eq!(page.markdown.as_deref(), Some("# Hello"));
        assert_eq!(page.raw_output.as_deref(), Some(block));
    }
}
