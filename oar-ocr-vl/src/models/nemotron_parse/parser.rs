//! Coordinate pairs enclose literal element bodies; the class follows the closing pair.
use crate::{
    DocumentBlock, PageDocument, ParseDiagnostic,
    render::{markdown::native_chart, table::latex, text::normalize_printed_text},
};
use regex::{Captures, Regex};
use std::{borrow::Cow, collections::HashSet, sync::LazyLock};

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

fn prepare_completion(text: &str) -> Cow<'_, str> {
    if text.contains("```") || text.contains("<class_Code>") || text.contains("<class_Formula>") {
        Cow::Borrowed(text)
    } else {
        PREFIX.replace_all(text, "$1")
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

pub(super) fn parse(raw: String, completion: &str) -> PageDocument {
    let completion = prepare_completion(completion);
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
    page.raw_output = Some(raw);
    page
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron_parse::PROMPT;

    #[test]
    fn empty_completion_renders_empty_markdown() {
        use crate::nemotron_parse::{Generation, parse_generation};
        use tokenizers::Tokenizer;

        // A subset of the checkpoint's byte-level vocabulary, including ordinary HTML characters.
        let tokenizer = Tokenizer::from_bytes(br#"{
            "model":{"type":"BPE","vocab":{"<s>":0,"</s>":2,"<predict_bbox>":50004,"<":50,"s":105,">":52,"x":110,"/":37},"merges":[]},
            "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}
        }"#).unwrap();
        let generation = Generation {
            bos_token_id: 0,
            eos_token_id: 2,
            max_new_tokens: 32,
            repetition_penalty: 1.0,
        };
        let prompt = [2, 0, 50004];
        for (completion, markdown, raw_suffix) in [
            (&[2][..], "", "</s>"),
            (
                &[0, 50, 105, 52, 110, 50, 37, 105, 52, 2],
                "<s>x</s>",
                "<s><s>x</s></s>",
            ),
        ] {
            let ids = [prompt.as_slice(), completion].concat();
            let page = parse_generation(&tokenizer, &ids, prompt.len(), &generation).unwrap();
            assert_eq!(page.markdown.as_deref(), Some(markdown));
            assert!(page.blocks.is_empty());
            assert_eq!(
                page.raw_output.unwrap(),
                format!("</s><s><predict_bbox>{raw_suffix}")
            );
        }
    }

    #[test]
    fn pixel_export_uses_decimal_source_and_honors_edited_blocks() {
        let completion = "<x_0.9746><y_1.2>caption<x_0.0234><y_0.0123><class_Caption>";
        let mut page = parse(format!("{PROMPT}{completion}</s>"), completion);
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
        let completion = format!("<x_0><y_0><class_Title>{block}{block}");
        let raw = format!("{PROMPT}{completion}</s>");
        let page = parse(raw.clone(), &completion);
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(page.blocks[0].bbox, [0.1, 0.2, 0.9, 0.4]);
        assert_eq!(page.markdown.as_deref(), Some("# Hello"));
        assert_eq!(page.raw_output.as_deref(), Some(raw.as_str()));
        assert_eq!(source_boxes(&raw).unwrap().count(), 1);
    }
}
