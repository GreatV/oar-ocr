//! Complete-page parser adapter for HPD-Parsing.

use super::{HpdGenerationConfig, HpdParsing};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

impl PageParser for HpdParsing {
    type Options = HpdGenerationConfig;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let raw = self
            .parse(std::slice::from_ref(image), options)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("HPD-Parsing returned no page result"))??;
        Ok(parse_output(raw, self.reference_inline_formulas))
    }
}

fn parse_output(raw: String, reference_inline_formulas: bool) -> PageDocument {
    use crate::{DocumentBlock, ParseDiagnostic};
    use regex::Regex;
    use std::sync::LazyLock;
    static HEADER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^\s*([a-zA-Z_]+)\s*\[\s*([-\d.,\s]+)\]").unwrap());
    let mut page = PageDocument::default();
    for segment in raw.split("<BLOCK>").skip(1) {
        let (header, content) = segment.split_once("<CHILD>").unwrap_or((segment, ""));
        let block_type = header
            .trim_start()
            .chars()
            .take_while(|ch| ch.is_ascii_alphabetic() || *ch == '_')
            .collect::<String>()
            .to_lowercase();
        let coords = HEADER.captures(header).and_then(|caps| {
            caps[2]
                .split(',')
                .map(|value| value.trim().parse::<f32>().ok())
                .collect::<Option<Vec<_>>>()
        });
        let mut bbox = [0.0, 0.0, 1.0, 1.0];
        let issue = match coords.as_deref() {
            Some([x1, y1, x2, y2])
                if [x1, y1, x2, y2].iter().all(|v| v.is_finite()) && x1 <= x2 && y1 <= y2 =>
            {
                // The official demo's hpd_postprocess.py documents this scale
                // from real inference and uses it to draw boxes on the source image:
                // https://huggingface.co/spaces/PaddlePaddle/hpd-parsing/blob/2d114ffbdb837944262fe9135cf480578259621d/hpd_postprocess.py
                let normalized = [x1, y1, x2, y2].map(|v| v / 1000.0);
                bbox = normalized.map(|v| v.clamp(0.0, 1.0));
                (bbox != normalized).then_some("Clamped HPD-Parsing bbox to the page")
            }
            _ => Some("Invalid HPD-Parsing header or bbox; retained content with a full-page bbox"),
        };
        if let Some(message) = issue {
            page.diagnostics.push(ParseDiagnostic {
                block_index: Some(page.blocks.len()),
                stage: "postprocess".into(),
                message: message.into(),
            });
        }
        page.blocks.push(DocumentBlock {
            block_type: if block_type.is_empty() {
                "text".into()
            } else {
                block_type
            },
            bbox,
            angle: None,
            content: Some(clean_content(content, reference_inline_formulas)),
        });
    }
    if !page.blocks.is_empty() {
        // The reference emits content verbatim in reading order, including headers,
        // but excludes chart and seal transcripts from Markdown.
        page.markdown = Some(
            page.blocks
                .iter()
                .filter(|block| !matches!(block.block_type.as_str(), "chart" | "seal"))
                .filter_map(|block| block.content.as_deref())
                .filter(|content| !content.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
    }
    page.raw_output = Some(raw);
    page
}

fn clean_content(content: &str, reference_inline_formulas: bool) -> String {
    use regex::Regex;
    use std::sync::LazyLock;
    static NESTED_HEADER: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\b\w+\s*\[\s*[-\d.,\s]+\]\s*<(?:FORK|CHILD|BLOCK)>").unwrap()
    });
    static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\tag\s*\{[^{}]*\}").unwrap());
    let text = NESTED_HEADER.replace_all(content, "");
    let text = text.replace("<FORK>", "").replace("<CHILD>", "").replace("<BLOCK>", "")
        .replace("The image is too blurry to recognize any text content.", "")
        .replace("The image contains no text or characters. It is a graphical element (a horizontal line with a vertical line) and does not contain any chart, graph, or data points that can be extracted. Therefore, the correct OCR output is an empty string.", "");
    let mut text = text.trim().to_string();
    if text == "[Non-Text]" {
        return String::new();
    }
    if text.starts_with(r"\[") && !text.ends_with(r"\]") {
        text.push_str("\n\\]");
    }
    if text.starts_with("<table>") && !text.ends_with("</table>") {
        text.push_str("</table>");
    }
    if reference_inline_formulas {
        if text.contains("\\[\n") && !text.contains(r"\\") {
            text = text.replace("\\[\n", r"\(").replace("\n\\]", r"\)");
        }
        text = text.replace(r"\) \(", "\\)\n\n\\(");
    }
    TAG.replace_all(&text, "").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hierarchical_output_keeps_boxes_and_reading_order() {
        let raw = "<BLOCK>text[0,0,1000,100]<CHILD>Hello<FORK><BLOCK>table[0,100,1000,800]<CHILD><table><tr><td>cell</td></tr></table>";
        let page = parse_output(raw.into(), true);
        assert_eq!(page.blocks.len(), 2);
        assert_eq!(page.blocks[1].bbox, [0.0, 0.1, 1.0, 0.8]);
        assert!(page.markdown.unwrap().contains("Hello"));
        assert_eq!(page.raw_output.as_deref(), Some(raw));
    }

    #[test]
    fn malformed_boxes_preserve_content_and_structural_cleaning_matches_reference() {
        let raw = concat!(
            "<BLOCK>text[-10.5,0,1010,100]<CHILD>before text[1, 2, 3, 4]<FORK>after",
            "<BLOCK>text[900,0,100,100]<CHILD>reversed",
            "<BLOCK>text[invalid]<CHILD>malformed",
            "<BLOCK>text[0,0,1000,100]<CHILD>[Non-Text]",
            "<BLOCK>chart[0,0,1000,100]<CHILD>chart transcript",
            "<BLOCK>seal[0,0,1000,100]<CHILD>seal transcript",
            "<BLOCK>text[0,0,1000,100]<CHILD>The image is too blurry to recognize any text content.",
            "<BLOCK>text[0,0,1000,100]<CHILD>The image contains no text or characters. It is a graphical element (a horizontal line with a vertical line) and does not contain any chart, graph, or data points that can be extracted. Therefore, the correct OCR output is an empty string.",
            "<BLOCK>formula[0,0,1000,100]<CHILD>\\[\na \\supset b\\tag{1}\n\\] \\[\nc",
            "<BLOCK>table[0,0,1000,100]<CHILD><table><tr><td>cell</td></tr>",
            "<BLOCK>image[0,100,1000,200]",
        );
        let page = parse_output(raw.into(), true);
        assert_eq!(page.blocks.len(), 11);
        assert_eq!(page.blocks[0].bbox, [0.0, 0.0, 1.0, 0.1]);
        assert_eq!(page.blocks[1].bbox, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(page.blocks[2].bbox, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(page.diagnostics.len(), 3);
        assert_eq!(page.diagnostics[2].block_index, Some(2));
        assert_eq!(
            page.markdown.as_deref(),
            Some(concat!(
                "before after\n\nreversed\n\nmalformed\n\n",
                "\\(a \\supset b\\)\n\n\\(c\\)\n\n<table><tr><td>cell</td></tr></table>",
            ))
        );
        assert_eq!(page.raw_output.as_deref(), Some(raw));
        assert_eq!(
            parse_output(raw.into(), false).blocks[8].content.as_deref(),
            Some("\\[\na \\supset b\n\\] \\[\nc\n\\]")
        );
    }
}
