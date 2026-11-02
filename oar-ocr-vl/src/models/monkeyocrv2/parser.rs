//! Complete-page parser adapter for MonkeyOCRv2.

use super::{DEFAULT_MAX_NEW_TOKENS, MonkeyOcrV2, MonkeyOcrV2Task};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// MonkeyOCRv2 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct MonkeyOcrV2ParseOptions {
    pub task: MonkeyOcrV2Task,
    pub max_new_tokens: usize,
}

impl Default for MonkeyOcrV2ParseOptions {
    fn default() -> Self {
        Self {
            task: MonkeyOcrV2Task::EndToEnd,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

impl PageParser for MonkeyOcrV2 {
    type Options = MonkeyOcrV2ParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let raw = self
            .generate(
                std::slice::from_ref(image),
                &[options.task],
                options.max_new_tokens,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("MonkeyOCRv2 returned no page result"))??;
        Ok(parse_output(raw, options.task))
    }
}

fn parse_output(raw: String, task: MonkeyOcrV2Task) -> PageDocument {
    use crate::render::{table::convert_otsl_to_html, text::strip_math_wrappers};
    use crate::{DocumentBlock, ParseDiagnostic};
    let mut page = PageDocument {
        raw_output: Some(raw.clone()),
        ..PageDocument::default()
    };
    if matches!(
        task,
        MonkeyOcrV2Task::Text | MonkeyOcrV2Task::Formula | MonkeyOcrV2Task::Table
    ) {
        page.markdown = Some(match task {
            MonkeyOcrV2Task::Table => convert_otsl_to_html(&raw),
            MonkeyOcrV2Task::Formula => format!("$$\n{}\n$$", strip_math_wrappers(&raw)),
            _ => raw.trim().to_string(),
        });
        return page;
    }
    // The official protocol uses Python literal dictionaries as well as JSON.
    // Decode strings without evaluating model output, then normalize the literals.
    let normalized = normalize_literals(&raw);
    let mut objects = Vec::new();
    let mut start = None;
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in normalized.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                quoted = false;
            }
            continue;
        }
        if ch == '"' {
            quoted = true;
        } else if ch == '{' {
            if depth == 0 {
                start = Some(index);
            }
            depth += 1;
        } else if ch == '}' && depth > 0 {
            depth -= 1;
            if depth == 0
                && let Some(start) = start.take()
                && let Ok(value) =
                    serde_json::from_str::<serde_json::Value>(&normalized[start..=index])
            {
                objects.push(value);
            }
        }
    }
    for object in objects {
        let Some(label) = object.get("label").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(coords) = object
            .get("bbox")
            .and_then(|v| v.as_array())
            .filter(|v| v.len() == 4)
        else {
            continue;
        };
        let Some(coords) = coords
            .iter()
            .map(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if coords.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let bbox = [
            coords[0].min(coords[2]),
            coords[1].min(coords[3]),
            coords[0].max(coords[2]),
            coords[1].max(coords[3]),
        ]
        .map(|v| (v as f32 / 1000.0).clamp(0.0, 1.0));
        let label = match label {
            "Title" => "doc_title",
            "Section-header" => "paragraph_title",
            "Page-header" => "header",
            "Page-footer" => "footer",
            "List-item" => "text",
            "Caption" => "figure_title",
            other => other,
        }
        .to_lowercase();
        let content = object
            .get("content")
            .and_then(|v| v.as_str())
            .map(|content| {
                if label == "table" {
                    convert_otsl_to_html(content)
                } else {
                    content.trim().to_string()
                }
            });
        page.blocks.push(DocumentBlock {
            block_type: label,
            bbox,
            angle: None,
            content,
        });
    }
    if page.blocks.is_empty() {
        page.diagnostics.push(ParseDiagnostic {
            block_index: None,
            stage: "postprocess".into(),
            message: "No valid layout records in MonkeyOCRv2 output".into(),
        });
    } else if task == MonkeyOcrV2Task::EndToEnd {
        page.markdown = Some(page.render_markdown());
    }
    page
}

fn normalize_literals(text: &str) -> String {
    let mut output = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\'' && ch != '"' {
            if ch.is_alphabetic() || ch == '_' {
                let mut word = String::from(ch);
                while chars
                    .peek()
                    .is_some_and(|ch| ch.is_alphanumeric() || *ch == '_')
                {
                    word.push(chars.next().unwrap());
                }
                output.push_str(match word.as_str() {
                    "None" => "null",
                    "True" => "true",
                    "False" => "false",
                    _ => &word,
                });
            } else {
                output.push(ch);
            }
            continue;
        }
        let quote = ch;
        let mut encoded = String::from("\"");
        while let Some(ch) = chars.next() {
            if ch == quote {
                break;
            }
            if ch == '\\' {
                match chars.next() {
                    Some('\'') => encoded.push('\''),
                    Some(next) => {
                        if !matches!(next, 'n' | 'r' | 't' | 'b' | 'f' | 'u' | '"' | '/' | '\\') {
                            encoded.push('\\');
                        }
                        encoded.push('\\');
                        encoded.push(next);
                    }
                    None => break,
                }
            } else if ch == '"' {
                encoded.push_str("\\\"");
            } else {
                encoded.push(ch);
            }
        }
        encoded.push('"');
        output.push_str(&encoded);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_literal_records_and_preserves_raw_output() {
        let raw = r#"[{'label':'Title','bbox':[0,0,1000,100],'content':"A title"}, {'label':'Text','bbox':[0,100,1000,200],'content':'a \\times b'}]"#;
        let page = parse_output(raw.into(), MonkeyOcrV2Task::EndToEnd);
        assert_eq!(page.blocks.len(), 2);
        assert_eq!(page.blocks[1].content.as_deref(), Some(r"a \times b"));
        assert!(page.markdown.unwrap().contains("# A title"));
        assert_eq!(page.raw_output.as_deref(), Some(raw));
        let unicode = parse_output(
            r#"[{"label":"Text","bbox":[0,0,1000,1000],"content":"\u4e2d\u6587"}]"#.into(),
            MonkeyOcrV2Task::EndToEnd,
        );
        assert_eq!(unicode.blocks[0].content.as_deref(), Some("中文"));
        let literals = parse_output(
            "[{'label':'Text','bbox':[0,0,1000,100], 'content':None, 'flags':[True,False]}, {'label':'Text','bbox':[0,100,1000,200],'content':'None True False TrueName'}]".into(),
            MonkeyOcrV2Task::EndToEnd,
        );
        assert_eq!(literals.blocks.len(), 2);
        assert_eq!(literals.blocks[0].content, None);
        assert_eq!(
            literals.blocks[1].content.as_deref(),
            Some("None True False TrueName")
        );
        assert_eq!(
            normalize_literals("TrueName Falsehood NoneValue"),
            "TrueName Falsehood NoneValue"
        );
    }
}
