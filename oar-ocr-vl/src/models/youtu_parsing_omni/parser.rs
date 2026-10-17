use super::processing::ImagePlan;
use super::{DEFAULT_MAX_NEW_TOKENS, YoutuParsingOmni};
use crate::document::page::{DocumentBlock, PageDocument, ParseDiagnostic};
use crate::render::table::convert_otsl_to_html;
use crate::{Error, PageParser};
use image::RgbImage;
use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::LazyLock;

/// Document parsing and greedy generation controls for Youtu-Parsing-Omni.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct YoutuParsingOmniParseOptions {
    /// Maximum number of generated tokens; defaults to the official 32768.
    pub max_new_tokens: usize,
    /// Optional image pixel ceiling; `None` keeps the checkpoint processor's limit.
    pub max_pixels: Option<usize>,
    /// Optional task instruction; `None` uses the official document JSON prompt.
    pub prompt: Option<String>,
}

impl Default for YoutuParsingOmniParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_pixels: None,
            prompt: None,
        }
    }
}

impl YoutuParsingOmniParseOptions {
    /// Set the generated-token budget.
    pub fn with_max_new_tokens(mut self, count: usize) -> Self {
        self.max_new_tokens = count;
        self
    }
    /// Bound the padded image area before encoding.
    pub fn with_max_pixels(mut self, pixels: usize) -> Self {
        self.max_pixels = Some(pixels);
        self
    }
    /// Override the document instruction for another image-based task.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }
}

impl PageParser for YoutuParsingOmni {
    type Options = YoutuParsingOmniParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let plan = self.image_plan(image, options)?;
        let (tokens, finished) = self.generate_one(image, options)?;
        let raw = self.decode_tokens(&tokens)?;
        let mut document = parse_output(&raw, plan);
        if !finished {
            document.diagnostics.push(ParseDiagnostic {
                block_index: None,
                stage: "generation".into(),
                message: format!(
                    "Youtu-Parsing-Omni generation reached {} tokens without EOS",
                    options.max_new_tokens
                ),
            });
        }
        Ok(document)
    }
}

static BOX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^<box><x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)></box>$").expect("static box regex")
});
static CELL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<(start|left|up|left_up)>\s*(\d+)\s*<content>").expect("static table regex")
});

fn bbox(raw: Option<&str>, plan: ImagePlan) -> Option<[f32; 4]> {
    let captures = BOX.captures(raw?.trim())?;
    let mut coords = [0f32; 4];
    for (index, out) in coords.iter_mut().enumerate() {
        *out = captures[index + 1].parse::<u32>().ok()? as f32 / 1000.0;
        if *out > 1.0 {
            return None;
        }
    }
    if coords[0] > coords[2] || coords[1] > coords[3] {
        return None;
    }
    for (index, value) in coords.iter_mut().enumerate() {
        let scale = if index.is_multiple_of(2) {
            plan.width as f32 / plan.resized_width as f32
        } else {
            plan.height as f32 / plan.resized_height as f32
        };
        *value = (*value * scale).clamp(0.0, 1.0);
    }
    Some(coords)
}

fn table(raw: &str) -> String {
    if !raw.contains("<content>") {
        return convert_otsl_to_html(raw);
    }
    let mut normalized = String::new();
    for line in raw.lines() {
        let cells = CELL.captures_iter(line).collect::<Vec<_>>();
        if cells.is_empty() {
            continue;
        }
        let mut column = 0;
        for (i, cell) in cells.iter().enumerate() {
            let col = cell[2].parse::<usize>().unwrap_or(column);
            if col > raw.len() {
                return raw.to_string();
            }
            while column < col {
                normalized.push_str("<ecel>");
                column += 1;
            }
            normalized.push_str(match &cell[1] {
                "start" => "<fcel>",
                "left" => "<lcel>",
                "up" => "<ucel>",
                _ => "<xcel>",
            });
            if &cell[1] == "start" {
                let start = cell.get(0).expect("matched cell").end();
                let end = cells.get(i + 1).map_or(line.len(), |next| {
                    next.get(0).expect("matched cell").start()
                });
                normalized.push_str(line[start..end].trim());
            }
            column += 1;
        }
        normalized.push_str("<nl>");
    }
    if normalized.is_empty() {
        raw.to_string()
    } else {
        convert_otsl_to_html(&normalized)
    }
}

fn content(element: &Value, category: &str) -> Option<String> {
    let content = element.get("content")?;
    let keys: &[&str] = match category {
        "table" => &["otsl", "html", "markdown", "text"],
        "formula" => &["latex", "text"],
        "chart" => &["markdown", "caption", "text"],
        "flowchart" => &["mermaid", "caption", "text"],
        "figure" | "geometric" => &["caption", "text", "geometric"],
        _ => &["text", "caption"],
    };
    let text = keys
        .iter()
        .find_map(|key| {
            content
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
        })?
        .trim();
    Some(match category {
        "table" => table(text),
        "formula" => {
            if text.starts_with('$') || text.starts_with("\\[") {
                text.to_string()
            } else {
                format!("$$\n{text}\n$$")
            }
        }
        "flowchart" if content.get("mermaid").is_some() => format!("```mermaid\n{text}\n```"),
        _ => text.to_string(),
    })
}

pub(super) fn parse_output(raw: &str, plan: ImagePlan) -> PageDocument {
    let mut document = PageDocument {
        raw_output: Some(raw.to_string()),
        ..Default::default()
    };
    let json = raw
        .trim()
        .strip_prefix("```json")
        .or_else(|| raw.trim().strip_prefix("```"))
        .unwrap_or(raw.trim())
        .trim();
    let json = json.strip_suffix("```").unwrap_or(json).trim();
    let value = match serde_json::from_str::<Value>(json) {
        Ok(value) => value,
        Err(error) => {
            document.diagnostics.push(ParseDiagnostic {
                block_index: None,
                stage: "postprocessing".into(),
                message: format!("invalid Youtu-Parsing-Omni JSON: {error}"),
            });
            return document;
        }
    };
    let Some(elements) = value.get("elements").and_then(Value::as_array) else {
        document.diagnostics.push(ParseDiagnostic {
            block_index: None,
            stage: "postprocessing".into(),
            message: "Youtu-Parsing-Omni JSON lacks an elements array".into(),
        });
        return document;
    };
    let mut order = Vec::with_capacity(elements.len());
    let mut seen = HashSet::new();
    if let Some(reading_order) = value.get("reading_order").and_then(Value::as_array) {
        for id in reading_order.iter().filter_map(Value::as_str) {
            if let Some(index) = elements
                .iter()
                .position(|e| e.get("id").and_then(Value::as_str) == Some(id))
                && seen.insert(index)
            {
                order.push(index);
            }
        }
    }
    order.extend((0..elements.len()).filter(|index| !seen.contains(index)));
    for index in order {
        let element = &elements[index];
        let Some(category) = element.get("category").and_then(Value::as_str) else {
            document.diagnostics.push(ParseDiagnostic {
                block_index: None,
                stage: "postprocessing".into(),
                message: format!("Youtu-Parsing-Omni element {index} lacks a category"),
            });
            continue;
        };
        let bbox = bbox(element.get("bbox").and_then(Value::as_str), plan).unwrap_or_else(|| {
            document.diagnostics.push(ParseDiagnostic {
                block_index: Some(document.blocks.len()),
                stage: "postprocessing".into(),
                message: format!("Youtu-Parsing-Omni element {index} has invalid box coordinates"),
            });
            [0.0, 0.0, 1.0, 1.0]
        });
        document.blocks.push(DocumentBlock {
            block_type: category.to_string(),
            bbox,
            angle: None,
            content: content(element, category),
        });
    }
    document.markdown = Some(
        document
            .blocks
            .iter()
            .filter_map(|b| b.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n"),
    );
    document
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omni_schema_preserves_layout_order_and_table_spans() {
        let raw = r#"```json
{"elements":[{"id":"t","category":"table","bbox":"<box><x_0><y_500><x_1000><y_1000></box>","content":{"otsl":"<start>0<content>A<left>1<content>\n<up>0<content><left_up>1<content>"}},{"id":"h","category":"title","bbox":"<box><x_100><y_0><x_900><y_100></box>","content":{"text":"Heading"}},{"id":"f","category":"formula","bbox":"<box><x_0><y_100><x_1000><y_400></box>","content":{"latex":"x^2"}}],"reading_order":["h","f","t","t"]}
```"#;
        let plan = ImagePlan {
            width: 64,
            height: 64,
            resized_width: 64,
            resized_height: 32,
        };
        let page = parse_output(raw, plan);
        assert!(page.diagnostics.is_empty());
        assert_eq!(
            page.blocks
                .iter()
                .map(|b| b.block_type.as_str())
                .collect::<Vec<_>>(),
            ["title", "formula", "table"]
        );
        assert_eq!(page.blocks[0].bbox, [0.1, 0.0, 0.9, 0.2]);
        assert_eq!(page.blocks[1].content.as_deref(), Some("$$\nx^2\n$$"));
        let html = page.blocks[2].content.as_deref().unwrap();
        assert!(html.contains("rowspan=\"2\""), "{html}");
        assert!(html.contains("colspan=\"2\""), "{html}");
        assert_eq!(page.raw_output.as_deref(), Some(raw));
        let invalid = parse_output("```json\n{\"elements\":[", plan);
        assert!(invalid.blocks.is_empty());
        assert_eq!(invalid.diagnostics.len(), 1);
        assert!(invalid.raw_output.is_some());
    }
}
