//! Bounded document-JSON recovery from the evaluation protocol's behavior.

use crate::document::decoding::{
    JsonFenceOptions, JsonTrailing, decode_json, escape_string_controls, read_string_body,
    strip_json_fence,
};
use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;

pub(super) fn decode(raw: &str) -> Result<(Value, Vec<String>), String> {
    static TOKENS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:<\|[^|]*\|>\s*)+$").unwrap());
    let mut messages = Vec::new();
    let stripped = TOKENS.replace(raw.trim(), "");
    let text = strip_json_fence(
        &stripped,
        JsonFenceOptions {
            bare_language: true,
            ..Default::default()
        },
    );
    let trimmed = crate::render::text::clean_repetition(
        text,
        false,
        &crate::render::text::RepetitionOptions {
            algorithm: crate::render::text::RepeatAlgorithm::StrongestSuffix,
            max_period: Some(512),
            min_repeats: 3,
            min_repeat_chars: 80,
            search_window: Some(4096),
            retain_partial_tail: false,
            ..Default::default()
        },
    );
    if trimmed != text {
        messages.push("Trimmed a repeated JSON tail before decoding".into());
    }
    if let Ok((value, _)) = decode_json::<Value>(&trimmed, JsonTrailing::Allow) {
        return if value.is_object() {
            Ok((value, messages))
        } else {
            Err("Youtu-Parsing-Omni expects a top-level JSON object".into())
        };
    }
    // Repair quoted strings and recover complete category/content records.
    let repaired = escape_string_controls(&trimmed);
    if let Ok(value) = serde_json::from_str::<Value>(&repaired) {
        messages.push("Repaired JSON string escapes/control characters".into());
        return Ok((
            if value.is_array() {
                json!({"elements":value})
            } else {
                value
            },
            messages,
        ));
    }
    static CATEGORY: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#""category"\s*:\s*"([A-Za-z_]+)""#).unwrap());
    static CONTENT: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#""(text|latex|otsl|markdown|caption|html)"\s*:\s*""#).unwrap()
    });
    let categories: Vec<_> = CATEGORY.captures_iter(&trimmed).collect();
    let mut elements = Vec::new();
    for (index, category) in categories.iter().enumerate() {
        let start = category.get(0).unwrap().end();
        let end = categories
            .get(index + 1)
            .map_or(trimmed.len(), |c| c.get(0).unwrap().start());
        let segment = &trimmed[start..end];
        let mut fields = serde_json::Map::new();
        for key in CONTENT.captures_iter(segment) {
            let name = &key[1];
            if fields.contains_key(name) {
                continue;
            }
            let body = read_string_body(&segment[key.get(0).unwrap().end()..]);
            let quoted = format!("\"{body}\"");
            let value =
                serde_json::from_str::<String>(&escape_string_controls(&quoted)).unwrap_or(body);
            fields.insert(name.into(), Value::String(value));
        }
        elements.push(json!({"category": &category[1], "content": fields}));
    }
    if elements.is_empty() {
        return Err(
            "Invalid Youtu-Parsing-Omni JSON; no document records could be recovered".into(),
        );
    }
    messages.push(
        "Recovered category/content records from malformed JSON; id and bbox unavailable".into(),
    );
    Ok((json!({"elements":elements,"reading_order":[]}), messages))
}
