//! Bounded document-JSON recovery from the evaluation protocol's behavior.

use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;

pub(super) fn decode(raw: &str) -> Result<(Value, Vec<String>), String> {
    static TOKENS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:<\|[^|]*\|>\s*)+$").unwrap());
    let mut messages = Vec::new();
    let stripped = TOKENS.replace(raw.trim(), "");
    let mut text = stripped.trim();
    if let Some(body) = text.strip_prefix("```") {
        text = body.strip_prefix("json").unwrap_or(body).trim_start();
        text = text.strip_suffix("```").unwrap_or(text).trim_end();
    } else if let Some(body) = text.strip_prefix("json\n") {
        text = body;
    }
    let trimmed = trim_tail(text);
    if trimmed != text {
        messages.push("Trimmed a repeated JSON tail before decoding".into());
    }
    let mut stream = serde_json::Deserializer::from_str(&trimmed).into_iter::<Value>();
    if let Some(Ok(value)) = stream.next() {
        return if value.is_object() {
            Ok((value, messages))
        } else {
            Err("Youtu-Parsing-Omni expects a top-level JSON object".into())
        };
    }
    // Independently implemented repair: protect stray LaTeX backslashes and
    // literal control characters, then recover category/content records if
    // the surrounding document or final record is incomplete.
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

fn escape_string_controls(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            quoted = !quoted;
            out.push(ch);
        } else if quoted && ch == '\\' {
            if chars
                .peek()
                .is_some_and(|&c| matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u'))
            {
                out.push(ch);
                out.push(chars.next().unwrap());
            } else {
                out.push_str("\\\\");
            }
        } else if quoted && ch.is_control() {
            let encoded = serde_json::to_string(&ch.to_string()).unwrap();
            out.push_str(&encoded[1..encoded.len() - 1]);
        } else {
            out.push(ch);
        }
    }
    out
}

fn read_string_body(text: &str) -> String {
    let mut body = String::new();
    let mut chars = text.char_indices();
    while let Some((index, ch)) = chars.next() {
        if ch == '\\' {
            body.push(ch);
            if let Some((_, next)) = chars.next() {
                body.push(next);
            }
        } else if ch == '"' {
            if text[index + 1..].trim_start().starts_with([',', '}', ']']) {
                break;
            }
            body.push_str("\\\"");
        } else {
            body.push(ch);
        }
    }
    body
}

fn trim_tail(text: &str) -> String {
    let chars: Vec<_> = text.chars().collect();
    let window = &chars[chars.len().saturating_sub(4096)..];
    if window.len() < 80 {
        return text.into();
    }
    let matching = |slice: &[char], period: usize| {
        (0..slice.len() - period)
            .take_while(|&i| slice[slice.len() - 1 - i] == slice[slice.len() - 1 - i - period])
            .count()
    };
    let mut best = (0, 0);
    for period in 1..=512.min(window.len() / 3) {
        let span = matching(window, period) + period;
        if span >= 80 && span >= 3 * period && span > best.0 {
            best = (span, period);
        }
    }
    let period = best.1;
    if period == 0 {
        return text.into();
    }
    let span = matching(&chars, period) + period;
    let remainder = span % period;
    chars[..chars.len() - span]
        .iter()
        .chain(&chars[chars.len() - remainder - period..chars.len() - remainder])
        .collect()
}
