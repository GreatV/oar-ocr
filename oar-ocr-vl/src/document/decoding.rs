//! Shared document-JSON decoding and string recovery.

use serde::de::DeserializeOwned;
use serde_json::Value;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum JsonTrailing {
    #[default]
    Reject,
    Allow,
}

pub(crate) fn decode_json<T: DeserializeOwned>(
    input: &str,
    trailing: JsonTrailing,
) -> Result<(T, usize), serde_json::Error> {
    match trailing {
        JsonTrailing::Reject => serde_json::from_str(input).map(|value| (value, input.len())),
        JsonTrailing::Allow => {
            let mut stream = serde_json::Deserializer::from_str(input).into_iter::<T>();
            match stream.next() {
                Some(value) => value.map(|value| (value, stream.byte_offset())),
                None => serde_json::from_str(input).map(|value| (value, input.len())),
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct JsonFenceOptions {
    pub uppercase_language: bool,
    pub bare_language: bool,
}

pub(crate) fn strip_json_fence(input: &str, options: JsonFenceOptions) -> &str {
    let text = input.trim();
    if let Some(body) = text.strip_prefix("```") {
        let body = body
            .strip_prefix("json")
            .or_else(|| {
                options
                    .uppercase_language
                    .then(|| body.strip_prefix("JSON"))
                    .flatten()
            })
            .unwrap_or(body)
            .trim_start();
        body.strip_suffix("```").unwrap_or(body).trim_end()
    } else if options.bare_language {
        text.strip_prefix("json\n").unwrap_or(text)
    } else {
        text
    }
}

pub(crate) fn complete_array_prefix(input: &str) -> Vec<Value> {
    let mut values = Vec::new();
    if let Some(mut tail) = input.strip_prefix('[') {
        loop {
            tail = tail.trim_start();
            if tail.is_empty() || tail.starts_with(']') {
                break;
            }
            let Ok((value, consumed)) = decode_json(tail, JsonTrailing::Allow) else {
                break;
            };
            values.push(value);
            tail = tail[consumed..].trim_start();
            if let Some(next) = tail.strip_prefix(',') {
                tail = next;
            } else {
                break;
            }
        }
    }
    values
}

pub(crate) fn complete_objects(input: &str) -> Vec<Value> {
    let mut objects = Vec::new();
    let mut start = None;
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in input.char_indices() {
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
                && let Ok((value, _)) = decode_json(&input[start..=index], JsonTrailing::Reject)
            {
                objects.push(value);
            }
        }
    }
    objects
}

pub(crate) fn escape_string_controls(text: &str) -> String {
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

pub(crate) fn read_string_body(text: &str) -> String {
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

pub(crate) fn normalize_python_literals(text: &str) -> String {
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
