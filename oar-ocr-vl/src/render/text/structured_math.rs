//! Ordered formula repairs shared by document recognition adapters.
//!
//! Follows MiniCPM-OCR's Apache-2.0 postprocess/latex_repair.py.

use regex::{Captures, NoExpand, Regex};
use std::sync::LazyLock;

fn escaped(text: &str, index: usize) -> bool {
    text.as_bytes()[..index]
        .iter()
        .rev()
        .take_while(|&&b| b == b'\\')
        .count()
        % 2
        == 1
}

pub(crate) fn repair_display_math(input: &str) -> String {
    let tokens: Vec<_> = input.split("$$").collect();
    let mut output = String::new();
    let mut inside = false;
    for (i, segment) in tokens.iter().enumerate() {
        if i > 0 {
            if inside {
                output.push_str("$$");
                inside = false;
            } else if i < tokens.len() - 1 && !(segment.trim().is_empty() && i + 1 < tokens.len()) {
                output.push_str("$$");
                inside = true;
            }
        }
        output.push_str(segment);
    }
    static EMPTY: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\\[\s*\\\]|\\\(\s*\\\)").unwrap());
    static SPANS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)\\\[(.*?)\\\]|\$\$(.*?)\$\$").unwrap());
    let output = EMPTY.replace_all(&output, " ");
    SPANS
        .replace_all(&output, |caps: &Captures<'_>| {
            let bracket = caps.get(1).is_some();
            let body = caps
                .get(if bracket { 1 } else { 2 })
                .map_or("", |m| m.as_str());
            let stripped = body.trim_end();
            let body = if stripped.ends_with('\\')
                && (stripped.len() == 1 || stripped.as_bytes()[stripped.len() - 2] != b'\\')
            {
                format!(
                    "{}{}",
                    &stripped[..stripped.len() - 1],
                    &body[stripped.len()..]
                )
            } else {
                body.to_string()
            };
            if !body.trim().is_empty() && spacing_only(&body) {
                return " ".into();
            }
            if bracket {
                format!("\\[{body}\\]")
            } else {
                format!("$${body}$$")
            }
        })
        .into_owned()
}

fn spacing_only(body: &str) -> bool {
    static SPACE_COMMAND: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\\(?:quad|qquad|enspace|thinspace|medspace|thickspace|negthinspace|negmedspace|negthickspace)").unwrap()
    });
    static DIMENSION: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\(?:hspace|vspace)\s*\{[^{}]*\}").unwrap());
    static SYMBOL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\[,;:! ]").unwrap());
    let mut text = String::new();
    let mut position = 0;
    for token in SPACE_COMMAND.find_iter(body) {
        if body
            .as_bytes()
            .get(token.end())
            .is_some_and(u8::is_ascii_alphabetic)
        {
            continue;
        }
        text.push_str(&body[position..token.start()]);
        position = token.end();
    }
    text.push_str(&body[position..]);
    let text = DIMENSION.replace_all(&text, "");
    SYMBOL
        .replace_all(&text, "")
        .chars()
        .all(|c| c.is_whitespace() || matches!(c, '{' | '}' | '~'))
}

pub(crate) fn repair_balanced_formula(input: &str, fix_delimiters: bool) -> String {
    let mut braces = Vec::new();
    let mut removed = std::collections::BTreeSet::new();
    for (i, c) in input.char_indices() {
        if escaped(input, i) {
            continue;
        }
        if c == '{' {
            braces.push(i);
        } else if c == '}' && braces.pop().is_none() {
            removed.insert(i);
        }
    }
    removed.extend(braces);
    let text: String = input
        .char_indices()
        .filter_map(|(i, c)| (!removed.contains(&i)).then_some(c))
        .collect();
    let mut text = balance_left_right(&text, fix_delimiters);
    const ENVIRONMENTS: [&str; 12] = [
        "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "Vmatrix", "cases",
        "aligned", "gathered", "align", "align*",
    ];
    for env in ENVIRONMENTS {
        let begin = format!("\\begin{{{env}}}");
        let end = format!("\\end{{{env}}}");
        let opens = text.matches(&begin).count();
        let closes = text.matches(&end).count();
        if closes > opens {
            let format = text
                .find(&begin)
                .and_then(|i| text[i + begin.len()..].strip_prefix('{'))
                .and_then(|after| {
                    after
                        .split_once('}')
                        .map(|(format, _)| format!("{{{format}}}"))
                })
                .unwrap_or_else(|| {
                    if env == "array" {
                        "{c}".into()
                    } else {
                        String::new()
                    }
                });
            text = format!(
                "{}{text}",
                format!("{begin}{format} ").repeat(closes - opens)
            );
        } else if opens > closes {
            text.push_str(&format!(" {end}").repeat(opens - closes));
        }
    }
    static ARRAY_CLOSE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:\\end\{array\}\s*\\right\.\s*){2,}").unwrap());
    static BRACKET_CLOSE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?:\\\]\s*){2,}").unwrap());
    text = ARRAY_CLOSE
        .replace_all(&text, NoExpand(r"\end{array} \right."))
        .into_owned();
    text = BRACKET_CLOSE
        .replace_all(&text, NoExpand(r"\]"))
        .into_owned();
    text.trim_end_matches('\\').trim_end().into()
}

fn delimiter_end(text: &str, start: usize) -> usize {
    let after = &text[start..];
    for command in [
        r"\lceil",
        r"\rceil",
        r"\lfloor",
        r"\rfloor",
        r"\backslash",
        r"\uparrow",
        r"\downarrow",
        r"\Uparrow",
        r"\Downarrow",
        r"\langle",
        r"\rangle",
        r"\{",
        r"\}",
        r"\|",
        r"\.",
    ] {
        if after.starts_with(command) {
            return start + command.len();
        }
    }
    if after.starts_with(['(', ')', '[', ']', '{', '}', '/', '|', '.']) {
        return start + 1;
    }
    start
}

fn balance_left_right(input: &str, fix_delimiters: bool) -> String {
    static COMMAND: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\(?:left|right)").unwrap());
    let mut text = String::new();
    let mut position = 0;
    for command in COMMAND.find_iter(input) {
        if input
            .as_bytes()
            .get(command.end())
            .is_some_and(u8::is_ascii_alphabetic)
        {
            continue;
        }
        text.push_str(&input[position..command.end()]);
        if fix_delimiters && delimiter_end(input, command.end()) == command.end() {
            text.push('.');
        }
        position = command.end();
    }
    text.push_str(&input[position..]);
    let commands: Vec<_> = COMMAND
        .find_iter(&text)
        .filter(|m| {
            !text
                .as_bytes()
                .get(m.end())
                .is_some_and(u8::is_ascii_alphabetic)
        })
        .map(|m| (m.start(), m.end(), m.as_str() == r"\left"))
        .collect();
    let left = commands.iter().filter(|c| c.2).count();
    if left * 2 != commands.len() {
        let mut output = String::new();
        let mut position = 0;
        for (start, end, _) in commands {
            output.push_str(&text[position..start]);
            position = end + usize::from(text[end..].starts_with('.'));
        }
        output.push_str(&text[position..]);
        return output;
    }
    let mut stack = Vec::new();
    let mut depth = 0usize;
    let mut moves = Vec::new();
    let mut i = 0;
    while i < text.len() {
        if escaped(&text, i) {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
            continue;
        }
        if let Some(&(_, end, is_left)) = commands.iter().find(|c| c.0 == i) {
            if is_left {
                stack.push((i, depth));
            } else if let Some((left, left_depth)) = stack.pop()
                && left_depth != depth
            {
                let mut count = left_depth;
                let target = text[left..].char_indices().find_map(|(offset, ch)| {
                    if escaped(&text, left + offset) {
                        return None;
                    }
                    if ch == '{' {
                        count += 1;
                    } else if ch == '}' {
                        if count == left_depth {
                            return Some(left + offset);
                        }
                        count -= 1;
                    }
                    None
                });
                if let Some(target) = target {
                    let end = delimiter_end(&text, end).max(end);
                    moves.push((i, end, target));
                }
            }
            i = delimiter_end(&text, end).max(end);
            continue;
        }
        match text.as_bytes()[i] {
            b'{' => depth += 1,
            b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
        i += text[i..].chars().next().map_or(1, char::len_utf8);
    }
    // Apply relocations over original offsets to keep UTF-8 and neighboring tokens intact.
    let mut output = String::new();
    let mut position = 0;
    let mut events = std::collections::BTreeMap::<usize, Vec<&str>>::new();
    for &(start, end, target) in &moves {
        events.entry(target).or_default().push(&text[start..end]);
    }
    let mut i = 0;
    while i < text.len() {
        if let Some(tokens) = events.get(&i) {
            for token in tokens {
                output.push_str(token);
            }
        }
        if let Some(&(_, end, _)) = moves.iter().find(|m| m.0 == i) {
            i = end;
            position = i;
            continue;
        }
        let length = text[i..].chars().next().map_or(1, char::len_utf8);
        output.push_str(&text[i..i + length]);
        i += length;
        position = i;
    }
    if let Some(tokens) = events.get(&position) {
        for token in tokens {
            output.push_str(token);
        }
    }
    output
}
