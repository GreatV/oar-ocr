//! Mermaid reply cleanup using the MinerU utility client's repair rules.
// Adapted from mineru-vl-utils post_process/mermaid_*.py and
// image_analysis_postprocess.py, Apache-2.0, commit c9797cde8f54cc48306196fc6aebe73ea31dd4a3.

use regex::{Captures, Regex};
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

macro_rules! rx {
    ($pattern:literal) => {{
        static PATTERN: LazyLock<Regex> = LazyLock::new(|| Regex::new($pattern).unwrap());
        &*PATTERN
    }};
}

pub(crate) fn format_mermaid(content: &str) -> String {
    if content.trim().is_empty() || repetitive(content) {
        return String::new();
    }
    let content = legalize_ids(content);
    let content = repair_lines(&content);
    let content = content.trim();
    let content = rx!(r"(?s)```mermaid\s*(.*?)\s*```")
        .captures(content)
        .or_else(|| rx!(r"(?s)```\s*(.*?)\s*```").captures(content))
        .map_or(content, |c| c.get(1).unwrap().as_str())
        .trim();
    let content = if content.to_lowercase().starts_with("mermaid") {
        content[7..].trim()
    } else {
        content
    };
    let content = rx!(r"(?i)^(grap|grapg|graphh)\b").replace(content, "graph");
    let content = rx!(r"(?i)^(flowchar|flowchartt)\b").replace(&content, "flowchart");
    let content = rx!(r"<\s*-\s*-\s*>").replace_all(&content, "__MERMAID_BIDIR_ARROW__");
    let content = rx!(r"-\s+->|--\s+>|-\s+-\s+>").replace_all(&content, "-->");
    let content = rx!(r"([^-=])\s+->\s+").replace_all(&content, "$1 --> ");
    let content = content.replace("__MERMAID_BIDIR_ARROW__", "<-->");
    let node_pattern = rx!(
        r"(?s)([a-zA-Z0-9_\x{4e00}-\x{9fa5}\-]+(?:[ \t]+[a-zA-Z0-9_\x{4e00}-\x{9fa5}\-]+)*)[ \t]*\[(.*?)\]"
    );
    let content = node_pattern.replace_all(&content, |c: &Captures<'_>| {
        let id = c[1].trim();
        let text = &c[2];
        if id.to_lowercase().starts_with("subgraph ") || text.starts_with("\"[") {
            return c[0].into();
        }
        let id = rx!(r"[ \t]+").replace_all(id, "_");
        if text.is_empty() {
            return format!("{id}[]");
        }
        let text = text
            .strip_prefix('"')
            .and_then(|t| t.strip_suffix('"'))
            .unwrap_or(text);
        format!(
            "{id}[\"{}\"]",
            text.replace('"', "&quot;").replace('\n', "<br>")
        )
    });
    format!("```mermaid\n{content}\n```")
}

fn normalized_newlines(content: &str) -> String {
    let content = rx!(r"\\+n").replace_all(content, "\n");
    rx!(r"(?i)<br\s*/?>")
        .replace_all(&content, "\n")
        .into_owned()
}

fn repetitive(content: &str) -> bool {
    let content = normalized_newlines(content);
    let mut labels = HashMap::new();
    let mut edges = HashMap::new();
    for c in rx!(r#"(?s)([A-Za-z0-9_\x{4e00}-\x{9fa5}\-]+)\s*\["((?:[^"\\]|\\.)*)"\]"#)
        .captures_iter(&content)
    {
        let label = &c[2];
        let lines: Vec<_> = label
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        if bad_id(&c[1]) || super::repetition::has_repeated_lines(&lines.join("\n"), 8, 20, 0) {
            return true;
        }
        if label.chars().count() >= 2000
            && !lines.is_empty()
            && lines.iter().copied().collect::<HashSet<_>>().len() * 5 <= lines.len()
        {
            return true;
        }
    }
    let mut free = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_lowercase();
        let labeled_edge = rx!(
            r#"^\s*([A-Za-z0-9_\x{4e00}-\x{9fa5}\-]+)\s*-->\s*([A-Za-z0-9_\x{4e00}-\x{9fa5}\-]+)\s*\["((?:[^"\\]|\\.)*)"\]\s*$"#
        );
        let edge = rx!(
            r"^\s*([A-Za-z0-9_\x{4e00}-\x{9fa5}\-]+)\s*-->\s*(?:\|[^|\n]*\|\s*)?([A-Za-z0-9_\x{4e00}-\x{9fa5}\-]+)(?:\s*(?:\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}))?\s*$"
        );
        if let Some(c) = labeled_edge.captures(line)
            && !c[3].trim().is_empty()
            && increment(&mut labels, c[3].trim()) >= 8
        {
            return true;
        }
        if let Some(c) = edge.captures(line)
            && (bad_id(&c[1])
                || bad_id(&c[2])
                || increment(&mut edges, &format!("{}-->{}", &c[1], &c[2])) >= 8)
        {
            return true;
        }
        if trimmed.matches("-->").count() >= 2 {
            for segment in trimmed.split("-->") {
                let segment = segment.trim();
                let segment = if let Some(label) = segment.strip_prefix('|') {
                    label
                        .split_once('|')
                        .map_or(segment, |(_, tail)| tail.trim())
                } else {
                    segment
                };
                let id = segment
                    .split(['[', '(', '{'])
                    .next()
                    .unwrap_or_default()
                    .trim();
                if bad_id(id) {
                    return true;
                }
            }
        }
        if trimmed.is_empty()
            || lower.starts_with("graph ")
            || lower.starts_with("flowchart ")
            || lower.starts_with("subgraph ")
            || lower == "end"
            || lower.starts_with("direction ")
            || lower.starts_with("%%")
            || trimmed.contains("-->")
        {
            continue;
        }
        let line = trimmed
            .trim_start_matches(['[', ']', '(', ')', '"', '\'', ' ', '>'])
            .trim_end_matches(['[', ']', '(', ')', '"', '\'', ' ', ',', '.', ';', ':']);
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() {
            free.push(line);
        }
    }
    super::repetition::has_repeated_lines(&free.join("\n"), 8, 20, 0)
}

fn increment(counts: &mut HashMap<String, usize>, key: &str) -> usize {
    let count = counts.entry(key.into()).or_default();
    *count += 1;
    *count
}

fn bad_id(id: &str) -> bool {
    if id.chars().count() > 128 {
        return true;
    }
    let mut counts = HashMap::new();
    for token in rx!(r"[A-Z][a-z]|[A-Z]{2}").find_iter(id) {
        increment(&mut counts, token.as_str());
    }
    if counts.is_empty() {
        for run in rx!(r"[A-Za-z]{8,}").find_iter(id) {
            for pair in run.as_str().as_bytes().windows(2) {
                increment(&mut counts, std::str::from_utf8(pair).unwrap());
            }
        }
    }
    counts.values().any(|&count| count >= 8)
}

fn split_top_level<'a>(text: &'a str, delimiter: &str) -> Vec<&'a str> {
    let mut output = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    let mut pipe = false;
    for (index, c) in text.char_indices() {
        if index < start {
            continue;
        }
        if escaped {
            escaped = false;
            continue;
        }
        if quoted {
            if c == '\\' {
                escaped = true;
            } else if c == '"' {
                quoted = false;
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth = depth.saturating_sub(1),
            '|' if depth == 0 => pipe = !pipe,
            _ if depth == 0 && !pipe && text[index..].starts_with(delimiter) => {
                output.push(&text[start..index]);
                start = index + delimiter.len();
            }
            _ => {}
        }
    }
    output.push(&text[start..]);
    output
}

fn valid_id(id: &str) -> bool {
    rx!(r"^[A-Za-z_][A-Za-z0-9_]*$").is_match(id)
}

fn endpoint(
    text: &str,
    ids: &mut HashMap<String, String>,
    declared: &mut HashSet<String>,
) -> String {
    let raw = text.trim();
    if raw.is_empty() || raw.contains(['[', ']', '(', ')', '{', '}', '"', '|']) || valid_id(raw) {
        return raw.into();
    }
    let id = ids
        .entry(raw.into())
        .or_insert_with(|| {
            let words: Vec<_> = rx!(r"[A-Za-z0-9]+")
                .find_iter(raw)
                .map(|m| m.as_str())
                .collect();
            let mut base = String::new();
            for (index, word) in words.iter().enumerate() {
                if index == 0 {
                    base.push_str(word);
                } else {
                    let (first, rest) = word.split_at(1);
                    base.push_str(&first.to_uppercase());
                    base.push_str(&rest.to_lowercase());
                }
            }
            if base.is_empty() {
                base = raw
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
            }
            if base.is_empty() {
                base = "node".into();
            }
            if base.starts_with(|c: char| c.is_ascii_digit()) {
                base.insert(0, 'n');
            }
            let mut id = base.clone();
            let mut suffix = 2;
            while declared.contains(&id) {
                id = format!("{base}{suffix}");
                suffix += 1;
            }
            id
        })
        .clone();
    if declared.insert(id.clone()) {
        format!("{id}[\"{raw}\"]")
    } else {
        id
    }
}

fn legalize_ids(content: &str) -> String {
    let mut declared: HashSet<_> = content
        .lines()
        .filter_map(|line| {
            rx!(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[\[\(\{]")
                .captures(line)
                .map(|c| c[1].to_string())
        })
        .collect();
    let mut ids = HashMap::new();
    content
        .lines()
        .map(|line| {
            if !line.contains("-->")
                || line.trim_start().starts_with("%%")
                || rx!(r"<\s*-\s*-\s*>").is_match(line)
            {
                return line.into();
            }
            let mut statements = split_top_level(line, ";");
            let trailing =
                statements.len() > 1 && statements.last().is_some_and(|s| s.trim().is_empty());
            if trailing {
                statements.pop();
            }
            let fixed: Vec<_> = statements
                .into_iter()
                .filter(|s| !s.trim().is_empty())
                .map(|statement| {
                    let chain = split_top_level(statement.trim(), "-->");
                    if chain.len() < 2 || statement.trim_start().starts_with("%%") {
                        return statement.trim().into();
                    }
                    let mut fixed = endpoint(chain[0], &mut ids, &mut declared);
                    for segment in &chain[1..] {
                        let segment = segment.trim();
                        if let Some((label, target)) = segment
                            .strip_prefix('|')
                            .and_then(|s| s.split_once('|'))
                            .filter(|(_, t)| !t.trim().is_empty())
                        {
                            let label = rx!(r" {2,}")
                                .replace_all(&label.replace(['(', ')'], ""), " ")
                                .trim()
                                .to_string();
                            fixed.push_str(&format!(
                                " -->|{label}| {}",
                                endpoint(target, &mut ids, &mut declared)
                            ));
                        } else {
                            fixed.push_str(&format!(
                                " --> {}",
                                endpoint(segment, &mut ids, &mut declared)
                            ));
                        }
                    }
                    fixed
                })
                .collect();
            if fixed.is_empty() {
                line.into()
            } else {
                format!("  {}{}", fixed.join("; "), if trailing { ";" } else { "" })
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn repair_lines(content: &str) -> String {
    let mut seen = HashSet::new();
    content
        .lines()
        .filter_map(|line| {
            if line.contains("-->")
                && unquoted_pipes(line) == 1
                && !line.trim_start().starts_with("%%")
                && rx!(r"-->\s*[^|\n]*\|\s*[^|\n]+$").is_match(line)
            {
                return None;
            }
            let Some(c) = rx!(r"^(\s*)subgraph\s+(.+?)\s*$").captures(line) else {
                return Some(line.into());
            };
            let body = &c[2];
            if let Some(id) = rx!(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[\[\(]").captures(body) {
                seen.insert(id[1].to_string());
                return Some(line.into());
            }
            let first = body.split_whitespace().next().unwrap_or_default();
            if valid_id(first) {
                seen.insert(first.into());
                return Some(line.into());
            }
            let base = rx!(r"[^A-Za-z0-9_]+").replace_all(body, "_");
            let base = rx!(r"_+")
                .replace_all(&base, "_")
                .trim_matches('_')
                .to_string();
            let mut base = if base.is_empty() { "sg".into() } else { base };
            if base.starts_with(|c: char| c.is_ascii_digit()) {
                base = format!("sg_{base}");
            }
            let mut id = base.clone();
            let mut suffix = 2;
            while seen.contains(&id) {
                id = format!("{base}_{suffix}");
                suffix += 1;
            }
            seen.insert(id.clone());
            Some(format!(
                "{}subgraph {id}[\"{}\"]",
                &c[1],
                body.replace('"', "\\\"")
            ))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unquoted_pipes(line: &str) -> usize {
    let mut single = false;
    let mut double = false;
    let mut escaped = false;
    let mut count = 0;
    for c in line.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '\'' if !double => single = !single,
            '"' if !single => double = !double,
            '|' if !single && !double => count += 1,
            _ => {}
        }
    }
    count
}
