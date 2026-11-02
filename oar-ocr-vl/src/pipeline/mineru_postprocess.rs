//! Semantic post-processing for the autoregressive MinerU 2.5 family.

use super::mineru_layout::cover_ratio;
use crate::DocumentBlock;
use regex::{Captures, Regex};
use std::collections::BTreeMap;
use std::sync::LazyLock;

fn clean_equation(text: &str) -> String {
    static BIG: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(\\(?:bigg|Bigg|big|Big)[lmr]?) ?\{(\\(?:rangle|langle|[{}|])|[()|\[\]])\}")
            .unwrap()
    });
    static DOUBLE_SUBSCRIPT: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"_\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*_\s*\{(?:[^{}]|\{[^{}]*\})*\}").unwrap()
    });
    let text = text.trim();
    let text = text.strip_prefix(r"\[").unwrap_or(text);
    let text = text.strip_suffix(r"\]").unwrap_or(text).trim();
    // Ported from MinerU-Team/mineru-vl-utils post_process (Apache-2.0):
    // equation_left_right.py, equation_double_subscript.py, equation_big.py.
    let text = fix_left_right(text);
    let text = DOUBLE_SUBSCRIPT.replace_all(&text, "");
    let text = BIG
        .replace_all(&text, |caps: &Captures<'_>| {
            let delimiter = &caps[2];
            let space = if matches!(delimiter, r"\rangle" | r"\langle") {
                " "
            } else {
                ""
            };
            format!("{}{delimiter}{space}", &caps[1])
        })
        .replace(r"\bigtimes", r"\times")
        .replace(r"\eqqcolon", "=:")
        .replace(r"\coloneqq", ":=")
        .replace('<', "< ");
    // Remove unmatched, unescaped braces, retaining every recognized symbol.
    let mut stack = Vec::new();
    let mut remove = std::collections::HashSet::new();
    let mut slashes = 0;
    for (index, ch) in text.char_indices() {
        if slashes % 2 == 0 {
            if ch == '{' {
                stack.push(index);
            } else if ch == '}' && stack.pop().is_none() {
                remove.insert(index);
            }
        }
        slashes = if ch == '\\' { slashes + 1 } else { 0 };
    }
    remove.extend(stack);
    text.char_indices()
        .filter_map(|(index, ch)| (!remove.contains(&index)).then_some(ch))
        .collect()
}

fn split_preserving(text: &str, regex: &Regex) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for token in regex.find_iter(text) {
        pieces.push(text[start..token.start()].to_string());
        pieces.push(token.as_str().to_string());
        start = token.end();
    }
    pieces.push(text[start..].to_string());
    pieces
}

fn clean_text(text: &str) -> String {
    // MinerU post_process/text_{display2inline,inline_spacing,
    // move_underscores_outside}.py (Apache-2.0).
    static DISPLAY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\\\[(.*?)\\\]").unwrap());
    static NUMBERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\A[–\d\-,\s]+\z").unwrap());
    static INLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)\\\((.*?)\\\)").unwrap());
    static MACRO: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(\\(?:cong|to|times|subset|in))([a-zA-Z])").unwrap());
    static BLANK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_{3,}").unwrap());
    let text = DISPLAY.replace_all(text, |c: &Captures<'_>| {
        if NUMBERS.is_match(&c[1]) {
            c[0].into()
        } else {
            format!("\\({}\\)", &c[1])
        }
    });
    INLINE
        .replace_all(&text, |c: &Captures<'_>| {
            let inner = &c[1];
            let inner = MACRO.replace_all(inner, |m: &Captures<'_>| {
                let tail = &inner[m.get(0).unwrap().end()..];
                if matches!(&m[0], r"\top" | r"\int" | r"\inf")
                    || tail.starts_with(|ch: char| ch.is_ascii_alphabetic())
                {
                    m[0].into()
                } else {
                    format!("{} {}", &m[1], &m[2])
                }
            });
            if !BLANK.is_match(&inner) {
                return format!("\\({inner}\\)");
            }
            split_preserving(&inner, &BLANK)
                .into_iter()
                .filter(|s| !s.trim().is_empty())
                .map(|s| {
                    if BLANK.is_match(&s) {
                        s
                    } else {
                        format!("\\({s}\\)")
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .into_owned()
}

fn paired_delimiters(left: &str, right: &str) -> bool {
    let (Some(left), Some(right)) = (left.strip_prefix(r"\left"), right.strip_prefix(r"\right"))
    else {
        return false;
    };
    [
        (r"\lbrace", r"\rbrace"),
        (r"\lVert", r"\lVert"),
        (r"\lvert", r"\lvert"),
        (r"\vert", r"\vert"),
        (r"\Vert", r"\Vert"),
        (r"\lfloor", r"\rfloor"),
        (r"\lbrack", r"\rbrack"),
        (r"\langle", r"\rangle"),
        ("|", "|"),
        (r"\|", r"\|"),
        ("[", "]"),
        ("]", "["),
        ("(", ")"),
        (")", "("),
        (r"\{", r"\}"),
        (r"\}", r"\{"),
        ("/", "/"),
    ]
    .iter()
    .any(|&(l, r)| (left == l || left == ".") && (right == r || right == "."))
}

fn fix_left_right(text: &str) -> String {
    // Keep the reference's token patterns, including its broad final right
    // alternative; only recognized delimiter tokens are subsequently repaired.
    const LEFT: &str = r"\\left(?:\\(?:lbrace|lVert|lvert|rvert|rVert|vert|Vert|lfloor|lbrack|langle)|\||\\\||\[|\]|\(|\\\{|\\\}|\.)";
    const RIGHT: &str = r"\\right(?:\\(?:rbrace|lVert|lvert|rvert|rVert|vert|Vert|rfloor|rbrack|rangle)|\||\\\||\]|\[|\)|\\\}|\\\{|.)";
    static L: LazyLock<Regex> = LazyLock::new(|| Regex::new(LEFT).unwrap());
    static R: LazyLock<Regex> = LazyLock::new(|| Regex::new(RIGHT).unwrap());
    static TOKENS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(&format!("{LEFT}|{RIGHT}")).unwrap());
    static SEPARATORS: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"&|\\\\|\\begin\{array\}\s*\{[a-zA-Z\s]*\}|\\end\{array\}").unwrap()
    });
    if L.find_iter(text).count() == R.find_iter(text).count()
        || text.matches(r"\begin{array}").count() != text.matches(r"\end{array}").count()
    {
        return text.into();
    }
    let mut nodes: Vec<_> = split_preserving(text.trim(), &SEPARATORS)
        .into_iter()
        .filter(|node| !node.is_empty())
        .map(|node| node.trim().to_string())
        .collect();
    let mut arrays = Vec::new();
    let mut next_array = 0usize;
    let mut groups = BTreeMap::<Option<(usize, usize)>, Vec<usize>>::new();
    for (index, node) in nodes.iter().enumerate() {
        if node.contains(r"\begin{array}") {
            next_array += 1;
            arrays.push((next_array, 0));
        } else if node == r"\end{array}" {
            if arrays.pop().is_none() {
                return text.into();
            }
        } else if matches!(node.as_str(), "&" | r"\\") {
            if let Some((_, cell)) = arrays.last_mut() {
                *cell += 1;
            }
        } else {
            groups
                .entry(arrays.last().copied())
                .or_default()
                .push(index);
        }
    }
    if !arrays.is_empty() {
        return text.into();
    }
    for indices in groups.values() {
        let left: usize = indices
            .iter()
            .map(|&i| L.find_iter(&nodes[i]).count())
            .sum();
        let right: usize = indices
            .iter()
            .map(|&i| R.find_iter(&nodes[i]).count())
            .sum();
        if left == right {
            continue;
        }
        let mut spans: Vec<Vec<String>> = indices
            .iter()
            .map(|&i| {
                split_preserving(&nodes[i], &TOKENS)
                    .into_iter()
                    .filter(|s| !s.trim().is_empty())
                    .collect()
            })
            .collect();
        let mut unmatched: Vec<(usize, usize)> = Vec::new();
        for (span_index, span) in spans.iter().enumerate() {
            for (token_index, token) in span.iter().enumerate() {
                let delimiter = token
                    .strip_prefix(r"\left")
                    .or_else(|| token.strip_prefix(r"\right"));
                if !delimiter.is_some_and(|d| {
                    matches!(d, r"\rvert" | r"\rVert")
                        || paired_delimiters(token, r"\right.")
                        || paired_delimiters(r"\left.", token)
                }) {
                    continue;
                }
                if unmatched
                    .last()
                    .is_some_and(|&(s, t)| paired_delimiters(&spans[s][t], token))
                {
                    unmatched.pop();
                } else {
                    unmatched.push((span_index, token_index));
                }
            }
        }
        for (s, t) in unmatched {
            let token = &mut spans[s][t];
            *token = if token.starts_with(r"\left") {
                format!("{token} \\right.")
            } else {
                format!("\\left. {token}")
            };
        }
        for (&index, span) in indices.iter().zip(spans) {
            nodes[index] = span.concat();
        }
    }
    nodes.concat()
}

pub(crate) fn postprocess_blocks(mut blocks: Vec<DocumentBlock>) -> Vec<DocumentBlock> {
    static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\tag\s*\{([^}]*)\}").unwrap());
    for block in &mut blocks {
        if block.block_type == "list_item" {
            block.block_type = "text".into();
        }
        if block.block_type == "equation"
            && let Some(content) = &mut block.content
        {
            *content = clean_equation(content);
        } else if block.block_type == "text"
            && let Some(content) = &mut block.content
        {
            *content = clean_text(content);
        }
    }
    let mut absorbed = vec![false; blocks.len()];
    for index in 0..blocks.len() {
        if blocks[index].block_type != "equation_block" {
            continue;
        }
        let spans: Vec<_> = blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| {
                block.block_type == "equation" && cover_ratio(blocks[index].bbox, block.bbox) > 0.9
            })
            .map(|(i, _)| i)
            .collect();
        if spans.len() < 2 {
            continue;
        }
        let tag_count: usize = spans
            .iter()
            .map(|&i| {
                TAG.find_iter(blocks[i].content.as_deref().unwrap_or_default())
                    .count()
            })
            .sum();
        let mut contents = Vec::new();
        for i in spans {
            absorbed[i] = true;
            let content = blocks[i].content.as_deref().unwrap_or_default();
            contents.push(if tag_count > 1 {
                TAG.replace_all(content, "($1)").into_owned()
            } else {
                content.to_string()
            });
        }
        blocks[index].block_type = "equation".into();
        blocks[index].content = Some(format!(
            "\\begin{{array}}{{l}} {} \\\\ \\end{{array}}",
            contents.join(" \\\\ ")
        ));
    }
    blocks
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut block)| {
            if absorbed[index] || block.block_type == "equation_block" {
                return None;
            }
            if block
                .content
                .as_deref()
                .is_some_and(|s| s.trim() == "[Non-Text]")
                && !matches!(
                    block.block_type.as_str(),
                    "table" | "image" | "chart" | "equation" | "list" | "image_block"
                )
            {
                if matches!(block.block_type.as_str(), "header" | "footer") {
                    block.content = Some(String::new());
                } else {
                    return None;
                }
            }
            if block.block_type == "equation"
                && let Some(content) = &mut block.content
                && !content.is_empty()
            {
                *content = format!("\\[\n{}\n\\]", content.trim());
            }
            Some(block)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equation_containers_merge_spans_without_multiple_tags() {
        let block = |kind: &str, bbox, text: &str| DocumentBlock {
            block_type: kind.into(),
            bbox,
            content: Some(text.into()),
            angle: None,
        };
        let blocks = postprocess_blocks(vec![
            block("equation_block", [0., 0., 1., 1.], ""),
            block("equation", [0., 0., 1., 0.4], r"\[x\coloneqq 1\tag{1}\]"),
            block("equation", [0., 0.5, 1., 1.], r"y=2\tag{2}"),
        ]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_type, "equation");
        assert_eq!(
            blocks[0].content.as_deref(),
            Some("\\[\n\\begin{array}{l} x:= 1(1) \\\\ y=2(2) \\\\ \\end{array}\n\\]")
        );
        assert_eq!(
            clean_equation(r"\big{\rangle}x \Big{(}y\Big{)} \big{\{}z\big{\}} \big{\|}"),
            r"\big\rangle x \Big(y\Big) \big\{z\big\} \big\|"
        );
        assert_eq!(clean_equation(r"x_{a}_{b}+\left(y"), r"x+\left( \right.y");
        assert_eq!(
            clean_equation(r"\begin{array}{ll}\left(x & y\end{array}"),
            r"\begin{array}{ll}\left( \right.x&y\end{array}"
        );
        let empty = postprocess_blocks(vec![block("equation", [0., 0., 1., 1.], r"\[ \]")]);
        assert_eq!(empty[0].content.as_deref(), Some(""));
        assert_eq!(
            clean_text(r"a \[x\tox\] and \[1, 2\]"),
            r"a \(x\to x\) and \[1, 2\]"
        );
        assert_eq!(clean_text(r"\(x___y\)"), r"\(x\) ___ \(y\)");
    }
}
