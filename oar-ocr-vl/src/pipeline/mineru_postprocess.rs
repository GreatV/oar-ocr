//! Shared semantic post-processing for the two MinerU decoders.

use super::mineru_layout::cover_ratio;
use crate::DocumentBlock;
use regex::Regex;
use std::sync::LazyLock;

fn clean_equation(text: &str) -> String {
    static BIG: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(\\(?:big|Big|bigg|Bigg)[lmr]?)\s*\{(\\[a-zA-Z]+|\\\||[\[\]])\}").unwrap()
    });
    let text = text.trim();
    let text = text.strip_prefix(r"\[").unwrap_or(text);
    let text = text.strip_suffix(r"\]").unwrap_or(text).trim();
    let text = BIG
        .replace_all(text, "$1$2")
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

pub(crate) fn postprocess_blocks(mut blocks: Vec<DocumentBlock>) -> Vec<DocumentBlock> {
    static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\tag\s*\{([^}]*)\}").unwrap());
    for block in &mut blocks {
        if block.block_type == "equation"
            && let Some(content) = &mut block.content
        {
            *content = clean_equation(content);
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
    }
}
