//! Semantic post-processing for the autoregressive MinerU 2.5 family.

use super::geometry::cover_ratio;
use crate::DocumentBlock;
use crate::render::text::{
    FormulaOptions, FormulaRepair, MathStripping, format_formula_with_options,
    repair_math_in_text as clean_text,
};
use regex::Regex;
use std::sync::LazyLock;

fn clean_equation(text: &str) -> String {
    format_formula_with_options(
        text,
        &FormulaOptions {
            opening: "",
            closing: "",
            stripping: MathStripping::DisplayBrackets,
            repair: FormulaRepair::Equation,
            normalize_mu: false,
            newline_to_break: false,
            clean_tokens: false,
        },
    )
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
