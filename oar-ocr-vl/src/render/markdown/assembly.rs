//! Formula-number and text composition for Markdown projections.
//!
//! Neighbor rules follow PaddleX and GLM-OCR (Apache-2.0).

use crate::api::recognition::FormulaNumberComposition;
use crate::document::structure::{LayoutElement, LayoutElementType};
use std::borrow::Cow;

fn label(element: &LayoutElement) -> &str {
    element
        .label
        .as_deref()
        .unwrap_or(element.element_type.as_str())
}
fn formula(element: &LayoutElement) -> bool {
    matches!(label(element), "formula" | "display_formula")
}

pub(crate) fn compose_formula_numbers(
    elements: &[LayoutElement],
    mode: FormulaNumberComposition,
) -> Cow<'_, [LayoutElement]> {
    if matches!(
        mode,
        FormulaNumberComposition::None | FormulaNumberComposition::SeparateParagraph
    ) {
        return Cow::Borrowed(elements);
    }
    let mut output = Vec::new();
    let mut i = 0;
    while i < elements.len() {
        let item = &elements[i];
        let next = elements.get(i + 1);
        let pair = if formula(item) && next.is_some_and(|e| label(e) == "formula_number") {
            next.map(|number| (item, number))
        } else if mode == FormulaNumberComposition::AdjacentTag
            && label(item) == "formula_number"
            && next.is_some_and(formula)
        {
            next.map(|formula| (formula, item))
        } else {
            None
        };
        if let Some((formula, number)) = pair {
            let mut combined = formula.clone();
            if let Some(content) = &formula.text {
                let number = number.text.as_deref().unwrap_or("");
                combined.text = Some(if mode == FormulaNumberComposition::FollowingStarTag {
                    format!("$${} \\tag*{{{number}}}$$", content.replace("$$", ""))
                } else if let Some(body) = content.strip_suffix("\n$$") {
                    let number = number.trim();
                    let number = number
                        .strip_prefix('(')
                        .and_then(|n| n.strip_suffix(')'))
                        .or_else(|| number.strip_prefix('（').and_then(|n| n.strip_suffix('）')))
                        .unwrap_or(number);
                    format!("{body} \\tag{{{number}}}\n$$")
                } else {
                    content.clone()
                });
            }
            output.push(combined);
            i += 2;
            continue;
        }
        if !(mode == FormulaNumberComposition::AdjacentTag && label(item) == "formula_number") {
            output.push(item.clone());
        }
        i += 1;
    }
    Cow::Owned(output)
}

fn text_like(element: &LayoutElement) -> bool {
    !matches!(
        element.element_type,
        LayoutElementType::Image
            | LayoutElementType::HeaderImage
            | LayoutElementType::FooterImage
            | LayoutElementType::Chart
            | LayoutElementType::Table
            | LayoutElementType::Formula
            | LayoutElementType::FormulaNumber
            | LayoutElementType::Region
    )
}

#[derive(Clone, Copy)]
pub(crate) struct BulletAlignment {
    pub page_width: f32,
    pub grid: u32,
    pub tolerance: f64,
}

pub(crate) fn prepare_text(
    elements: &[LayoutElement],
    hyphens: bool,
    alignment: Option<BulletAlignment>,
) -> Cow<'_, [LayoutElement]> {
    if !hyphens && alignment.is_none() {
        return Cow::Borrowed(elements);
    }
    let mut elements = elements.to_vec();
    if hyphens {
        let mut omitted = vec![false; elements.len()];
        for i in 0..elements.len() {
            if omitted[i] || !text_like(&elements[i]) {
                continue;
            }
            let content = elements[i].text.as_deref().unwrap_or("").trim_end();
            let Some(before) = content.strip_suffix('-') else {
                continue;
            };
            let Some(j) = (i + 1..elements.len()).find(|&j| text_like(&elements[j])) else {
                continue;
            };
            let after = elements[j].text.as_deref().unwrap_or("").trim_start();
            if !after.chars().next().is_some_and(char::is_lowercase) {
                continue;
            }
            let (Some(left), Some(right)) = (
                before.split_whitespace().last(),
                after.split_whitespace().next(),
            ) else {
                continue;
            };
            let word = format!("{left}{right}").to_lowercase();
            if (3..=31).contains(&word.len())
                && word.as_bytes()[0].is_ascii_lowercase()
                && word
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || matches!(b, b'\'' | b'-'))
                && !word.contains("--")
                && !word.contains("''")
            {
                elements[i].text = Some(format!("{before}{after}"));
                omitted[j] = true;
            }
        }
        elements = elements
            .into_iter()
            .zip(omitted)
            .filter_map(|(e, omit)| (!omit).then_some(e))
            .collect();
    }
    if let Some(alignment) = alignment {
        // GLM's neighboring text blocks use page-relative thousandth coordinates.
        for i in 1..elements.len().saturating_sub(1) {
            if [i - 1, i, i + 1]
                .into_iter()
                .all(|i| label(&elements[i]) == "text")
                && elements[i - 1]
                    .text
                    .as_deref()
                    .is_some_and(|s| s.starts_with("- "))
                && elements[i + 1]
                    .text
                    .as_deref()
                    .is_some_and(|s| s.starts_with("- "))
                && !elements[i].text.as_deref().unwrap_or("").starts_with("- ")
            {
                let left = |i: usize| {
                    (f64::from(elements[i].bbox.x_min()) / f64::from(alignment.page_width)
                        * f64::from(alignment.grid))
                    .trunc()
                };
                if (left(i) - left(i - 1)).abs() <= alignment.tolerance
                    && (left(i) - left(i + 1)).abs() <= alignment.tolerance
                {
                    elements[i].text =
                        Some(format!("- {}", elements[i].text.as_deref().unwrap_or("")));
                }
            }
        }
    }
    Cow::Owned(elements)
}

pub(crate) fn normalize_bullet(text: &str) -> String {
    let text = text.trim();
    let mut output = if text.starts_with(['·', '•']) || text.starts_with("* ") {
        format!(
            "- {}",
            text.chars()
                .next()
                .map_or(text, |c| &text[c.len_utf8()..])
                .trim_start()
        )
    } else {
        text.to_string()
    };
    static PAREN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"^[（(]([0-9]+|[A-Za-z])[）)](.*)$").unwrap()
    });
    static NUMBER: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"^([0-9]+|[A-Za-z])([.)）])(.*)$").unwrap());
    output = PAREN
        .replace(&output, |c: &regex::Captures<'_>| {
            format!("({}) {}", &c[1], c[2].trim_start())
        })
        .into_owned();
    output = NUMBER
        .replace(&output, |c: &regex::Captures<'_>| {
            format!(
                "{}{} {}",
                &c[1],
                if &c[2] == "）" { ")" } else { &c[2] },
                c[3].trim_start()
            )
        })
        .into_owned();
    if output.starts_with("```") && !output.ends_with("```") {
        output.push_str("\n```");
    }
    output
}
