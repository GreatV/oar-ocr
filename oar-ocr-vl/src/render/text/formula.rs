//! Parameterized math delimiters, formula wrappers, and ordered LaTeX repairs.

use super::{clean_special_tokens, fix_latex_brackets, strip_math_wrappers};
use regex::{Captures, Regex};
use std::collections::BTreeMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum BracketMatching {
    #[default]
    EachPair,
    AnyPair,
    All,
    Spans,
    NonemptySpans,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DelimiterOptions {
    pub matching: BracketMatching,
    pub inline: (&'static str, &'static str),
    pub display: (&'static str, &'static str),
    pub remove_dollars: bool,
    pub collapse_brackets: bool,
    pub unwrap_dollar_brackets: bool,
}

impl Default for DelimiterOptions {
    fn default() -> Self {
        Self {
            matching: BracketMatching::EachPair,
            inline: (" $ ", " $ "),
            display: (" $$ ", " $$ "),
            remove_dollars: false,
            collapse_brackets: false,
            unwrap_dollar_brackets: false,
        }
    }
}

pub(crate) fn has_bracket_math(text: &str) -> bool {
    (text.contains(r"\(") && text.contains(r"\)")) || (text.contains(r"\[") && text.contains(r"\]"))
}

pub(crate) fn normalize_math_delimiters(input: &str, options: &DelimiterOptions) -> String {
    if matches!(options.matching, BracketMatching::AnyPair) && !has_bracket_math(input) {
        return input.into();
    }
    let mut text = if options.remove_dollars {
        input.replace('$', "")
    } else {
        input.into()
    };
    if options.collapse_brackets {
        text = text.replace(r"\[\[", r"\[").replace(r"\]\]", r"\]");
    }
    if matches!(
        options.matching,
        BracketMatching::Spans | BracketMatching::NonemptySpans
    ) {
        static DISPLAY_DOLLARS: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)\$\$?\s*\\\[\s*(.*?)\s*\\\]\s*\$\$?").unwrap());
        static INLINE_DOLLARS: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)\$\s*\\\(\s*(.*?)\s*\\\)\s*\$").unwrap());
        static DISPLAY: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)\\\[\s*(.*?)\s*\\\]").unwrap());
        static INLINE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)\\\(\s*(.*?)\s*\\\)").unwrap());
        static NONEMPTY_DISPLAY: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)\\\[(.+?)\\\]").unwrap());
        static NONEMPTY_INLINE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?s)\\\((.+?)\\\)").unwrap());
        let nonempty = matches!(options.matching, BracketMatching::NonemptySpans);
        let display = if nonempty {
            &*NONEMPTY_DISPLAY
        } else {
            &*DISPLAY
        };
        let inline = if nonempty {
            &*NONEMPTY_INLINE
        } else {
            &*INLINE
        };
        let body = |caps: &Captures<'_>| {
            if nonempty {
                caps[1].trim().to_string()
            } else {
                caps[1].to_string()
            }
        };
        if options.unwrap_dollar_brackets {
            text = DISPLAY_DOLLARS
                .replace_all(&text, |caps: &Captures<'_>| {
                    format!("{}{}{}", options.display.0, &caps[1], options.display.1)
                })
                .into_owned();
            text = INLINE_DOLLARS
                .replace_all(&text, |caps: &Captures<'_>| {
                    format!("{}{}{}", options.inline.0, &caps[1], options.inline.1)
                })
                .into_owned();
        }
        text = display
            .replace_all(&text, |caps: &Captures<'_>| {
                format!("{}{}{}", options.display.0, body(caps), options.display.1)
            })
            .into_owned();
        return inline
            .replace_all(&text, |caps: &Captures<'_>| {
                format!("{}{}{}", options.inline.0, body(caps), options.inline.1)
            })
            .into_owned();
    }
    if matches!(
        options.matching,
        BracketMatching::AnyPair | BracketMatching::All
    ) || (text.contains(r"\(") && text.contains(r"\)"))
    {
        text = text
            .replace(r"\(", options.inline.0)
            .replace(r"\)", options.inline.1);
    }
    if matches!(
        options.matching,
        BracketMatching::AnyPair | BracketMatching::All
    ) || (text.contains(r"\[") && text.contains(r"\]"))
    {
        text = text
            .replace(r"\[", options.display.0)
            .replace(r"\]", options.display.1);
    }
    text
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum MathStripping {
    #[default]
    All,
    Dollars,
    OuterDollars,
    DisplayBrackets,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum FormulaRepair {
    #[default]
    SimpleDelimiters,
    Equation,
    EquationNumber,
    Balanced {
        fix_delimiters: bool,
    },
    None,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FormulaOptions {
    pub opening: &'static str,
    pub closing: &'static str,
    pub stripping: MathStripping,
    pub repair: FormulaRepair,
    pub normalize_mu: bool,
    pub newline_to_break: bool,
    pub clean_tokens: bool,
}

impl Default for FormulaOptions {
    fn default() -> Self {
        Self {
            opening: "$$",
            closing: "$$",
            stripping: MathStripping::All,
            repair: FormulaRepair::SimpleDelimiters,
            normalize_mu: true,
            newline_to_break: true,
            clean_tokens: true,
        }
    }
}

pub(crate) fn format_formula_with_options(input: &str, options: &FormulaOptions) -> String {
    let mut text = if options.clean_tokens {
        clean_special_tokens(input)
    } else {
        input.into()
    };
    if options.normalize_mu {
        text = text.replace(r"\upmu", r"\mu");
    }
    text = match options.stripping {
        MathStripping::All => text
            .replace(r"\[", "")
            .replace(r"\]", "")
            .replace(r"\(", "")
            .replace(r"\)", "")
            .trim()
            .trim_matches('$')
            .into(),
        MathStripping::Dollars => text.trim().trim_matches('$').trim().into(),
        MathStripping::OuterDollars => strip_math_wrappers(&text).into(),
        MathStripping::DisplayBrackets => {
            let body = text.trim();
            let body = body.strip_prefix(r"\[").unwrap_or(body);
            body.strip_suffix(r"\]").unwrap_or(body).trim().into()
        }
    };
    if options.newline_to_break {
        text = text.replace('\n', "\\\\\n");
    }
    let mut number = None;
    text = match options.repair {
        FormulaRepair::SimpleDelimiters => fix_latex_brackets(&text),
        FormulaRepair::Equation => repair_equation(&text),
        FormulaRepair::EquationNumber => {
            let (body, extracted) = extract_equation_number(&text);
            number = extracted;
            body
        }
        FormulaRepair::Balanced { fix_delimiters } => {
            super::structured_math::repair_balanced_formula(&text, fix_delimiters)
        }
        FormulaRepair::None => text,
    };
    format!(
        "{}{}{}{}",
        options.opening,
        text.trim(),
        options.closing,
        number.unwrap_or_default()
    )
}

pub(crate) fn ensure_math_wrappers(text: &str, opening: &str, closing: &str) -> String {
    let mut text = text.trim().to_string();
    if text.is_empty() {
        return text;
    }
    if !text.starts_with(opening) {
        text = format!("{opening}\n{text}");
    }
    if !text.ends_with(closing) {
        text.push_str(&format!("\n{closing}"));
    }
    text
}

fn repair_equation(text: &str) -> String {
    static BIG: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(\\(?:bigg|Bigg|big|Big)[lmr]?) ?\{(\\(?:rangle|langle|[{}|])|[()|\[\]])\}")
            .unwrap()
    });
    static DOUBLE_SUBSCRIPT: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"_\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*_\s*\{(?:[^{}]|\{[^{}]*\})*\}").unwrap()
    });
    let text = text.trim();
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

pub(crate) fn repair_math_in_text(text: &str) -> String {
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
    // Match and repair delimiter tokens within each array cell.
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

fn extract_equation_number(text: &str) -> (String, Option<String>) {
    // MonkeyOCRv2 core_runner.py process_formula (Apache-2.0).
    static QUAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:\\quad\s*){5,}").unwrap());
    static QQUAD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:\\qquad\s*){5,}").unwrap());
    static TAG: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$|\\tag\{([^{}]*)\}\s*$").unwrap()
    });
    static BEGIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\A\\begin\{([^}]+)\}").unwrap());
    let text = QUAD.replace_all(text.trim().trim_matches('$').trim(), r"\quad ");
    let mut content = QQUAD.replace_all(&text, r"\qquad ").trim().to_string();
    let mut extracted = None;
    let mut environment = None;
    for pass in 0..2 {
        if let Some(caps) = TAG.captures(&content) {
            // Read numbers from both equation-spacing and tag forms.
            extracted = caps
                .get(1)
                .or_else(|| caps.get(2))
                .map(|m| m.as_str().to_string());
            content.truncate(caps.get(0).unwrap().start());
            content = content.trim_end().into();
        }
        if pass == 0
            && let Some(caps) = BEGIN.captures(&content)
        {
            let name = caps[1].to_string();
            content = content[caps.get(0).unwrap().end()..].trim_start().into();
            if let Some(body) = content.trim_end().strip_suffix(&format!("\\end{{{name}}}")) {
                content = body.trim_end().into();
            }
            environment = Some(name);
        }
    }
    if let Some(name) = environment {
        content = format!("\\begin{{{name}}}\n{content}\n\\end{{{name}}}");
    }
    (content, extracted)
}
