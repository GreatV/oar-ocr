//! Raw Unlimited-OCR-2 output to Markdown, ported from the official
//! `utils/process_unlimitedOCR2.py` (MIT).
//!
//! The model emits layout blocks rather than Markdown:
//!
//! ```text
//! <PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>text ...
//! <BLOCK>text[x1,y1,x2,y2]<FORK>body ...
//! <BLOCK>cross_table[bbox]<FORK><table>...</table><PAGE><BLOCK>append_table[bbox]<FORK>...
//! ```
//!
//! Boxes are on a 0-1000 grid of the page. `cross_<t>`/`append_<t>` chains
//! continue a block across pages and `<MERGE>` joins same-page columns; the
//! converter groups them and concatenates each group in reading order.

use once_cell::sync::Lazy;
use regex::{Captures, Regex};

static PAGE_SPLIT_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)<\s*/?\s*PAGE\s*\d*\s*>"));
static MERGE_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)<\s*/?\s*MERGE\s*>"));
static STRAY_TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
    re(concat!(
        r"(?i)<[|｜]/?[a-zA-Z_]+[|｜]>|<image>",
        r"|<\s*/?\s*PAGE\s*\d*\s*>|<\s*/?\s*BLOCK\s*>|<\s*/?\s*FORK\s*>",
        r"|<\s*/?\s*LEVEL\s*\d*\s*[>}]?|<\s*/?\s*MERGE\s*>",
    ))
});
// <BLOCK>label [<LEVELn>|[bbox]] [<LEVELn>|[bbox]] [<FORK>]; LEVEL-first or bbox-first.
static BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    re(concat!(
        r"(?i)<\s*BLOCK\s*>\s*([a-zA-Z_]+)",
        r"(?:",
        r"\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?(?:\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)",
        r"(?:\s*[,，]\s*-?\d+)*\s*\])?",
        r"|",
        r"\s*\[\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)\s*[,，]\s*(-?\d+)",
        r"(?:\s*[,，]\s*-?\d+)*\s*\](?:\s*<\s*/?\s*LEVEL\s*(\d*)\s*[>}]?)?",
        r")?",
        r"\s*(?:<\s*FORK\s*>)?",
    ))
});
// The official pattern's `(?<=<FORK>)|(?<=<PAGE>)|(?<=\n)|^` lookbehind is
// replaced by anchored matches at those positions (see `fix_bare_blocks`).
static BARE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    re(concat!(
        r"(?i)^(?:<[|｜]\s*det\s*[|｜]>)?",
        r"((?:[a-zA-Z_][a-zA-Z_0-9]*)(?:\s*<\s*/?\s*LEVEL\s*\d*\s*[>}]?)?\s*",
        r"\[\s*-?\d+(?:\s*[,，]\s*-?\d+){3,}[^\]]*\])",
    ))
});
static BARE_ANCHOR_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)<FORK>|<PAGE>|\n"));
static FORK_BEFORE_BLOCK_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)\n\s*<FORK>\s*(<BLOCK>)"));
static TR_RE: Lazy<Regex> = Lazy::new(|| re(r"(?is)<tr\b[^>]*>.*?</tr>"));
static TABLE_OPEN_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)<table\b[^>]*>"));
static SPAN_ATTR_RE: Lazy<Regex> = Lazy::new(|| re(r#"(?i)\b(colspan|rowspan)\s*=\s*"([^"]*)""#));
static TAG_RE: Lazy<Regex> = Lazy::new(|| re(r"<[^>]+>"));
static CELL_RE: Lazy<Regex> = Lazy::new(|| re(r"(?i)<t[hd]\b"));
static DIGITS_RE: Lazy<Regex> = Lazy::new(|| re(r"\d+"));
static WHITESPACE_RE: Lazy<Regex> = Lazy::new(|| re(r"\s+"));
static BARE_BOX_LINE_RE: Lazy<Regex> =
    Lazy::new(|| re(r"(?m)^\s*\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]\s*$"));
static TRAILING_SPACE_RE: Lazy<Regex> = Lazy::new(|| re(r"[ \t]+\n"));
static BLANK_LINES_RE: Lazy<Regex> = Lazy::new(|| re(r"\n{3,}"));

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("static regex")
}

const IMAGE_LABELS: [&str; 6] = ["image", "figure", "picture", "chart", "img", "seal"];
const NOISE_CONTENT: [&str; 4] = ["[non-text]", "[unreadable]", "[image]", "[figure]"];
const CHAIN_TYPES: [&str; 2] = ["text", "table"];
const MAX_HEADER_ROWS: usize = 5;
const BASE_LABELS: [&str; 17] = [
    "text",
    "title",
    "header",
    "footer",
    "page_number",
    "image",
    "figure",
    "chart",
    "seal",
    "table",
    "image_caption",
    "image_footnote",
    "page_footnote",
    "equation",
    "ref_text",
    "aside_text",
    "algorithm",
];
/// Label of text outside any `<BLOCK>` declaration.
const RAW: &str = "__raw__";

/// One layout block of the raw output.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Block {
    /// Lower-cased label as emitted, including `cross_`/`append_` prefixes.
    pub label: String,
    /// `[x1, y1, x2, y2]` on the page's 0-1000 grid.
    pub bbox: Option<[i64; 4]>,
    pub level: Option<usize>,
    pub content: String,
    /// 1-based page number.
    pub page: usize,
    merge_prev: bool,
}

/// `cross_text`/`append_text` -> `text`, `append_table`/`cross_table` -> `table`.
pub(crate) fn base_type(label: &str) -> &str {
    label
        .strip_prefix("append_")
        .or_else(|| label.strip_prefix("cross_"))
        .unwrap_or(label)
}

fn is_bare_label(label: &str) -> bool {
    let label = label.to_lowercase();
    let base = label
        .strip_prefix("cross_")
        .or_else(|| label.strip_prefix("append_"))
        .unwrap_or(&label);
    BASE_LABELS.contains(&label.as_str()) || BASE_LABELS.contains(&base)
}

/// Add the missing `<BLOCK>` prefix to bare block declarations (known
/// labels only).
fn fix_bare_blocks(text: &str) -> String {
    let mut anchors = vec![0];
    anchors.extend(BARE_ANCHOR_RE.find_iter(text).map(|m| m.end()));
    let mut out = String::with_capacity(text.len());
    let (mut copied, mut scanned) = (0, 0);
    for anchor in anchors {
        if anchor < scanned {
            continue;
        }
        let Some(caps) = BARE_BLOCK_RE.captures(&text[anchor..]) else {
            continue;
        };
        // Like `re.sub`, a match is consumed even when it is kept as is.
        scanned = anchor + caps.get(0).expect("match").end();
        let declaration = caps.get(1).expect("group 1").as_str();
        let label_end = declaration
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(declaration.len());
        if !is_bare_label(&declaration[..label_end]) {
            continue;
        }
        out.push_str(&text[copied..anchor]);
        out.push_str("<BLOCK>");
        out.push_str(declaration);
        copied = scanned;
    }
    out.push_str(&text[copied..]);
    FORK_BEFORE_BLOCK_RE.replace_all(&out, "\n$1").into_owned()
}

fn group_int(caps: &Captures, index: usize) -> Option<i64> {
    caps.get(index).map(|m| m.as_str().parse().unwrap_or(0))
}

fn group_level(caps: &Captures, index: usize) -> Option<usize> {
    caps.get(index)
        .filter(|m| !m.as_str().is_empty())
        .and_then(|m| m.as_str().parse().ok())
}

fn bbox(caps: &Captures, first: usize) -> Option<[i64; 4]> {
    caps.get(first)?;
    Some([0, 1, 2, 3].map(|k| group_int(caps, first + k).unwrap_or(0)))
}

/// Split on `<PAGE>` and parse every page's blocks in reading order.
pub(crate) fn parse_blocks(text: &str) -> Vec<Block> {
    let text = fix_bare_blocks(text);
    let mut segments: Vec<&str> = PAGE_SPLIT_RE.split(&text).collect();
    if segments.first().is_some_and(|s| s.trim().is_empty()) {
        segments.remove(0);
    }
    let mut blocks = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let page = index + 1;
        if segment.trim().is_empty() {
            continue;
        }
        let matches: Vec<Captures> = BLOCK_RE.captures_iter(segment).collect();
        if matches.is_empty() {
            blocks.push(Block {
                label: RAW.into(),
                bbox: None,
                level: None,
                content: MERGE_RE.replace_all(segment, "").trim().to_string(),
                page,
                merge_prev: false,
            });
            continue;
        }
        for (j, caps) in matches.iter().enumerate() {
            let whole = caps.get(0).expect("match");
            let gap_start = if j == 0 {
                0
            } else {
                matches[j - 1].get(0).expect("match").end()
            };
            let gap = &segment[gap_start..whole.start()];
            if j == 0 {
                let pre = MERGE_RE.replace_all(gap, "");
                let pre = STRAY_TOKEN_RE.replace_all(&pre, "");
                let pre = pre.trim();
                if !pre.is_empty() {
                    blocks.push(Block {
                        label: RAW.into(),
                        bbox: None,
                        level: None,
                        content: pre.to_string(),
                        page,
                        merge_prev: false,
                    });
                }
            }
            let end = matches
                .get(j + 1)
                .map_or(segment.len(), |next| next.get(0).expect("match").start());
            let (bbox, level) = if caps.get(2).is_some() {
                (bbox(caps, 3), group_level(caps, 2))
            } else {
                (bbox(caps, 7), group_level(caps, 11))
            };
            blocks.push(Block {
                label: caps[1].to_lowercase(),
                bbox,
                level,
                content: MERGE_RE
                    .replace_all(&segment[whole.end()..end], "")
                    .trim()
                    .to_string(),
                page,
                merge_prev: MERGE_RE.is_match(gap),
            });
        }
    }
    blocks
}

/// Merge strictly adjacent, same-page, same-`<LEVELn>` title blocks.
fn merge_adjacent_titles(blocks: Vec<Block>) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::with_capacity(blocks.len());
    for block in blocks {
        if let Some(prev) = out.last_mut()
            && block.label == "title"
            && prev.label == "title"
            && block.page == prev.page
            && block.level.is_some()
            && block.level == prev.level
        {
            prev.content = [prev.content.as_str(), block.content.as_str()]
                .into_iter()
                .filter(|x| !x.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            continue;
        }
        out.push(block);
    }
    out
}

/// Union-find grouping of append/cross chains and same-page `<MERGE>`
/// columns; groups come back ordered by their first block.
fn build_groups(blocks: &[Block]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..blocks.len()).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    }
    let chain =
        |block: &Block| block.label != RAW && CHAIN_TYPES.contains(&base_type(&block.label));
    let table_pages: std::collections::HashSet<usize> = blocks
        .iter()
        .filter(|b| chain(b) && base_type(&b.label) == "table")
        .map(|b| b.page)
        .collect();
    let table_adjacent = |tail: usize, current: usize| {
        current.saturating_sub(tail) <= 1 || (tail + 1..current).all(|p| !table_pages.contains(&p))
    };

    let mut open_chain: std::collections::HashMap<&str, usize> = Default::default();
    for (i, block) in blocks.iter().enumerate() {
        if !chain(block) {
            continue;
        }
        let label = block.label.as_str();
        let base = base_type(label);
        if label.starts_with("append_") {
            let Some(&tail) = open_chain.get(base) else {
                continue;
            };
            if base == "table" && !table_adjacent(blocks[tail].page, block.page) {
                open_chain.remove(base);
                continue;
            }
            union(&mut parent, tail, i);
            open_chain.insert(base, i);
        } else if label.starts_with("cross_") || base == "table" {
            open_chain.insert(base, i);
        }
    }

    let texty = |block: &Block| block.label == "title" || base_type(&block.label) == "text";
    for i in 1..blocks.len() {
        let (prev, block) = (&blocks[i - 1], &blocks[i]);
        if block.merge_prev && prev.page == block.page && texty(block) && texty(prev) {
            union(&mut parent, i - 1, i);
        }
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot: std::collections::HashMap<usize, usize> = Default::default();
    for i in 0..blocks.len() {
        let root = find(&mut parent, i);
        match slot.get(&root) {
            Some(&g) => groups[g].push(i),
            None => {
                slot.insert(root, groups.len());
                groups.push(vec![i]);
            }
        }
    }
    groups
}

fn row_key(row: &str) -> (usize, String) {
    let text: String = TAG_RE
        .replace_all(row, " ")
        .chars()
        .map(|c| match c as u32 {
            0xFF01..=0xFF5E => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
            _ => c,
        })
        .collect();
    let compact: String = text.split_whitespace().collect();
    (CELL_RE.find_iter(row).count(), compact.to_lowercase())
}

/// Merge cross-page table fragments, skipping repeated leading header rows.
fn merge_table_fragments(fragments: &[&str]) -> String {
    let fragments: Vec<&str> = fragments
        .iter()
        .copied()
        .filter(|f| !f.trim().is_empty())
        .collect();
    if fragments.len() <= 1 {
        return fragments.first().map(|f| f.to_string()).unwrap_or_default();
    }
    let base_rows: Vec<&str> = TR_RE.find_iter(fragments[0]).map(|m| m.as_str()).collect();
    if base_rows.is_empty() {
        return fragments.join("\n\n");
    }
    let base_keys: Vec<_> = base_rows.iter().map(|r| row_key(r)).collect();
    let mut rows = base_rows.clone();
    let mut extra = Vec::new();
    for fragment in &fragments[1..] {
        let fragment_rows: Vec<&str> = TR_RE.find_iter(fragment).map(|m| m.as_str()).collect();
        if fragment_rows.is_empty() {
            extra.push(fragment.trim());
            continue;
        }
        let mut skip = 0;
        for row in fragment_rows.iter().take(MAX_HEADER_ROWS) {
            if skip < base_keys.len() && row_key(row) == base_keys[skip] {
                skip += 1;
            } else {
                break;
            }
        }
        rows.extend(&fragment_rows[skip..]);
    }
    let open = TABLE_OPEN_RE
        .find(fragments[0])
        .map_or("<table>", |m| m.as_str());
    let mut merged = format!("{open}{}</table>", rows.concat());
    if !extra.is_empty() {
        merged.push_str("\n\n");
        merged.push_str(&extra.join("\n\n"));
    }
    merged
}

/// Render each group to a Markdown item (figure, table, heading, or text).
fn render_blocks(blocks: &[Block], groups: &[Vec<usize>], keep_image_refs: bool) -> Vec<String> {
    let mut items = Vec::new();
    for members in groups {
        let head = &blocks[members[0]];
        let base = if head.label == RAW {
            RAW
        } else {
            base_type(&head.label)
        };
        let contents: Vec<&str> = members
            .iter()
            .map(|&k| blocks[k].content.as_str())
            .filter(|c| !c.is_empty() && !NOISE_CONTENT.contains(&c.to_lowercase().as_str()))
            .collect();
        if base == RAW {
            if let Some(first) = contents.first() {
                items.push(first.to_string());
            }
            continue;
        }
        if IMAGE_LABELS.contains(&base) {
            if keep_image_refs && let Some(bbox) = head.bbox {
                let [x1, y1, x2, y2] = bbox.map(|v| v.clamp(0, 1000));
                items.push(format!("![](page{}_{x1}_{y1}_{x2}_{y2}.jpg)", head.page));
            }
            if let Some(first) = contents.first() {
                items.push(first.to_string());
            }
            continue;
        }
        if contents.is_empty() {
            continue;
        }
        match base {
            "title" => {
                let joined = contents.concat();
                if STRAY_TOKEN_RE.replace_all(&joined, "").trim().is_empty() {
                    continue;
                }
                let level = head.level.unwrap_or(1).max(1);
                items.push(format!("{} {joined}", "#".repeat(level)));
            }
            "table" if contents.len() > 1 => items.push(merge_table_fragments(&contents)),
            _ => items.push(contents.concat()),
        }
    }
    items
}

fn head_level(item: &str) -> usize {
    let s = item.trim_start();
    s.len() - s.trim_start_matches('#').len()
}

fn is_heading(item: &str) -> bool {
    item.trim_start().starts_with('#')
}

/// Re-level headings under the first one when several share the top level.
fn single_root_titles(items: Vec<&str>) -> Vec<String> {
    let headings: Vec<usize> = (0..items.len()).filter(|&i| is_heading(items[i])).collect();
    let mut out: Vec<String> = items.iter().map(|s| s.to_string()).collect();
    if headings.len() <= 1 {
        return out;
    }
    let levels: Vec<usize> = headings.iter().map(|&i| head_level(items[i])).collect();
    let min_level = *levels.iter().min().expect("non-empty");
    if levels.iter().filter(|&&l| l == min_level).count() <= 1 {
        return out;
    }
    let first_level = levels[0];
    for (k, &i) in headings.iter().enumerate() {
        let body = items[i].trim_start()[levels[k]..].trim_start_matches(' ');
        let level = if k == 0 {
            1
        } else {
            (levels[k] + 2).saturating_sub(first_level).max(2)
        };
        out[i] = format!("{} {body}", "#".repeat(level));
    }
    out
}

fn fix_span_attrs(text: &str) -> String {
    SPAN_ATTR_RE
        .replace_all(text, |caps: &Captures| {
            let value = caps[2].trim();
            if value.parse::<i64>().is_ok() {
                caps[0].to_string()
            } else {
                let number = DIGITS_RE.find(value).map_or("1", |m| m.as_str());
                format!("{}=\"{number}\"", &caps[1])
            }
        })
        .into_owned()
}

/// Convert a raw Unlimited-OCR-2 output (one or many pages) to Markdown
/// like the official converter, which joins all pages into one document.
/// Grounded figures become `![](page{N}_x1_y1_x2_y2.jpg)` references on the
/// 0-1000 grid when `keep_image_refs` is set (the official output), and are
/// dropped otherwise; figure captions emitted inside the block are kept.
pub fn to_markdown(raw: &str, keep_image_refs: bool) -> String {
    let mut text = raw.trim();
    if let Some(rest) = text.strip_prefix("```markdown") {
        text = rest.trim();
    }
    if let Some(rest) = text.strip_suffix("```") {
        text = rest.trim();
    }
    let blocks = merge_adjacent_titles(parse_blocks(text));
    let groups = build_groups(&blocks);
    let items: Vec<String> = render_blocks(&blocks, &groups, keep_image_refs)
        .into_iter()
        .filter(|item| {
            !(is_heading(item) && item.trim_start().trim_start_matches('#').trim().is_empty())
        })
        .map(|item| {
            if !is_heading(&item) {
                return item;
            }
            let s = item.trim_start();
            let level = head_level(s);
            let body = WHITESPACE_RE.replace_all(&s[level..], " ");
            format!("{} {}", "#".repeat(level), body.trim())
        })
        .collect();
    let text = items
        .join("\n\n")
        .replace("\\coloneqq", ":=")
        .replace("\\eqqcolon", "=:");
    let text = STRAY_TOKEN_RE.replace_all(&text, "");
    let text = fix_span_attrs(&text);
    let text = BARE_BOX_LINE_RE.replace_all(&text, "");
    let text = TRAILING_SPACE_RE.replace_all(&text, "\n");
    let text = BLANK_LINES_RE.replace_all(&text, "\n\n");
    single_root_titles(text.trim().split("\n\n").collect()).join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_official_converter() {
        // Expected output of the official `convert` on this input: title
        // merging and re-levelling, a bare block, `<MERGE>` columns, a
        // cross-page text chain, a cross-page table with its repeated header
        // dropped, figure references, and stray-token cleanup.
        let raw = r#"<PAGE><BLOCK>title<LEVEL1>[10,10,500,40]<FORK>Annual   Report
<BLOCK>title<LEVEL1>[10,45,500,60]<FORK>2025
<BLOCK>text[10,70,480,300]<FORK>First column ends here and
<MERGE><BLOCK>text[520,70,990,300]<FORK>continues in the second.
<BLOCK>image[100,320,900,600]<FORK>Figure 1: Revenue.
<BLOCK>chart[100,620,900,700]<FORK>[non-text]
title<LEVEL2>[10,710,400,730]<FORK>Details\eqqcolon x
<BLOCK>cross_table[10,740,990,990]<FORK><table><tr><td>Year</td><td>Value</td></tr><tr><td>2024</td><td>1</td></tr></table>
<BLOCK>cross_text[10,995,990,1000]<FORK>A sentence that crosses
<PAGE><BLOCK>append_text[10,5,990,30]<FORK> the page break.
<BLOCK>append_table[10,40,990,300]<FORK><table><tr><td>Ｙｅａｒ</td><td>Value</td></tr><tr><td>2025</td><td colspan="2x">2</td></tr></table>
<BLOCK>title[10,310,500,330]<LEVEL1><FORK>Second   Chapter
<BLOCK>other[0,0,0,0]<FORK>[12, 34, 56, 78]
stray <|ref|>tail<|/ref|> text
<PAGE>
<PAGE>Loose text on page four.
"#;
        let expected = r#"# Annual Report 2025

First column ends here andcontinues in the second.

![](page1_100_320_900_600.jpg)

Figure 1: Revenue.

![](page1_100_620_900_700.jpg)

### Details=: x

<table><tr><td>Year</td><td>Value</td></tr><tr><td>2024</td><td>1</td></tr><tr><td>2025</td><td colspan="2">2</td></tr></table>

A sentence that crossesthe page break.

## Second Chapter

stray tail text

Loose text on page four."#;
        assert_eq!(to_markdown(raw, true), expected);
        assert!(!to_markdown(raw, false).contains("![]("));
    }
}
