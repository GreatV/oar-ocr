//! Page Markdown from the ordered lines and the order model's layout heads:
//! a new block starts wherever the same-block logit between consecutive
//! lines is negative, and each block renders by its class.

use super::order::{BLOCK_CLASSES, Square, group_class, median_f32};
use regex::Regex;
use std::sync::LazyLock;

const HEADER: usize = 2;
const FOOTER: usize = 3;
const SECTION: usize = 1;
const LIST: usize = 4;
pub(super) const TABLE: usize = 5;

/// A line that is one formula.
static WHOLE_MATH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\$[^$]+\$$").expect("valid regex"));
/// A list marker, followed by text or alone on its line.
static ITEM: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)").expect("valid regex")
});

/// Han, kana, CJK punctuation and fullwidth forms: written without spaces.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x30FF | 0x3400..=0x9FFF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF | 0x20000..=0x2FA1F)
}

/// Joins the lines of one paragraph, undoing end-of-line hyphenation and
/// adding no space between two CJK characters.
pub(super) fn join_lines<S: AsRef<str>>(texts: &[S]) -> String {
    let mut out = String::new();
    for t in texts {
        let t = t.as_ref().trim();
        let Some(first) = t.chars().next() else {
            continue;
        };
        let mut tail = out.chars().rev();
        let (last, before) = (tail.next(), tail.next());
        if last == Some('-')
            && before.is_some_and(char::is_alphabetic)
            && first.is_alphabetic()
            && first.is_lowercase()
        {
            out.pop();
            out.push_str(t);
        } else if last.is_some_and(is_cjk) && is_cjk(first) {
            out.push_str(t);
        } else {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(t);
        }
    }
    out
}

/// Runs of the reading order with a positive same-block logit between
/// neighbours.
pub(super) fn split_blocks(order: &[usize], same: &Square) -> Vec<Vec<usize>> {
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    for &i in order {
        match blocks.last_mut() {
            Some(b) if same.at(*b.last().expect("blocks are non-empty"), i) > 0.0 => b.push(i),
            _ => blocks.push(vec![i]),
        }
    }
    blocks
}

fn escape_html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Cell lines grouped into rows by vertical center, each row left to right.
fn rows(boxes: &[[f32; 4]]) -> Vec<Vec<usize>> {
    let cy: Vec<f32> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let mut heights: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let h = median_f32(&mut heights);
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    idx.sort_by(|&i, &j| cy[i].total_cmp(&cy[j]));
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for i in idx {
        match rows.last_mut() {
            Some(r)
                if {
                    let mean = r.iter().map(|&k| cy[k] as f64).sum::<f64>() / r.len() as f64;
                    (cy[i] - mean as f32).abs() < 0.5 * h
                } =>
            {
                r.push(i)
            }
            _ => rows.push(vec![i]),
        }
    }
    for r in &mut rows {
        r.sort_by(|&a, &b| boxes[a][0].total_cmp(&boxes[b][0]));
    }
    rows
}

fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Joins the line rows of wrapped cells into one table row: a gap under half
/// the median row gap and 0.6 line heights, with fewer lines than the row
/// above.
fn merge_wrapped(rows: Vec<Vec<usize>>, boxes: &[[f32; 4]]) -> Vec<Vec<usize>> {
    if rows.len() < 3 {
        return rows;
    }
    let mut heights: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let h = median_f32(&mut heights) as f64;
    let gaps: Vec<f64> = rows
        .windows(2)
        .map(|w| {
            let top = w[1]
                .iter()
                .map(|&k| boxes[k][1])
                .fold(f32::INFINITY, f32::min);
            let bottom = w[0]
                .iter()
                .map(|&k| boxes[k][3])
                .fold(f32::NEG_INFINITY, f32::max);
            (top - bottom) as f64
        })
        .collect();
    let med = median_f64(&mut gaps.clone());
    let mut out = vec![rows[0].clone()];
    let mut prev = &rows[0];
    for (g, r) in gaps.iter().zip(&rows[1..]) {
        if *g < 0.5 * med && *g < 0.6 * h && r.len() < prev.len() {
            out.last_mut().expect("out is non-empty").extend(r);
        } else {
            out.push(r.clone());
        }
        prev = r;
    }
    out
}

/// An HTML table from cell lines, and the titles that go before it in the
/// Markdown: a first row that is one line across most of the table is a
/// title. The first remaining row is the header.
pub(super) fn table_html(boxes: &[[f32; 4]], texts: &[String]) -> (Vec<String>, String) {
    let rows = rows(boxes);
    let span_x = boxes.iter().map(|b| b[2]).fold(f32::NEG_INFINITY, f32::max)
        - boxes.iter().map(|b| b[0]).fold(f32::INFINITY, f32::min);
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if rows.len() > 2 && rows[0].len() == 1 && width > 1 {
        let t = rows[0][0];
        if boxes[t][2] - boxes[t][0] > 0.6 * span_x {
            let keep: Vec<usize> = (0..texts.len()).filter(|&k| k != t).collect();
            let kb: Vec<[f32; 4]> = keep.iter().map(|&k| boxes[k]).collect();
            let kt: Vec<String> = keep.iter().map(|&k| texts[k].clone()).collect();
            let (mut titles, html) = table_html(&kb, &kt);
            titles.insert(0, texts[t].clone());
            return (titles, html);
        }
    }
    let full: Vec<&Vec<usize>> = rows.iter().filter(|r| r.len() == width).collect();
    let rows = merge_wrapped(rows.clone(), boxes);
    // Column bands from the rows with the most cells: those rarely span.
    let bands: Vec<(f32, f32)> = (0..width)
        .map(|c| {
            let mut l: Vec<f32> = full.iter().map(|r| boxes[r[c]][0]).collect();
            let mut r: Vec<f32> = full.iter().map(|r| boxes[r[c]][2]).collect();
            (median_f32(&mut l), median_f32(&mut r))
        })
        .collect();
    // Widened to the midpoints between neighbours, so every x is in one band.
    let mut edges = vec![f32::NEG_INFINITY];
    edges.extend((0..width.saturating_sub(1)).map(|c| (bands[c].1 + bands[c + 1].0) / 2.0));
    edges.push(f32::INFINITY);
    let band_of = |x: f32| edges.partition_point(|&e| e <= x) as i64 - 1;
    let last_col = width as i64 - 1;
    let mut out = vec!["<table>".to_string()];
    for (ri, r) in rows.iter().enumerate() {
        let tag = if ri == 0 && rows.len() > 1 {
            "th"
        } else {
            "td"
        };
        let mut cells: std::collections::HashMap<usize, (usize, String)> = Default::default();
        let mut order = r.clone();
        order.sort_by(|&a, &b| {
            boxes[a][1]
                .total_cmp(&boxes[b][1])
                .then(boxes[a][0].total_cmp(&boxes[b][0]))
        });
        for k in order {
            let (x0, x1) = (boxes[k][0], boxes[k][2]);
            let (first, last) = (band_of(x0), band_of(x1));
            // A cell spans every band it covers by more than half of the band.
            let mut span: Vec<usize> = (first.max(0)..=last.min(last_col))
                .map(|c| c as usize)
                .filter(|&c| {
                    let (bl, br) = bands[c];
                    x1.min(br) - x0.max(bl) > 0.5 * (br - bl)
                })
                .collect();
            if span.is_empty() {
                span.push(first.max(0).min(last_col) as usize);
            }
            let c0 = span[0];
            let cell = match cells.remove(&c0) {
                None => (span.len(), texts[k].clone()),
                Some((n, text)) => (n.max(span.len()), format!("{text} {}", texts[k])),
            };
            cells.insert(c0, cell);
        }
        let mut tds = String::new();
        let mut c = 0;
        while c < width {
            if let Some((n, text)) = cells.get(&c) {
                let attr = if *n > 1 {
                    format!(" colspan=\"{n}\"")
                } else {
                    String::new()
                };
                tds.push_str(&format!("<{tag}{attr}>{}</{tag}>", escape_html(text)));
                c += n;
            } else {
                tds.push_str(&format!("<{tag}></{tag}>"));
                c += 1;
            }
        }
        out.push(format!("<tr>{tds}</tr>"));
    }
    out.push("</table>".to_string());
    (Vec::new(), out.join("\n"))
}

/// One layout block of a page, in reading order.
pub(super) struct Block {
    pub lines: Vec<usize>,
    pub class: usize,
    /// The block's Markdown; `None` for left-out page furniture.
    pub markdown: Option<String>,
    /// A table block's HTML, without the titles its Markdown puts before it.
    pub html: Option<String>,
}

/// Splits the reading order into blocks and renders each. `cells` holds a
/// line's table cells `(box, text)` when it splits into several.
pub(super) fn page_blocks(
    boxes: &[[f32; 4]],
    texts: &[String],
    order: &[usize],
    cls: &[f32],
    same: &Square,
    keep_furniture: bool,
    cells: &[Vec<([f32; 4], String)>],
) -> Vec<Block> {
    split_blocks(order, same)
        .into_iter()
        .map(|blk| {
            let class = group_class(cls, &blk);
            let lines: Vec<&str> = blk.iter().map(|&i| texts[i].as_str()).collect();
            // Every table block gets HTML (the page JSON content); its
            // Markdown renders as a table only with two or more lines, as in
            // the reference.
            let table = (class == TABLE).then(|| {
                let (mut cb, mut ct) = (Vec::new(), Vec::new());
                for &i in &blk {
                    if cells[i].is_empty() {
                        cb.push(boxes[i]);
                        ct.push(texts[i].clone());
                    } else {
                        for (b, t) in &cells[i] {
                            cb.push(*b);
                            ct.push(t.clone());
                        }
                    }
                }
                table_html(&cb, &ct)
            });
            let markdown = if matches!(class, HEADER | FOOTER) && !keep_furniture {
                None
            } else if lines.iter().all(|t| WHOLE_MATH.is_match(t.trim())) {
                // Display equations.
                Some(
                    lines
                        .iter()
                        .map(|t| {
                            let t = t.trim();
                            format!("$${}$$", &t[1..t.len() - 1])
                        })
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            } else if class == SECTION {
                Some(format!("## {}", join_lines(&lines)))
            } else if let Some((titles, html)) = table.as_ref().filter(|_| blk.len() > 1) {
                let mut parts = titles.clone();
                parts.push(html.clone());
                Some(parts.join("\n\n"))
            } else if class == LIST {
                let mut items: Vec<Vec<&str>> = Vec::new();
                for t in &lines {
                    match items.last_mut() {
                        Some(item) if !ITEM.is_match(t) => item.push(t),
                        _ => items.push(vec![t]),
                    }
                }
                Some(
                    items
                        .iter()
                        .map(|it| join_lines(it))
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            } else {
                Some(join_lines(&lines))
            };
            Block {
                lines: blk,
                class,
                markdown,
                html: table.map(|(_, html)| html),
            }
        })
        .collect()
}

/// The Markdown of a page: its rendered blocks, blank-line separated.
pub(super) fn page_markdown(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|b| b.markdown.as_deref())
        .filter(|m| !m.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The block label of a class id.
pub(super) fn label(class: usize) -> &'static str {
    BLOCK_CLASSES[class]
}
