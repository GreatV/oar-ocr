//! Page Markdown from the ordered lines and the order model's layout heads:
//! a new block starts wherever the same-block logit between consecutive
//! lines is negative, and each block renders by its class.

use super::order::{BLOCK_CLASSES, Square, cluster, group_class, median_f32};
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

/// The order model's table heads for some lines: same row, same column and
/// same cell logits, `[3, n, n]` row-major.
pub(super) struct TableHeads<'a> {
    pub n: usize,
    pub data: &'a [f32],
}

impl TableHeads<'_> {
    /// The `[n, n]` logits of head `k` between the lines `ids` (row-major).
    fn sub(&self, k: usize, ids: &[usize]) -> Vec<f32> {
        let base = k * self.n * self.n;
        ids.iter()
            .flat_map(|&i| ids.iter().map(move |&j| self.data[base + i * self.n + j]))
            .collect()
    }
}

/// The lines of one cell are stacked (they overlap horizontally). When the
/// cell head joins side-by-side lines in more than 10% of its cells, it has
/// failed on this table and the geometric assembly is used instead.
fn heads_plausible(boxes: &[[f32; 4]], same_cell: &[f32]) -> bool {
    let cells = cluster(&Square {
        n: boxes.len(),
        data: same_cell,
    });
    let bad = cells
        .iter()
        .filter(|c| {
            c.len() >= 2
                && c.iter().any(|&i| {
                    c.iter()
                        .any(|&j| boxes[i][2].min(boxes[j][2]) - boxes[i][0].max(boxes[j][0]) < 0.0)
                })
        })
        .count();
    bad as f64 <= 0.1 * cells.len() as f64
}

/// An HTML table from the order model's table heads (`tab`: the heads of
/// the table's own lines): cells from the same-cell head, then rows and
/// columns of cells from the mean same-row and same-column logits.
fn table_html_heads(boxes: &[[f32; 4]], texts: &[&str], tab: &[Vec<f32>; 3]) -> String {
    let n = boxes.len();
    let cells = cluster(&Square { n, data: &tab[2] });
    let m = cells.len();
    let cb: Vec<[f32; 4]> = cells
        .iter()
        .map(|c| {
            let bx = |k: usize| c.iter().map(move |&i| boxes[i][k]);
            [
                bx(0).fold(f32::INFINITY, f32::min),
                bx(1).fold(f32::INFINITY, f32::min),
                bx(2).fold(f32::NEG_INFINITY, f32::max),
                bx(3).fold(f32::NEG_INFINITY, f32::max),
            ]
        })
        .collect();
    let ctext: Vec<String> = cells
        .iter()
        .map(|c| {
            let mut c = c.clone();
            c.sort_by(|&a, &b| {
                boxes[a][1]
                    .total_cmp(&boxes[b][1])
                    .then(boxes[a][0].total_cmp(&boxes[b][0]))
            });
            c.iter().map(|&k| texts[k]).collect::<Vec<_>>().join(" ")
        })
        .collect();
    let cell_logits = |k: usize| -> Vec<f32> {
        let mut out = vec![0f32; m * m];
        for a in 0..m {
            for b in a + 1..m {
                let pairs: Vec<f32> = cells[a]
                    .iter()
                    .flat_map(|&i| cells[b].iter().map(move |&j| tab[k][i * n + j]))
                    .collect();
                let v = super::mean_f32(&pairs) as f32;
                out[a * m + b] = v;
                out[b * m + a] = v;
            }
        }
        out
    };
    let groups = |k: usize, edge: usize| {
        let mut g = cluster(&Square {
            n: m,
            data: &cell_logits(k),
        });
        let key = |g: &Vec<usize>| g.iter().map(|&c| cb[c][edge]).fold(f32::INFINITY, f32::min);
        g.sort_by(|a, b| key(a).total_cmp(&key(b)));
        g
    };
    let rows = groups(0, 1);
    let cols = groups(1, 0);
    let mut row_of = vec![0; m];
    let mut col_of = vec![0; m];
    for (r, g) in rows.iter().enumerate() {
        for &c in g {
            row_of[c] = r;
        }
    }
    for (k, g) in cols.iter().enumerate() {
        for &c in g {
            col_of[c] = k;
        }
    }
    let centers: Vec<f32> = cols
        .iter()
        .map(|g| {
            median_f32(
                &mut g
                    .iter()
                    .map(|&c| (cb[c][0] + cb[c][2]) / 2.0)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let h = median_f32(&mut boxes.iter().map(|b| b[3] - b[1]).collect::<Vec<_>>());
    let mut grid: std::collections::HashMap<(usize, usize), (usize, String)> = Default::default();
    for c in 0..m {
        let c0 = col_of[c];
        let mut span = 1;
        while c0 + span < cols.len()
            && cb[c][0] - h <= centers[c0 + span]
            && centers[c0 + span] <= cb[c][2] + h
        {
            span += 1;
        }
        // Two cells on one slot keep both texts.
        let cell = match grid.remove(&(row_of[c], c0)) {
            Some((n, text)) => (n.max(span), format!("{text} {}", ctext[c])),
            None => (span, ctext[c].clone()),
        };
        grid.insert((row_of[c], c0), cell);
    }
    let mut out = vec!["<table>".to_string()];
    for r in 0..rows.len() {
        let tag = if r == 0 && rows.len() > 1 { "th" } else { "td" };
        let mut tds = String::new();
        let mut c = 0;
        while c < cols.len() {
            if let Some((span, text)) = grid.get(&(r, c)) {
                // A span stops at the next occupied slot, so close columns
                // keep their cells (the reference skips them, dropping text).
                let span = (1..*span)
                    .find(|&k| grid.contains_key(&(r, c + k)))
                    .unwrap_or(*span);
                let attr = if span > 1 {
                    format!(" colspan=\"{span}\"")
                } else {
                    String::new()
                };
                tds.push_str(&format!("<{tag}{attr}>{}</{tag}>", escape_html(text)));
                c += span;
            } else {
                tds.push_str(&format!("<{tag}></{tag}>"));
                c += 1;
            }
        }
        out.push(format!("<tr>{tds}</tr>"));
    }
    out.push("</table>".to_string());
    out.join("\n")
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

/// The order model's layout heads for a page's lines.
pub(super) struct Heads<'a> {
    /// Block class logits, `[n, classes]`.
    pub cls: &'a [f32],
    /// Same-block logits.
    pub same: Square<'a>,
    /// The table heads, when tables are built from them.
    pub tab: Option<TableHeads<'a>>,
}

/// Splits the reading order into blocks and renders each. `cells` holds a
/// line's table cells `(box, text)` when it splits into several.
pub(super) fn page_blocks(
    boxes: &[[f32; 4]],
    texts: &[String],
    order: &[usize],
    heads: &Heads,
    keep_furniture: bool,
    cells: &[Vec<([f32; 4], String)>],
) -> Vec<Block> {
    let (cls, tab) = (heads.cls, heads.tab.as_ref());
    split_blocks(order, &heads.same)
        .into_iter()
        .map(|blk| {
            let class = group_class(cls, &blk);
            let lines: Vec<&str> = blk.iter().map(|&i| texts[i].as_str()).collect();
            // Every table block gets HTML (the page JSON content); its
            // Markdown renders as a table only with two or more lines, as in
            // the reference. With the table heads, a table of two or more
            // lines is built from them when they look sound.
            let heads = tab
                .filter(|_| class == TABLE && blk.len() > 1)
                .and_then(|tab| {
                    let bx: Vec<[f32; 4]> = blk.iter().map(|&i| boxes[i]).collect();
                    let sub = [0, 1, 2].map(|k| tab.sub(k, &blk));
                    heads_plausible(&bx, &sub[2])
                        .then(|| (Vec::new(), table_html_heads(&bx, &lines, &sub)))
                });
            let table = heads.or_else(|| {
                (class == TABLE).then(|| {
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
                })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Two close columns: each wide left cell reaches the right column's
    /// center within a line height, but the right cells still render.
    #[test]
    fn head_spans_stop_at_occupied_cells() {
        let boxes = [
            [0., 0., 20., 10.],
            [22., 0., 26., 10.],
            [0., 20., 20., 30.],
            [22., 20., 26., 30.],
        ];
        let (row, col) = ([0, 0, 1, 1], [0, 1, 0, 1]);
        let logits = |same: &dyn Fn(usize, usize) -> bool| -> Vec<f32> {
            (0..16)
                .map(|k| if same(k / 4, k % 4) { 5.0 } else { -5.0 })
                .collect()
        };
        let tab = [
            logits(&|i, j| row[i] == row[j]),
            logits(&|i, j| col[i] == col[j]),
            logits(&|i, j| i == j),
        ];
        assert_eq!(
            table_html_heads(&boxes, &["A", "B", "C", "D"], &tab),
            "<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td>C</td><td>D</td></tr>\n</table>"
        );
    }
}
