//! Render chart metadata and numeric tables, associating adjacent captions by page geometry.
use crate::render::table::latex;
use regex::{Captures, Regex};
use std::{collections::HashMap, fmt::Write, sync::LazyLock};

static CHART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)(<chart\b[^>]*>)(.*?)(</chart>)").unwrap());
static TITLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^title:[ \t]*([^\n]*)\n?").unwrap());
static TABLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^\|[^\n]*\|[ \t]*\n\|[ \t:|\-]+\|[ \t]*\n(?:\|[^\n]*\|[ \t]*(?:\n|$))+")
        .unwrap()
});
static PAIR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\s*([+−-]?\d[\d,. ]*(?:[kKmMbB])?)\s*\(\s*([+−-]?\d[\d,. ]*(?:[kKmMbB])?\s*%)\s*\)\s*$",
    )
    .unwrap()
});
static PANEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^panel:\s*([^;\n]+)\s*;[^;\n]+;\s*([^\n]+)").unwrap());
static MARKER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"table\[([^\]\n]+)\]:\s*$").unwrap());

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn pipe_table(raw: &str, title: &str, panel: &str, subtitles: &[&str]) -> Option<String> {
    if raw.contains(r"\|") || raw.contains('`') {
        return None;
    }
    let rows: Vec<Vec<_>> = raw
        .trim()
        .lines()
        .map(|l| {
            l.trim()
                .trim_matches('|')
                .split('|')
                .map(str::trim)
                .collect()
        })
        .collect();
    let head = rows.first()?;
    if rows.len() < 3
        || rows.iter().any(|r| r.len() != head.len())
        || rows[1].iter().any(|s| {
            let delim = s.replace(' ', "");
            let bare = delim.trim_matches(':');
            bare.is_empty() || !bare.bytes().all(|b| b == b'-')
        })
    {
        return None;
    }
    let split: Vec<_> = (0..head.len())
        .map(|c| {
            let values: Vec<_> = rows[2..]
                .iter()
                .map(|r| r[c])
                .filter(|s| !s.is_empty())
                .collect();
            c != 0 && values.len() >= 2 && values.iter().all(|s| PAIR.is_match(s))
        })
        .collect();
    let divided = split.contains(&true);
    if !divided && title.is_empty() && panel.is_empty() {
        return None;
    }
    let mut caption = [title, panel]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" — ");
    if !caption.is_empty() {
        for &s in subtitles {
            if !caption.contains(s) {
                caption.push_str(" — ");
                caption.push_str(s);
            }
        }
    }
    let mut html = String::from("<table>");
    if !caption.is_empty() {
        let _ = write!(html, "<caption>{}</caption>", escape(&caption));
    }
    html.push_str("<thead><tr>");
    for (i, col) in head.iter().enumerate() {
        let span = if split[i] {
            " colspan=\"2\""
        } else if divided {
            " rowspan=\"2\""
        } else {
            ""
        };
        let _ = write!(html, "<th{span} scope=\"col\">{}</th>", escape(col));
    }
    html.push_str("</tr>");
    if divided {
        html.push_str("<tr>");
        for _ in split.iter().filter(|v| **v) {
            html.push_str("<th scope=\"col\">Value</th><th scope=\"col\">Percent</th>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</thead><tbody>");
    for row in &rows[2..] {
        html.push_str("<tr>");
        for (i, value) in row.iter().enumerate() {
            if split[i] {
                if let Some(m) = PAIR.captures(value) {
                    let _ = write!(
                        html,
                        "<td>{}</td><td>{}</td>",
                        escape(m[1].trim()),
                        escape(m[2].trim())
                    );
                } else {
                    html.push_str("<td></td><td></td>");
                }
            } else {
                let _ = write!(html, "<td>{}</td>", escape(value));
            }
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>\n");
    Some(html)
}

pub(crate) fn render(text: &str, caption: Option<&str>) -> String {
    let rendered = CHART.replace_all(text, |m: &Captures<'_>| {
        let body = &m[2];
        if body.contains("<chart") {
            return m[0].to_owned();
        }
        let title = TITLE
            .captures(body)
            .map(|m| m[1].trim().to_owned())
            .unwrap_or_default();
        let title = match caption {
            Some(v) if title.is_empty() => v.to_owned(),
            Some(v) if !title.contains(v) => format!("{title} — {v}"),
            _ => title,
        };
        let panels: HashMap<_, _> = PANEL
            .captures_iter(body)
            .map(|m| (m[1].trim().to_owned(), m[2].trim().to_owned()))
            .collect();
        let subtitles: Vec<_> = body
            .lines()
            .filter_map(|l| l.strip_prefix("subtitle:"))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        let mut changed = false;
        let converted = TABLE.replace_all(body, |t: &Captures<'_>| {
            let before = &body[..t.get(0).unwrap().start()];
            let panel = MARKER
                .captures(before)
                .and_then(|p| panels.get(p[1].trim()))
                .map_or("", String::as_str);
            if let Some(v) = pipe_table(&t[0], &title, panel, &subtitles) {
                changed = true;
                v
            } else {
                t[0].to_owned()
            }
        });
        if !changed {
            return m[0].to_owned();
        }
        format!("{}{}{}", &m[1], TITLE.replace(&converted, ""), &m[3])
    });
    latex::to_html(&rendered, true)
}

fn nearest(candidates: &mut [(f64, usize)]) -> Option<usize> {
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let &(gap, id) = candidates.first()?;
    (!candidates.get(1).is_some_and(|v| (v.0 - gap).abs() < 1e-9)).then_some(id)
}

pub(crate) struct Region<'a> {
    pub class: &'a str,
    pub coords: [f64; 4],
    pub body: &'a str,
}

pub(crate) fn relocate(blocks: &[Region<'_>], fragments: &mut [String]) {
    static MARKUP: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"\*\*|__|~~|`|<[/a-zA-Z!]|\\[A-Za-z&%_]|\]\(|(?:^|\W)_[^_\n]+_(?:\W|$)")
            .unwrap()
    });
    let mut targets: HashMap<usize, Vec<(f64, usize)>> = HashMap::new();
    for (i, b) in blocks.iter().enumerate() {
        let body = b.body.replace("<tbc>", "");
        let text = body.trim();
        if b.class != "Caption" || text.is_empty() || text.contains('\n') || MARKUP.is_match(text) {
            continue;
        }
        let [x0, y0, x1, y1] = b.coords;
        let mut candidates = Vec::new();
        for j in [i.checked_sub(1), i.checked_add(1)].into_iter().flatten() {
            let Some(other) = blocks.get(j) else { continue };
            if other.class != "Chart" {
                continue;
            }
            let [a, top, c, bottom] = other.coords;
            if y1 > top {
                continue;
            }
            let gap = (top - y1).max(y0 - bottom);
            let width = (x1 - x0).min(c - a);
            if width > 0. && (0.0..=0.03).contains(&gap) && x1.min(c) - x0.max(a) >= 0.8 * width {
                candidates.push((gap, j));
            }
        }
        if let Some(j) = nearest(&mut candidates) {
            targets.entry(j).or_default().push((candidates[0].0, i));
        }
    }
    for (j, mut candidates) in targets {
        let Some(i) = nearest(&mut candidates) else {
            continue;
        };
        if blocks[j].body.matches("<chart").count() != 1 {
            continue;
        }
        let body = blocks[i].body.replace("<tbc>", "");
        let title = body.trim();
        let chart = blocks[j].body.replace("<tbc>", "");
        let derived = render(chart.trim(), Some(title));
        if derived
            .split("<caption>")
            .skip(1)
            .filter_map(|s| s.split_once("</caption>"))
            .any(|(s, _)| s.contains(&escape(title)))
        {
            fragments[i].clear();
            fragments[j] = derived;
        }
    }
}
