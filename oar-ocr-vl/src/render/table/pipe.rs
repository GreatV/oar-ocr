//! Markdown grid tables used by structured region replies.

use regex::Regex;
use std::sync::LazyLock;

pub(crate) fn separator(line: &str) -> bool {
    static SEPARATOR: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$").unwrap());
    SEPARATOR.is_match(line)
}

pub(crate) fn cells(line: &str) -> Option<Vec<String>> {
    split_cells(line, false)
}

fn split_cells(line: &str, protect_code: bool) -> Option<Vec<String>> {
    let mut line = line.trim();
    if !line.contains('|') {
        return None;
    }
    line = line.strip_prefix('|').unwrap_or(line);
    line = line.strip_suffix('|').unwrap_or(line);
    let mut output = Vec::new();
    let mut cell = String::new();
    let mut escaped = false;
    let mut in_code = false;
    for c in line.chars() {
        if escaped {
            if c != '|' {
                cell.push('\\');
            }
            cell.push(c);
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '`' && protect_code {
            in_code = !in_code;
            cell.push(c);
        } else if c == '|' && !in_code {
            output.push(cell.trim().to_string());
            cell.clear();
        } else {
            cell.push(c);
        }
    }
    if escaped {
        cell.push('\\');
    }
    output.push(cell.trim().to_string());
    Some(output)
}

pub(crate) fn markdown_table_to_html(content: &str) -> Option<String> {
    let lines: Vec<_> = content.trim().lines().map(str::trim).collect();
    if lines.len() < 2 || lines.iter().any(|line| line.is_empty()) || !separator(lines[1]) {
        return None;
    }
    let header = cells(lines[0])?;
    if header.len() < 2 || cells(lines[1])?.len() != header.len() {
        return None;
    }
    let rows: Vec<_> = lines[2..]
        .iter()
        .map(|line| cells(line).filter(|row| row.len() == header.len()))
        .collect::<Option<_>>()?;
    let cell = |tag: &str, text: &str| {
        format!(
            "<{tag}>{}</{tag}>",
            html_escape::encode_text(text)
                .replace('"', "&quot;")
                .replace('\'', "&#x27;")
        )
    };
    let mut html = String::from("<table><tr>");
    for value in &header {
        html.push_str(&cell("th", value));
    }
    html.push_str("</tr>");
    for row in rows {
        html.push_str("<tr>");
        for value in row {
            html.push_str(&cell("td", &value));
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    Some(html)
}

pub(crate) fn malformed_table(content: &str) -> bool {
    let lines: Vec<_> = content.lines().collect();
    let count = |line: &str| {
        let line = line.trim();
        let line = line.strip_prefix('|').unwrap_or(line);
        let line = line.strip_suffix('|').unwrap_or(line);
        line.split('|').count()
    };
    let candidate = |line: &str| line.contains('|') && count(line) >= 2;
    let mut found = false;
    for (index, line) in lines.iter().enumerate().filter(|(_, line)| separator(line)) {
        let Some(header) = lines[..index]
            .iter()
            .rev()
            .find(|line| !line.trim().is_empty())
        else {
            return true;
        };
        let width = count(header);
        if !candidate(header) || width < 2 || count(line) != width {
            return true;
        }
        for line in lines[index + 1..]
            .iter()
            .take_while(|line| !line.trim().is_empty() && candidate(line))
        {
            if count(line) != width {
                return true;
            }
        }
        found = true;
    }
    !found && lines.iter().filter(|line| candidate(line)).count() >= 2
}

pub(crate) fn largest_table(content: &str) -> Option<String> {
    let mut best = None;
    let mut block = Vec::new();
    let mut separator_seen = false;
    for line in content.lines() {
        let line = line.trim().replace(['｜', '¦'], "|").replace(r"\|", "|");
        if let Some(row) = split_cells(&line, true).filter(|row| row.len() >= 2) {
            if block.is_empty() {
                separator_seen = false;
            } else if !separator_seen {
                separator_seen = separator_cells(&row);
            } else if !separator_cells(&row) && row.iter().any(|cell| noisy_cell(cell)) {
                continue;
            }
            block.push(row);
        } else {
            finish_block(&mut block, &mut best);
            separator_seen = false;
        }
    }
    finish_block(&mut block, &mut best);
    best
}

fn separator_cells(row: &[String]) -> bool {
    !row.is_empty()
        && row.iter().all(|cell| {
            let cell = cell.replace(' ', "");
            let cell = cell.strip_prefix(':').unwrap_or(&cell);
            let cell = cell.strip_suffix(':').unwrap_or(cell);
            cell.len() >= 3 && cell.bytes().all(|c| c == b'-')
        })
}

fn finish_block(block: &mut Vec<Vec<String>>, best: &mut Option<String>) {
    for separator in 1..block.len() {
        if !separator_cells(&block[separator]) || separator_cells(&block[separator - 1]) {
            continue;
        }
        let header = &block[separator - 1];
        let mut end = separator + 1;
        while end < block.len() && block[end].len() == header.len() && !separator_cells(&block[end])
        {
            end += 1;
        }
        retain_table(best, header, &block[separator + 1..end]);
    }
    let mut start = 0;
    while start < block.len() {
        if separator_cells(&block[start]) {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < block.len()
            && block[end].len() == block[start].len()
            && !separator_cells(&block[end])
        {
            end += 1;
        }
        retain_table(best, &block[start], &block[start + 1..end]);
        start = end;
    }
    block.clear();
}

fn retain_table(best: &mut Option<String>, header: &[String], rows: &[Vec<String>]) {
    if header.len() < 2 || rows.is_empty() {
        return;
    }
    let mut lines = vec![
        format_row(header),
        format_row(&vec!["---".into(); header.len()]),
    ];
    lines.extend(rows.iter().map(|row| format_row(row)));
    let candidate = lines.join("\n");
    let score = |table: &str| {
        (
            table.lines().count(),
            split_cells(table.lines().next().unwrap_or_default(), true).map_or(0, |row| row.len()),
            table.chars().count(),
        )
    };
    if best
        .as_ref()
        .is_none_or(|old| score(&candidate) > score(old))
    {
        *best = Some(candidate);
    }
}

fn format_row(row: &[String]) -> String {
    let cells: Vec<_> = row
        .iter()
        .map(|cell| {
            let mut cell = cell.split_whitespace().collect::<Vec<_>>().join(" ");
            if cell.matches('`').count() % 2 == 1 {
                cell.push('`');
            }
            cell.replace('|', r"\|")
        })
        .collect();
    format!("| {} |", cells.join(" | "))
}

fn noisy_cell(cell: &str) -> bool {
    if cell.chars().count() > 300
        || cell.matches("::").count() > 20
        || cell.matches(';').count() > 30
    {
        return true;
    }
    static FRAGMENT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\s*(?:::|;|,|\|)\s*").unwrap());
    let mut counts = std::collections::HashMap::new();
    for fragment in FRAGMENT.split(cell) {
        let fragment = fragment
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        if fragment.chars().count() < 3 {
            continue;
        }
        let count = counts.entry(fragment).or_insert(0);
        *count += 1;
        if *count >= 8 {
            return true;
        }
    }
    false
}

pub(crate) fn repair_chart_table(content: &str) -> String {
    let compact = content.split_whitespace().collect::<Vec<_>>().join(" ");
    let whole = if compact.chars().count() >= 80 {
        let doubled = compact.repeat(2);
        let first = compact.chars().next().map_or(0, char::len_utf8);
        let period = doubled[first..]
            .find(&compact)
            .map(|p| p + first)
            .unwrap_or(compact.len());
        period < compact.len() && compact.len().is_multiple_of(period)
    } else {
        false
    };
    let mut previous = None;
    let mut run = 0;
    let characters = content.chars().any(|c| {
        run = if Some(c) == previous { run + 1 } else { 1 };
        previous = Some(c);
        c.is_ascii_alphanumeric() && run >= 20
    });
    static TOKEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|[^\s]").unwrap());
    let body = content
        .lines()
        .filter(|line| !separator(line))
        .collect::<Vec<_>>()
        .join("\n");
    let mut previous = String::new();
    let mut run = 0;
    let tokens = TOKEN.find_iter(&body).any(|m| {
        let token = m.as_str().to_lowercase();
        run = if token == previous { run + 1 } else { 1 };
        previous = token;
        run >= 8
    });
    let mut in_table = false;
    let mut seen_separator = false;
    let mut cell_issue = false;
    for line in content.lines() {
        match cells(line) {
            None => {
                in_table = false;
                seen_separator = false;
            }
            Some(_) if !in_table => {
                in_table = true;
            }
            Some(_) if !seen_separator => {
                seen_separator = separator(line);
            }
            Some(row) => {
                cell_issue |= row.iter().any(|c| noisy_cell(c));
            }
        }
    }
    let lines = crate::render::text::has_repeated_lines(content, 2, 0, 30)
        || crate::render::text::has_repeated_lines(&body, 0, 4, 30)
        || crate::render::text::has_repeated_lines(content, 3, 0, 0);
    if content.contains(r"\|")
        || content.contains("\n\n\n\n\n")
        || whole
        || characters
        || tokens
        || cell_issue
        || lines
    {
        largest_table(content).unwrap_or_else(|| content.into())
    } else {
        content.into()
    }
}
