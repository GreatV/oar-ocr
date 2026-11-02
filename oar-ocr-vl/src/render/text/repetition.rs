//! Shared repetition algorithms and their length/table gates.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum RepeatAlgorithm {
    #[default]
    Legacy,
    ExactSuffix,
    PartialSuffix,
    StrongestSuffix,
    Content,
    Paragraph,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RepetitionOptions {
    pub algorithm: RepeatAlgorithm,
    pub minimum_chars: usize,
    pub table_minimum_chars: Option<usize>,
    pub exempt_tables: bool,
    pub preserve_complete_tables: bool,
    pub min_period: usize,
    pub max_period: Option<usize>,
    pub min_repeats: usize,
    pub min_repeat_chars: usize,
    pub search_window: Option<usize>,
    pub retain_partial_tail: bool,
    pub line_threshold: usize,
    pub char_threshold: usize,
    pub min_len: usize,
    pub consecutive_lines: usize,
    pub trim: bool,
    pub trim_result: bool,
}

impl Default for RepetitionOptions {
    fn default() -> Self {
        Self {
            algorithm: RepeatAlgorithm::Legacy,
            minimum_chars: 0,
            table_minimum_chars: None,
            exempt_tables: false,
            preserve_complete_tables: false,
            min_period: 1,
            max_period: None,
            min_repeats: 10,
            min_repeat_chars: 0,
            search_window: None,
            retain_partial_tail: true,
            line_threshold: 10,
            char_threshold: 10,
            min_len: 10,
            consecutive_lines: 3,
            trim: false,
            trim_result: false,
        }
    }
}

pub(crate) fn has_repeated_lines(
    input: &str,
    consecutive: usize,
    total: usize,
    minimum_chars: usize,
) -> bool {
    let mut counts = HashMap::new();
    let mut previous = "";
    let mut run = 0;
    for line in input.lines().map(str::trim) {
        if line.is_empty() || line.chars().count() < minimum_chars {
            previous = "";
            run = 0;
            continue;
        }
        run = if line == previous { run + 1 } else { 1 };
        previous = line;
        let count = counts.entry(line).or_insert(0usize);
        *count += 1;
        if (consecutive > 0 && run >= consecutive) || (total > 0 && *count >= total) {
            return true;
        }
    }
    false
}

pub(crate) fn clean_repetition(input: &str, is_table: bool, options: &RepetitionOptions) -> String {
    let text = if options.trim { input.trim() } else { input };
    let chars: Vec<_> = text.chars().collect();
    let minimum = if is_table {
        options.table_minimum_chars.unwrap_or(options.minimum_chars)
    } else {
        options.minimum_chars
    };
    if chars.len() < minimum
        || (is_table && options.exempt_tables)
        || (options.preserve_complete_tables
            && text.trim().starts_with("<table")
            && text.trim().ends_with("</table>"))
    {
        return if options.trim_result {
            text.trim().into()
        } else {
            text.into()
        };
    }
    let result = match options.algorithm {
        RepeatAlgorithm::Legacy => truncate_legacy(
            text,
            options.line_threshold,
            options.char_threshold,
            options.min_len,
        ),
        RepeatAlgorithm::Paragraph => clean_paragraph_repetition(text, options),
        RepeatAlgorithm::Content => clean_repeated_content(text, &chars, options),
        _ => clean_suffix(text, &chars, options),
    };
    if options.trim_result {
        result.trim().into()
    } else {
        result
    }
}

fn clean_suffix(text: &str, chars: &[char], options: &RepetitionOptions) -> String {
    let repeats = options.min_repeats.max(2);
    let window_start = chars
        .len()
        .saturating_sub(options.search_window.unwrap_or(chars.len()));
    let window = &chars[window_start..];
    let max_period = options
        .max_period
        .unwrap_or(window.len() / repeats)
        .min(window.len() / repeats);
    let mut best = (0, 0);
    for period in options.min_period.max(1)..=max_period {
        let span = if matches!(options.algorithm, RepeatAlgorithm::ExactSuffix) {
            let suffix = &window[window.len() - period..];
            window
                .rchunks_exact(period)
                .take_while(|chunk| *chunk == suffix)
                .count()
                * period
        } else {
            suffix_span(window, period)
        };
        if span < repeats * period || span < options.min_repeat_chars {
            continue;
        }
        if !matches!(options.algorithm, RepeatAlgorithm::StrongestSuffix) {
            best = (span, period);
            break;
        }
        if span > best.0 {
            best = (span, period);
        }
    }
    let period = best.1;
    if period == 0 {
        return text.into();
    }
    let span = if matches!(options.algorithm, RepeatAlgorithm::StrongestSuffix) {
        suffix_span(chars, period)
    } else {
        best.0
    };
    let remainder = span % period;
    if options.retain_partial_tail {
        chars[..chars.len() - span + period]
            .iter()
            .chain(&chars[chars.len() - remainder..])
            .collect()
    } else {
        chars[..chars.len() - span]
            .iter()
            .chain(&chars[chars.len() - remainder - period..chars.len() - remainder])
            .collect()
    }
}

fn suffix_span(chars: &[char], period: usize) -> usize {
    (0..chars.len() - period)
        .take_while(|&i| chars[chars.len() - 1 - i] == chars[chars.len() - 1 - i - period])
        .count()
        + period
}

pub(super) fn find_shortest_repeating_substring(chars: &[char]) -> Option<&[char]> {
    let n = chars.len();
    for i in 1..=n / 2 {
        if n.is_multiple_of(i) {
            let unit = &chars[..i];
            let mut matches = true;
            for start in (0..n).step_by(i) {
                if &chars[start..start + i] != unit {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Some(unit);
            }
        }
    }
    None
}

fn find_repeating_suffix(
    chars: &[char],
    min_len: usize,
    min_repeats: usize,
) -> Option<(&[char], &[char], usize)> {
    let n = chars.len();
    for i in (min_len..=n / min_repeats).rev() {
        let total = i * min_repeats;
        if n < total {
            continue;
        }
        let unit = &chars[n - i..n];
        let mut matches = true;
        let start = n - total;
        for offset in 0..min_repeats {
            let chunk_start = start + offset * i;
            if &chars[chunk_start..chunk_start + i] != unit {
                matches = false;
                break;
            }
        }
        if matches {
            let mut count = 0;
            let mut end = n;
            while end >= i && &chars[end - i..end] == unit {
                count += 1;
                end -= i;
            }
            let prefix = &chars[..end];
            return Some((prefix, unit, count));
        }
    }
    None
}

/// Detect and truncate repetitive content.
fn truncate_legacy(
    content: &str,
    line_threshold: usize,
    char_threshold: usize,
    min_len: usize,
) -> String {
    let stripped = content.trim();
    if stripped.is_empty() {
        return content.to_string();
    }
    let chars: Vec<char> = stripped.chars().collect();
    let stripped_chars = chars.len();

    if !stripped.contains('\n')
        && stripped_chars > 100
        && let Some((prefix, unit, count)) = find_repeating_suffix(&chars, 8, 5)
        && unit.len() * count > stripped_chars / 2
    {
        return prefix.iter().collect();
    }

    if !stripped.contains('\n')
        && stripped_chars > min_len
        && let Some(unit) = find_shortest_repeating_substring(&chars)
    {
        let count = stripped_chars / unit.len();
        if count >= char_threshold {
            return unit.iter().collect();
        }
    }

    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return content.to_string();
    }

    let total_lines = lines.len();
    if total_lines < line_threshold {
        return content.to_string();
    }

    let mut counts = HashMap::new();
    for line in &lines {
        *counts.entry(*line).or_insert(0usize) += 1;
    }

    if let Some((most_common, count)) = counts.into_iter().max_by_key(|(_, c)| *c)
        && count >= line_threshold
        && (count as f32 / total_lines as f32) >= 0.8
    {
        return most_common.to_string();
    }

    content.to_string()
}

fn clean_repeated_content(text: &str, chars: &[char], options: &RepetitionOptions) -> String {
    // GLM SDK utils/result_postprocess_utils.py (Apache-2.0).
    for start in 0..chars.len() {
        for period in options.min_period
            ..=options
                .max_period
                .unwrap_or((chars.len() - start) / options.min_repeats)
                .min((chars.len() - start) / options.min_repeats)
        {
            let unit = &chars[start..start + period];
            if (1..options.min_repeats)
                .all(|i| chars[start + i * period..start + (i + 1) * period] == *unit)
            {
                return chars[..start + period].iter().collect();
            }
        }
    }
    let lines: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for line in &lines {
        *counts.entry(line.trim()).or_default() += 1;
    }
    if let Some(common) = lines
        .iter()
        .map(|line| line.trim())
        .max_by_key(|line| counts[line])
        && counts[common] >= options.line_threshold
        && counts[common] * 5 >= lines.len() * 4
    {
        for (index, _) in lines.iter().enumerate() {
            if lines[index].trim() == common
                && lines[index..]
                    .iter()
                    .take(options.consecutive_lines)
                    .filter(|line| line.trim() == common)
                    .count()
                    == options.consecutive_lines
            {
                let mut seen = 0;
                let mut kept = Vec::new();
                for line in text.lines() {
                    kept.push(line);
                    if !line.trim().is_empty() {
                        seen += 1;
                    }
                    if seen == index + 1 {
                        return kept.join("\n");
                    }
                }
            }
        }
    }
    text.into()
}

pub(crate) fn truncate_word_repetition(
    text: &str,
    min_words: usize,
    max_words: usize,
    repeats: usize,
    keep_closures: bool,
) -> String {
    use regex::Regex;
    use std::sync::LazyLock;
    static WORDS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\S+").unwrap());
    static CLOSURES: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\end\{[^}]*\}|\\\]|\$\$|</(?:table|tr|td|th)>").unwrap());
    let words: Vec<_> = WORDS.find_iter(text).collect();
    if repeats < 2 || words.len() < min_words.saturating_mul(repeats) {
        return text.into();
    }
    let lowered: Vec<_> = words.iter().map(|w| w.as_str().to_lowercase()).collect();
    let mut cut = text.len();
    for width in min_words..=max_words.min(words.len() / repeats) {
        for start in 0..=words.len() - width * repeats {
            if (1..repeats).all(|r| {
                lowered[start..start + width] == lowered[start + r * width..start + (r + 1) * width]
            }) {
                cut = cut.min(words[start].start());
                break;
            }
        }
    }
    if cut == text.len() {
        return text.into();
    }
    let mut result = text[..cut].trim_end().to_string();
    if keep_closures {
        let closing: Vec<_> = CLOSURES
            .find_iter(&text[cut..])
            .map(|m| m.as_str())
            .collect();
        if !closing.is_empty() {
            result.push(' ');
            result.push_str(&closing.join(" "));
        }
    }
    result
}

fn clean_paragraph_repetition(text: &str, options: &RepetitionOptions) -> String {
    let stripped = text.trim();
    let chars: Vec<_> = stripped.chars().collect();
    if !stripped.contains('\n') && chars.len() > 100 {
        for width in options.min_period..=chars.len() / options.min_repeats {
            let unit = &chars[chars.len() - width..];
            let mut count = 0;
            let mut position = chars.len();
            while position >= width && &chars[position - width..position] == unit {
                count += 1;
                position -= width;
            }
            if count >= options.min_repeats && width * count > chars.len() / 2 {
                return chars[..position].iter().collect();
            }
        }
    }
    if !stripped.contains('\n') && chars.len() > 10 {
        for width in 1..=chars.len() / 2 {
            let count = chars.len() / width;
            if (0..width * count).all(|i| chars[i] == chars[i % width]) {
                if count >= options.char_threshold {
                    return chars[..width].iter().collect();
                }
                break;
            }
        }
    }
    let mut counts = HashMap::new();
    let mut top = ("", 0usize);
    let mut total = 0;
    for line in stripped.lines().map(str::trim).filter(|s| !s.is_empty()) {
        total += 1;
        let count = counts.entry(line).or_insert(0);
        *count += 1;
        if *count > top.1 {
            top = (line, *count);
        }
    }
    if total >= options.line_threshold
        && top.1 >= options.line_threshold
        && top.1 as f64 / total as f64 >= 0.8
    {
        return top.0.into();
    }
    text.into()
}
