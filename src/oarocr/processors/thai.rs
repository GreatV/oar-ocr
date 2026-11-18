//! Thai canonicalization and line filtering from OpenThai2.0-OCR-Text 2.1.3.
use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

const VOWELS: &str = "ัิีึืฺุู็";
const TONES: &str = "่้๊๋์๎";
const LEGACY: &str = "ฐิีึื่้๊๋์่้๊๋์ญัํ็่้๊๋์ฺุู";
static SARA_AM: LazyLock<Regex> = LazyLock::new(|| Regex::new("([่-๋]?)ํ([่-๋]?)า").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t\x0c\x0b]+").unwrap());

fn group(c: char) -> u8 {
    if VOWELS.contains(c) {
        1
    } else if TONES.contains(c) {
        2
    } else if c == 'ํ' {
        3
    } else {
        0
    }
}
pub(crate) fn canonical(text: &str) -> String {
    let norm: Vec<_> = text
        .chars()
        .filter_map(|c| match c {
            '\u{f700}'..='\u{f71a}' => LEGACY.chars().nth(c as usize - 0xf700),
            '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{00ad}' | '\u{feff}' => None,
            '\u{a0}' => Some(' '),
            other => Some(other),
        })
        .nfc()
        .collect();
    let mut ordered = String::new();
    let mut i = 0;
    while i < norm.len() {
        if group(norm[i]) == 0 {
            ordered.push(norm[i]);
            i += 1;
            continue;
        }
        let start = i;
        while i < norm.len() && group(norm[i]) != 0 {
            i += 1;
        }
        let mut marks = norm[start..i].to_vec();
        marks.sort_by_key(|&c| group(c));
        marks.dedup();
        ordered.extend(marks);
    }
    let joined = SARA_AM
        .replace_all(&ordered, "${1}${2}ำ")
        .replace("เเ", "แ")
        .replace("ำำ", "ำ")
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let text = joined
        .split('\n')
        .map(|line| SPACES.replace_all(line, " ").trim().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    let chars: Vec<_> = text.trim().chars().collect();
    let mut out = String::new();
    for (i, &ch) in chars.iter().enumerate() {
        let prev = i
            .checked_sub(1)
            .and_then(|p| chars.get(p))
            .copied()
            .unwrap_or(' ');
        let next = chars.get(i + 1).copied().unwrap_or(' ');
        let thai = if next == '.' {
            chars.get(i + 2).copied().unwrap_or(' ')
        } else {
            next
        };
        let ch = if matches!(ch, 'w' | 'W')
            && !prev.is_ascii_alphabetic()
            && matches!(thai,'ก'..='ฮ'|'เ'..='ไ')
        {
            'พ'
        } else if matches!(ch, 'o' | 'O' | '0')
            && (matches!(prev, '๐'..='๙') || matches!(next, '๐'..='๙'))
        {
            '๐'
        } else {
            ch
        };
        out.push(ch);
    }
    out
}

pub(crate) fn digits(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '๐'..='๙')
}
pub(crate) fn mixed(s: &str) -> usize {
    s.split_whitespace()
        .filter(|s| s.chars().any(|c| matches!(c, 'ก'..='๎')))
        .map(|s| s.bytes().filter(u8::is_ascii_alphabetic).count())
        .sum()
}

pub(crate) fn drop_reason(
    text: &str,
    conf: f64,
    threshold: f32,
    height: Option<f32>,
) -> Option<&'static str> {
    if text.is_empty() {
        return Some("empty");
    }
    if conf < f64::from(threshold) {
        return Some("low_confidence");
    }
    let mut bare = text.chars().filter(|&c| c != ' ');
    let count = bare.clone().count();
    if count < 2 && !(count == 1 && bare.all(digits) && height.is_some_and(|h| h as f64 >= 0.62)) {
        return Some("too_short");
    }
    if conf < 0.9 {
        let cs: Vec<_> = text.chars().collect();
        let mut thai = 0;
        let mut bad = 0;
        let cons = |c: Option<&char>| matches!(c, Some('ก'..='ฮ'));
        for (i, &c) in cs.iter().enumerate() {
            if !('\u{e00}'..='\u{e7f}').contains(&c) {
                continue;
            }
            thai += 1;
            let prev = i.checked_sub(1).and_then(|j| cs.get(j));
            let valid = cons(prev)
                || (prev.is_some_and(|&c| group(c) > 0)
                    && cons(i.checked_sub(2).and_then(|j| cs.get(j))));
            if (group(c) > 0 && !valid) || (matches!(c, 'เ'..='ไ') && !cons(cs.get(i + 1))) {
                bad += 1;
            }
        }
        if thai > 0 && bad as f64 / thai as f64 > 0.05 {
            return Some("invalid_thai");
        }
    }
    None
}
pub(crate) fn base(text: &str) -> String {
    text.chars()
        .filter(|&c| !matches!(c,'ั'|'ิ'..='ฺ'|'็'..='๎') && !c.is_whitespace())
        .map(|c| if c == 'ำ' { 'า' } else { c })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_digits_survive_only_at_line_height() {
        assert_eq!(drop_reason("๗", 0.95, 0.8, Some(0.63)), None);
        assert_eq!(drop_reason("7", 0.95, 0.8, Some(0.61)), Some("too_short"));
        assert_eq!(drop_reason("ก", 0.99, 0.8, Some(1.)), Some("too_short"));
    }

    #[test]
    fn script_fixes_respect_word_and_digit_context() {
        assert_eq!(canonical("W.ศ. ๗o.๘๒ New York"), "พ.ศ. ๗๐.๘๒ New York");
        assert_eq!(canonical("นํ้า\u{200b}"), "น้ำ");
    }
}
