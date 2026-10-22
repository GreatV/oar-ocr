//! Right-to-left text for recognizers that read every line left to right.
//!
//! A CTC recognizer emits characters in the order it sees them, so a line
//! of Arabic or Hebrew comes out in visual order. [`flip`] turns visual
//! order back into logical order (and logical into visual: it is its own
//! inverse):
//!
//! - A right-to-left line (more RTL letters than LTR letters) reverses the
//!   order of its units. A unit is a run of LTR text (Latin words, digits,
//!   and the spaces and punctuation between them), which keeps its own
//!   order, or one other character cluster.
//! - A left-to-right line keeps its order, and each run of RTL text
//!   reverses.
//!
//! A cluster is a base character with its combining marks, so marks stay
//! after their base. A bracket at right-to-left level changes to its
//! mirror image. This is the reference's small subset of the Unicode bidi
//! algorithm: no embeddings or isolates, neutrals resolved by runs.

use unicode_bidi::{BidiClass, bidi_class};
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

const MIRROR: [(char, char); 12] = [
    ('(', ')'),
    (')', '('),
    ('[', ']'),
    (']', '['),
    ('{', '}'),
    ('}', '{'),
    ('<', '>'),
    ('>', '<'),
    ('«', '»'),
    ('»', '«'),
    ('‹', '›'),
    ('›', '‹'),
];

pub(super) fn is_rtl(c: char) -> bool {
    matches!(bidi_class(c), BidiClass::R | BidiClass::AL)
}

/// Python's `str.isalpha` for one character: a letter of any case or kind.
pub(super) fn is_letter(c: char) -> bool {
    c.general_category_group() == GeneralCategoryGroup::Letter
}

fn is_mark(c: char) -> bool {
    matches!(
        c.general_category(),
        GeneralCategory::NonspacingMark | GeneralCategory::EnclosingMark
    )
}

pub(super) fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl)
}

/// Whether a line reads right to left: more RTL letters than LTR letters.
fn rtl_base(text: &str) -> bool {
    let r = text.chars().filter(|&c| is_rtl(c)).count();
    let l = text
        .chars()
        .filter(|&c| bidi_class(c) == BidiClass::L)
        .count();
    r > l
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Rtl,
    Ltr,
    Neutral,
}

fn kind(cluster: &str) -> Kind {
    let c = cluster.chars().next().expect("clusters are non-empty");
    if is_rtl(c) {
        Kind::Rtl
    } else if matches!(bidi_class(c), BidiClass::L | BidiClass::EN | BidiClass::AN) {
        // Digits (European and Arabic-Indic) read left to right in RTL text.
        Kind::Ltr
    } else {
        Kind::Neutral
    }
}

/// `[start, end)` spans from a `strong` cluster to the last `strong` cluster
/// before one of kind `stop`.
fn runs(kinds: &[Kind], strong: Kind, stop: Kind) -> Vec<(usize, usize)> {
    let (mut spans, mut start, mut last) = (Vec::new(), None, 0);
    for i in 0..=kinds.len() {
        let k = kinds.get(i).copied();
        if k == Some(strong) {
            start.get_or_insert(i);
            last = i;
        } else if (k == Some(stop) || k.is_none())
            && let Some(s) = start.take()
        {
            spans.push((s, last + 1));
        }
    }
    spans
}

fn mirror(cluster: &str) -> String {
    let mut chars = cluster.chars();
    let c = chars.next().expect("clusters are non-empty");
    let m = MIRROR.iter().find(|(a, _)| *a == c).map_or(c, |&(_, b)| b);
    std::iter::once(m).chain(chars).collect()
}

/// Visual order to logical order (and back). `base_rtl` is the paragraph
/// direction when known; `None` decides by the line's own letters.
pub(super) fn flip(text: &str, base_rtl: Option<bool>) -> String {
    if !has_rtl(text) {
        return text.to_string();
    }
    let mut cls: Vec<String> = Vec::new();
    for c in text.chars() {
        match cls.last_mut() {
            Some(last) if is_mark(c) => last.push(c),
            _ => cls.push(c.to_string()),
        }
    }
    let kinds: Vec<Kind> = cls.iter().map(|c| kind(c)).collect();
    if base_rtl.unwrap_or_else(|| rtl_base(text)) {
        // LTR runs stay whole, every other cluster is its own unit; the
        // units reverse.
        let mut units: Vec<String> = Vec::new();
        let mut i = 0;
        let n = cls.len();
        for (a, b) in runs(&kinds, Kind::Ltr, Kind::Rtl)
            .into_iter()
            .chain([(n, n)])
        {
            units.extend(cls[i..a].iter().map(|c| mirror(c)));
            if a < b {
                units.push(cls[a..b].concat());
            }
            i = b;
        }
        return units.into_iter().rev().collect();
    }
    // A left-to-right line: each RTL run reverses in place.
    let mut out = cls.clone();
    for (a, b) in runs(&kinds, Kind::Rtl, Kind::Ltr) {
        for (o, c) in out[a..b].iter_mut().zip(cls[a..b].iter().rev()) {
            *o = mirror(c);
        }
    }
    out.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Visual-order recognizer output back in logical order: Hebrew words
    /// reverse, the Latin run and the number keep their order, and the
    /// bracket pair mirrors; flipping again gives the visual text back.
    #[test]
    fn flip_restores_logical_order() {
        let visual = "(ABC 12) םולש";
        let logical = flip(visual, None);
        assert_eq!(logical, "שלום (ABC 12)");
        assert_eq!(flip(&logical, None), visual);
    }
}
