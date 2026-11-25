//! Script tests and reading cleanup for lines routed among script recognizers.
//!
//! Adapted from LiteParse's ocria-nano runtime (Apache-2.0). See `reading/NOTICE`.

/// Number of recognizer groups: Latin, CJK, right-to-left and other scripts.
pub(crate) const GROUPS: usize = 4;

fn in_ranges(c: char, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&(a, b)| (a..=b).contains(&(c as u32)))
}

/// Characters that show a reading really is in `group`'s script.
pub(crate) fn script_chars(group: usize, text: &str) -> usize {
    text.chars()
        .filter(|&c| match group {
            0 => c.is_ascii_alphabetic() || in_ranges(c, &[(0x00c0, 0x024f), (0x0370, 0x04ff)]),
            1 => in_ranges(
                c,
                &[
                    (0x3040, 0x30ff),
                    (0x3400, 0x9fff),
                    (0xac00, 0xd7af),
                    (0xf900, 0xfaff),
                ],
            ),
            2 => in_ranges(c, &[(0x0590, 0x08ff), (0xfb1d, 0xfdff), (0xfe70, 0xfeff)]),
            _ => in_ranges(
                c,
                &[
                    (0x0900, 0x0dff),
                    (0x0e00, 0x0eff),
                    (0x0f00, 0x109f),
                    (0x10a0, 0x10ff),
                    (0x0530, 0x058f),
                    (0x1200, 0x139f),
                    (0x1780, 0x17ff),
                ],
            ),
        })
        .count()
}

/// The most common group, the lowest index on ties.
pub(crate) fn majority(groups: &[usize]) -> usize {
    let mut counts = [0usize; GROUPS];
    for &g in groups {
        counts[g] += 1;
    }
    let mut best = 0;
    for i in 1..GROUPS {
        if counts[i] > counts[best] {
            best = i;
        }
    }
    best
}

/// Cyrillic and Greek lookalikes of Latin letters.
const HOMO_FROM: &str = "аеорсухкіјѕԁԛԝАВЕКМНОРСТХІЈЅΑΒΕΖΗΙΚΜΝΟΡΤΥΧοαν";
const HOMO_TO: &str = "aeopcyxkijsdqwABEKMHOPCTXIJSABEZHIKMNOPTYXoav";
/// Lookalikes from the other scripts a Latin line picks up: the Latin recognizer's own foreign classes (Thai น ท ร, Armenian ա) and another recognizer reading a Latin line ("REPՍBLQՍE", "C๐mmunicatoทร"). Zero digits that look like "o" count here, because inside a Latin word they are the letter.
const HOMO_EXTRA_FROM: &str = "ՍսՕօոհցզաԼՏนทรอบ๐ບນຣ໐၀ဝ";
const HOMO_EXTRA_TO: &str = "UuOonhgqwLSunsououusooo";
/// Latin and Greek lookalikes of Cyrillic letters, for a Cyrillic word the Latin recognizer wrote partly in Latin ("aрганізацый", "ФOPYMA", "ПяТAГΑ").
const CYR_FROM: &str = "aceopxyiABCEHKMOPTXYIΑΒΓΕΗΚΜΟΠΡΤΧαο";
const CYR_TO: &str = "асеорхуіАВСЕНКМОРТХУІАВГЕНКМОПРТХао";
/// First code point of each run of decimal digits the recognizers can write besides ASCII.
const DIGIT_ZEROS: [u32; 18] = [
    0x0660, 0x06f0, 0x0966, 0x09e6, 0x0a66, 0x0ae6, 0x0b66, 0x0be6, 0x0c66, 0x0ce6, 0x0d66, 0x0de6,
    0x0e50, 0x0ed0, 0x0f20, 0x1040, 0x1090, 0x17e0,
];

fn is_latin_letter(c: char) -> bool {
    c.is_ascii_alphabetic() || (c.is_alphabetic() && ('\u{00c0}'..='\u{024f}').contains(&c))
}

fn is_cyrillic(c: char) -> bool {
    ('\u{0400}'..='\u{04ff}').contains(&c)
}

fn is_greek_or_cyrillic(c: char) -> bool {
    ('\u{0370}'..='\u{04ff}').contains(&c)
}

/// Latin letters in `text`, not counting Greek or Cyrillic.
pub(crate) fn latin_letters(text: &str) -> usize {
    text.chars().filter(|&c| is_latin_letter(c)).count()
}

fn other_letters(text: &str) -> usize {
    text.chars()
        .filter(|&c| c.is_alphabetic() && !is_latin_letter(c))
        .count()
}

fn mapped(from: &str, to: &str, c: char) -> Option<char> {
    from.chars().zip(to.chars()).find(|m| m.0 == c).map(|m| m.1)
}

/// The Latin letter a lookalike stands for in a Latin word. Small-cap and italic Cyrillic forms (т н м п г л) only look Latin in context; т н м take the word's case.
fn latin_lookalike(c: char, upper: bool) -> Option<char> {
    mapped(HOMO_FROM, HOMO_TO, c)
        .or_else(|| mapped(HOMO_EXTRA_FROM, HOMO_EXTRA_TO, c))
        .or(match c {
            'т' => Some(if upper { 'T' } else { 't' }),
            'н' => Some(if upper { 'H' } else { 'h' }),
            'м' => Some(if upper { 'M' } else { 'm' }),
            'п' | 'л' => Some('n'),
            'г' => Some('r'),
            _ => None,
        })
}

/// Clean a reading of other scripts' lookalike characters.
///
/// In a line that is mostly Latin, lookalikes become the Latin letter inside a word with at least as many Latin letters as other letters ("REPՍBLQՍE" to "REPUBLQUE") and in a word of lookalikes alone ("В - Linear"), and other scripts' digits become ASCII digits ("(UA०८" to "(UA08"). Lines mostly in another script are left alone.
pub(crate) fn clean_latin(text: &str) -> String {
    let latin = latin_letters(text);
    let latin_line = latin > 0 && latin >= 2 * other_letters(text);
    let cyrillic_line = text.chars().filter(|&c| is_cyrillic(c)).count() > latin;
    let mut out = String::with_capacity(text.len());
    for (k, word) in text.split(char::is_whitespace).enumerate() {
        if k > 0 {
            out.push(' ');
        }
        let word_latin = latin_letters(word);
        let word_other = other_letters(word);
        let upper = word.chars().filter(|c| c.is_uppercase()).count()
            > word.chars().filter(|c| c.is_lowercase()).count();
        // Greek letters standing alone are real ("α = 0.05"), never lookalikes.
        let all_lookalike = word_other > 0
            && word.chars().all(|c| {
                !c.is_alphabetic()
                    || is_latin_letter(c)
                    || (latin_lookalike(c, upper).is_some()
                        && !('\u{0370}'..='\u{03ff}').contains(&c))
            });
        // Cyrillic and Greek inside a Latin word fold anywhere. The other scripts' lookalikes fold only in a Latin line, since Thai and Lao glue Latin words to their own without a space ("Windowsนี้").
        let greek_cyrillic = word.chars().filter(|&c| is_greek_or_cyrillic(c)).count();
        let fold_greek_cyrillic = greek_cyrillic > 0 && word_latin >= greek_cyrillic;
        let fold = latin_line && word_other > 0 && (word_latin >= word_other || all_lookalike);
        // The other way round: a Cyrillic word with Latin or Greek lookalikes inside.
        let cyrillic = word.chars().filter(|&c| is_cyrillic(c)).count();
        let lookalikes_only = word
            .chars()
            .filter(|&c| c.is_alphabetic() && !is_cyrillic(c))
            .all(|c| CYR_FROM.contains(c));
        let to_cyrillic = (cyrillic >= 2 && cyrillic > word_latin + (greek_cyrillic - cyrillic))
            || (cyrillic >= 1 && cyrillic_line && lookalikes_only);
        for c in word.chars() {
            let c = if to_cyrillic {
                mapped(CYR_FROM, CYR_TO, c).unwrap_or(c)
            } else if fold || (fold_greek_cyrillic && is_greek_or_cyrillic(c)) {
                latin_lookalike(c, upper).unwrap_or(c)
            } else {
                c
            };
            let digit = DIGIT_ZEROS
                .iter()
                .find(|&&zero| (zero..zero + 10).contains(&(c as u32)));
            match digit {
                Some(&zero) if latin_line => out.push(char::from(b'0' + (c as u32 - zero) as u8)),
                _ => out.push(c),
            }
        }
    }
    out
}

/// A single short token ("1", "12", "x3"): no space, at least one letter or digit, at most six characters.
pub(crate) fn is_short_token(text: &str) -> bool {
    let t = text.trim();
    !t.is_empty()
        && t.chars().count() <= 6
        && !t.chars().any(char::is_whitespace)
        && t.chars().any(char::is_alphanumeric)
}

/// A detector false positive on a picture: no letter or digit, or one or two unsure characters.
pub(crate) fn is_junk(text: &str, confidence: f32) -> bool {
    let word = text.chars().filter(|c| c.is_alphanumeric()).count();
    word == 0 || (word <= 2 && confidence < 0.8)
}

/// Reorder a visual-order reading into logical order with a right-to-left paragraph level.
pub(crate) fn visual_to_logical(text: &str) -> String {
    use unicode_bidi::{BidiInfo, Level};
    let is_rtl =
        |c: char| ('\u{0590}'..='\u{08ff}').contains(&c) || ('\u{fb1d}'..='\u{feff}').contains(&c);
    if !text.chars().any(is_rtl) {
        return text.to_owned();
    }
    let info = BidiInfo::new(text, Some(Level::rtl()));
    info.paragraphs
        .iter()
        .map(|p| info.reorder_line(p, p.range.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookalikes_fold_inside_latin_words_of_latin_lines() {
        for (read, clean) in [
            ("pt6а-42 Привет", "pt6a-42 Привет"),
            ("REPՍBLQՍE FRANÇAISE", "REPUBLQUE FRANÇAISE"),
            ("C๐mmunicatoทร Security", "Communicatons Security"),
            ("В - Linear 1- Center", "B - Linear 1- Center"),
            ("GSTIN: 27ADC300६/१ZK", "GSTIN: 27ADC3006/1ZK"),
            ("ZQX272003тOKEN OнMS", "ZQX272003TOKEN OHMS"),
            ("airborлe пoise to гeduce", "airborne noise to reduce"),
            ("aрганізацый ФOPYMA ПяТAГΑ", "арганізацый ФОРУМА ПяТАГА"),
        ] {
            assert_eq!(clean_latin(read), clean);
        }
    }

    #[test]
    fn other_script_lines_and_lone_greek_letters_are_left_alone() {
        for line in [
            "significant at α = 0.05 level",
            "Привет мир, как дела",
            "ReturnToWorkSA ਦਾ ਨਿਯਮ ਸਮੂਹ (Regulation Group) ਰਿਟਰਨ ਟੂ ਵਰਕ",
            "การบริหารความเสี่ยง ๒๕๖๗",
            "ติดตั้ง Windowsนี้ ได้ที่ร้าน",
        ] {
            assert_eq!(clean_latin(line), line);
        }
    }
}
