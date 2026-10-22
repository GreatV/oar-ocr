//! Line crops for the recognizers, width-bucketed batches, CTC token
//! decoding, and the splitting of detected lines at wide gaps.

use super::cv::{Interp, resize};
use super::rtl::{flip, has_rtl};
use image::GrayImage;
use regex::Regex;
use std::sync::LazyLock;
use unicode_normalization::UnicodeNormalization;

const PAD_Y: f64 = 0.2;
pub(super) const PAD_X: f64 = 0.3;
/// Batch widths are padded to a multiple of this (the timestep stride).
const WIDTH_MULTIPLE: usize = 8;
const MAX_WIDTH: usize = 4096;

/// A recognizer input: an ink map (0 = paper, 1 = ink), `height` rows of
/// `width` values.
pub(super) struct Crop {
    pub width: usize,
    pub data: Vec<f32>,
}

/// The 2nd, 50th and 98th percentiles (numpy's linear interpolation) of
/// 8-bit pixels, from their histogram.
fn percentiles(pixels: impl Iterator<Item = u8>) -> [f64; 3] {
    let mut hist = [0u64; 256];
    let mut n = 0u64;
    for v in pixels {
        hist[v as usize] += 1;
        n += 1;
    }
    let mut cum = hist;
    for i in 1..256 {
        cum[i] += cum[i - 1];
    }
    // The first value whose cumulative count passes `i`.
    let value = |i: u64| cum.partition_point(|&c| c <= i) as f64;
    [0.02, 0.5, 0.98].map(|q| {
        let x = q * (n - 1) as f64;
        let i = x.floor();
        let g = x - i;
        let lo = value(i as u64);
        let hi = value((i as u64 + 1).min(n - 1));
        let d = hi - lo;
        // numpy's lerp, from the nearer end.
        if g >= 0.5 {
            hi - d * (1.0 - g)
        } else {
            lo + d * g
        }
    })
}

/// Light text on a dark ground (a banner, a dark table head): the ground,
/// the median of the line box, lies nearer the dark end than the text does,
/// so the bright tail is the longer one. The recognizers read dark text on
/// light paper, so such a line is read inverted.
pub(super) fn is_dark_line(gray: &GrayImage, b: [f32; 4]) -> bool {
    let (pw, ph) = (gray.width() as i64, gray.height() as i64);
    let [x0, y0, x1, y1] = b.map(|v| v as i64);
    let (sx0, sy0, sx1, sy1) = (x0.max(0), y0.max(0), (x1 + 1).min(pw), (y1 + 1).min(ph));
    if sx1 <= sx0 || sy1 <= sy0 {
        return false;
    }
    let src = gray.as_raw();
    let pixels = (sy0..sy1).flat_map(|y| {
        let row = (y * pw) as usize;
        src[row + sx0 as usize..row + sx1 as usize].iter().copied()
    });
    let [lo, med, hi] = percentiles(pixels);
    hi - med > 2.0 * (med - lo) + 8.0
}

/// Crops one line: the box padded by `PAD_X` line heights across and, for a
/// window of `window` line bands, enough above and below that its middle
/// band is the one-band crop; scaled to `window * height` rows at its aspect
/// ratio. A `dark` line (see [`is_dark_line`]) is read inverted, with a dark
/// ground outside the page.
pub(super) fn crop_line(
    gray: &GrayImage,
    b: [f32; 4],
    height: usize,
    window: usize,
    dark: bool,
) -> Crop {
    let pad_y = (window as f64 * (1.0 + 2.0 * PAD_Y) - 1.0) / 2.0;
    let height = height * window;
    let [x0, y0, x1, y1] = b.map(f64::from);
    let h = (y1 - y0).max(1.0);
    let (ix0, iy0) = (
        (x0 - PAD_X * h).floor() as i64,
        (y0 - pad_y * h).floor() as i64,
    );
    let (ix1, iy1) = (
        (x1 + PAD_X * h).ceil() as i64,
        (y1 + pad_y * h).ceil() as i64,
    );
    let cw = (ix1 - ix0).max(1);
    let ch = (iy1 - iy0).max(1);
    let (pw, ph) = (gray.width() as i64, gray.height() as i64);
    let ground = if dark { 0 } else { 255 };
    let mut crop = GrayImage::from_pixel(cw as u32, ch as u32, image::Luma([ground]));
    let (sx0, sy0, sx1, sy1) = (ix0.max(0), iy0.max(0), ix1.min(pw), iy1.min(ph));
    if sx1 > sx0 && sy1 > sy0 {
        let src = gray.as_raw();
        let dst: &mut [u8] = &mut crop;
        for y in sy0..sy1 {
            let s = (y * pw + sx0) as usize;
            let d = ((y - iy0) * cw + (sx0 - ix0)) as usize;
            let n = (sx1 - sx0) as usize;
            dst[d..d + n].copy_from_slice(&src[s..s + n]);
        }
    }
    let out_w = ((cw as f64 * height as f64 / ch as f64).round_ties_even() as usize)
        .clamp(WIDTH_MULTIPLE, MAX_WIDTH);
    let interp = if ch as usize > height {
        Interp::Area
    } else {
        Interp::Linear
    };
    let resized = resize(&crop, out_w as u32, height as u32, interp);
    Crop {
        width: out_w,
        data: resized
            .as_raw()
            .iter()
            .map(|&v| {
                let ink = v as f32 / 255.0;
                if dark { ink } else { 1.0 - ink }
            })
            .collect(),
    }
}

/// Right-pads crops to one width (a multiple of 8): the `[B, 1, H, W]` data,
/// `W`, and each crop's own width.
pub(super) fn pad_batch(crops: &[&Crop], height: usize) -> (Vec<f32>, usize, Vec<i64>) {
    let max = crops.iter().map(|c| c.width).max().unwrap_or(0);
    let wm = max.div_ceil(WIDTH_MULTIPLE) * WIDTH_MULTIPLE;
    let mut out = vec![0f32; crops.len() * height * wm];
    for (i, c) in crops.iter().enumerate() {
        for y in 0..height {
            let d = (i * height + y) * wm;
            out[d..d + c.width].copy_from_slice(&c.data[y * c.width..(y + 1) * c.width]);
        }
    }
    (out, wm, crops.iter().map(|c| c.width as i64).collect())
}

/// Groups line indices into batches of similar width, narrowest first. A
/// batch closes at `max_batch` lines, at a line wider than `max_ratio` times
/// its first (widths under 64 count as 64), or when lines x width would pass
/// `max_area` columns.
pub(super) fn width_batches(
    widths: &[usize],
    max_batch: usize,
    max_ratio: f64,
    max_area: usize,
) -> Vec<Vec<usize>> {
    const MIN_WIDTH: usize = 64;
    let mut order: Vec<usize> = (0..widths.len()).collect();
    order.sort_by_key(|&i| widths[i]);
    let mut batches: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let w = widths[i].max(MIN_WIDTH);
        match batches.last_mut() {
            Some(b)
                if b.len() < max_batch
                    && w as f64 <= max_ratio * widths[b[0]].max(MIN_WIDTH) as f64
                    && (b.len() + 1) * w <= max_area =>
            {
                b.push(i)
            }
            _ => batches.push(vec![i]),
        }
    }
    batches
}

/// The recognizer's default batching.
pub(super) fn line_batches(crops: &[Crop]) -> Vec<Vec<usize>> {
    let widths: Vec<usize> = crops.iter().map(|c| c.width).collect();
    width_batches(&widths, 64, 1.3, 64 * 640)
}

/// One recognized token and its page x position.
#[derive(Clone, Debug)]
pub(super) struct Token {
    pub text: String,
    pub x: f64,
}

/// A recognized line: text, mean CTC peak confidence, and tokens.
/// `char_tokens` marks tokens that are characters of a script group's
/// recognizer or a whole math reading: their text joins with runs of
/// whitespace collapsed (see [`decode`]).
#[derive(Clone, Debug, Default)]
pub(super) struct LineText {
    pub text: String,
    pub conf: f64,
    pub tokens: Vec<Token>,
    pub char_tokens: bool,
}

/// Joins tokens as the main vocabulary decodes them: a LaTeX command followed
/// by a letter gets a space, so `\alpha` + `x` does not read `\alphax`.
pub(super) fn join_tokens<S: AsRef<str>>(tokens: &[S]) -> String {
    let mut out = String::new();
    for (k, t) in tokens.iter().enumerate() {
        let t = t.as_ref();
        out.push_str(t);
        let next = tokens.get(k + 1).map(|n| n.as_ref()).unwrap_or("");
        let command =
            t.chars().count() > 2 && t.starts_with('\\') && t[1..].chars().all(char::is_alphabetic);
        if command && next.chars().next().is_some_and(char::is_alphabetic) {
            out.push(' ');
        }
    }
    out
}

/// The text of a run of tokens. Character tokens compose Hangul jamo into
/// syllables, and right-to-left text read in visual order turns to logical
/// order (in the page's direction on a right-to-left page).
pub(super) fn decode(tokens: &[Token], char_tokens: bool, page_rtl: bool) -> String {
    if char_tokens {
        let joined: String = tokens.iter().map(|t| t.text.as_str()).collect();
        let text: String = joined
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .nfc()
            .collect();
        if has_rtl(&text) {
            flip(&text, page_rtl.then_some(true))
        } else {
            text
        }
    } else {
        join_tokens(&tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>())
    }
}

fn is_space(t: &Token) -> bool {
    t.text == " "
}

/// Cuts a line where neighbouring characters are more than `gap` line heights
/// apart, or across one of the gutter positions `at`. Returns
/// `(box, text, tokens)` per piece; one item when nothing is cut.
pub(super) fn split_line(
    b: [f32; 4],
    line: &LineText,
    gap: f64,
    at: &[f64],
    page_rtl: bool,
) -> Vec<([f32; 4], String, Vec<Token>)> {
    let h = ((b[3] - b[1]) as f64).max(1.0);
    let mut segs: Vec<Vec<Token>> = vec![Vec::new()];
    let mut prev: Option<f64> = None;
    for tk in &line.tokens {
        if !is_space(tk) {
            let x = tk.x;
            if let Some(p) = prev
                && (x - p > gap * h || at.iter().any(|&c| p < c && c < x))
            {
                segs.push(Vec::new());
            }
            prev = Some(x);
        }
        segs.last_mut().expect("segs is non-empty").push(tk.clone());
    }
    if segs.len() == 1 {
        return vec![(
            b,
            decode(&line.tokens, line.char_tokens, page_rtl),
            line.tokens.clone(),
        )];
    }
    let mut out = Vec::new();
    for sg in segs {
        let text = decode(&sg, line.char_tokens, page_rtl).trim().to_string();
        let xs: Vec<f64> = sg.iter().filter(|t| !is_space(t)).map(|t| t.x).collect();
        if !text.is_empty() && !xs.is_empty() {
            let lo = xs.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let bx = [
                (b[0] as f64).max(lo - 0.35 * h) as f32,
                b[1],
                (b[2] as f64).min(hi + 0.35 * h) as f32,
                b[3],
            ];
            out.push((bx, text, sg));
        }
    }
    out
}

/// A gap over this many line heights may be a column gutter.
const GUTTER_GAP: f64 = 1.6;
/// How many other line edges must line up with a gutter.
const GUTTER_EDGES: usize = 3;

/// Per line, the x positions of gaps that sit on a column gutter: a gap over
/// `GUTTER_GAP` line heights where at least `GUTTER_EDGES` other lines end at
/// its left side or start at its right side.
pub(super) fn gutter_cuts(boxes: &[[f32; 4]], lines: &[LineText]) -> Vec<Vec<f64>> {
    boxes
        .iter()
        .zip(lines)
        .enumerate()
        .map(|(i, (b, line))| {
            let h = ((b[3] - b[1]) as f64).max(1.0);
            let xs: Vec<f64> = line
                .tokens
                .iter()
                .filter(|t| !is_space(t))
                .map(|t| t.x)
                .collect();
            let mut cuts = Vec::new();
            for w in xs.windows(2) {
                let (a, c) = (w[0], w[1]);
                if c - a <= GUTTER_GAP * h {
                    continue;
                }
                let (gl, gr) = (a + 0.35 * h, c - 0.35 * h);
                let near = boxes
                    .iter()
                    .enumerate()
                    .filter(|&(j, o)| {
                        j != i
                            && (((o[2] as f64) - gl).abs() < 0.5 * h
                                || ((o[0] as f64) - gr).abs() < 0.5 * h)
                    })
                    .count();
                if near >= GUTTER_EDGES {
                    cuts.push((a + c) / 2.0);
                }
            }
            cuts
        })
        .collect()
}

/// Form fill-in dots, rules and specks.
const LEADER_CHARS: &str = ".·…‥_-–—=";
/// A lone short dash can be an empty table cell; other lone marks are not.
const DASHES: &str = "-–—";
/// List markers (not dashes: a "-" can be a table cell).
const BULLETS: &str = "•◦▪▫●○■□‣⁃∙*";
/// A numbered list marker: "1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)".
static ENUMERATOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(\d{1,3}|[A-Za-z])[.)]|\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\))$")
        .expect("valid regex")
});
/// A list marker and its item's text are at most this many line heights
/// apart.
const MARKER_GAP: f32 = 2.5;

/// Drops or joins lines that carry no text of their own:
///
/// - a line of leader characters only is dropped: 3 or more of them, a box
///   3 or more times wider than tall, or any dot or underscore alone (a
///   short lone dash stays: it can be an empty table cell);
/// - a lone list marker (a bullet, "1.", "(a)") joins the line that starts
///   within `MARKER_GAP` line heights after it on the same row (before it on
///   a right-to-left page). On a left-to-right page its tokens join too
///   (`space`: the main vocabulary has a space token), so a table line still
///   splits into cells at its gaps.
pub(super) fn tidy_lines(
    mut boxes: Vec<[f32; 4]>,
    mut lines: Vec<LineText>,
    rtl: bool,
    space: bool,
) -> (Vec<[f32; 4]>, Vec<LineText>) {
    let n = lines.len();
    let mut keep = vec![true; n];
    let h: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let w: Vec<f32> = boxes.iter().map(|b| b[2] - b[0]).collect();
    for (i, line) in lines.iter().enumerate() {
        let core: Vec<char> = line.text.chars().filter(|&c| c != ' ').collect();
        if !core.is_empty() && core.iter().all(|&c| LEADER_CHARS.contains(c)) {
            let wide = w[i] >= 3.0 * h[i];
            if core.len() >= 3 || wide || !core.iter().any(|&c| DASHES.contains(c)) {
                keep[i] = false;
            }
        }
    }
    for i in 0..n {
        let t = lines[i].text.trim().to_string();
        let mut chars = t.chars();
        let bullet = matches!((chars.next(), chars.next()), (Some(c), None) if BULLETS.contains(c));
        if !keep[i] || !(bullet || ENUMERATOR.is_match(&t)) {
            continue;
        }
        let mut best: Option<(usize, f32)> = None;
        for j in 0..n {
            if j == i || !keep[j] || lines[j].text.trim().is_empty() {
                continue;
            }
            let (bi, bj) = (boxes[i], boxes[j]);
            let ov = bi[3].min(bj[3]) - bi[1].max(bj[1]);
            if ov < 0.5 * h[i].min(h[j]) {
                continue;
            }
            let gap = if rtl { bi[0] - bj[2] } else { bj[0] - bi[2] };
            if -0.3 * h[i] <= gap
                && gap <= MARKER_GAP * h[i].max(h[j])
                && best.is_none_or(|(_, g)| gap < g)
            {
                best = Some((j, gap));
            }
        }
        let Some((j, _)) = best else {
            continue;
        };
        let marker = std::mem::take(&mut lines[i].tokens);
        let item = &lines[j];
        // Only tokens of one kind (characters or vocabulary entries) join.
        let space = space || item.char_tokens;
        if !rtl
            && !marker.is_empty()
            && !item.tokens.is_empty()
            && space
            && lines[i].char_tokens == item.char_tokens
        {
            let x = (boxes[i][2] + boxes[j][0]) as f64 / 2.0;
            let mut tokens = marker;
            tokens.push(Token {
                text: " ".to_string(),
                x,
            });
            tokens.append(&mut lines[j].tokens);
            lines[j].tokens = tokens;
        }
        lines[j].text = format!("{t} {}", lines[j].text.trim());
        let (bi, bj) = (boxes[i], boxes[j]);
        boxes[j] = [
            bi[0].min(bj[0]),
            bi[1].min(bj[1]),
            bi[2].max(bj[2]),
            bi[3].max(bj[3]),
        ];
        keep[i] = false;
    }
    boxes
        .into_iter()
        .zip(lines)
        .zip(keep)
        .filter_map(|(line, k)| k.then_some(line))
        .unzip()
}
