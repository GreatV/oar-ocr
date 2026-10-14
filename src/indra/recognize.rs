//! Line crops for the recognizers, width-bucketed batches, CTC token
//! decoding, and the splitting of detected lines at wide gaps.

use super::cv::{Interp, resize};
use image::GrayImage;

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

/// Crops one line: the box padded by `PAD_Y` / `PAD_X` line heights (white
/// outside the page), scaled to `height` rows at its aspect ratio.
pub(super) fn crop_line(gray: &GrayImage, b: [f32; 4], height: usize) -> Crop {
    let [x0, y0, x1, y1] = b.map(f64::from);
    let h = (y1 - y0).max(1.0);
    let (ix0, iy0) = (
        (x0 - PAD_X * h).floor() as i64,
        (y0 - PAD_Y * h).floor() as i64,
    );
    let (ix1, iy1) = (
        (x1 + PAD_X * h).ceil() as i64,
        (y1 + PAD_Y * h).ceil() as i64,
    );
    let cw = (ix1 - ix0).max(1);
    let ch = (iy1 - iy0).max(1);
    let (pw, ph) = (gray.width() as i64, gray.height() as i64);
    let mut crop = GrayImage::from_pixel(cw as u32, ch as u32, image::Luma([255]));
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
            .map(|&v| 1.0 - v as f32 / 255.0)
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
/// `char_tokens` marks tokens that are characters of a non-default
/// vocabulary (the Chinese recognizer) or a whole math reading: their text
/// joins with runs of whitespace collapsed.
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

/// The text of a run of tokens.
pub(super) fn decode(tokens: &[Token], char_tokens: bool) -> String {
    if char_tokens {
        let joined: String = tokens.iter().map(|t| t.text.as_str()).collect();
        joined.split_whitespace().collect::<Vec<_>>().join(" ")
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
            decode(&line.tokens, line.char_tokens),
            line.tokens.clone(),
        )];
    }
    let mut out = Vec::new();
    for sg in segs {
        let text = decode(&sg, line.char_tokens).trim().to_string();
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
