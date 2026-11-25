//! LayoutNet inputs: unit features, ruling lines and the page image.
//!
//! Each unit has 68 features: 21 of geometry, text and style, 32 Fourier terms of its centre, 8 grid-projection signals, 5 ruling-line flags, then a block id and a leaf id. The projection signals and both ids come from the page projection in `projection.rs`. Adapted from LiteParse's `table_model/features.rs` (Apache-2.0). See `processors/NOTICE`.

/// Grid-projection signals of one unit, absent when no projected item matched it.
#[derive(Clone, Debug, Default)]
pub struct Proj {
    /// Text the text features read instead of the unit's own (an OCR faint-token read).
    pub feat_text: Option<String>,
    pub flow: f32,
    pub lc: f32,
    pub rc: f32,
    pub cc: f32,
    pub snap: char,
    pub block: f32,
    pub leaf: f32,
}

/// A text unit with what the features need. `bx` is in the text's reading frame, `pbox` on the
/// rendered page (they differ only for sideways text); points, top-left origin.
#[derive(Clone, Debug)]
pub struct FUnit {
    pub bx: [f32; 4],
    pub pbox: [f32; 4],
    pub text: String,
    pub h: f32,
    pub bold: Option<bool>,
    pub proj: Option<Proj>,
}

/// Ruling lines in page points: horizontal `[y, x0, x1]`, vertical `[x, y0, y1]`.
#[derive(Clone, Debug, Default)]
pub struct Rules {
    pub h: Vec<[f32; 3]>,
    pub v: Vec<[f32; 3]>,
}

pub const N_FOURIER: usize = 8;
pub const FEAT_DIM: usize = 21 + 4 * N_FOURIER + 8 + 5 + 2;

fn median_f32(v: &[f32]) -> f32 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// tn_core._NUM: `^[\s(\[$€£¥+\-–−±<>≤≥~]*[\d.,]+[%)\]*†‡§a-z]*$` (case-insensitive).
fn is_numeric_text(t: &str) -> bool {
    let cs: Vec<char> = t.chars().collect();
    let lead = |c: char| c.is_whitespace() || "([$€£¥+-–−±<>≤≥~".contains(c);
    let body = |c: char| {
        c.is_ascii_digit() || c == '.' || c == ',' || (c.is_numeric() && c.is_ascii_digit())
    };
    let tail = |c: char| "%)]*†‡§".contains(c) || c.is_ascii_alphabetic();
    let mut i = 0;
    while i < cs.len() && lead(cs[i]) && !body(cs[i]) {
        i += 1;
    }
    // the body needs at least one char; regex backtracking can give leading chars back only if
    // they could be body chars, and none of the lead set is a digit, '.' or ','
    let start = i;
    while i < cs.len() && body(cs[i]) {
        i += 1;
    }
    if i == start {
        return false;
    }
    cs[i..].iter().all(|&c| tail(c))
}

/// Python str.isdigit (Unicode decimal / digit), close enough for features.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || matches!(c, '²' | '³' | '¹' | '⁰'..='⁹' | '₀'..='₉')
}

/// [N, 4]: a horizontal rule just above / below the unit, a vertical rule just left / right.
pub fn unit_rule_features(boxes: &[[f32; 4]], rules: &Rules, h: &[f32]) -> Vec<[f32; 4]> {
    let mut out = vec![[0f32; 4]; boxes.len()];
    for (i, b) in boxes.iter().enumerate() {
        let hh = h[i].max(0.5);
        for r in &rules.h {
            let ov = b[2].min(r[2]) - b[0].max(r[1]) > 0.0;
            if !ov {
                continue;
            }
            let above = b[1] - r[0];
            let below = r[0] - b[3];
            if above >= -0.3 * hh && above < 1.5 * hh {
                out[i][0] = 1.0;
            }
            if below >= -0.3 * hh && below < 1.5 * hh {
                out[i][1] = 1.0;
            }
        }
        for r in &rules.v {
            let ov = b[3].min(r[2]) - b[1].max(r[1]) > 0.0;
            if !ov {
                continue;
            }
            let left = b[0] - r[0];
            let right = r[0] - b[2];
            if left >= -0.3 * hh && left < 2.0 * hh {
                out[i][2] = 1.0;
            }
            if right >= -0.3 * hh && right < 2.0 * hh {
                out[i][3] = 1.0;
            }
        }
    }
    out
}

/// [N, N, 2]: rules strictly between two units (spanning both), counts capped at 3, scaled.
pub fn pair_rule_features(boxes: &[[f32; 4]], rules: &Rules) -> Vec<f32> {
    let n = boxes.len();
    let mut out = vec![0f32; n * n * 2];
    let cap = 3.0f32;
    let cy: Vec<f32> = boxes.iter().map(|b| (b[1] + b[3]) / 2.0).collect();
    let cx: Vec<f32> = boxes.iter().map(|b| (b[0] + b[2]) / 2.0).collect();
    // covers[i][r]: rule r runs under unit i's x-range (h) / beside its y-range (v)
    let hcov: Vec<Vec<bool>> = boxes
        .iter()
        .map(|b| {
            rules
                .h
                .iter()
                .map(|r| r[1] <= b[2] && r[2] >= b[0])
                .collect()
        })
        .collect();
    let vcov: Vec<Vec<bool>> = boxes
        .iter()
        .map(|b| {
            rules
                .v
                .iter()
                .map(|r| r[1] <= b[3] && r[2] >= b[1])
                .collect()
        })
        .collect();
    for i in 0..n {
        for j in 0..n {
            if !rules.h.is_empty() {
                let (lo, hi) = (cy[i].min(cy[j]), cy[i].max(cy[j]));
                let c = rules
                    .h
                    .iter()
                    .enumerate()
                    .filter(|(r, rr)| rr[0] > lo && rr[0] < hi && hcov[i][*r] && hcov[j][*r])
                    .count();
                out[(i * n + j) * 2] = (c as f32).min(cap) / cap;
            }
            if !rules.v.is_empty() {
                let (lo, hi) = (cx[i].min(cx[j]), cx[i].max(cx[j]));
                let c = rules
                    .v
                    .iter()
                    .enumerate()
                    .filter(|(r, rr)| rr[0] > lo && rr[0] < hi && vcov[i][*r] && vcov[j][*r])
                    .count();
                out[(i * n + j) * 2 + 1] = (c as f32).min(cap) / cap;
            }
        }
    }
    out
}

/// [N, FEAT_DIM] row-major (tn_core.unit_features with rules; `rules` None = unknown).
pub fn unit_features(units: &[FUnit], wh: (f32, f32), rules: Option<&Rules>) -> Vec<f32> {
    let (w_pg, h_pg) = wh;
    let n = units.len();
    let mut f = vec![0f32; n * FEAT_DIM];
    if n == 0 {
        return f;
    }
    let b: Vec<[f32; 4]> = units.iter().map(|u| u.bx).collect();
    let h: Vec<f32> = units.iter().map(|u| u.h.max(0.5)).collect();
    let hm = median_f32(&h);
    let mid: Vec<f32> = b.iter().map(|x| (x[1] + x[3]) / 2.0).collect();
    let rule_f = rules.map(|r| {
        let pb: Vec<[f32; 4]> = units.iter().map(|u| u.pbox).collect();
        let hs: Vec<f32> = units.iter().map(|u| u.h).collect();
        unit_rule_features(&pb, r, &hs)
    });
    for i in 0..n {
        let mut n_row = 0f32;
        let (mut gap_l, mut gap_r) = (20f32, 20f32);
        let (mut lmax, mut rmin) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut has_l, mut has_r) = (false, false);
        for j in 0..n {
            let same = (mid[j] - mid[i]).abs() < 0.5 * h[j].max(h[i]);
            if !same {
                continue;
            }
            n_row += 1.0;
            if j == i {
                continue;
            }
            if b[j][2] <= b[i][0] + 0.5 * h[i] {
                has_l = true;
                lmax = lmax.max(b[j][2]);
            }
            if b[j][0] >= b[i][2] - 0.5 * h[i] {
                has_r = true;
                rmin = rmin.min(b[j][0]);
            }
        }
        if has_l {
            gap_l = 20f32.min((b[i][0] - lmax) / hm);
        }
        if has_r {
            gap_r = 20f32.min((rmin - b[i][2]) / hm);
        }
        let cx = (b[i][0] + b[i][2]) / 2.0 / w_pg;
        let cy = (b[i][1] + b[i][3]) / 2.0 / h_pg;
        let t = units[i]
            .proj
            .as_ref()
            .and_then(|p| p.feat_text.as_deref())
            .unwrap_or(&units[i].text);
        let chars: Vec<char> = t.chars().collect();
        let len = chars.len() as f32;
        let n_alpha = chars.iter().filter(|c| c.is_alphabetic()).count() as f32;
        let dig = chars.iter().filter(|&&c| is_digit(c)).count() as f32 / len.max(1.0);
        let alpha = n_alpha / len.max(1.0);
        let upper = chars.iter().filter(|c| c.is_uppercase()).count() as f32 / n_alpha.max(1.0);
        let words = t.split_whitespace().count() as f32;
        let numeric = if is_numeric_text(t) { 1.0 } else { 0.0 };
        let tr = t.trim_end();
        let endp =
            if tr.ends_with('.') || tr.ends_with(':') || tr.ends_with(';') || tr.ends_with(',') {
                1.0
            } else {
                0.0
            };
        let row = &mut f[i * FEAT_DIM..(i + 1) * FEAT_DIM];
        let base = [
            cx,
            cy,
            (b[i][2] - b[i][0]) / w_pg,
            (b[i][3] - b[i][1]) / h_pg,
            b[i][0] / w_pg,
            b[i][2] / w_pg,
            b[i][1] / h_pg,
            b[i][3] / h_pg,
            h[i] / hm,
            len.ln_1p() / 5.0,
            words.ln_1p() / 3.0,
            dig,
            alpha,
            upper,
            numeric,
            endp,
            n_row.ln_1p() / 3.0,
            gap_l / 20.0,
            gap_r / 20.0,
            if units[i].bold == Some(true) {
                1.0
            } else {
                0.0
            },
            if units[i].bold.is_none() { 0.0 } else { 1.0 },
        ];
        row[..21].copy_from_slice(&base);
        let mut k = 21;
        for q in 0..N_FOURIER {
            let fr = (2f32).powi(q as i32) * std::f32::consts::PI;
            row[k] = (fr * cx).sin();
            row[k + 1] = (fr * cx).cos();
            row[k + 2] = (fr * cy).sin();
            row[k + 3] = (fr * cy).cos();
            k += 4;
        }
        if let Some(p) = &units[i].proj {
            row[k] = 1.0;
            row[k + 1] = p.flow;
            row[k + 2] = p.lc.ln_1p() / 3.0;
            row[k + 3] = p.rc.ln_1p() / 3.0;
            row[k + 4] = p.cc.ln_1p() / 3.0;
            row[k + 5] = (p.snap == 'L') as u8 as f32;
            row[k + 6] = (p.snap == 'R') as u8 as f32;
            row[k + 7] = (p.snap == 'C') as u8 as f32;
        }
        k += 8;
        if let Some(rf) = &rule_f {
            row[k..k + 4].copy_from_slice(&rf[i]);
            row[k + 4] = 1.0;
        }
        k += 5;
        row[k] = units[i].proj.as_ref().map_or(-1.0, |p| p.block);
        row[k + 1] = units[i].proj.as_ref().map_or(-1.0, |p| p.leaf);
    }
    f
}

/// Ruling lines from a grayscale page render (`gray` row-major, `w` x `h` pixels, `scale` pixels per
/// point): dark pixels (< 170) opened with a long thin kernel, so only runs longer than any glyph
/// survive; each 8-connected surviving component is one rule, unless it is thicker than 2.5 pt.
/// An opening with a 1 x L line keeps exactly the ink runs of length >= L along that axis.
pub fn detect_rules(gray: &[u8], w: usize, h: usize, scale: f32) -> Rules {
    let ink = InkBits::from_grey(gray, w, h);
    rules_from_runs(ink.runs(), scale)
}

/// Ink runs of each row or column as `(start, end inclusive)`.
type Runs = Vec<Vec<(usize, usize)>>;

/// Gray threshold below which a pixel is ink.
const DARK: u8 = 170;

/// A page's ink mask (grey < [`DARK`]) as one bitset per row, so the run scan only visits the
/// pixels where ink starts or stops (pages are mostly paper).
struct InkBits {
    w: usize,
    h: usize,
    words: usize,
    bits: Vec<u64>,
}

impl InkBits {
    fn from_grey(gray: &[u8], w: usize, h: usize) -> Self {
        let words = w.div_ceil(64);
        let mut bits = vec![0u64; words * h];
        for (row, dst) in gray
            .chunks_exact(w.max(1))
            .take(h)
            .zip(bits.chunks_exact_mut(words.max(1)))
        {
            for (k, px) in row.chunks(64).enumerate() {
                let mut word = 0u64;
                for (b, &g) in px.iter().enumerate() {
                    word |= ((g < DARK) as u64) << b;
                }
                dst[k] = word;
            }
        }
        InkBits { w, h, words, bits }
    }

    fn row(&self, y: usize) -> &[u64] {
        &self.bits[y * self.words..(y + 1) * self.words]
    }

    /// Horizontal ink runs per row and vertical ink runs per column, as `(start, end inclusive)`
    /// in scan order, keeping those at least 3% of the page side (15 px floor) long.
    #[allow(clippy::needless_range_loop)]
    fn runs(&self) -> (Runs, Runs) {
        let (w, h, words) = (self.w, self.h, self.words);
        let lh = 15usize.max((w as f32 * 0.03) as usize);
        let lv = 15usize.max((h as f32 * 0.03) as usize);
        let mut hruns: Vec<Vec<(usize, usize)>> = vec![Vec::new(); h];
        let mut vruns: Vec<Vec<(usize, usize)>> = vec![Vec::new(); w];
        // where each column's current vertical run began (valid while its bit was set last row)
        let mut open = vec![0usize; w];
        let zero = vec![0u64; words];
        for y in 0..h {
            let cur = self.row(y);
            let prev = if y == 0 { &zero[..] } else { self.row(y - 1) };
            // vertical runs start or end where a column's ink differs from the row above
            for k in 0..words {
                let mut changed = cur[k] ^ prev[k];
                while changed != 0 {
                    let x = k * 64 + changed.trailing_zeros() as usize;
                    changed &= changed - 1;
                    if cur[k] >> (x % 64) & 1 == 1 {
                        open[x] = y;
                    } else if y - open[x] >= lv {
                        vruns[x].push((open[x], y - 1));
                    }
                }
            }
            // horizontal runs: maximal stretches of set bits
            let mut x = next_bit(cur, 0, w, true);
            while x < w {
                let end = next_bit(cur, x, w, false);
                if end - x >= lh {
                    hruns[y].push((x, end - 1));
                }
                x = next_bit(cur, end, w, true);
            }
        }
        if h > 0 {
            let last = self.row(h - 1);
            for (x, &o) in open.iter().enumerate() {
                if last[x / 64] >> (x % 64) & 1 == 1 && h - o >= lv {
                    vruns[x].push((o, h - 1));
                }
            }
        }
        (hruns, vruns)
    }
}

/// The first position at or after `from` (below `w`) whose bit is `set`; `w` when there is none.
fn next_bit(row: &[u64], from: usize, w: usize, set: bool) -> usize {
    let mut k = from / 64;
    if k >= row.len() {
        return w;
    }
    let flip = if set { 0 } else { u64::MAX };
    let mut word = (row[k] ^ flip) & (u64::MAX << (from % 64));
    loop {
        if word != 0 {
            return (k * 64 + word.trailing_zeros() as usize).min(w);
        }
        k += 1;
        if k >= row.len() {
            return w;
        }
        word = row[k] ^ flip;
    }
}

/// Ruling lines from ink runs: each 8-connected component of runs on adjacent rows (columns) is
/// one rule, unless it is thicker than 2.5 pt.
#[allow(clippy::needless_range_loop)]
fn rules_from_runs((hruns, vruns): (Runs, Runs), scale: f32) -> Rules {
    let thick_max_pt = 2.5f32;
    let mut out = Rules::default();
    for (axis, runs) in [(0, hruns), (1, vruns)] {
        let lines_n = runs.len();
        // 8-connected components over runs of adjacent lines (union-find)
        let mut ids: Vec<(usize, usize, usize)> = Vec::new(); // (line, start, end)
        let mut first: Vec<usize> = vec![0; lines_n + 1];
        for (line, rs) in runs.iter().enumerate() {
            first[line] = ids.len();
            for &(s, e) in rs {
                ids.push((line, s, e));
            }
        }
        first[lines_n] = ids.len();
        let mut parent: Vec<usize> = (0..ids.len()).collect();
        fn root(p: &mut [usize], mut k: usize) -> usize {
            while p[k] != k {
                p[k] = p[p[k]];
                k = p[k];
            }
            k
        }
        for line in 1..lines_n {
            for a in first[line - 1]..first[line] {
                for b in first[line]..first[line + 1] {
                    let (_, s0, e0) = ids[a];
                    let (_, s1, e1) = ids[b];
                    if s1 <= e0 + 1 && e1 + 1 >= s0 {
                        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                        if ra != rb {
                            parent[ra] = rb;
                        }
                    }
                }
            }
        }
        // component extents, in root order (deterministic)
        let mut comp: Vec<Option<(usize, usize, usize, usize)>> = vec![None; ids.len()];
        for k in 0..ids.len() {
            let r = root(&mut parent, k);
            let (line, s, e) = ids[k];
            let c = comp[r].get_or_insert((usize::MAX, 0, usize::MAX, 0));
            c.0 = c.0.min(line);
            c.1 = c.1.max(line);
            c.2 = c.2.min(s);
            c.3 = c.3.max(e);
        }
        let round1 = |v: f32| (v * 10.0).round() / 10.0;
        for (l0, l1, s, e) in comp.into_iter().flatten() {
            let thick = (l1 - l0 + 1) as f32;
            if thick > thick_max_pt * scale {
                continue;
            }
            let along = (e - s + 1) as f32;
            let mid = (l0 as f32 + thick / 2.0) / scale;
            let (a0, a1) = (s as f32 / scale, (s as f32 + along) / scale);
            if axis == 0 {
                out.h.push([round1(mid), round1(a0), round1(a1)]);
            } else {
                out.v.push([round1(mid), round1(a0), round1(a1)]);
            }
        }
    }
    out
}

/// A page rendered for the image branch: grey (0 = black) with its longest side
/// `net::IMG_SIDE`, and its pixels per PDF point.
#[derive(Clone, Debug)]
pub struct PageImage {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
    pub k: f32,
}

impl PageImage {
    /// The model's input: ink on the square canvas, unit boxes (`pbox`, points) as fractions.
    pub fn input(&self, pboxes: &[[f32; 4]]) -> super::ImgInput {
        let side = super::IMG_SIDE;
        let mut canvas = vec![0f32; side * side];
        for y in 0..self.h.min(side) {
            for x in 0..self.w.min(side) {
                canvas[y * side + x] = (255.0 - self.px[y * self.w + x] as f32) / 255.0;
            }
        }
        let f = self.k / side as f32;
        let boxes = pboxes
            .iter()
            .map(|b| [b[0] * f, b[1] * f, b[2] * f, b[3] * f].map(|v| v.clamp(0.0, 1.0)))
            .collect();
        super::ImgInput {
            canvas,
            boxes,
            valid: (self.w as f32 / side as f32, self.h as f32 / side as f32),
        }
    }
}

/// Area-average resample (OpenCV INTER_AREA when shrinking) of a grey image to (tw, th).
pub fn resample_area(g: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    // separable box filter with fractional coverage: rows, then columns. Each output's taps are
    // (source index, weight) in source order, flattened, with `at[o]..at[o + 1]` its range.
    let weights = |n_in: usize, n_out: usize| -> (Vec<usize>, Vec<(usize, f32)>) {
        let f = n_in as f32 / n_out as f32;
        let mut at = vec![0];
        let mut taps = Vec::new();
        for o in 0..n_out {
            let (a, b) = (o as f32 * f, (o as f32 + 1.0) * f);
            let mut i = a.floor() as usize;
            while (i as f32) < b && i < n_in {
                let cov = (b.min(i as f32 + 1.0) - a.max(i as f32)).max(0.0);
                if cov > 0.0 {
                    taps.push((i, cov / f));
                }
                i += 1;
            }
            at.push(taps.len());
        }
        (at, taps)
    };
    let ((ax, wx), (ay, wy)) = (weights(w, tw), weights(h, th));
    let mut tmp = vec![0f32; h * tw];
    for (row, dst) in g.chunks_exact(w).zip(tmp.chunks_exact_mut(tw)) {
        for (x, d) in dst.iter_mut().enumerate() {
            let mut s = 0f32;
            for &(i, c) in &wx[ax[x]..ax[x + 1]] {
                s += row[i] as f32 * c;
            }
            *d = s;
        }
    }
    // columns: the same per-output sums in the same tap order, accumulated a row at a time
    let mut out = vec![0u8; th * tw];
    let mut acc = vec![0f32; tw];
    for (y, dst) in out.chunks_exact_mut(tw).enumerate() {
        acc.iter_mut().for_each(|a| *a = 0.0);
        for &(i, c) in &wy[ay[y]..ay[y + 1]] {
            for (a, &v) in acc.iter_mut().zip(&tmp[i * tw..(i + 1) * tw]) {
                *a += v * c;
            }
        }
        for (d, &a) in dst.iter_mut().zip(&acc) {
            *d = a.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}
