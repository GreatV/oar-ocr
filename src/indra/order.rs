//! Reading order: the order model's line features, the decoding of its
//! pairwise logits, and the XY-cut and top-down baselines it falls back on
//! inside blocks.

/// Block classes of the order model's layout head.
pub(super) const BLOCK_CLASSES: [&str; 10] = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
];
pub(super) const N_FEATS: usize = 16;
/// Table, Table-Of-Contents and Form blocks read row by row.
const GRID_CLASSES: [usize; 3] = [5, 8, 9];

/// A square matrix of logits, row-major.
pub(super) struct Square<'a> {
    pub n: usize,
    pub data: &'a [f32],
}

impl Square<'_> {
    pub(super) fn at(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.n + j]
    }
}

/// Median as numpy computes it on float32 data.
pub(super) fn median_f32(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Python's `str.isdigit` for one character, closely enough for features:
/// decimal and superscript-style digits, not fractions or numeral letters.
pub(super) fn is_digit(c: char) -> bool {
    c.is_numeric()
        && !matches!(c,
            '\u{BC}'..='\u{BE}' | '\u{2150}'..='\u{218B}' | '\u{09F4}'..='\u{09F9}'
            | '\u{3007}' | '\u{3021}'..='\u{3029}' | '\u{3038}'..='\u{303A}'
            | '\u{4E00}'..='\u{9FFF}')
}

/// `[N, 4]` boxes and their texts -> `[N, N_FEATS]` order-model features.
pub(super) fn line_features(boxes: &[[f32; 4]], page_wh: (u32, u32), texts: &[String]) -> Vec<f32> {
    let n = boxes.len();
    let (pw, ph) = (page_wh.0 as f32, page_wh.1 as f32);
    let mut heights: Vec<f32> = boxes.iter().map(|b| (b[3] - b[1]).max(1e-3)).collect();
    let med_h = median_f32(&mut heights);
    let mut out = Vec::with_capacity(n * N_FEATS);
    for (b, t) in boxes.iter().zip(texts) {
        let w = (b[2] - b[0]).max(1e-3);
        let h = (b[3] - b[1]).max(1e-3);
        out.extend_from_slice(&[
            b[0] / pw,
            b[1] / ph,
            b[2] / pw,
            b[3] / ph,
            (b[0] + b[2]) / (2.0 * pw),
            (b[1] + b[3]) / (2.0 * ph),
            w / pw,
            (h / med_h).ln(),
            (w / h).ln() / 4.0,
            (med_h as f64 / ph as f64 * 20.0) as f32,
        ]);
        let t = t.trim();
        let chars: Vec<char> = t.chars().collect();
        if chars.is_empty() {
            out.extend_from_slice(&[0.0; 6]);
            continue;
        }
        let len = chars.len() as f64;
        let head: String = chars.iter().take(3).collect();
        let head = head.trim_end_matches(['.', ')']);
        let numbered = !head.is_empty() && head.chars().all(is_digit) && chars.len() > 3;
        let last = chars[chars.len() - 1];
        out.extend_from_slice(&[
            (len.ln_1p() / 5.0) as f32,
            (chars.iter().filter(|c| is_digit(**c)).count() as f64 / len) as f32,
            ("•-–▪◦*·→".contains(chars[0]) || numbered) as u8 as f32,
            t.ends_with('-') as u8 as f32,
            ".!?:;".contains(last) as u8 as f32,
            chars[0].is_uppercase() as u8 as f32,
        ]);
    }
    out
}

/// `numpy.logaddexp(0, x)` in float32.
fn log1p_exp(x: f32) -> f32 {
    if x == 0.0 {
        std::f32::consts::LN_2
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// Order from a pairwise logit matrix (`a[i][j] > 0`: i before j): the
/// row-sum ranking, then up to `sweeps` rounds of insertion moves that each
/// put one line where it most raises the summed log-probability.
pub(super) fn decode_order(a: &Square, sweeps: usize) -> Vec<usize> {
    let n = a.n;
    if n <= 1 {
        return (0..n).collect();
    }
    let lp: Vec<f32> = a.data.iter().map(|&v| -log1p_exp(-v)).collect();
    let score: Vec<f32> = (0..n)
        .map(|i| {
            let row: f64 = (0..n).map(|j| lp[i * n + j] as f64).sum();
            let col: f64 = (0..n).map(|j| lp[j * n + i] as f64).sum();
            -(row as f32) + col as f32
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| score[i].total_cmp(&score[j]));
    let mut gains = vec![0f32; n];
    for _ in 0..sweeps {
        let mut improved = false;
        for x in order.clone() {
            let rest: Vec<usize> = order.iter().copied().filter(|&v| v != x).collect();
            // gains[k] = sum(lp[rest[..k], x]) + sum(lp[x, rest[k..]])
            let mut acc = 0f32;
            gains[0] = 0.0;
            for (k, &r) in rest.iter().enumerate() {
                acc += lp[r * n + x];
                gains[k + 1] = acc;
            }
            let mut acc = 0f32;
            for k in (0..rest.len()).rev() {
                acc += lp[x * n + rest[k]];
                gains[k] += acc;
            }
            let k = argmax(&gains[..n]);
            let cur = order
                .iter()
                .position(|&v| v == x)
                .expect("x is in the order");
            if gains[k] > gains[cur] + 1e-6 {
                let mut next = rest;
                next.insert(k, x);
                order = next;
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    order
}

/// Rows top to bottom (centers within half a line height share a row),
/// left to right.
pub(super) fn order_top_down(boxes: &[[f32; 4]]) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let mut heights: Vec<f32> = boxes.iter().map(|b| b[3] - b[1]).collect();
    let h = median_f32(&mut heights);
    let unit = (h * 0.5).max(1e-3);
    let key = |b: &[f32; 4]| (((b[1] + b[3]) / 2.0 / unit).round_ties_even() as i64, b[0]);
    let mut idx: Vec<usize> = (0..boxes.len()).collect();
    idx.sort_by(|&i, &j| {
        let (ri, xi) = key(&boxes[i]);
        let (rj, xj) = key(&boxes[j]);
        ri.cmp(&rj).then(xi.total_cmp(&xj))
    });
    idx
}

/// Recursive XY-cut: split at the widest whitespace gap on either axis (upper
/// or left part first); without a gap, top-down order.
pub(super) fn order_xycut(boxes: &[[f32; 4]]) -> Vec<usize> {
    fn best_cut(boxes: &[[f32; 4]], ids: &[usize], axis_y: bool) -> (f32, usize, Vec<usize>) {
        let (lo, hi) = if axis_y { (1, 3) } else { (0, 2) };
        let mut o: Vec<usize> = (0..ids.len()).collect();
        o.sort_by(|&i, &j| boxes[ids[i]][lo].total_cmp(&boxes[ids[j]][lo]));
        let mut end = boxes[ids[o[0]]][hi];
        let (mut best_gap, mut best_k) = (0f32, 0usize);
        for k in 1..o.len() {
            let gap = boxes[ids[o[k]]][lo] - end;
            if gap > best_gap {
                best_gap = gap;
                best_k = k;
            }
            end = end.max(boxes[ids[o[k]]][hi]);
        }
        (best_gap, best_k, o)
    }
    fn rec(boxes: &[[f32; 4]], ids: Vec<usize>) -> Vec<usize> {
        if ids.len() <= 1 {
            return ids;
        }
        let (gy, ky, oy) = best_cut(boxes, &ids, true);
        let (gx, kx, ox) = best_cut(boxes, &ids, false);
        if ky == 0 && kx == 0 {
            let sub: Vec<[f32; 4]> = ids.iter().map(|&i| boxes[i]).collect();
            return order_top_down(&sub).into_iter().map(|k| ids[k]).collect();
        }
        let (k, o) = if gy >= gx { (ky, oy) } else { (kx, ox) };
        let mut out = rec(boxes, o[..k].iter().map(|&i| ids[i]).collect());
        out.extend(rec(boxes, o[k..].iter().map(|&i| ids[i]).collect()));
        out
    }
    rec(boxes, (0..boxes.len()).collect())
}

/// Average-linkage clustering of pairwise "same group" logits: merge the two
/// groups with the highest mean logit while that mean is positive.
pub(super) fn cluster(logits: &Square) -> Vec<Vec<usize>> {
    let n = logits.n;
    if n == 0 {
        return Vec::new();
    }
    let mut s: Vec<f64> = logits.data.iter().map(|&v| v as f64).collect();
    for i in 0..n {
        s[i * n + i] = 0.0;
    }
    let mut size = vec![1f64; n];
    let mut alive = vec![true; n];
    let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut m = s.clone();
    for i in 0..n {
        m[i * n + i] = f64::NEG_INFINITY;
    }
    let row_argmax = |m: &[f64], i: usize| -> usize {
        let row = &m[i * n..(i + 1) * n];
        let mut best = 0;
        for (j, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = j;
            }
        }
        best
    };
    let mut best: Vec<usize> = (0..n).map(|i| row_argmax(&m, i)).collect();
    let mut best_v: Vec<f64> = (0..n).map(|i| m[i * n + best[i]]).collect();
    for _ in 0..n.saturating_sub(1) {
        let mut a = 0;
        for (i, v) in best_v.iter().enumerate() {
            if *v > best_v[a] {
                a = i;
            }
        }
        let b = best[a];
        if best_v[a] <= 0.0 {
            break;
        }
        let (a, b) = (a.min(b), a.max(b));
        for j in 0..n {
            s[a * n + j] += s[b * n + j];
        }
        for i in 0..n {
            s[i * n + a] += s[i * n + b];
        }
        s[a * n + a] = 0.0;
        size[a] += size[b];
        alive[b] = false;
        let moved = std::mem::take(&mut members[b]);
        members[a].extend(moved);
        let row: Vec<f64> = (0..n)
            .map(|j| {
                if !alive[j] || j == a {
                    f64::NEG_INFINITY
                } else {
                    s[a * n + j] / (size[a] * size[j])
                }
            })
            .collect();
        for j in 0..n {
            m[a * n + j] = row[j];
            m[j * n + a] = row[j];
            m[b * n + j] = f64::NEG_INFINITY;
            m[j * n + b] = f64::NEG_INFINITY;
        }
        best_v[b] = f64::NEG_INFINITY;
        best[a] = row_argmax(&m, a);
        best_v[a] = row[best[a]];
        for j in 0..n {
            if alive[j] && j != a && (best[j] == a || best[j] == b) {
                best[j] = row_argmax(&m, j);
                best_v[j] = m[j * n + best[j]];
            }
        }
        for j in 0..n {
            if alive[j] && j != a && row[j] > best_v[j] {
                best[j] = a;
                best_v[j] = row[j];
            }
        }
    }
    (0..n)
        .filter(|&i| alive[i])
        .map(|i| {
            let mut g = members[i].clone();
            g.sort_unstable();
            g
        })
        .collect()
}

/// Sums the class logits of a group of lines and returns the winning class.
pub(super) fn group_class(cls: &[f32], ids: &[usize]) -> usize {
    let nc = BLOCK_CLASSES.len();
    let mut sum = vec![0f32; nc];
    for &i in ids {
        for (c, s) in sum.iter_mut().enumerate() {
            *s += cls[i * nc + c];
        }
    }
    argmax(&sum)
}

/// Reading order with the layout heads: blocks from clustering the
/// same-block logits, lines inside a block by XY-cut (row by row in grid
/// blocks), blocks by the mean pairwise order logit between them.
pub(super) fn decode_blocks_order(
    a: &Square,
    same: &Square,
    boxes: &[[f32; 4]],
    cls: &[f32],
) -> Vec<usize> {
    let n = a.n;
    if n <= 1 {
        return (0..n).collect();
    }
    let blocks = cluster(same);
    let inner: Vec<Vec<usize>> = blocks
        .iter()
        .map(|b| {
            let bx: Vec<[f32; 4]> = b.iter().map(|&i| boxes[i]).collect();
            let local = if GRID_CLASSES.contains(&group_class(cls, b)) {
                order_top_down(&bx)
            } else {
                order_xycut(&bx)
            };
            local.into_iter().map(|k| b[k]).collect()
        })
        .collect();
    // Mean order logit between blocks p < q, made antisymmetric.
    let m = blocks.len();
    let mut up = vec![0f64; m * m];
    for p in 0..m {
        for q in p + 1..m {
            let mut sum = 0f64;
            for &i in &blocks[p] {
                for &j in &blocks[q] {
                    sum += a.at(i, j) as f64;
                }
            }
            up[p * m + q] = sum / (blocks[p].len() * blocks[q].len()) as f64;
        }
    }
    let ab: Vec<f32> = (0..m * m)
        .map(|k| {
            let (p, q) = (k / m, k % m);
            (up[p * m + q] - up[q * m + p]) as f32
        })
        .collect();
    decode_order(&Square { n: m, data: &ab }, 4)
        .into_iter()
        .flat_map(|k| inner[k].clone())
        .collect()
}
