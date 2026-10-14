//! Line detection pre- and post-processing: the page as a padded ink map,
//! and line boxes from the core-band probability and the top / bottom
//! distance maps.

use super::cv::{Interp, connected_components, resize};
use image::GrayImage;

/// Output stride of the detector.
const OUT_STRIDE: usize = 2;
/// Input height and width must divide by this.
const INPUT_MULTIPLE: usize = 32;
/// The core band is shorter than the line by this fraction of its height at
/// each end.
const CORE_X_SHRINK: f64 = 0.15;

/// The page scaled so its long side is `long_side` and padded to a multiple
/// of 32: the `[H, W]` ink map (0 = white), `H`, `W`, and the scale.
pub(super) fn page_to_input(gray: &GrayImage, long_side: usize) -> (Vec<f32>, usize, usize, f64) {
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let scale = long_side as f64 / w.max(h) as f64;
    let nh = ((h as f64 * scale).round_ties_even() as usize).max(1);
    let nw = ((w as f64 * scale).round_ties_even() as usize).max(1);
    let interp = if scale < 1.0 {
        Interp::Area
    } else {
        Interp::Linear
    };
    let scaled = resize(gray, nw as u32, nh as u32, interp);
    let ph = nh.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let pw = nw.div_ceil(INPUT_MULTIPLE) * INPUT_MULTIPLE;
    let mut out = vec![0f32; ph * pw];
    for (y, row) in scaled.as_raw().chunks_exact(nw).enumerate() {
        for (o, &v) in out[y * pw..y * pw + nw].iter_mut().zip(row) {
            *o = 1.0 - v as f32 / 255.0;
        }
    }
    (out, ph, pw, scale)
}

/// Line boxes `[x0, y0, x1, y1, score]` in input pixels from one page's
/// `[3, h, w]` output (core logit, top distance, bottom distance).
pub(super) fn decode(out: &[f32], h: usize, w: usize) -> Vec<[f32; 5]> {
    let plane = h * w;
    let prob: Vec<f32> = out[..plane]
        .iter()
        .map(|&v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    let mask: Vec<bool> = prob.iter().map(|&p| p > 0.5).collect();
    let (labels, stats) = connected_components(&mask, w, h);
    let n = stats.len();
    let (mut top, mut bot, mut score) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let ys = ((i / w) as f32 + 0.5) * OUT_STRIDE as f32;
        let c = l as usize - 1;
        top[c] += (ys - out[plane + i]) as f64;
        bot[c] += (ys + out[2 * plane + i]) as f64;
        score[c] += prob[i] as f64;
    }
    let s = OUT_STRIDE as f64;
    stats
        .iter()
        .enumerate()
        .filter(|(_, st)| st.area >= 2)
        .map(|(c, st)| {
            let cnt = st.area as f64;
            let (y0, y1) = (top[c] / cnt, bot[c] / cnt);
            let lh = (y1 - y0).max(1.0);
            let ex = (CORE_X_SHRINK * lh).min(0.5 * lh);
            [
                (st.left as f64 * s - ex) as f32,
                y0 as f32,
                (st.right as f64 * s + ex) as f32,
                y1 as f32,
                (score[c] / cnt) as f32,
            ]
        })
        .collect()
}
