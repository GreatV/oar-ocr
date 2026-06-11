//! SIMD-accelerated kernels for the pipeline's two hot CPU loops: per-pixel
//! image normalization and per-timestep CTC argmax decode. (Conv/matmul work
//! lives in ONNX Runtime.)
//!
//! Each kernel ships as a scalar reference (`*_scalar`, always compiled) and a
//! SIMD path (`*_simd`, behind the `simd` feature) built on [`wide`] and
//! runtime-dispatched via [`multiversion`], so one binary selects AVX2/SSE/NEON
//! at runtime without `target-cpu=native`. Entry points pick the SIMD path when
//! the feature is on, else the scalar one.
//!
//! The SIMD paths are **bit-identical** to the scalar reference: same
//! `value * alpha + beta` form (plain multiply-add, *not* FMA, which rounds
//! differently) and same "last index wins on ties" argmax. Parity tests at the
//! bottom of the module enforce this, so toggling `simd` never changes output.
//!
//! [`wide`]: https://docs.rs/wide
//! [`multiversion`]: https://docs.rs/multiversion

/// Writes a normalized image in CHW (channel-major) layout into `out`.
///
/// For each output channel `c`, plane `c` is `out[c * plane .. (c + 1) * plane]`
/// where `plane = width * height`, and
/// `out[c * plane + p] = rgb[p * 3 + src_channels[c]] as f32 * alpha[c] + beta[c]`.
///
/// `rgb` is the tightly packed interleaved RGB buffer (`image::RgbImage::as_raw`),
/// length `width * height * 3`. `out` must have length `3 * plane`.
#[inline]
pub fn normalize_chw_into(
    rgb: &[u8],
    width: usize,
    height: usize,
    src_channels: [usize; 3],
    alpha: &[f32; 3],
    beta: &[f32; 3],
    out: &mut [f32],
) {
    #[cfg(feature = "simd")]
    {
        normalize_chw_simd(rgb, width, height, src_channels, alpha, beta, out);
    }
    #[cfg(not(feature = "simd"))]
    {
        normalize_chw_scalar(rgb, width, height, src_channels, alpha, beta, out);
    }
}

/// Writes a normalized image in HWC (interleaved) layout into `out`.
///
/// `out[p * 3 + c] = rgb[p * 3 + src_channels[c]] as f32 * alpha[c] + beta[c]`.
/// `out` must have length `width * height * 3`.
#[inline]
pub fn normalize_hwc_into(
    rgb: &[u8],
    width: usize,
    height: usize,
    src_channels: [usize; 3],
    alpha: &[f32; 3],
    beta: &[f32; 3],
    out: &mut [f32],
) {
    // HWC is interleaved (and BGR swaps per triple), so it doesn't vectorize
    // cleanly; the scalar raw-slice path is used regardless of the `simd` feature.
    normalize_hwc_scalar(rgb, width, height, src_channels, alpha, beta, out);
}

/// Returns the `(index, value)` of the maximum element of `row`.
///
/// On ties the **last** maximal index is returned, matching
/// [`Iterator::max_by`] semantics (which the previous decode loop relied on).
/// Returns `None` for an empty row.
#[inline]
pub fn argmax(row: &[f32]) -> Option<(usize, f32)> {
    #[cfg(feature = "simd")]
    {
        argmax_simd(row)
    }
    #[cfg(not(feature = "simd"))]
    {
        argmax_scalar(row)
    }
}

// Scalar reference implementations: the active fallback when `simd` is off, and
// the oracle for the parity tests.
#[cfg(any(not(feature = "simd"), test))]
#[inline]
fn normalize_chw_scalar(
    rgb: &[u8],
    width: usize,
    height: usize,
    src_channels: [usize; 3],
    alpha: &[f32; 3],
    beta: &[f32; 3],
    out: &mut [f32],
) {
    let plane = width * height;
    for c in 0..3 {
        let (a, b, sc) = (alpha[c], beta[c], src_channels[c]);
        let dst = &mut out[c * plane..c * plane + plane];
        for (p, d) in dst.iter_mut().enumerate() {
            *d = rgb[p * 3 + sc] as f32 * a + b;
        }
    }
}

#[inline]
fn normalize_hwc_scalar(
    rgb: &[u8],
    width: usize,
    height: usize,
    src_channels: [usize; 3],
    alpha: &[f32; 3],
    beta: &[f32; 3],
    out: &mut [f32],
) {
    let plane = width * height;
    for p in 0..plane {
        let base = p * 3;
        for c in 0..3 {
            out[base + c] = rgb[base + src_channels[c]] as f32 * alpha[c] + beta[c];
        }
    }
}

// Fallback when `simd` is off; also the oracle in `argmax_matches_scalar_*`.
#[cfg(any(not(feature = "simd"), test))]
#[inline]
fn argmax_scalar(row: &[f32]) -> Option<(usize, f32)> {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &v)| (idx, v))
}

// SIMD implementations (feature = "simd").

#[cfg(feature = "simd")]
fn normalize_chw_simd(
    rgb: &[u8],
    width: usize,
    height: usize,
    src_channels: [usize; 3],
    alpha: &[f32; 3],
    beta: &[f32; 3],
    out: &mut [f32],
) {
    let plane = width * height;
    for c in 0..3 {
        let dst = &mut out[c * plane..c * plane + plane];
        normalize_plane_simd(rgb, src_channels[c], alpha[c], beta[c], dst);
    }
}

/// Fills one contiguous CHW plane: `dst[p] = rgb[p * 3 + sc] as f32 * a + b`.
///
/// The strided u8 gather is scalar (3-channel deinterleave has no portable SIMD
/// gather), but the `* a + b` arithmetic and the store run 8 lanes wide. Uses a
/// plain multiply-add (not `mul_add`) to stay bit-identical to the scalar path.
#[cfg(feature = "simd")]
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.2", "aarch64+neon"))]
fn normalize_plane_simd(rgb: &[u8], sc: usize, a: f32, b: f32, dst: &mut [f32]) {
    use wide::f32x8;

    let va = f32x8::splat(a);
    let vb = f32x8::splat(b);
    let mut chunks = dst.chunks_exact_mut(8);
    let mut p = 0usize;
    for chunk in &mut chunks {
        let gathered = [
            rgb[p * 3 + sc] as f32,
            rgb[(p + 1) * 3 + sc] as f32,
            rgb[(p + 2) * 3 + sc] as f32,
            rgb[(p + 3) * 3 + sc] as f32,
            rgb[(p + 4) * 3 + sc] as f32,
            rgb[(p + 5) * 3 + sc] as f32,
            rgb[(p + 6) * 3 + sc] as f32,
            rgb[(p + 7) * 3 + sc] as f32,
        ];
        let v = f32x8::new(gathered) * va + vb;
        chunk.copy_from_slice(&v.to_array());
        p += 8;
    }
    for d in chunks.into_remainder() {
        *d = rgb[p * 3 + sc] as f32 * a + b;
        p += 1;
    }
}

#[cfg(feature = "simd")]
fn argmax_simd(row: &[f32]) -> Option<(usize, f32)> {
    if row.is_empty() {
        return None;
    }
    // 8-wide max-reduce for the value, then a forward scan keeping the *last*
    // index equal to it — same tie-breaking as `Iterator::max_by`.
    let max_val = reduce_max_simd(row);
    let mut idx = 0usize;
    for (i, &v) in row.iter().enumerate() {
        // `v >= max_val` holds only at the maximum (NaN-free input).
        if v >= max_val {
            idx = i;
        }
    }
    Some((idx, max_val))
}

#[cfg(feature = "simd")]
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.2", "aarch64+neon"))]
fn reduce_max_simd(row: &[f32]) -> f32 {
    use wide::f32x8;

    let mut acc = f32x8::splat(f32::NEG_INFINITY);
    let mut chunks = row.chunks_exact(8);
    for chunk in &mut chunks {
        // chunk has exactly 8 elements.
        let v = f32x8::new([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        acc = acc.max(v);
    }
    let mut best = f32::NEG_INFINITY;
    for lane in acc.to_array() {
        best = best.max(lane);
    }
    for &v in chunks.remainder() {
        best = best.max(v);
    }
    best
}

// CRNN recognition normalize: `(v / 255.0 - 0.5) / 0.5` in BGR order.

/// BGR source-channel order for the CRNN tensor (dst channel 0 ← src B, etc.).
const CRNN_SRC: [usize; 3] = [2, 1, 0];

/// Fills a CRNN CHW input tensor from a resized RGB crop.
///
/// Each channel is normalized as `(v / 255.0 - 0.5) / 0.5` — the PaddleOCR CRNN
/// convention — and emitted in BGR channel order. `dst` is the full
/// `3 * img_h * tensor_width` tensor; columns `[resized_w, tensor_width)` of
/// every row keep their prior value (the caller's zero padding).
///
/// `rgb` is the tightly packed `resized_w * img_h * 3` interleaved buffer
/// (`image::RgbImage::as_raw`). The kernel is bit-identical to the scalar
/// reference: it keeps the exact `(v / 255 - 0.5) / 0.5` op order (a true
/// division, not a reciprocal multiply), so recognition input is unchanged.
#[inline]
pub fn normalize_crnn_chw_into(
    rgb: &[u8],
    resized_w: usize,
    img_h: usize,
    tensor_width: usize,
    dst: &mut [f32],
) {
    let plane = img_h * tensor_width;
    for (c, &sc) in CRNN_SRC.iter().enumerate() {
        let cbase = c * plane;
        for y in 0..img_h {
            let src_row = &rgb[y * resized_w * 3..y * resized_w * 3 + resized_w * 3];
            let row_off = cbase + y * tensor_width;
            let dst_row = &mut dst[row_off..row_off + resized_w];
            #[cfg(feature = "simd")]
            crnn_row_simd(src_row, sc, dst_row);
            #[cfg(not(feature = "simd"))]
            crnn_row_scalar(src_row, sc, dst_row);
        }
    }
}

// Fallback when `simd` is off. The CRNN parity test uses an independent
// full-tensor oracle rather than this per-row helper, so it is not needed in
// `simd` test builds.
#[cfg(not(feature = "simd"))]
#[inline]
fn crnn_row_scalar(src_row: &[u8], sc: usize, dst_row: &mut [f32]) {
    for (x, d) in dst_row.iter_mut().enumerate() {
        *d = (src_row[x * 3 + sc] as f32 / 255.0 - 0.5) / 0.5;
    }
}

#[cfg(feature = "simd")]
#[multiversion::multiversion(targets("x86_64+avx2+fma", "x86_64+sse4.2", "aarch64+neon"))]
fn crnn_row_simd(src_row: &[u8], sc: usize, dst_row: &mut [f32]) {
    use wide::f32x8;

    let mut chunks = dst_row.chunks_exact_mut(8);
    let mut x = 0usize;
    for chunk in &mut chunks {
        let g = [
            src_row[x * 3 + sc] as f32,
            src_row[(x + 1) * 3 + sc] as f32,
            src_row[(x + 2) * 3 + sc] as f32,
            src_row[(x + 3) * 3 + sc] as f32,
            src_row[(x + 4) * 3 + sc] as f32,
            src_row[(x + 5) * 3 + sc] as f32,
            src_row[(x + 6) * 3 + sc] as f32,
            src_row[(x + 7) * 3 + sc] as f32,
        ];
        // Exact `(v / 255.0 - 0.5) / 0.5`, lane-wise.
        let v = (f32x8::new(g) / 255.0 - 0.5) / 0.5;
        chunk.copy_from_slice(&v.to_array());
        x += 8;
    }
    for d in chunks.into_remainder() {
        *d = (src_row[x * 3 + sc] as f32 / 255.0 - 0.5) / 0.5;
        x += 1;
    }
}

// UVDoc rectification output: CHW float planes -> interleaved RGB u8.

/// Converts a BGR-ordered CHW float plane-triple into an interleaved RGB `u8`
/// image, computing `(v * scale).clamp(0.0, 255.0) as u8` per element.
///
/// `c0`/`c1`/`c2` are the three contiguous channel planes (each `plane` long,
/// in the model's BGR order). `out` is `plane * 3` interleaved bytes, emitted as
/// RGB (`r <- c2`, `g <- c1`, `b <- c0`).
///
/// This is intentionally a tight contiguous scalar loop, *not* a hand-vectorized
/// kernel: the output is an interleaved (strided) RGB scatter, which an explicit
/// `wide` gather/scatter measured ~1.8x *slower* than letting the autovectorizer
/// handle this `chunks_exact_mut(3)` form. The win over the previous
/// `put_pixel` + 4-D strided `ndarray` indexing comes from the contiguous
/// plane access alone (~1.8x faster than the old loop). Kept here next to the
/// other post-processing kernels for cohesion.
#[inline]
pub fn scale_clamp_bgr_planes_to_rgb(
    c0: &[f32],
    c1: &[f32],
    c2: &[f32],
    scale: f32,
    out: &mut [u8],
) {
    for (p, px) in out.chunks_exact_mut(3).enumerate() {
        px[0] = to_u8(c2[p], scale);
        px[1] = to_u8(c1[p], scale);
        px[2] = to_u8(c0[p], scale);
    }
}

#[inline]
fn to_u8(v: f32, scale: f32) -> u8 {
    (v * scale).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(width: usize, height: usize) -> Vec<u8> {
        let n = width * height * 3;
        // Deterministic pseudo-random bytes covering the full range.
        (0..n).map(|i| ((i * 37 + 11) % 256) as u8).collect()
    }

    #[test]
    fn chw_simd_matches_scalar_rgb_and_bgr() {
        let (w, h) = (37, 19); // not a multiple of 8 -> exercises the tail
        let rgb = make_rgb(w, h);
        let alpha = [1.0 / 255.0, 0.5, 2.0];
        let beta = [-0.485, 0.1, -1.0];
        for src in [[0, 1, 2], [2, 1, 0]] {
            let mut a = vec![0.0f32; w * h * 3];
            let mut b = vec![0.0f32; w * h * 3];
            normalize_chw_scalar(&rgb, w, h, src, &alpha, &beta, &mut a);
            normalize_chw_into(&rgb, w, h, src, &alpha, &beta, &mut b);
            assert_eq!(
                a, b,
                "CHW SIMD must be bit-identical to scalar (src={src:?})"
            );
        }
    }

    #[test]
    fn hwc_matches_scalar_rgb_and_bgr() {
        let (w, h) = (23, 7);
        let rgb = make_rgb(w, h);
        let alpha = [1.0 / 255.0, 0.5, 2.0];
        let beta = [-0.485, 0.1, -1.0];
        for src in [[0, 1, 2], [2, 1, 0]] {
            let mut a = vec![0.0f32; w * h * 3];
            let mut b = vec![0.0f32; w * h * 3];
            normalize_hwc_scalar(&rgb, w, h, src, &alpha, &beta, &mut a);
            normalize_hwc_into(&rgb, w, h, src, &alpha, &beta, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn argmax_matches_scalar_including_ties() {
        // Tail-exercising length, with deterministic values.
        let row: Vec<f32> = (0..101).map(|i| ((i * 17) % 13) as f32 * 0.5).collect();
        assert_eq!(argmax(&row), argmax_scalar(&row));

        // Explicit ties: the maximum 9.0 appears at indices 3 and 6 -> last wins.
        let tied = vec![1.0f32, 2.0, 5.0, 9.0, 4.0, 8.0, 9.0, 0.0];
        assert_eq!(argmax_scalar(&tied), Some((6, 9.0)));
        assert_eq!(argmax(&tied), Some((6, 9.0)));

        // Single element and empty.
        assert_eq!(argmax(&[42.0]), Some((0, 42.0)));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn crnn_simd_matches_scalar_with_padding() {
        // resized_w not a multiple of 8, tensor_width > resized_w (padding).
        let (resized_w, img_h, tensor_width) = (21usize, 32usize, 40usize);
        let rgb = make_rgb(resized_w, img_h);

        let mut a = vec![0.0f32; 3 * img_h * tensor_width];
        let mut b = vec![0.0f32; 3 * img_h * tensor_width];
        // Scalar reference, emulating the original crnn.rs loop exactly.
        let plane = img_h * tensor_width;
        for y in 0..img_h {
            for x in 0..resized_w {
                let s = (y * resized_w + x) * 3;
                let idx = y * tensor_width + x;
                a[idx] = (rgb[s + 2] as f32 / 255.0 - 0.5) / 0.5;
                a[plane + idx] = (rgb[s + 1] as f32 / 255.0 - 0.5) / 0.5;
                a[2 * plane + idx] = (rgb[s] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        normalize_crnn_chw_into(&rgb, resized_w, img_h, tensor_width, &mut b);
        assert_eq!(
            a, b,
            "CRNN SIMD must be bit-identical to scalar (incl. padding)"
        );
    }

    #[test]
    fn uvdoc_conversion_matches_reference() {
        let plane = 8 * 7 + 3;
        // Values spanning below 0 and above 255 to exercise both clamp bounds.
        let c0: Vec<f32> = (0..plane).map(|i| (i as f32) * 1.9 - 5.0).collect();
        let c1: Vec<f32> = (0..plane).map(|i| (i as f32) * -2.3 + 260.0).collect();
        let c2: Vec<f32> = (0..plane).map(|i| (i as f32) * 0.7).collect();
        let scale = 1.0;

        // Independent reference matching the original uvdoc per-pixel arithmetic.
        let mut expected = vec![0u8; plane * 3];
        for p in 0..plane {
            expected[p * 3] = (c2[p] * scale).clamp(0.0, 255.0) as u8;
            expected[p * 3 + 1] = (c1[p] * scale).clamp(0.0, 255.0) as u8;
            expected[p * 3 + 2] = (c0[p] * scale).clamp(0.0, 255.0) as u8;
        }
        let mut got = vec![0u8; plane * 3];
        scale_clamp_bgr_planes_to_rgb(&c0, &c1, &c2, scale, &mut got);
        assert_eq!(expected, got);
    }
}
