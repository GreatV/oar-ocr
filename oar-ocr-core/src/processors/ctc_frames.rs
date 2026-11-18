//! CTC run endpoints for recognizers that project words onto a source quadrilateral.
use super::{CTCLabelDecode, simd::argmax};
/// One collapsed nonblank CTC run with normalized horizontal extent.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct CtcFrame {
    pub ch: char,
    pub left: f64,
    pub right: f64,
    pub p: f32,
}
/// Recognition result with optional first-frame scores and positions.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CtcReading {
    pub text: String,
    pub confidence: f64,
    pub frames: Option<Vec<CtcFrame>>,
    /// Timestep indices returned by the standard decoder when word boxes are enabled.
    pub column_indices: Vec<usize>,
    /// Number of timesteps including right-side padding.
    pub sequence_length: usize,
    /// Fraction of the padded input occupied by this crop.
    pub width_fraction: f32,
}

impl CTCLabelDecode {
    pub(crate) fn decode_frame_spans(
        &self,
        probs: &[f32],
        width: u32,
        batch_w: u32,
        steps: usize,
    ) -> CtcReading {
        let chars = self.get_character_list();
        let mut frames: Vec<CtcFrame> = Vec::new();
        let mut prev = None;
        for (t, row) in probs.chunks_exact(chars.len()).enumerate() {
            let (id, p) = argmax(row).expect("non-empty verified vocabulary");
            // NumPy selects the first class on a tie. A softmax score above one half is unique.
            let id = if p <= 0.5 {
                row[..id].iter().position(|&x| x == p).unwrap_or(id)
            } else {
                id
            };
            let position =
                |t: usize| (t as f64 * (batch_w as f64 / steps as f64) / width as f64).min(1.);
            if id != 0 {
                if prev == Some(id) {
                    frames.last_mut().unwrap().right = position(t + 1);
                } else {
                    frames.push(CtcFrame {
                        ch: chars[id],
                        left: position(t),
                        right: position(t + 1),
                        p,
                    });
                }
            }
            prev = Some(id);
        }
        let confidence = if frames.is_empty() {
            0.
        } else {
            frames.iter().map(|f| f.p as f64).sum::<f64>() / frames.len() as f64
        };
        CtcReading {
            text: frames.iter().map(|f| f.ch).collect(),
            confidence,
            frames: Some(frames),
            ..Default::default()
        }
    }
}
