//! Dynamic-width recognition batches with optional background flattening and CTC run endpoints.
use super::CRNNModel;
use crate::{
    core::{OCRError, inference::TensorInput},
    processors::{
        CtcReading,
        raster::{self, Interp},
    },
};
use image::{GrayImage, RgbImage};
use ndarray::Array4;

/// Pixel normalization and width policy for line recognition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RecognitionPreprocessing {
    /// Use the recognizer's configured RGB resize.
    #[default]
    Standard,
    /// Remove local column backgrounds after resizing to a 48-pixel height.
    Flatten,
}
/// Output reduction strategy for a CTC recognizer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum CtcDecoding {
    /// Apply the default CTC decoder.
    #[default]
    Standard,
    /// Keep the first score and both horizontal endpoints of each nonblank run.
    FrameSpans,
}
fn prepare(img: &RgbImage) -> GrayImage {
    let w = (48 * img.width() as u64).div_ceil(img.height() as u64) as u32;
    let resized = raster::resize_rgb(img, w.max(1), 48);
    raster::flatten(&raster::grayscale(&resized))
}

impl CRNNModel {
    pub(crate) fn read_lines(
        &self,
        crops: &[impl std::borrow::Borrow<RgbImage>],
        policy: RecognitionPreprocessing,
        decoding: CtcDecoding,
        batch_size: usize,
        return_word_boxes: bool,
    ) -> Result<Vec<CtcReading>, OCRError> {
        let prepared: Vec<_> = if policy == RecognitionPreprocessing::Flatten {
            crops.iter().map(|img| prepare(img.borrow())).collect()
        } else {
            Vec::new()
        };
        let width = |i: usize| {
            if prepared.is_empty() {
                (self.resizer.rec_image_shape[1] as f32
                    * (crops[i].borrow().width() as f32 / crops[i].borrow().height().max(1) as f32))
                    .ceil() as u32
            } else {
                prepared[i].width()
            }
        };
        let mut order: Vec<_> = (0..crops.len()).collect();
        order.sort_by_key(|&i| width(i));
        let mut result = vec![CtcReading::default(); crops.len()];
        for ids in order.chunks(batch_size.max(1)) {
            let flattened = policy == RecognitionPreprocessing::Flatten;
            let w = width(*ids.last().unwrap()).clamp(16, 3200);
            let input = if flattened {
                let mut input = Array4::zeros((ids.len(), 3, 48, w as usize));
                for (n, &i) in ids.iter().enumerate() {
                    let rw = width(i).min(w);
                    let resized = raster::resize(&prepared[i], rw, 48, Interp::Linear);
                    for (x, y, p) in resized.enumerate_pixels() {
                        let val = (p[0] as f32 / 255. - 0.5) / 0.5;
                        for c in 0..3 {
                            input[[n, c, y as usize, x as usize]] = val;
                        }
                    }
                }
                input
            } else {
                self.preprocess_refs(&ids.iter().map(|&i| crops[i].borrow()).collect::<Vec<_>>())?
            };
            let w = input.dim().3 as u32;
            let model = &self.inference;
            model.infer_first_output_f32(
                &[(model.input_name(), TensorInput::Array4(&input))],
                |shape, probs| {
                    let &[batch, steps, classes] = shape else {
                        return Err(OCRError::invalid_input("CTC recognizer expects [N,T,V]"));
                    };
                    if batch != ids.len()
                        || steps == 0
                        || classes != self.decoder.get_character_list().len()
                    {
                        return Err(OCRError::invalid_input(format!(
                            "unexpected CTC recognizer output {shape:?} for {} dictionary classes",
                            self.decoder.get_character_list().len()
                        )));
                    }
                    if decoding == CtcDecoding::Standard {
                        let view = ndarray::ArrayView3::from_shape((batch, steps, classes), probs)
                            .map_err(|e| OCRError::invalid_input(e.to_string()))?;
                        let standard = self.postprocess(&view, return_word_boxes);
                        for (j, ((&i, text), score)) in ids
                            .iter()
                            .zip(standard.texts)
                            .zip(standard.scores)
                            .enumerate()
                        {
                            result[i] = CtcReading {
                                text,
                                confidence: f64::from(score),
                                column_indices: standard
                                    .char_col_indices
                                    .get(j)
                                    .cloned()
                                    .unwrap_or_default(),
                                sequence_length: steps,
                                width_fraction: width(i).min(w) as f32 / w as f32,
                                ..Default::default()
                            };
                        }
                        return Ok(());
                    }
                    for (&i, row) in ids.iter().zip(probs.chunks_exact(steps * classes)) {
                        result[i] = self
                            .decoder
                            .decode_frame_spans(row, width(i).min(w), w, steps);
                    }
                    Ok(())
                },
            )?;
        }
        Ok(result)
    }
}
