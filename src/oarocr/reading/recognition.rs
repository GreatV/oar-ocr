//! Read, split and route line crops through the shared recognition adapters.
use super::{
    CropStrategy, OAROCR, Reading,
    geometry::{self, Quad, Transform},
    text,
};
use super::{CtcDecoding, RecognitionPreprocessing};
use image::{RgbImage, imageops};
use oar_ocr_core::core::{
    constants::DEFAULT_REC_IMAGE_SHAPE,
    traits::{adapter::ModelAdapter, task::ImageTaskInput},
};
use oar_ocr_core::{
    core::OCRError,
    domain::adapters::TextRecognitionAdapter,
    processors::{BoundingBox, raster},
};
use std::sync::Arc;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};
impl OAROCR {
    pub(super) fn recognize(
        &self,
        crops: &[Arc<RgbImage>],
        model: &TextRecognitionAdapter,
    ) -> Result<Vec<Reading>, OCRError> {
        let batch_size = self
            .region_batch_size
            .unwrap_or_else(|| model.recommended_batch_size())
            .max(1);
        if self.stages.recognition_preprocessing != RecognitionPreprocessing::Standard
            || self.stages.ctc_decoding != CtcDecoding::Standard
        {
            return Ok(model
                .read_lines(
                    crops,
                    self.stages.recognition_preprocessing,
                    self.stages.ctc_decoding,
                    batch_size,
                )?
                .into_iter()
                .map(|r| Reading {
                    text: r.text,
                    confidence: r.confidence,
                    positions: (self.stages.ctc_decoding == CtcDecoding::Standard
                        && self.return_word_box)
                        .then(|| super::WordPositions {
                            columns: r.column_indices,
                            positions: Vec::new(),
                            steps: r.sequence_length,
                            ratio: r.width_fraction,
                            max_ratio: 1.,
                        }),
                    frames: r.frames,
                    available: true,
                    ..Default::default()
                })
                .collect());
        }
        let aspect = |i: usize| crops[i].width() as f32 / crops[i].height().max(1) as f32;
        let mut indices: Vec<_> = (0..crops.len()).collect();
        indices.sort_by(|&a, &b| aspect(a).total_cmp(&aspect(b)));
        let mut readings = vec![Reading::default(); crops.len()];
        for ids in indices.chunks(batch_size) {
            let base = DEFAULT_REC_IMAGE_SHAPE[2] as f32 / DEFAULT_REC_IMAGE_SHAPE[1] as f32;
            let max_ratio = ids.iter().fold(base, |acc, &i| acc.max(aspect(i)));
            let input =
                ImageTaskInput::from_arc_images(ids.iter().map(|&i| crops[i].clone()).collect());
            let rec = model.execute(input, None)?;
            for (j, (&i, text)) in ids.iter().zip(rec.texts).enumerate() {
                readings[i] = Reading {
                    text,
                    confidence: f64::from(rec.scores.get(j).copied().unwrap_or_default()),
                    positions: self.return_word_box.then(|| super::WordPositions {
                        columns: rec.char_col_indices.get(j).cloned().unwrap_or_default(),
                        positions: rec.char_positions.get(j).cloned().unwrap_or_default(),
                        steps: rec.sequence_lengths.get(j).copied().unwrap_or_default(),
                        ratio: aspect(i),
                        max_ratio,
                    }),
                    available: true,
                    ..Default::default()
                };
            }
        }
        Ok(readings)
    }
    pub(super) fn crop_line(&self, img: &RgbImage, q: &Quad) -> Result<RgbImage, OCRError> {
        if self.stages.crop_strategy == CropStrategy::PerspectiveCubic {
            Ok(geometry::crop(img, q))
        } else {
            oar_ocr_core::utils::get_rotate_crop_image(img, &super::as_bbox(q).points)
        }
    }
    pub(super) fn read_boxes(
        &self,
        img: &RgbImage,
        boxes: &[Quad],
    ) -> Result<Vec<Reading>, OCRError> {
        let mut crops = boxes
            .iter()
            .map(|q| self.crop_line(img, q).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let angles = self.orient_crops(&mut crops)?;
        let mut readings = self.read_pool(crops)?;
        for (r, angle) in readings.iter_mut().zip(angles) {
            r.orientation = angle;
        }
        Ok(readings)
    }

    pub(super) fn orient_crops(
        &self,
        crops: &mut [Arc<RgbImage>],
    ) -> Result<Vec<Option<f32>>, OCRError> {
        if self.pipeline.text_line_orientation_adapter.is_none() {
            return Ok(vec![None; crops.len()]);
        }
        let angles = self.line_orientations(crops.to_vec())?;
        for (crop, angle) in crops.iter_mut().zip(&angles) {
            if *angle == Some(180.) {
                *crop = Arc::new(imageops::rotate180(crop.as_ref()));
            }
        }
        Ok(angles)
    }

    pub(super) fn read_pool(&self, mut raw: Vec<Arc<RgbImage>>) -> Result<Vec<Reading>, OCRError> {
        let n = raw.len();
        let mut turned = vec![false; n];
        let tall: Vec<_> = (0..raw.len())
            .filter(|&i| {
                self.stages.recognize_tall_rotations
                    && 2 * raw[i].height() as u64 >= 3 * raw[i].width() as u64
            })
            .collect();
        for &i in &tall {
            raw.push(Arc::new(imageops::rotate270(raw[i].as_ref())));
        }
        for &i in &tall {
            raw.push(Arc::new(imageops::rotate90(raw[i].as_ref())));
        }
        let mut out = self.recognize(&raw, &self.pipeline.text_recognition_adapter)?;
        let alternatives = out.split_off(n);
        for (&i, (ccw, cw)) in tall.iter().zip(
            alternatives
                .iter()
                .zip(alternatives.iter().skip(tall.len())),
        ) {
            for alt in [ccw, cw] {
                if alt.confidence > out[i].confidence {
                    out[i] = alt.clone();
                    out[i].frames = None;
                    out[i].positions = None;
                    turned[i] = true;
                }
            }
        }
        if let Some((rec2, super::SecondaryRouting::MixedThaiLatin)) =
            &self.pipeline.secondary_recognizer
        {
            let ids: Vec<_> = out
                .iter()
                .enumerate()
                .filter(|(i, r)| !turned[*i] && text::mixed(&r.text) > 0)
                .map(|(i, _)| i)
                .collect();
            let crops: Vec<_> = ids.iter().map(|&i| raw[i].clone()).collect();
            for (i, new) in ids.into_iter().zip(self.recognize(&crops, rec2)?) {
                if text::mixed(&new.text) < text::mixed(&out[i].text) {
                    out[i] = new;
                }
            }
        }
        Ok(out)
    }
}

pub(super) fn split(img: &RgbImage, q: &Quad) -> Option<Vec<Quad>> {
    let (w, h) = img.dimensions();
    if h < 4 || w < 4 * h {
        return None;
    }
    let flat = raster::flatten(&raster::grayscale(img));
    let mut runs = Vec::new();
    let mut start = None;
    let mut blank = 0;
    for x in 0..w {
        if (0..h).any(|y| flat[(x, y)][0] < 128) {
            if let Some(s) = start {
                if blank as f64 >= 1.2 * h as f64 {
                    runs.push((s, x - blank));
                    start = Some(x);
                }
            } else {
                start = Some(x);
            }
            blank = 0;
        } else if start.is_some() {
            blank += 1;
        }
    }
    if let Some(s) = start {
        runs.push((s, w - blank));
    }
    if runs.is_empty()
        || (runs.len() == 1
            && (runs[0].0 as f64) < 1.2 * h as f64
            && ((w - runs[0].1) as f64) < 1.2 * h as f64)
    {
        return None;
    }
    Some(
        runs.into_iter()
            .map(|(a, b)| {
                geometry::horizontal_slice(
                    q,
                    ((a as f64 - 0.15 * h as f64) / w as f64).max(0.) as f32,
                    ((b as f64 + 0.15 * h as f64) / w as f64).min(1.) as f32,
                )
            })
            .collect(),
    )
}

pub(super) fn words(read: &Reading, q: &Quad, back: &Transform, thai: bool) -> Vec<BoundingBox> {
    let quad = q.map(|p| p.cast::<f64>());
    let whole = || vec![super::box_from_pixels(geometry::back_quad(quad, back))];
    let Some(frames) = &read.frames else {
        return whole();
    };
    let mut tokens = Vec::new();
    let mut gaps: Vec<f64> = Vec::new();
    let mut token = Vec::new();
    for &f in frames {
        if f.ch.is_whitespace() {
            let center = (f.left + f.right) / 2.;
            if !token.is_empty() {
                tokens.push(std::mem::take(&mut token));
                gaps.push(center);
            } else if let Some(g) = gaps.last_mut() {
                *g = (*g + center) / 2.;
            }
        } else {
            token.push(f);
        }
    }
    if !token.is_empty() {
        tokens.push(token);
    }
    if tokens.is_empty() {
        return whole();
    }
    let mut steps = Vec::new();
    for tok in &tokens {
        let starts: Vec<_> = tok
            .iter()
            .filter(|f| {
                unicode_normalization::char::canonical_combining_class(f.ch) == 0
                    && f.ch.general_category() != GeneralCategory::NonspacingMark
            })
            .map(|f| f.left)
            .collect();
        steps.extend(
            starts
                .windows(2)
                .filter_map(|p| (p[1] > p[0]).then_some(p[1] - p[0])),
        );
    }
    let adv = if steps.is_empty() {
        (tokens.last().unwrap().last().unwrap().right - tokens[0][0].left)
            / tokens.iter().map(Vec::len).sum::<usize>() as f64
    } else {
        crate::indra::median_f64(&mut steps)
    };
    let mut words = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let raw = t.iter().map(|f| f.ch).collect::<String>();
        let text = if thai { text::canonical(&raw) } else { raw };
        if text.is_empty() {
            continue;
        }
        let (first, last) = (t[0].left, t.last().unwrap().right);
        let left = if i == 0 {
            (first - adv).max(0.)
        } else {
            first.min((gaps[i - 1] + 0.2 * adv).max(first - 0.5 * adv))
        };
        let right = if i + 1 < tokens.len() {
            last.max((gaps[i] - 0.2 * adv).min(last + 0.5 * adv))
        } else {
            (last + 0.5 * adv).min(1.)
        };
        let bbox = geometry::back_quad(geometry::horizontal_slice(&quad, left, right), back);
        words.push(super::box_from_pixels(bbox));
    }
    if words.is_empty() { whole() } else { words }
}
