//! Read a page's lines with the script recognizers a router model picks.
//!
//! Adapted from LiteParse's ocria-nano runtime (Apache-2.0). See the NOTICE beside this file.
use super::{OAROCR, PageRead, Reading, geometry::Quad, pool::quad};
use crate::oarocr::processors::scripts::{
    GROUPS, clean_latin, is_short_token, latin_letters, majority, script_chars, visual_to_logical,
};
use image::{RgbImage, imageops};
use oar_ocr_core::{
    core::OCRError, domain::adapters::TextRecognitionAdapter,
    models::classification::LineRouterModel, processors::BoundingBox,
};
use std::sync::Arc;

/// Router and recognizers for the script groups beyond the pipeline's primary Latin recognizer.
#[derive(Debug)]
pub(in crate::oarocr) struct ScriptModels {
    pub(in crate::oarocr) router: LineRouterModel,
    /// CJK, right-to-left and other-script recognizers, each with whether it writes right-to-left text in visual order.
    pub(in crate::oarocr) recognizers: [(TextRecognitionAdapter, bool); GROUPS - 1],
}

/// A routed reading this confident is a real reading, not the 0.5-0.7 junk of the wrong recognizer.
const KEEP_CONF: f32 = 0.75;
/// A faint candidate is kept only at this confidence: rule fragments and specks read as "|", "l" or a letter pair at 0.5-0.7.
const FAINT_CONF: f32 = 0.9;

fn conf(read: &Reading) -> f32 {
    read.confidence as f32
}

fn mean(reads: impl Iterator<Item = f32>) -> f64 {
    let (sum, n) = reads.fold((0f64, 0usize), |(s, n), x| (s + f64::from(x), n + 1));
    sum / n as f64
}

impl OAROCR {
    fn read_group(
        &self,
        scripts: &ScriptModels,
        group: usize,
        crops: &[Arc<RgbImage>],
    ) -> Result<Vec<Reading>, OCRError> {
        let Some((model, visual_rtl)) = group.checked_sub(1).map(|g| &scripts.recognizers[g])
        else {
            return self.recognize(crops, &self.pipeline.text_recognition_adapter);
        };
        let mut reads = self.recognize(crops, model)?;
        if *visual_rtl {
            for read in &mut reads {
                read.text = visual_to_logical(&read.text);
            }
        }
        Ok(reads)
    }

    /// Readings per box, whether the page reads upside down, and the page's script group.
    fn route_lines(
        &self,
        scripts: &ScriptModels,
        img: &RgbImage,
        boxes: &[Quad],
    ) -> Result<(Vec<Reading>, bool, usize), OCRError> {
        let mut crops = boxes
            .iter()
            .map(|q| self.crop_line(img, q).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let routing = scripts.router.route(&crops)?;
        let (mut groups, share) = (routing.groups, routing.shares);
        if routing.upside_down {
            for crop in &mut crops {
                *crop = Arc::new(imageops::rotate180(crop.as_ref()));
            }
        }
        let n = crops.len();
        let subset = |ids: &[usize]| ids.iter().map(|&i| crops[i].clone()).collect::<Vec<_>>();
        let routed = |groups: &[usize], group: usize| {
            (0..n).filter(|&i| groups[i] == group).collect::<Vec<_>>()
        };
        let mut out = vec![Reading::default(); n];
        for group in 0..GROUPS {
            let ids = routed(&groups, group);
            if !ids.is_empty() {
                for (&i, read) in ids
                    .iter()
                    .zip(self.read_group(scripts, group, &subset(&ids))?)
                {
                    out[i] = read;
                }
            }
        }
        let mut dominant = majority(&groups);
        if dominant == 0 {
            // A whole page of another script sent to the Latin recognizer reads as confident-looking junk at 0.5-0.7; the right recognizer then scores 0.83-0.90.
            const RESCUE_CONF: f64 = 0.75;
            const RESCUE_MARGIN: f64 = 0.15;
            // Only scripts the router saw a fair share of are tried. Low-confidence Latin regions such as chart labels carry almost none, and trying every script there switched them to other-script junk.
            const RESCUE_SHARE: f64 = 0.2;
            let latin = routed(&groups, 0);
            if !latin.is_empty() && share[1..].iter().any(|&s| s >= RESCUE_SHARE) {
                let latin_conf = mean(latin.iter().map(|&i| conf(&out[i])));
                if latin_conf < RESCUE_CONF {
                    let sample: Vec<usize> = latin.iter().copied().take(40).collect();
                    let (mut best, mut best_conf) = (None, latin_conf + RESCUE_MARGIN);
                    for group in (1..GROUPS).filter(|&g| share[g] >= RESCUE_SHARE) {
                        let reads = self.read_group(scripts, group, &subset(&sample))?;
                        let c = mean(reads.iter().map(conf));
                        if c >= best_conf {
                            (best, best_conf) = (Some(group), c);
                        }
                    }
                    if let Some(group) = best {
                        let reads = self.read_group(scripts, group, &subset(&latin))?;
                        for (&i, read) in latin.iter().zip(reads) {
                            // A bilingual page keeps its confident Latin lines.
                            if !(conf(&out[i]) >= KEEP_CONF && conf(&out[i]) >= conf(&read)) {
                                out[i] = read;
                                groups[i] = group;
                            }
                        }
                        dominant = group;
                    }
                }
            }
            // When the page stays Latin, every Latin-routed line on a page carrying another script's share is re-read, not only weak ones: Armenian reads as plausible Latin at 0.75+. A switch needs the margin and the script's own characters, so English lines on a bilingual page stay Latin.
            if dominant == 0 {
                const LINE_MARGIN: f32 = 0.1;
                const LINE_SHARE: f64 = 0.15;
                const LINE_CONF: f32 = 0.8;
                for group in (1..GROUPS).filter(|&g| share[g] >= LINE_SHARE) {
                    let ids = routed(&groups, 0);
                    if ids.is_empty() {
                        break;
                    }
                    let reads = self.read_group(scripts, group, &subset(&ids))?;
                    for (&i, read) in ids.iter().zip(reads) {
                        // Specks and smudges on old scans read as 2-character CJK at 0.1-0.77, a margin above the Latin junk they replace.
                        if conf(&read) >= conf(&out[i]) + LINE_MARGIN
                            && conf(&read) >= LINE_CONF
                            && script_chars(group, &read.text) >= 3
                            && script_chars(group, &read.text) > latin_letters(&read.text)
                        {
                            out[i] = read;
                            groups[i] = group;
                        }
                    }
                }
            }
        }
        // Lines routed away from the page's script are read with the page's recognizer too, and the off-script reading must win clearly.
        let odd: Vec<usize> = (0..n).filter(|&i| groups[i] != dominant).collect();
        if !odd.is_empty() {
            let reads = self.read_group(scripts, dominant, &subset(&odd))?;
            for (&i, read) in odd.iter().zip(reads) {
                let own = script_chars(groups[i], &out[i].text);
                // A confident reading that is mostly its routed script stands: the page's recognizer cannot write that script and still scores high.
                let routed_ok =
                    conf(&out[i]) >= KEEP_CONF && own >= 2 && own > latin_letters(&out[i].text);
                if !routed_ok && (conf(&out[i]) < conf(&read) + 0.1 || own < 3) {
                    out[i] = read;
                }
            }
        }
        // Every line is cleaned: lines the page's Latin recognizer re-read keep their routed group, and the other recognizers write Latin lines with lookalikes.
        for read in &mut out {
            read.text = clean_latin(&read.text);
        }
        Ok((out, routing.upside_down, dominant))
    }

    /// Read detected lines and faint candidates, returning the page and whether it reads upside down.
    pub(super) fn read_script_page(
        &self,
        scripts: &ScriptModels,
        img: &RgbImage,
        mut polygons: Vec<BoundingBox>,
        faint: Vec<BoundingBox>,
    ) -> Result<(PageRead, bool), OCRError> {
        let mut boxes: Vec<Quad> = polygons.iter().map(quad).collect();
        let (mut reads, upside_down, dominant) = if boxes.is_empty() {
            (Vec::new(), false, 0)
        } else {
            self.route_lines(scripts, img, &boxes)?
        };
        if !faint.is_empty() && !upside_down {
            // A lone glyph is too short for the router, so the page's recognizer reads it.
            let crops = faint
                .iter()
                .map(|b| self.crop_line(img, &quad(b)).map(Arc::new))
                .collect::<Result<Vec<_>, _>>()?;
            let tokens = self.read_group(scripts, dominant, &crops)?;
            for (bbox, mut read) in faint.into_iter().zip(tokens) {
                read.text = clean_latin(&read.text);
                if conf(&read) >= FAINT_CONF && is_short_token(&read.text) {
                    read.faint = true;
                    boxes.push(quad(&bbox));
                    polygons.push(bbox);
                    reads.push(read);
                }
            }
        }
        let rows = (0..reads.len()).map(|i| vec![i]).collect();
        Ok((
            PageRead {
                polygons,
                boxes,
                reads,
                rows,
            },
            upside_down,
        ))
    }
}
