//! Region-recognition adapter for GLM-OCR.

use super::GlmOcr;
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

fn prompt(task: RecognitionTask) -> &'static str {
    match task {
        RecognitionTask::Ocr | RecognitionTask::Chart => "Text Recognition:",
        RecognitionTask::Table => "Table Recognition:",
        RecognitionTask::Formula => "Formula Recognition:",
    }
}

fn clean(output: String) -> String {
    let text = output.trim();
    if text.starts_with("<table") && text.ends_with("</table>") {
        return text.into();
    }
    let chars: Vec<_> = text.chars().collect();
    if chars.len() < 2048 {
        return text.into();
    }
    // GLM SDK clean_repeated_content (Apache-2.0) keeps the prefix and one
    // repeat unit, including multiline units.
    for start in 0..chars.len() {
        for period in 10..=(chars.len() - start) / 10 {
            let unit = &chars[start..start + period];
            if (1..10).all(|i| chars[start + i * period..start + (i + 1) * period] == *unit) {
                return chars[..start + period].iter().collect();
            }
        }
    }
    let lines: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let mut counts = std::collections::HashMap::<&str, usize>::new();
    for line in &lines {
        *counts.entry(line.trim()).or_default() += 1;
    }
    if let Some(common) = lines
        .iter()
        .map(|line| line.trim())
        .max_by_key(|line| counts[line])
        && counts[common] >= 10
        && counts[common] * 5 >= lines.len() * 4
    {
        for (index, _) in lines.iter().enumerate() {
            if lines[index].trim() == common
                && lines[index..]
                    .iter()
                    .take(3)
                    .filter(|line| line.trim() == common)
                    .count()
                    == 3
            {
                let mut seen = 0;
                let mut kept = Vec::new();
                for line in text.lines() {
                    kept.push(line);
                    if !line.trim().is_empty() {
                        seen += 1;
                    }
                    if seen == index + 1 {
                        return kept.join("\n");
                    }
                }
            }
        }
    }
    text.into()
}

fn prepare_region(image: &RgbImage) -> Result<RgbImage, Error> {
    let (h, w) = (image.height() as f64, image.width() as f64);
    if h == 0. || w == 0. {
        return Err(Error::invalid_input("GLM-OCR cannot process an empty crop"));
    }
    // SDK utils/image_utils.py: t=t_factor=2, factor=28; no preliminary
    // small-side scaling. The model processor runs after this bicubic resize.
    let (mut rh, mut rw) = (
        (h / 28.).round_ties_even() * 28.,
        (w / 28.).round_ties_even() * 28.,
    );
    if 2. * rh * rw > 71_372_800. {
        let beta = (2. * h * w / 71_372_800.).sqrt();
        rh = (h / beta / 28.).floor() * 28.;
        rw = (w / beta / 28.).floor() * 28.;
    } else if 2. * rh * rw < 12_544. {
        let beta = (12_544. / (2. * h * w)).sqrt();
        rh = (h * beta / 28.).ceil() * 28.;
        rw = (w * beta / 28.).ceil() * 28.;
    }
    crate::runtime::image::pil_bicubic_resize(image, rw.max(28.) as u32, rh.max(28.) as u32)
}

impl RecognitionBackend for GlmOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let image = prepare_region(&image)?;
        self.generate(&[image], &[prompt(task)], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("GLM-OCR: no result returned"))?
            .map(clean)
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let prompts: Vec<_> = tasks.iter().copied().map(prompt).collect();
        let images = images
            .iter()
            .map(prepare_region)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .map(|result| result.map(clean))
            .collect())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            truncate_repetitive_output: false,
            ..BackendCapabilities::default()
        }
    }

    fn postprocess_region_output(
        &self,
        _label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        Some(if task == RecognitionTask::Formula {
            format!("$$\n{}\n$$", crate::render::text::strip_math_wrappers(raw))
        } else {
            raw.trim().to_string()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_crops_and_cleanup_preserve_thin_regions_and_legitimate_tables() {
        assert_eq!(
            prepare_region(&RgbImage::new(400, 20))
                .unwrap()
                .dimensions(),
            (392, 28)
        );
        assert_eq!(
            prepare_region(&RgbImage::new(175, 10))
                .unwrap()
                .dimensions(),
            (336, 28)
        );
        let table = format!("<table>{}</table>", "<tr><td>x</td></tr>".repeat(120));
        assert_eq!(clean(table.clone()), table);
        assert_eq!(
            clean("a\na\na\na\na\na\na\na\na\na".into()),
            "a\na\na\na\na\na\na\na\na\na"
        );
        assert_eq!(
            clean(format!("prefix {}", "abcdefghij".repeat(210))),
            "prefix abcdefghij"
        );
    }
}
