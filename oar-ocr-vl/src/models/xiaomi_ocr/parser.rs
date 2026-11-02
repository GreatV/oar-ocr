//! Complete-page parser adapter for Xiaomi-OCR-0.

use super::model::{DEFAULT_MAX_NEW_TOKENS, XiaomiOcr, finalize_markdown};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// Xiaomi-OCR-0 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct XiaomiOcrParseOptions {
    pub max_new_tokens: usize,
}

impl Default for XiaomiOcrParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

impl PageParser for XiaomiOcr {
    type Options = XiaomiOcrParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let image = fit_page(image)?;
        let tokens = self
            .generate_tokens(std::slice::from_ref(&image), options.max_new_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("Xiaomi-OCR-0 returned no page result"))??;
        // `raw_output` keeps the official tail-cleanup view of the model
        // output; `markdown` additionally converts OTSL table blocks to HTML
        // (the official whole-page post-processing).
        let raw_output = self.decode_tokens(&tokens)?;
        let markdown = finalize_markdown(&raw_output);
        Ok(PageDocument {
            markdown: Some(markdown),
            raw_output: Some(raw_output),
            ..PageDocument::default()
        })
    }
}

fn fit_page(image: &RgbImage) -> Result<RgbImage, Error> {
    const MAX_EDGE: u32 = 2048;
    let (width, height) = image.dimensions();
    if width.max(height) <= MAX_EDGE {
        return Ok(image.clone());
    }
    let aspect = width as f64 / height as f64;
    // Pillow thumbnail chooses floor/ceil by aspect error, with floor winning
    // ties. Use full Lanczos rather than its optional reducing_gap shortcut.
    let (w, h) = if width >= height {
        let target = MAX_EDGE as f64 / aspect;
        let error = |h: f64| {
            if h == 0. {
                0.
            } else {
                (aspect - MAX_EDGE as f64 / h).abs()
            }
        };
        let h = if error(target.floor()) <= error(target.ceil()) {
            target.floor()
        } else {
            target.ceil()
        };
        (MAX_EDGE, h.max(1.) as u32)
    } else {
        let target = MAX_EDGE as f64 * aspect;
        let w = if (aspect - target.floor() / MAX_EDGE as f64).abs()
            <= (aspect - target.ceil() / MAX_EDGE as f64).abs()
        {
            target.floor()
        } else {
            target.ceil()
        };
        (w.max(1.) as u32, MAX_EDGE)
    };
    crate::runtime::image::pil_lanczos_resize(image, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_official_generation_limit() {
        let options = XiaomiOcrParseOptions::default();
        assert_eq!(options.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(DEFAULT_MAX_NEW_TOKENS, 4_096);
        assert_eq!(
            fit_page(&RgbImage::new(3000, 1500)).unwrap().dimensions(),
            (2048, 1024)
        );
    }
}
