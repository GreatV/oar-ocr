//! Complete-page parser adapter for OCRia-1.

use super::{DEFAULT_MAX_NEW_TOKENS, DEFAULT_MAX_PIXELS, Ocria, page_markdown};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;
use std::borrow::Cow;

/// OCRia-1 complete-page parsing options.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OcriaParseOptions {
    /// Maximum number of generated tokens per page; defaults to 24,576.
    pub max_new_tokens: usize,
    /// Pixel budget of the page image; a larger page is downscaled to it with a Lanczos filter. Defaults to 4.1 megapixels, the budget of the published benchmark numbers. `None` sends the page as it is, which costs more image tokens.
    pub max_pixels: Option<u64>,
}

impl Default for OcriaParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            max_pixels: Some(DEFAULT_MAX_PIXELS),
        }
    }
}

impl OcriaParseOptions {
    /// Set the per-page generation limit.
    pub fn with_max_new_tokens(mut self, value: usize) -> Self {
        self.max_new_tokens = value;
        self
    }

    /// Set the page image budget in pixels, or `None` to send pages as they are.
    pub fn with_max_pixels(mut self, value: Option<u64>) -> Self {
        self.max_pixels = value;
        self
    }
}

impl PageParser for Ocria {
    type Options = OcriaParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let image = fit_page(image, options.max_pixels)?;
        let tokens = self
            .generate_tokens(std::slice::from_ref(&*image), options.max_new_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("OCRia-1 returned no page result"))??;
        let raw_output = self.decode_tokens(&tokens)?;
        Ok(PageDocument {
            markdown: Some(page_markdown(&raw_output)),
            raw_output: Some(raw_output),
            ..PageDocument::default()
        })
    }
}

/// Size of a page inside a pixel budget: both sides scale by the square root of budget over area and round half to even, as the model card's downscale does. A page within the budget keeps its size.
fn fitted_size(width: u32, height: u32, max_pixels: u64) -> (u32, u32) {
    let area = u64::from(width) * u64::from(height);
    if area <= max_pixels {
        return (width, height);
    }
    let scale = (max_pixels as f64 / area as f64).powf(0.5);
    let side = |v: u32| ((f64::from(v) * scale).round_ties_even() as u32).max(1);
    (side(width), side(height))
}

fn fit_page(image: &RgbImage, max_pixels: Option<u64>) -> Result<Cow<'_, RgbImage>, Error> {
    let (width, height) = image.dimensions();
    let (w, h) = match max_pixels {
        Some(max_pixels) => fitted_size(width, height, max_pixels),
        None => (width, height),
    };
    if (w, h) == (width, height) {
        return Ok(Cow::Borrowed(image));
    }
    crate::runtime::image::pil_lanczos_resize(image, w, h).map(Cow::Owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_are_fitted_to_the_benchmark_budget() {
        let options = OcriaParseOptions::default();
        assert_eq!(options.max_new_tokens, 24_576);
        assert_eq!(options.max_pixels, Some(4_100_000));
        // A letter page at 300 dpi comes down to about 210 dpi.
        assert_eq!(fitted_size(2550, 3300, 4_100_000), (1780, 2303));
        // A page inside the budget is never upscaled.
        assert_eq!(fitted_size(1240, 1754, 4_100_000), (1240, 1754));
    }
}
