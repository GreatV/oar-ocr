//! PDF processing utilities for examples using pure Rust `hayro` library.

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;

use hayro::Pdf;

/// Error type for PDF processing.
#[derive(Debug)]
pub enum PdfError {
    Io(std::io::Error),
    Hayro(String),
    PageNotFound(usize),
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Hayro(e) => write!(f, "PDF error: {}", e),
            Self::PageNotFound(n) => write!(f, "Page {} not found", n),
        }
    }
}

impl std::error::Error for PdfError {}

impl From<std::io::Error> for PdfError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A rendered PDF page with its image data.
#[derive(Debug, Clone)]
pub struct RenderedPage {
    pub page_number: usize,
    pub width: u32,
    pub height: u32,
    pub image: image::RgbImage,
}

/// PDF document handler that provides in-memory page rendering.
///
/// This uses the pure Rust `hayro` library for PDF rendering.
pub struct PdfDocument {
    pdf: Pdf,
    page_count: usize,
}

impl PdfDocument {
    /// Opens a PDF file from the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PdfError> {
        let path = path.as_ref();
        if !is_pdf_file(path) {
            return Err(PdfError::Hayro("Not a PDF file".to_string()));
        }

        let data = std::fs::read(path)?;
        let pdf_data = Arc::new(data);
        let pdf = Pdf::new(pdf_data)
            .map_err(|e| PdfError::Hayro(format!("Failed to parse PDF: {:?}", e)))?;
        let page_count = pdf.pages().len();

        Ok(Self { pdf, page_count })
    }

    /// Returns the number of pages in the PDF.
    pub fn page_count(&self) -> usize {
        self.page_count
    }

    /// Renders a PDF page to an in-memory RGB image.
    ///
    /// This uses pure Rust rendering via the `hayro` library.
    pub fn render_page(&self, page_num: usize) -> Result<RenderedPage, PdfError> {
        use hayro::RenderSettings;

        if page_num < 1 || page_num > self.page_count {
            return Err(PdfError::PageNotFound(page_num));
        }

        let page_index = page_num - 1;
        let page = self
            .pdf
            .pages()
            .get(page_index)
            .ok_or_else(|| PdfError::Hayro(format!("Failed to get page {}", page_num)))?;

        // Get page dimensions from media box
        let media_box = page.media_box();
        let width = (media_box.x1 - media_box.x0) as f32;
        let height = (media_box.y1 - media_box.y0) as f32;

        if width <= 0.0 || height <= 0.0 {
            return Err(PdfError::Hayro(format!(
                "Invalid page size: {}x{}",
                width, height
            )));
        }

        // Use a reasonable scale factor for better quality
        let scale = 2.0;

        // Create render settings
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            ..Default::default()
        };

        // Render the page using hayro's render function
        let interpreter_settings = hayro::InterpreterSettings::default();
        let pixmap = hayro::render(page, &interpreter_settings, &settings);

        // Convert pixmap to RGB image
        let rgba_data = pixmap.data_as_u8_slice();
        let mut rgb_data =
            Vec::with_capacity(pixmap.width() as usize * pixmap.height() as usize * 3);

        // Convert RGBA to RGB
        for chunk in rgba_data.chunks(4) {
            rgb_data.push(chunk[0]); // R
            rgb_data.push(chunk[1]); // G
            rgb_data.push(chunk[2]); // B
            // Skip A (alpha)
        }

        let rgb_image = image::RgbImage::from_raw(
            u32::from(pixmap.width()),
            u32::from(pixmap.height()),
            rgb_data,
        )
        .ok_or_else(|| PdfError::Hayro("Failed to convert pixmap to image".to_string()))?;

        Ok(RenderedPage {
            page_number: page_num,
            width: rgb_image.width(),
            height: rgb_image.height(),
            image: rgb_image,
        })
    }

    /// Renders a PDF page with specified maximum dimensions.
    ///
    /// The page is scaled to fit within the specified bounds while preserving aspect ratio.
    pub fn render_page_sized(
        &self,
        page_num: usize,
        max_width: u32,
        max_height: u32,
    ) -> Result<RenderedPage, PdfError> {
        use hayro::RenderSettings;

        if page_num < 1 || page_num > self.page_count {
            return Err(PdfError::PageNotFound(page_num));
        }

        let page_index = page_num - 1;
        let page = self
            .pdf
            .pages()
            .get(page_index)
            .ok_or_else(|| PdfError::Hayro(format!("Failed to get page {}", page_num)))?;

        // Get page dimensions from media box
        let media_box = page.media_box();
        let width = (media_box.x1 - media_box.x0) as f32;
        let height = (media_box.y1 - media_box.y0) as f32;

        if width <= 0.0 || height <= 0.0 {
            return Err(PdfError::Hayro(format!(
                "Invalid page size: {}x{}",
                width, height
            )));
        }

        // Calculate scale to fit within bounds
        let scale_x = max_width as f32 / width;
        let scale_y = max_height as f32 / height;
        let scale = scale_x.min(scale_y).min(3.0); // Cap at 3x for quality

        // Create render settings
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            ..Default::default()
        };

        // Render the page using hayro's render function
        let interpreter_settings = hayro::InterpreterSettings::default();
        let pixmap = hayro::render(page, &interpreter_settings, &settings);

        // Convert pixmap to RGB image
        let rgba_data = pixmap.data_as_u8_slice();
        let mut rgb_data =
            Vec::with_capacity(pixmap.width() as usize * pixmap.height() as usize * 3);

        // Convert RGBA to RGB
        for chunk in rgba_data.chunks(4) {
            rgb_data.push(chunk[0]); // R
            rgb_data.push(chunk[1]); // G
            rgb_data.push(chunk[2]); // B
            // Skip A (alpha)
        }

        let rgb_image = image::RgbImage::from_raw(
            u32::from(pixmap.width()),
            u32::from(pixmap.height()),
            rgb_data,
        )
        .ok_or_else(|| PdfError::Hayro("Failed to convert pixmap to image".to_string()))?;

        Ok(RenderedPage {
            page_number: page_num,
            width: rgb_image.width(),
            height: rgb_image.height(),
            image: rgb_image,
        })
    }

    /// Renders all pages in the PDF.
    pub fn render_all(&self) -> Result<Vec<RenderedPage>, PdfError> {
        let mut pages = Vec::with_capacity(self.page_count);
        for page_num in 1..=self.page_count {
            pages.push(self.render_page(page_num)?);
        }
        Ok(pages)
    }

    /// Renders all pages with specified maximum dimensions.
    pub fn render_all_sized(
        &self,
        max_width: u32,
        max_height: u32,
    ) -> Result<Vec<RenderedPage>, PdfError> {
        let mut pages = Vec::with_capacity(self.page_count);
        for page_num in 1..=self.page_count {
            pages.push(self.render_page_sized(page_num, max_width, max_height)?);
        }
        Ok(pages)
    }
}

/// Checks if a file is a PDF based on its extension.
pub fn is_pdf_file<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pdf_file() {
        assert!(is_pdf_file("test.pdf"));
        assert!(is_pdf_file("test.PDF"));
        assert!(!is_pdf_file("test.jpg"));
        assert!(!is_pdf_file("test.png"));
    }
}
