use anyhow::{Context, Result, anyhow, ensure};
use hayro::{RenderCache, RenderSettings, hayro_syntax::Pdf};
use image::RgbImage;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
};

#[derive(Clone, Debug)]
pub(crate) struct PageRanges(Vec<(usize, usize)>);

impl FromStr for PageRanges {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let number = |value: &str| {
            value
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0)
                .ok_or_else(|| "page numbers must be positive integers".to_string())
        };
        value
            .split(',')
            .map(|part| {
                let (first, last) = match part.split_once('-') {
                    Some((first, last)) => (number(first)?, number(last)?),
                    None => {
                        let page = number(part)?;
                        (page, page)
                    }
                };
                if first > last {
                    return Err("page ranges must be ascending, e.g. 1-3,5".into());
                }
                Ok((first, last))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Self)
    }
}

impl PageRanges {
    fn select(&self, count: usize) -> Result<Vec<usize>> {
        ensure!(
            self.0.iter().all(|(_, last)| *last <= count),
            "requested page exceeds the PDF's {count} pages"
        );
        Ok((1..=count)
            .filter(|page| {
                self.0
                    .iter()
                    .any(|(first, last)| (*first..=*last).contains(page))
            })
            .collect())
    }
}

pub(crate) struct Input {
    pub(crate) path: PathBuf,
    pub(crate) pages: Vec<usize>,
    pdf: Option<Pdf>,
}

impl Input {
    pub(crate) fn open(path: &Path, ranges: Option<&PageRanges>) -> Result<Self> {
        ensure!(
            path.is_file(),
            "input {} is not a file; pass individual images or PDFs, not directories",
            path.display()
        );
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        Read::by_ref(&mut file).take(5).read_to_end(&mut data)?;
        let pdf =
            if data == b"%PDF-" {
                file.read_to_end(&mut data)?;
                Some(Pdf::new(Arc::new(data)).map_err(|error| {
                    anyhow!("could not parse PDF {}: {error:?}", path.display())
                })?)
            } else {
                None
            };
        let pages = if let Some(pdf) = &pdf {
            let count = pdf.pages().len();
            ensure!(count > 0, "PDF {} contains no pages", path.display());
            match ranges {
                Some(ranges) => ranges.select(count)?,
                None => (1..=count).collect(),
            }
        } else {
            vec![1]
        };
        Ok(Self {
            path: path.to_path_buf(),
            pages,
            pdf,
        })
    }

    pub(crate) fn is_pdf(&self) -> bool {
        self.pdf.is_some()
    }

    pub(crate) fn render(&self, number: usize, dpi: f32) -> Result<RgbImage> {
        let Some(pdf) = &self.pdf else {
            return image::ImageReader::open(&self.path)?
                .with_guessed_format()?
                .decode()
                .map(|image| image.to_rgb8())
                .with_context(|| format!("could not decode image {}", self.path.display()));
        };
        let page = &pdf.pages()[number - 1];
        let scale = dpi / 72.0;
        let (width, height) = page.render_dimensions();
        ensure!(
            [width * scale, height * scale]
                .iter()
                .all(|size| size.is_finite() && *size >= 1.0 && *size < 65536.0),
            "PDF page {number} dimensions exceed the renderer's 1–65535 pixel range; adjust --dpi"
        );
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            bg_color: hayro::vello_cpu::color::palette::css::WHITE,
            ..Default::default()
        };
        // A per-page cache releases decoded images before rendering the next page.
        let cache = RenderCache::new();
        let pixmap = hayro::render(page, &cache, &Default::default(), &settings);
        let data = pixmap
            .data_as_u8_slice()
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|rgba| rgba[..3].iter().copied())
            .collect();
        RgbImage::from_raw(u32::from(pixmap.width()), u32::from(pixmap.height()), data)
            .context("could not convert rendered PDF page to RGB")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_ranges_validate_and_select_in_document_order() {
        let ranges: PageRanges = "5,1-3,2".parse().unwrap();
        assert_eq!(ranges.select(6).unwrap(), [1, 2, 3, 5]);
        assert!(ranges.select(4).is_err());
        for invalid in ["", "0", "3-1", "1-", "1,", "1-2-3", "two"] {
            assert!(invalid.parse::<PageRanges>().is_err());
        }
    }
}
