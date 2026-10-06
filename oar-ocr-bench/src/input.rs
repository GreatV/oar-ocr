use crate::{hash_parts, manifest::Inputs};
use anyhow::{Context, Result, ensure};
use image::RgbImage;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};

pub(crate) struct Page {
    pub(crate) id: String,
    pub(crate) image: RgbImage,
    pub(crate) sha256: String,
}

fn image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff" | "webp" | "gif"
            )
        })
}
fn walk(path: &Path, paths: &mut BTreeSet<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(path)
        .with_context(|| format!("read image directory {}", path.display()))?
    {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            walk(&entry.path(), paths)?;
        } else if kind.is_file() && image_file(&entry.path()) {
            paths.insert(entry.path());
        }
    }
    Ok(())
}
fn add(pages: &mut Vec<Page>, id: String, image: RgbImage) {
    let sha256 = hash_parts(&[
        &image.width().to_le_bytes(),
        &image.height().to_le_bytes(),
        image.as_raw(),
    ]);
    pages.push(Page { id, sha256, image });
}

pub(crate) fn load(root: &Path, inputs: &Inputs) -> Result<Vec<Page>> {
    let mut files: BTreeSet<PathBuf> = inputs
        .images
        .iter()
        .chain(&inputs.pdfs)
        .map(|p| root.join(p))
        .collect();
    for dir in &inputs.image_dirs {
        walk(&root.join(dir), &mut files)?;
    }
    let limit = inputs.max_pages.unwrap_or(usize::MAX);
    let mut pages = Vec::new();
    for path in files {
        if pages.len() >= limit {
            break;
        }
        let id = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
        {
            let pdf = hayro::hayro_syntax::Pdf::new(Arc::new(std::fs::read(&path)?))
                .map_err(|error| anyhow::anyhow!("read PDF {id}: {error:?}"))?;
            let cache = hayro::RenderCache::new();
            let scale = inputs.pdf_scale.unwrap_or(2.0);
            let settings = hayro::RenderSettings {
                x_scale: scale,
                y_scale: scale,
                bg_color: hayro::vello_cpu::color::palette::css::WHITE,
                ..Default::default()
            };
            for (index, page) in pdf.pages().iter().enumerate() {
                if pages.len() >= limit {
                    break;
                }
                let pixmap = hayro::render(page, &cache, &Default::default(), &settings);
                let data: Vec<_> = pixmap
                    .data_as_u8_slice()
                    .chunks_exact(4)
                    .flat_map(|p| p[..3].iter().copied())
                    .collect();
                let image = RgbImage::from_raw(pixmap.width().into(), pixmap.height().into(), data)
                    .context("invalid PDF raster")?;
                add(&mut pages, format!("{id}#page:{}", index + 1), image);
            }
        } else {
            ensure!(image_file(&path), "unsupported image input {id}");
            let image = image::ImageReader::open(&path)?
                .with_guessed_format()?
                .decode()
                .with_context(|| format!("decode {id}"))?
                .to_rgb8();
            add(&mut pages, id, image);
        }
    }
    ensure!(!pages.is_empty(), "input set contains no pages");
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn image_content_takes_precedence_over_extension() {
        let dir = tempfile::tempdir().unwrap();
        RgbImage::new(2, 3)
            .save_with_format(dir.path().join("page.png"), image::ImageFormat::Jpeg)
            .unwrap();
        let pages = load(
            dir.path(),
            &Inputs {
                images: vec!["page.png".into()],
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pages[0].image.dimensions(), (2, 3));
    }

    #[test]
    fn directory_order_and_limit_are_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b.png", "a.png"] {
            RgbImage::new(2, 3).save(dir.path().join(name)).unwrap();
        }
        let inputs = Inputs {
            image_dirs: vec![".".into()],
            max_pages: Some(1),
            ..Default::default()
        };
        let pages = load(dir.path(), &inputs).unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].id, "a.png");
        assert_eq!(pages[0].sha256.len(), 64);
    }
}
