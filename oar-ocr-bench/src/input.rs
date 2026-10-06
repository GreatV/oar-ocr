use crate::manifest::Inputs;
use anyhow::{Context, Result, ensure};
use image::RgbImage;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub(crate) struct Page {
    pub(crate) id: String,
    pub(crate) image: RgbImage,
}

fn image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff" | "webp"
            )
        })
}

/// Builds an input set from `--input` paths: image files or directories.
pub(crate) fn from_paths(root: &Path, paths: &[PathBuf]) -> Result<Inputs> {
    let mut inputs = Inputs::default();
    for path in paths {
        let value = path.to_string_lossy().into_owned();
        if root.join(path).is_dir() {
            inputs.image_dirs.push(value);
        } else {
            ensure!(
                root.join(path).is_file() && image_file(path),
                "unsupported input {}",
                path.display()
            );
            inputs.images.push(value);
        }
    }
    Ok(inputs)
}

/// Loads pages in sorted path order, up to `max_pages`. Directories contribute
/// their top-level image files; subdirectories are not searched.
pub(crate) fn load(root: &Path, inputs: &Inputs) -> Result<Vec<Page>> {
    let mut files: BTreeSet<PathBuf> = inputs.images.iter().map(|p| root.join(p)).collect();
    for dir in &inputs.image_dirs {
        let dir = root.join(dir);
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("read directory {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_file() && image_file(&path) {
                files.insert(path);
            }
        }
    }
    let pages = files
        .into_iter()
        .take(inputs.max_pages.unwrap_or(usize::MAX))
        .map(|path| {
            let id = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let image = image::ImageReader::open(&path)?
                .with_guessed_format()?
                .decode()
                .with_context(|| format!("decode {id}"))?
                .to_rgb8();
            Ok(Page { id, image })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!pages.is_empty(), "input set contains no pages");
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directories_are_sorted_and_limited() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("nested")).unwrap();
        for name in ["b.png", "a.png", "nested/c.png"] {
            RgbImage::new(2, 3).save(dir.path().join(name)).unwrap();
        }
        let inputs = from_paths(dir.path(), &[".".into()]).unwrap();
        assert_eq!(inputs.image_dirs, ["."]);
        let all = load(dir.path(), &inputs).unwrap();
        assert_eq!(
            all.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
            ["a.png", "b.png"]
        );
        let pages = load(
            dir.path(),
            &Inputs {
                max_pages: Some(1),
                ..inputs
            },
        )
        .unwrap();
        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].id, "a.png");
        assert!(from_paths(dir.path(), &["missing.png".into()]).is_err());
    }
}
