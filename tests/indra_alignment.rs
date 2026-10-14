//! End-to-end alignment of the native Indra OCR pipeline against the
//! `indra-ocr` Python reference.
//!
//! The reference fixture is produced out of tree with the reference package
//! (`OrtOCR` on the same model folder, reading each image with
//! `read_images`). It is a JSON object with `pages`, each holding `image`
//! (path relative to the model directory or absolute), `math` (whether the
//! math reader was on), `rotation`, `skew`, `lines` (`[[x0, y0, x1, y1],
//! text]` in reading order) and `markdown`. Lines, boxes, reading order and
//! Markdown must match the reference exactly for color PNG input, whose gray
//! conversion is reproduced bit for bit; JPEG decoders differ.
//!
//! Both inputs come from environment variables and the test skips when
//! either is unset:
//!
//! ```bash
//! INDRA_MODEL_DIR=<model_dir> \
//! INDRA_ALIGNMENT_FIXTURE=<fixture.json> \
//!     cargo test --release --test indra_alignment -- --nocapture
//! ```

use std::path::PathBuf;

use oar_ocr::indra::{IndraOcr, IndraOcrBuilder};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    pages: Vec<Page>,
}

#[derive(Deserialize)]
struct Page {
    image: String,
    math: bool,
    rotation: u32,
    skew: f32,
    lines: Vec<([f32; 4], String)>,
    markdown: String,
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("INDRA_MODEL_DIR") else {
        eprintln!("skipping: INDRA_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("INDRA_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: INDRA_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    let build = |math: bool| -> IndraOcr {
        IndraOcrBuilder::new(&model_dir)
            .math(math)
            .build()
            .expect("models load")
    };
    let (plain, with_math) = (build(false), build(true));
    for expected in &fixture.pages {
        let image_path = if PathBuf::from(&expected.image).is_absolute() {
            PathBuf::from(&expected.image)
        } else {
            model_dir.join(&expected.image)
        };
        let image = image::open(&image_path)
            .unwrap_or_else(|err| panic!("failed to open {}: {}", image_path.display(), err))
            .to_rgb8();
        let ocr = if expected.math { &with_math } else { &plain };
        let page = ocr.predict(vec![image]).expect("page reads").remove(0);
        let name = &expected.image;
        assert_eq!(page.rotation, expected.rotation, "{name}: rotation");
        assert!(
            (page.skew - expected.skew).abs() < 1e-3,
            "{name}: skew {} vs {}",
            page.skew,
            expected.skew
        );
        assert_eq!(page.lines.len(), expected.lines.len(), "{name}: line count");
        for (i, (line, (bbox, text))) in page.lines.iter().zip(&expected.lines).enumerate() {
            assert_eq!(&line.text, text, "{name}: line {i} text");
            let worst = line
                .bbox
                .iter()
                .zip(bbox)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 0.1 + 1e-3,
                "{name}: line {i} box {:?} vs {bbox:?}",
                line.bbox
            );
        }
        assert_eq!(page.markdown, expected.markdown, "{name}: markdown");
        eprintln!("{name}: {} lines match", page.lines.len());
    }
}
