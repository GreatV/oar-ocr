//! Numerical alignment of Xiaomi-OCR-0, MinerU2.5, TeleOCR and PaddleOCR-VL
//! against their transformers references.
//!
//! Each fixture is produced out of tree with transformers, loading the same
//! checkpoint in float32 with eager attention on CPU (TeleOCR and
//! PaddleOCR-VL through the checkpoint's own remote code, which needs
//! transformers 4.57), and building the inputs through the checkpoint's
//! Pillow image processor (`use_fast=False`) and chat template with the
//! instruction the test passes. Use a lossless PNG input so both sides
//! decode identical pixels. The first-step logits come from a plain
//! forward pass (`model(**inputs).logits[0, -1]`) and the token ids from
//! greedy `generate(do_sample=False, max_new_tokens=16)`, keeping the
//! checkpoint's repetition penalty (1.05 for TeleOCR). A fixture is a JSON
//! object with `image` (path relative to the model directory or absolute),
//! `first_step_top3` (`[token_id, logit]` triples), and `generated_ids`.
//!
//! Every test reads `<PREFIX>_MODEL_DIR` and `<PREFIX>_ALIGNMENT_FIXTURE`
//! and skips when either is unset; the prefixes are `XIAOMIOCR`, `MINERU`,
//! `TELEOCR` and `PADDLEOCRVL`:
//!
//! ```bash
//! MINERU_MODEL_DIR=<model_dir> \
//! MINERU_ALIGNMENT_FIXTURE=<fixture.json> \
//!     cargo test -p oar-ocr-vl --test reference_alignment mineru -- --nocapture
//! ```

use std::path::PathBuf;

use candle_core::{DType, Device};
use image::RgbImage;
use oar_ocr_vl::RuntimeConfig;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    image: String,
    first_step_top3: Vec<[f64; 2]>,
    generated_ids: Vec<u32>,
}

struct Case {
    model_dir: PathBuf,
    image: RgbImage,
    fixture: Fixture,
}

impl Case {
    fn load(prefix: &str) -> Option<Self> {
        let model_var = format!("{prefix}_MODEL_DIR");
        let fixture_var = format!("{prefix}_ALIGNMENT_FIXTURE");
        let (Some(model_dir), Some(fixture_path)) =
            (std::env::var_os(&model_var), std::env::var_os(&fixture_var))
        else {
            eprintln!("skipping: {model_var} or {fixture_var} is not set");
            return None;
        };
        let model_dir = PathBuf::from(model_dir);
        let fixture: Fixture = serde_json::from_str(
            &std::fs::read_to_string(&fixture_path)
                .unwrap_or_else(|err| panic!("failed to read {fixture_path:?}: {err}")),
        )
        .expect("fixture parses");
        assert_eq!(
            fixture.first_step_top3.len(),
            3,
            "fixture must hold exactly three first-step logits"
        );
        let image_path = model_dir.join(&fixture.image);
        let image = image::open(&image_path)
            .unwrap_or_else(|err| panic!("failed to open {}: {err}", image_path.display()))
            .to_rgb8();
        Some(Self {
            model_dir,
            image,
            fixture,
        })
    }

    fn runtime() -> RuntimeConfig {
        RuntimeConfig::new(Device::Cpu).with_dtype(DType::F32)
    }

    /// The traced run must be asked for one token more than the fixture
    /// holds: the reference stops at `max_new_tokens` without an EOS, so
    /// ours is compared on the reference prefix.
    fn budget(&self) -> usize {
        self.fixture.generated_ids.len() + 1
    }

    fn assert_matches(&self, tokens: &[u32], step_top: &[[(u32, f32); 3]]) {
        let expected_ids = &self.fixture.generated_ids;
        assert!(
            tokens.len() >= expected_ids.len(),
            "ours stopped early: {tokens:?}"
        );
        assert_eq!(
            &tokens[..expected_ids.len()],
            &expected_ids[..],
            "greedy tokens"
        );

        let mut worst = 0f64;
        for (rank, ((actual_id, actual_value), [expected_id, expected_value])) in step_top[0]
            .iter()
            .zip(&self.fixture.first_step_top3)
            .enumerate()
        {
            assert_eq!(*actual_id, *expected_id as u32, "first-step rank {rank} id");
            let delta = (*actual_value as f64 - expected_value).abs();
            worst = worst.max(delta);
            let (abs_tol, rel_tol) = if rank == 0 { (0.35, 2e-2) } else { (0.5, 5e-2) };
            assert!(
                delta <= abs_tol || delta / expected_value.abs().max(f64::MIN_POSITIVE) <= rel_tol,
                "first-step rank {rank} logit {actual_value} vs reference {expected_value}"
            );
        }
        eprintln!(
            "alignment: {} tokens match, worst first-step logit delta {worst:.3}",
            expected_ids.len()
        );
    }
}

#[test]
fn xiaomi_ocr() {
    use oar_ocr_vl::xiaomi_ocr::{DEFAULT_PROMPT, XiaomiOcr};
    let Some(case) = Case::load("XIAOMIOCR") else {
        return;
    };
    let model = XiaomiOcr::from_dir_with_runtime(&case.model_dir, Case::runtime()).expect("load");
    let trace = model
        .generate_traced(&case.image, DEFAULT_PROMPT, case.budget())
        .expect("traced generation");
    case.assert_matches(&trace.tokens, &trace.step_top);
}

#[test]
fn mineru() {
    use oar_ocr_vl::MinerU;
    let Some(case) = Case::load("MINERU") else {
        return;
    };
    let model = MinerU::from_dir_with_runtime(&case.model_dir, Case::runtime()).expect("load");
    let trace = model
        .generate_traced(&case.image, "\nText Recognition:", case.budget())
        .expect("traced generation");
    case.assert_matches(&trace.tokens, &trace.step_top);
}

#[test]
fn teleocr() {
    use oar_ocr_vl::{TeleOcr, TeleOcrTask};
    let Some(case) = Case::load("TELEOCR") else {
        return;
    };
    let model = TeleOcr::from_dir_with_runtime(&case.model_dir, Case::runtime()).expect("load");
    let trace = model
        .generate_traced(&case.image, TeleOcrTask::Text.prompt(), case.budget())
        .expect("traced generation");
    case.assert_matches(&trace.tokens, &trace.step_top);
}

#[test]
fn paddleocr_vl() {
    use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
    let Some(case) = Case::load("PADDLEOCRVL") else {
        return;
    };
    let model = PaddleOcrVl::from_dir_with_runtime(&case.model_dir, Case::runtime()).expect("load");
    let trace = model
        .generate_traced(&case.image, PaddleOcrVlTask::Ocr, case.budget())
        .expect("traced generation");
    case.assert_matches(&trace.tokens, &trace.step_top);
}
