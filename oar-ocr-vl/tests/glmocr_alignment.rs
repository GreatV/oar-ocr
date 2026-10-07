//! Numerical alignment of the native GLM-OCR implementation against the
//! transformers reference (GlmOcr arch).
//!
//! The reference fixture is produced out of tree with transformers, loading
//! `GlmOcrForConditionalGeneration` from the same checkpoint in float32 with eager
//! attention on CPU, and building the inputs through the checkpoint's
//! processor and chat template with the prompt the Rust implementation uses.
//! The first-step logits come from a plain forward pass
//! (`model(**inputs).logits[0, -1]`), not from `generate()` scores, and the
//! token ids from greedy `generate(do_sample=False, max_new_tokens=16)`. The
//! fixture is a JSON object with `image` (path relative to the model
//! directory or absolute), `first_step_top3` (`[token_id, logit]`
//! triples), and `generated_ids`.
//!
//! Both inputs come from environment variables and the test skips when
//! either is unset:
//!
//! ```bash
//! GLMOCR_MODEL_DIR=<model_dir> \
//! GLMOCR_ALIGNMENT_FIXTURE=<fixture.json> \
//!     cargo test -p oar-ocr-vl --test glmocr_alignment -- --nocapture
//! ```

use std::path::PathBuf;

use candle_core::{DType, Device};
use oar_ocr_vl::RuntimeConfig;
use oar_ocr_vl::glmocr::GlmOcr;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    image: String,
    first_step_top3: Vec<[f64; 2]>,
    generated_ids: Vec<u32>,
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("GLMOCR_MODEL_DIR") else {
        eprintln!("skipping: GLMOCR_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("GLMOCR_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: GLMOCR_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    let image_path = if PathBuf::from(&fixture.image).is_absolute() {
        PathBuf::from(&fixture.image)
    } else {
        model_dir.join(&fixture.image)
    };
    let image = image::open(&image_path)
        .unwrap_or_else(|err| panic!("failed to open {}: {}", image_path.display(), err))
        .to_rgb8();

    let model = GlmOcr::from_dir_with_runtime(
        &model_dir,
        RuntimeConfig::new(Device::Cpu).with_dtype(DType::F32),
    )
    .expect("model loads");

    // The reference truncates at max_new_tokens without an EOS, so compare
    // the reference prefix; ours may legitimately continue past it.
    let trace = model
        .generate_traced(
            &image,
            "Extract all readable content from the image.",
            fixture.generated_ids.len() + 1,
        )
        .expect("traced generation");
    assert!(
        trace.tokens.len() >= fixture.generated_ids.len(),
        "ours stopped early: {:?}",
        trace.tokens
    );
    assert_eq!(
        &trace.tokens[..fixture.generated_ids.len()],
        &fixture.generated_ids[..],
        "greedy tokens"
    );

    let actual = &trace.step_top[0];
    let expected: Vec<(u32, f32)> = fixture
        .first_step_top3
        .iter()
        .map(|[id, value]| (*id as u32, *value as f32))
        .collect();
    assert_eq!(
        expected.len(),
        3,
        "fixture must hold exactly three first-step logits"
    );
    // The fixture's first-step logits come from a plain reference forward
    // pass (generate()'s scores path shifts values on this architecture);
    // against those, ids and values both agree.
    assert_eq!(actual[0].0, expected[0].0, "first-step greedy id");
    let mut worst = 0f64;
    for (rank, ((actual_id, actual_value), (expected_id, expected_value))) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(actual_id, expected_id, "first-step rank {rank} id");
        let delta = (actual_value - expected_value).abs() as f64;
        worst = worst.max(delta);
        let (abs_tol, rel_tol) = if rank == 0 { (0.35, 2e-2) } else { (0.5, 5e-2) };
        assert!(
            delta <= abs_tol
                || delta / (*expected_value as f64).abs().max(f64::MIN_POSITIVE) <= rel_tol,
            "first-step rank {rank} logit {actual_value} vs reference {expected_value}"
        );
    }
    eprintln!(
        "alignment: {} tokens, worst first-step logit delta {worst:.3}",
        fixture.generated_ids.len()
    );
}
