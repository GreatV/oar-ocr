//! CPU float32 alignment against the checkpoint's HF implementation.
//!
//! Produce the fixture outside the repository with the checkpoint's image-first
//! chat template, Text prompt, and explicit min_pixels=200704/max_pixels=921600.
//! Record expanded signed prompt IDs, grid, normalized pixel sum, the first
//! forward pass's top-3 logits, and 16 greedy tokens (repetition_penalty=1.0).
//! The reference uses eager text attention; logits must come from forward(),
//! not generation scores. No post-processing is applied on either side.
//!
//! Self-skips unless both OXR_MODEL_DIR and OXR_ALIGNMENT_FIXTURE are set.
//! The fixture image path may be absolute or relative to the model directory.

use std::path::PathBuf;

use candle_core::{DType, Device};
use oar_ocr_vl::{Oxr, OxrGenerationOptions, OxrTask, RuntimeConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    image: String,
    min_pixels: u32,
    max_pixels: u32,
    grid_thw: [usize; 3],
    pixel_values_sum: f64,
    prompt_ids: Vec<i64>,
    first_step_top3: [(u32, f32); 3],
    generated_ids: Vec<u32>,
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("OXR_MODEL_DIR") else {
        eprintln!("skipping: OXR_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("OXR_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: OXR_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    assert_eq!((fixture.min_pixels, fixture.max_pixels), (200704, 921600));
    assert_eq!(
        fixture.generated_ids.len(),
        16,
        "fixture must hold 16 tokens"
    );
    let image_path = PathBuf::from(&fixture.image);
    let image_path = if image_path.is_absolute() {
        image_path
    } else {
        model_dir.join(image_path)
    };
    let image = image::open(&image_path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", image_path.display()))
        .to_rgb8();
    let model = Oxr::from_dir_with_runtime(
        &model_dir,
        RuntimeConfig::new(Device::Cpu).with_dtype(DType::F32),
    )
    .expect("model loads");
    let options = OxrGenerationOptions::default()
        .with_min_pixels(fixture.min_pixels)
        .with_max_pixels(fixture.max_pixels)
        .with_max_new_tokens(16);
    let inputs = model
        .preprocess_image(&image, &options)
        .expect("preprocess");
    assert_eq!(inputs.grid_thw, fixture.grid_thw, "image grid");
    let pixel_sum: f64 = inputs
        .pixel_values
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&value| value as f64)
        .sum();
    let relative_error =
        (pixel_sum - fixture.pixel_values_sum).abs() / fixture.pixel_values_sum.abs().max(1.0);
    assert!(
        relative_error < 1e-6,
        "pixel sum {pixel_sum} != reference {} (relative error {relative_error})",
        fixture.pixel_values_sum
    );
    drop(inputs);

    let trace = model
        .generate_traced(&image, OxrTask::Text, &options)
        .expect("traced generation");
    assert_eq!(trace.prompt_ids, fixture.prompt_ids, "expanded chat prompt");
    assert_eq!(trace.tokens, fixture.generated_ids, "16 greedy tokens");
    let actual = trace.step_top.first().expect("first-step logits");
    let mut worst = 0f32;
    for (rank, ((actual_id, actual_value), (expected_id, expected_value))) in
        actual.iter().zip(&fixture.first_step_top3).enumerate()
    {
        assert_eq!(actual_id, expected_id, "first-step rank {rank} id");
        let delta = (actual_value - expected_value).abs();
        worst = worst.max(delta);
        // Same initial tolerances as ovisocr2_alignment; tighten only after
        // measuring a reference run, never relax to hide a token mismatch.
        let (abs_tol, rel_tol) = if rank == 0 { (0.35, 2e-2) } else { (0.5, 5e-2) };
        assert!(
            delta <= abs_tol || delta / expected_value.abs().max(f32::MIN_POSITIVE) <= rel_tol,
            "first-step rank {rank} logit {actual_value} vs reference {expected_value}"
        );
    }
    eprintln!("alignment: 16 tokens match, worst first-step logit delta {worst:.3}");
}
