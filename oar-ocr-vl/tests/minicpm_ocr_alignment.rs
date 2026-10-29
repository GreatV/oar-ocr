//! CPU float32 alignment against the Transformers MiniCPM-OCR implementation.
//!
//! Produce the fixture outside the repository from a lossless PNG large
//! enough to be sliced, with the chat template (`enable_thinking=False`),
//! the processor defaults, and eager attention. Record the slice target
//! sizes, the pixel-value sums, the prompt IDs, the first forward pass's
//! top-3 logits, and 16 greedy tokens. No post-processing on either side.
//!
//! Self-skips unless both MINICPM_OCR_MODEL_DIR and
//! MINICPM_OCR_ALIGNMENT_FIXTURE are set.

use std::path::PathBuf;

use candle_core::{DType, Device};
use oar_ocr_vl::{MiniCpmOcr, MiniCpmOcrGenerationOptions, MiniCpmOcrTask, RuntimeConfig};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    image: String,
    prompt: String,
    target_sizes: Vec<[usize; 2]>,
    pixel_values_abs_sum: f64,
    prompt_ids: Vec<u32>,
    first_step_top3: [(u32, f32); 3],
    generated_ids: Vec<u32>,
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("MINICPM_OCR_MODEL_DIR") else {
        eprintln!("skipping: MINICPM_OCR_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("MINICPM_OCR_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: MINICPM_OCR_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    assert_eq!(fixture.prompt, MiniCpmOcrTask::Text.prompt());
    assert_eq!(
        fixture.generated_ids.len(),
        16,
        "fixture must hold 16 tokens"
    );
    let image = image::open(PathBuf::from(&fixture.image))
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", fixture.image))
        .to_rgb8();
    let model = MiniCpmOcr::from_dir_with_runtime(
        PathBuf::from(model_dir),
        RuntimeConfig::new(Device::Cpu).with_dtype(DType::F32),
    )
    .expect("model loads");
    let options = MiniCpmOcrGenerationOptions::default().with_max_new_tokens(16);

    let inputs = model
        .preprocess_image(&image, &options)
        .expect("preprocess");
    let sizes: Vec<[usize; 2]> = inputs
        .slices
        .iter()
        .map(|slice| [slice.grid_hw.0, slice.grid_hw.1])
        .collect();
    assert_eq!(sizes, fixture.target_sizes, "slice grids");
    let abs_sum: f64 = inputs
        .slices
        .iter()
        .flat_map(|slice| {
            slice
                .pixel_values
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        })
        .map(|value| value.abs() as f64)
        .sum();
    // The reference resizes with torchvision's uint8 bicubic, this crate with
    // Pillow's; on the measured fixture they agreed to 2e-7.
    let relative_error =
        (abs_sum - fixture.pixel_values_abs_sum).abs() / fixture.pixel_values_abs_sum;
    assert!(
        relative_error < 1e-5,
        "pixel |sum| {abs_sum} vs reference {} (relative error {relative_error})",
        fixture.pixel_values_abs_sum
    );
    assert_eq!(
        model.prompt_ids(&inputs, &fixture.prompt).expect("prompt"),
        fixture.prompt_ids,
        "expanded chat prompt"
    );
    drop(inputs);

    let trace = model
        .generate_traced(&image, MiniCpmOcrTask::Text, &options)
        .expect("traced generation");
    assert_eq!(trace.tokens, fixture.generated_ids, "16 greedy tokens");
    let actual = trace.step_top.first().expect("first-step logits");
    let mut worst = 0f32;
    for (rank, ((actual_id, actual_value), (expected_id, expected_value))) in
        actual.iter().zip(&fixture.first_step_top3).enumerate()
    {
        assert_eq!(actual_id, expected_id, "first-step rank {rank} id");
        let delta = (actual_value - expected_value).abs();
        worst = worst.max(delta);
        let (abs_tol, rel_tol) = if rank == 0 { (0.35, 2e-2) } else { (0.5, 5e-2) };
        assert!(
            delta <= abs_tol || delta / expected_value.abs().max(f32::MIN_POSITIVE) <= rel_tol,
            "first-step rank {rank} logit {actual_value} vs reference {expected_value}"
        );
    }
    eprintln!(
        "alignment: pixel relative error {relative_error:.2e}, 16 tokens match, worst first-step logit delta {worst:.4}"
    );
}
