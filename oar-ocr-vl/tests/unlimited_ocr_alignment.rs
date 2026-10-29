//! Float32 alignment against the checkpoint's remote-code implementation.
//!
//! Produce the fixture outside the repository from lossless PNG pages with
//! float32 weights and the dense DASH path (`dash_router.use_triton =
//! False`), feeding the encoder the reference's bf16-rounded pixels as f32.
//! Record the absolute pixel sums of each encoder call (tiles, then global
//! view, per page), the prompt IDs, the first forward pass's top-3 logits,
//! and 16 greedy tokens from `infer` with its defaults.
//!
//! Self-skips unless both UNLIMITED_OCR_MODEL_DIR and
//! UNLIMITED_OCR_ALIGNMENT_FIXTURE are set; UNLIMITED_OCR_DEVICE selects
//! the device (`cpu` by default).

use std::path::PathBuf;

use candle_core::{DType, Device};
use oar_ocr_vl::unlimited_ocr::processing::DEFAULT_PROMPT;
use oar_ocr_vl::{
    RuntimeConfig, UnlimitedOcr, UnlimitedOcrGenerationOptions, UnlimitedOcrViewMode,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    images: Vec<String>,
    global_and_tile_abs_sums: Vec<f64>,
    prompt_ids: Vec<u32>,
    first_step_top3: [(u32, f32); 3],
    generated_ids: Vec<u32>,
}

fn abs_sum(tensor: &candle_core::Tensor) -> f64 {
    tensor
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&v| v.abs() as f64)
        .sum()
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("UNLIMITED_OCR_MODEL_DIR") else {
        eprintln!("skipping: UNLIMITED_OCR_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("UNLIMITED_OCR_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: UNLIMITED_OCR_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let device = match std::env::var("UNLIMITED_OCR_DEVICE").as_deref() {
        Ok("cuda") => Device::new_cuda(0).expect("CUDA device"),
        _ => Device::Cpu,
    };
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    assert_eq!(
        fixture.generated_ids.len(),
        16,
        "fixture must hold 16 tokens"
    );
    let pages: Vec<_> = fixture
        .images
        .iter()
        .map(|path| image::open(path).expect("page opens").to_rgb8())
        .collect();
    let model = UnlimitedOcr::from_dir_with_runtime(
        PathBuf::from(model_dir),
        RuntimeConfig::new(device).with_dtype(DType::F32),
    )
    .expect("model loads");

    let inputs = model
        .preprocess_pages(&pages, UnlimitedOcrViewMode::Auto)
        .expect("preprocess");
    // The reference encodes a page's tiles before its global view.
    let sums: Vec<f64> = inputs
        .iter()
        .flat_map(|page| page.tiles.iter().chain([&page.global]).map(abs_sum))
        .collect();
    assert_eq!(
        sums.len(),
        fixture.global_and_tile_abs_sums.len(),
        "encoder calls"
    );
    for (actual, expected) in sums.iter().zip(&fixture.global_and_tile_abs_sums) {
        let relative = (actual - expected).abs() / expected.max(1.0);
        assert!(
            relative < 1e-5,
            "pixel |sum| {actual} vs reference {expected}"
        );
    }
    assert_eq!(
        model.prompt_ids(&inputs, DEFAULT_PROMPT).expect("prompt"),
        fixture.prompt_ids,
        "prompt ids"
    );
    drop(inputs);

    let options = UnlimitedOcrGenerationOptions::default().with_max_new_tokens(16);
    let trace = model.generate_traced(&pages, &options).expect("generation");
    let actual = trace.step_top.first().expect("first-step logits");
    let mut worst = 0f32;
    for (rank, ((actual_id, actual_value), (expected_id, expected_value))) in
        actual.iter().zip(&fixture.first_step_top3).enumerate()
    {
        assert_eq!(actual_id, expected_id, "first-step rank {rank} id");
        worst = worst.max((actual_value - expected_value).abs());
    }
    assert!(worst < 0.05, "first-step logits differ by {worst}");
    assert_eq!(trace.tokens, fixture.generated_ids, "16 greedy tokens");
    eprintln!("alignment: 16 tokens match, worst first-step logit delta {worst:.4}");
}
