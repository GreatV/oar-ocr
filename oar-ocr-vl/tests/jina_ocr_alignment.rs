//! Numerical alignment of the native jina-ocr-v1 implementation against the
//! transformers reference (remote code, `model.generate`, float32, eager
//! attention).
//!
//! The fixture is produced out-of-tree and is intentionally not committed;
//! point `JINA_OCR_ALIGNMENT_FIXTURE` at one to enable the comparison. The
//! fixture's `image` entry is resolved relative to the model directory
//! unless it is absolute:
//!
//! ```bash
//! JINA_OCR_MODEL_DIR=<model_dir> \
//! JINA_OCR_ALIGNMENT_FIXTURE=<fixture.json> \
//!     cargo test -p oar-ocr-vl --test jina_ocr_alignment -- --nocapture
//! ```

use std::path::PathBuf;

use candle_core::{DType, Device};
use oar_ocr_vl::RuntimeConfig;
use oar_ocr_vl::jina_ocr::JinaOcr;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    image: String,
    spatial_crop: Vec<Vec<i64>>,
    crop_count: usize,
    /// Sum of the normalized global-view pixels (f32, reference).
    global_pixel_sum: f64,
    /// Sum of the normalized tile pixels (f32, reference).
    tiles_pixel_sum: f64,
    /// Reference-generated ids, including the closing EOS when `hit_eos`.
    generated_ids: Vec<u32>,
    hit_eos: bool,
    generated_text: String,
    /// Raw (pre-ngram-ban) top-3 `(token, logit)` pairs per decoding step.
    step_top: Vec<Vec<(u32, f32)>>,
}

fn tensor_sum(tensor: &candle_core::Tensor) -> f64 {
    tensor
        .to_dtype(DType::F64)
        .and_then(|t| t.sum_all())
        .and_then(|s| s.to_scalar())
        .expect("tensor sum")
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("JINA_OCR_MODEL_DIR") else {
        eprintln!("skipping: JINA_OCR_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("JINA_OCR_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: JINA_OCR_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read fixture {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    let image_path = if PathBuf::from(&fixture.image).is_absolute() {
        PathBuf::from(&fixture.image)
    } else {
        model_dir.join(&fixture.image)
    };
    let image = image::open(&image_path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", image_path.display()))
        .to_rgb8();

    // Float32 on CPU mirrors the fixture's reference configuration.
    let model = JinaOcr::from_dir_with_runtime(
        &model_dir,
        RuntimeConfig::new(Device::Cpu).with_dtype(DType::F32),
    )
    .expect("model loads");

    // Preprocessing: tile grid and tile count must match exactly, and the
    // normalized pixels must agree with the reference within a small relative
    // tolerance (the resampling kernels are not bit-identical).
    let inputs = oar_ocr_vl::jina_ocr::processing::preprocess_image(
        &image,
        model.processor_config(),
        &Device::Cpu,
        DType::F32,
    )
    .expect("preprocess");
    assert_eq!(
        (inputs.tile_grid.0 as i64, inputs.tile_grid.1 as i64),
        (fixture.spatial_crop[0][0], fixture.spatial_crop[0][1])
    );
    assert_eq!(inputs.tiles.dims()[0], fixture.crop_count);
    for (name, actual, expected) in [
        (
            "global",
            tensor_sum(&inputs.global_view),
            fixture.global_pixel_sum,
        ),
        ("tiles", tensor_sum(&inputs.tiles), fixture.tiles_pixel_sum),
    ] {
        let relative_error = (actual - expected).abs() / expected.abs().max(f64::MIN_POSITIVE);
        assert!(
            relative_error < 1e-3,
            "{name} pixel sum {actual} != reference {expected} (relative error {relative_error})"
        );
    }

    // Greedy decoding must reproduce the reference token-for-token (the
    // reference ids keep the closing EOS, the trace does not).
    let trace = model
        .generate_traced(&image, fixture.generated_ids.len())
        .expect("generation trace");
    let expected_tokens: Vec<u32> = if fixture.hit_eos {
        fixture.generated_ids[..fixture.generated_ids.len() - 1].to_vec()
    } else {
        fixture.generated_ids.clone()
    };
    assert_eq!(trace.hit_eos, fixture.hit_eos, "EOS flag");
    assert_eq!(trace.tokens, expected_tokens, "greedy tokens");
    let text = model.decode_tokens(&trace.tokens).expect("decode");
    assert_eq!(text, fixture.generated_text);

    // Step-wise logits: the rank-0 (greedy pick) token id must match exactly
    // at every step, including the EOS step, and the values within a tolerance
    // that accounts for CPU summation-order differences. Lower ranks can swap
    // when two candidates sit within that noise band, so their ids are only
    // compared when they agree and the value check is skipped otherwise.
    assert_eq!(trace.step_top.len(), fixture.step_top.len(), "step count");
    let mut worst = [0f64; 3];
    for (step, (actual, expected)) in trace.step_top.iter().zip(&fixture.step_top).enumerate() {
        assert_eq!(
            expected.len(),
            3,
            "fixture step {step} must carry 3 entries"
        );
        assert_eq!(
            actual[0].0, expected[0].0,
            "step {step} greedy token id: ours {actual:?} vs reference {expected:?}"
        );
        for (rank, ((actual_id, actual_value), (expected_id, expected_value))) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            if rank > 0 && actual_id != expected_id {
                continue;
            }
            let delta = (actual_value - expected_value).abs() as f64;
            worst[rank] = worst[rank].max(delta);
            let expected_value = *expected_value as f64;
            // Rank 0 is the greedy decision and stays tight; lower ranks only
            // need to track the reference, since their f32 values drift with
            // accumulated KV-cache rounding and near-tie ordering swaps.
            let (abs_tol, rel_tol) = if rank == 0 { (0.35, 2e-2) } else { (0.5, 5e-2) };
            assert!(
                delta <= abs_tol || delta / expected_value.abs().max(f64::MIN_POSITIVE) <= rel_tol,
                "step {step} rank {rank} logit {actual_value} vs reference {expected_value}"
            );
        }
    }
    eprintln!(
        "alignment: {} steps, worst logit delta per rank {worst:?}",
        trace.step_top.len()
    );
}
