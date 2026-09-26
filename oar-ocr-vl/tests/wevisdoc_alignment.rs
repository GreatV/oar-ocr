//! Numerical alignment of the native WeVisDoc implementation against the
//! transformers 4.57 reference.
//!
//! The reference fixture is produced by running
//! `Qwen3VLForConditionalGeneration.generate` (float32, eager attention,
//! greedy) over any page image and dumping the sequence, per-step top-3
//! logits, and preprocessor outputs. It is generated out-of-tree and is
//! intentionally not committed; point `WEVISDOC_ALIGNMENT_FIXTURE` at one to
//! enable the comparison.
//!
//! Both inputs come from environment variables and the test skips when
//! either is unset:
//!
//! ```bash
//! WEVISDOC_MODEL_DIR=<model_dir> \
//! WEVISDOC_ALIGNMENT_FIXTURE=<fixture.json> \
//!     cargo test -p oar-ocr-vl --test wevisdoc_alignment -- --nocapture
//! ```

use std::path::PathBuf;

use candle_core::{DType, Device};
use oar_ocr_vl::RuntimeConfig;
use oar_ocr_vl::wevisdoc::WeVisDoc;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    /// Test image, resolved inside the model directory (e.g. an asset shipped
    /// with the checkpoint).
    image: String,
    grid_thw: [usize; 3],
    pixel_values_sum: f64,
    generated_ids: Vec<u32>,
    generated_text: String,
}

#[test]
fn matches_python_reference_end_to_end() {
    let Some(model_dir) = std::env::var_os("WEVISDOC_MODEL_DIR") else {
        eprintln!("skipping: WEVISDOC_MODEL_DIR is not set");
        return;
    };
    let Some(fixture_path) = std::env::var_os("WEVISDOC_ALIGNMENT_FIXTURE") else {
        eprintln!("skipping: WEVISDOC_ALIGNMENT_FIXTURE is not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|err| panic!("failed to read fixture {fixture_path:?}: {err}")),
    )
    .expect("fixture parses");
    let image_path = model_dir.join(&fixture.image);
    let image = image::open(&image_path)
        .unwrap_or_else(|err| panic!("failed to open {}: {err}", image_path.display()))
        .to_rgb8();

    // Float32 on CPU mirrors the fixture's reference configuration.
    let model = WeVisDoc::from_dir_with_runtime(
        &model_dir,
        RuntimeConfig::new(Device::Cpu).with_dtype(DType::F32),
    )
    .expect("model loads");

    // Preprocessing: the grid must match exactly and the pixel values must
    // agree with the reference (sum compared with a relative tolerance: the
    // fast processor fuses rescale and normalize in one pass).
    let inputs = oar_ocr_vl::wevisdoc::processing::preprocess_image(
        &image,
        model.image_processor_config(),
        &model.config().vision_config,
        &Device::Cpu,
        DType::F32,
    )
    .expect("preprocess");
    assert_eq!(
        inputs.grid_thw,
        (
            fixture.grid_thw[0],
            fixture.grid_thw[1],
            fixture.grid_thw[2]
        )
    );
    let pixels_sum: f32 = inputs
        .pixel_values
        .sum_all()
        .and_then(|sum| sum.to_scalar())
        .expect("pixel sum");
    let relative_error =
        (pixels_sum as f64 - fixture.pixel_values_sum).abs() / fixture.pixel_values_sum.abs();
    assert!(
        relative_error < 1e-3,
        "pixel values sum {pixels_sum} != reference {} (relative error {relative_error})",
        fixture.pixel_values_sum
    );

    // Greedy decoding must reproduce the reference token-for-token, which
    // exercises the vision tower, DeepStack injection, interleaved MRoPE, and
    // the KV-cache decode loop in one pass.
    let tokens = model
        .generate_tokens(std::slice::from_ref(&image), fixture.generated_ids.len())
        .expect("generation batch")
        .pop()
        .expect("one image")
        .expect("generation succeeds");
    assert_eq!(tokens, fixture.generated_ids, "greedy tokens");
    let text = model.tokenizer().decode(&tokens, true).expect("decode");
    assert_eq!(text.trim(), fixture.generated_text.trim());
}
