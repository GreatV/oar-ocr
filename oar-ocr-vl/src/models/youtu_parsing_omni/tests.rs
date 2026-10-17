use super::{config::Config, text::TextModel, vision::VisionModel};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder, linear_no_bias};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    config: Config,
    pixels: Vec<f32>,
    embeddings: Vec<f32>,
    vision: Vec<f32>,
    logits: Vec<f32>,
}

fn assert_close(actual: &Tensor, expected: &[f32]) {
    let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(error < 2e-5, "reference max absolute error {error}");
}

#[test]
fn encoder_and_latent_decoder_match_python_reference() -> candle_core::Result<()> {
    // The fixture comes from the checkpoint's Python model definition with
    // deterministic tiny weights; encoder flash attention is replaced by SDPA.
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
    reference.config.validate().unwrap();
    let device = Device::Cpu;
    let vb = VarBuilder::from_slice_safetensors(
        include_bytes!("fixtures/tiny.safetensors"),
        DType::F32,
        &device,
    )?;
    let vision = VisionModel::load(&reference.config.omni_config, vb.pp("model.omni_model"))?;
    let pixels = Tensor::from_vec(reference.pixels, (16, 12), &device)?;
    assert_close(&vision.forward(&pixels, 4, 4)?, &reference.vision);

    let config = &reference.config.text_config;
    let text = TextModel::load(config, vb.pp("model.language_model"))?;
    let head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
    let embeds = Tensor::from_vec(reference.embeddings, (1, 7, 8), &device)?;
    let full = head.forward(&text.forward(&embeds, 0)?)?;
    assert_close(&full, &reference.logits);
    text.clear_cache();
    let mut parts = vec![head.forward(&text.forward(&embeds.narrow(1, 0, 4)?, 0)?)?];
    for offset in 4..7 {
        parts.push(head.forward(&text.forward(&embeds.narrow(1, offset, 1)?, offset)?)?);
    }
    assert_close(&Tensor::cat(&parts, 1)?, &reference.logits);
    text.clear_cache();
    assert_close(
        &head.forward(&text.forward(&embeds, 0)?)?,
        &reference.logits,
    );
    Ok(())
}
