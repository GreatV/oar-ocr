//! Model detection from a checkpoint directory's `config.json`.

use crate::api::any_page_parser::AnyPageParserModel;
use crate::api::error::Error;
use serde_json::Value;

/// `config.json` `architectures` entries that identify exactly one supported
/// model on their own, because no stock checkpoint of that family uses them.
const SINGLE_ARCHITECTURE_RULES: &[(&str, AnyPageParserModel)] = &[
    (
        "HunYuanVLForConditionalGeneration",
        AnyPageParserModel::HunyuanOcr,
    ),
    ("DeepseekOCRForCausalLM", AnyPageParserModel::JinaOcr),
    (
        "MinerUDiffusionForConditionalGeneration",
        AnyPageParserModel::MinerUDiffusion,
    ),
    ("MonkeyOCRv2ForCausalLM", AnyPageParserModel::MonkeyOcrV2),
    (
        "PaddleOCRVLForConditionalGeneration",
        AnyPageParserModel::PaddleOcrVl,
    ),
    ("GlmOcrForConditionalGeneration", AnyPageParserModel::GlmOcr),
];

// Architectures below are generic backbone names that stock checkpoints of
// the same family also carry, so each one needs the supported model's marker
// before it may load; without the marker detection refuses the config instead
// of guessing. See generic_backbone_refusal.

const INTERNVL_ARCHITECTURE: &str = "InternVLChatModel";
const QWEN2_VL_ARCHITECTURE: &str = "Qwen2VLForConditionalGeneration";
const QWEN2_5_VL_ARCHITECTURE: &str = "Qwen2_5_VLForConditionalGeneration";
const QWEN3_VL_ARCHITECTURE: &str = "Qwen3VLForConditionalGeneration";

/// HPD-Parsing's fork/child decode tokens; stock InternVL configs have none.
const HPD_FORK_TOKEN_KEY: &str = "fork_token_id";
/// MinerU2.5's layout-output delimiter; the stock Qwen2(-VL) tokenizer has no
/// such special token.
const MINERU_LAYOUT_TOKEN: &str = "<|md_start|>";
/// TeleOCR's windowed vision tower; stock Qwen2.5-VL configs have neither
/// window_size nor fullatt_block_indexes.
const TELEOCR_WINDOW_KEY: &str = "/vision_config/window_size";

/// OvisOCR2 and Xiaomi-OCR-0 both carry `Qwen3_5ForConditionalGeneration`
/// (they share a Qwen3.5 text tower), so their different vision towers
/// disambiguate them through `vision_config.model_type`.
const QWEN3_5_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const OVIS_VISION_MODEL_TYPE: &str = "qwen3_5";
const XIAOMI_VISION_MODEL_TYPE: &str = "qwen3_5_vision";

/// Detects the supported model described by a parsed `config.json` and the
/// directory's special tokenizer tokens.
///
/// Every error names what the config contains and what this crate supports,
/// so an unsupported or ambiguous checkpoint never falls back silently.
pub(crate) fn detect_model(
    config: &Value,
    special_tokens: &[String],
) -> Result<AnyPageParserModel, Error> {
    let architectures = config_architectures(config);
    let mut candidates: Vec<AnyPageParserModel> = Vec::new();
    for architecture in &architectures {
        let candidate = match *architecture {
            QWEN3_5_ARCHITECTURE => Some(qwen3_5_model(config)?),
            INTERNVL_ARCHITECTURE => Some(internvl_model(config)?),
            QWEN2_VL_ARCHITECTURE => Some(qwen2_vl_model(special_tokens)?),
            QWEN2_5_VL_ARCHITECTURE => Some(qwen2_5_vl_model(config)?),
            QWEN3_VL_ARCHITECTURE => Some(qwen3_vl_model()?),
            _ => SINGLE_ARCHITECTURE_RULES
                .iter()
                .find(|(arch, _)| arch == architecture)
                .map(|(_, model)| *model),
        };
        if let Some(model) = candidate
            && !candidates.contains(&model)
        {
            candidates.push(model);
        }
    }
    let found = format!(
        "architectures={architectures:?}, model_type={:?}",
        config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("<missing>")
    );
    match candidates.as_slice() {
        [] => Err(Error::config(format!(
            "unsupported model config ({found}); supported architectures: {}",
            supported_architectures()
        ))),
        [model] => Ok(*model),
        many => Err(Error::config(format!(
            "ambiguous model config ({found}) matched several supported models: {}",
            many.iter()
                .map(|model| model.name())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

/// Resolves the shared Qwen3.5 architecture by its vision tower.
fn qwen3_5_model(config: &Value) -> Result<AnyPageParserModel, Error> {
    let vision_model_type = config
        .pointer("/vision_config/model_type")
        .and_then(Value::as_str);
    match vision_model_type {
        Some(value) if value == OVIS_VISION_MODEL_TYPE => Ok(AnyPageParserModel::OvisOcr2),
        Some(value) if value == XIAOMI_VISION_MODEL_TYPE => Ok(AnyPageParserModel::XiaomiOcr),
        _ => Err(Error::config(format!(
            "unsupported {QWEN3_5_ARCHITECTURE} config (vision_config.model_type={vision_model_type:?}); \
             expected {OVIS_VISION_MODEL_TYPE:?} (OvisOCR2) or {XIAOMI_VISION_MODEL_TYPE:?} (Xiaomi-OCR-0)"
        ))),
    }
}

/// InternVL is a generic backbone; HPD-Parsing's fork decode gives its config
/// a marker no stock InternVL checkpoint carries.
fn internvl_model(config: &Value) -> Result<AnyPageParserModel, Error> {
    if config.get(HPD_FORK_TOKEN_KEY).is_some() {
        return Ok(AnyPageParserModel::HpdParsing);
    }
    Err(generic_backbone_refusal(
        INTERNVL_ARCHITECTURE,
        "HPD-Parsing (identified by its config.json fork_token_id)",
    ))
}

/// Qwen2-VL is a generic backbone; MinerU2.5's layout delimiter is a special
/// tokenizer token the stock tokenizer does not have.
fn qwen2_vl_model(special_tokens: &[String]) -> Result<AnyPageParserModel, Error> {
    if special_tokens
        .iter()
        .any(|token| token == MINERU_LAYOUT_TOKEN)
    {
        return Ok(AnyPageParserModel::MinerU);
    }
    Err(generic_backbone_refusal(
        QWEN2_VL_ARCHITECTURE,
        "MinerU2.5 (identified by its <|md_start|> tokenizer token)",
    ))
}

/// Qwen2.5-VL is a generic backbone; TeleOCR's windowed vision tower is the
/// marker.
fn qwen2_5_vl_model(config: &Value) -> Result<AnyPageParserModel, Error> {
    if config.pointer(TELEOCR_WINDOW_KEY).is_some() {
        return Ok(AnyPageParserModel::TeleOcr);
    }
    Err(generic_backbone_refusal(
        QWEN2_5_VL_ARCHITECTURE,
        "TeleOCR (identified by its vision_config.window_size)",
    ))
}

/// Qwen3-VL is a generic backbone, and WeVisDoc's published config matches a
/// stock Qwen3-VL checkpoint field for field, so nothing in the directory can
/// tell them apart: this architecture always loads through an explicit
/// [`AnyPageParserModel`].
fn qwen3_vl_model() -> Result<AnyPageParserModel, Error> {
    Err(generic_backbone_refusal(
        QWEN3_VL_ARCHITECTURE,
        "WeVisDoc (whose config is indistinguishable from a stock Qwen3-VL checkpoint)",
    ))
}

/// Refuses a generic backbone architecture that lacks the supported model's
/// marker, so a stock checkpoint never loads as the wrong parser.
fn generic_backbone_refusal(architecture: &str, supported: &str) -> Error {
    Error::config(format!(
        "{architecture} is a generic backbone architecture and carries no supported \
         model's marker; only {supported} is built on it — load that model explicitly \
         with AnyPageParserLoadOptions::with_model"
    ))
}

/// Reads the `architectures` array; missing or non-string entries are skipped.
fn config_architectures(config: &Value) -> Vec<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// One-line summary of every supported architecture, for error messages.
fn supported_architectures() -> String {
    let mut entries: Vec<String> = SINGLE_ARCHITECTURE_RULES
        .iter()
        .map(|(architecture, model)| format!("{architecture} ({})", model.name()))
        .collect();
    for (architecture, supported) in [
        (
            INTERNVL_ARCHITECTURE,
            "HPD-Parsing with config.json fork_token_id",
        ),
        (
            QWEN2_VL_ARCHITECTURE,
            "MinerU2.5 with its <|md_start|> tokenizer token",
        ),
        (
            QWEN2_5_VL_ARCHITECTURE,
            "TeleOCR with vision_config.window_size",
        ),
        (
            QWEN3_VL_ARCHITECTURE,
            "WeVisDoc via AnyPageParserLoadOptions::with_model only, as its config \
             matches a stock Qwen3-VL checkpoint",
        ),
        (
            QWEN3_5_ARCHITECTURE,
            "OvisOCR2 with vision_config.model_type=qwen3_5, or Xiaomi-OCR-0 with \
             vision_config.model_type=qwen3_5_vision",
        ),
    ] {
        entries.push(format!("{architecture} ({supported})"));
    }
    entries.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn real_checkpoints_map_to_their_models() {
        // Architecture names no stock family checkpoint uses.
        let unique = [
            (
                "HunYuanVLForConditionalGeneration",
                AnyPageParserModel::HunyuanOcr,
            ),
            ("DeepseekOCRForCausalLM", AnyPageParserModel::JinaOcr),
            (
                "MinerUDiffusionForConditionalGeneration",
                AnyPageParserModel::MinerUDiffusion,
            ),
            ("MonkeyOCRv2ForCausalLM", AnyPageParserModel::MonkeyOcrV2),
            (
                "PaddleOCRVLForConditionalGeneration",
                AnyPageParserModel::PaddleOcrVl,
            ),
            ("GlmOcrForConditionalGeneration", AnyPageParserModel::GlmOcr),
        ];
        for (architecture, expected) in unique {
            assert_eq!(
                detect_model(&json!({ "architectures": [architecture] }), &[]).unwrap(),
                expected,
                "{architecture}"
            );
        }
        // Generic backbone names load only with the supported model's marker.
        let hpd = json!({ "architectures": ["InternVLChatModel"], "fork_token_id": 151679 });
        assert_eq!(
            detect_model(&hpd, &[]).unwrap(),
            AnyPageParserModel::HpdParsing
        );
        let mineru = ["<|vision_start|>".to_string(), "<|md_start|>".to_string()];
        assert_eq!(
            detect_model(
                &json!({ "architectures": ["Qwen2VLForConditionalGeneration"] }),
                &mineru
            )
            .unwrap(),
            AnyPageParserModel::MinerU
        );
        let teleocr = json!({
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "vision_config": { "window_size": 112 },
        });
        assert_eq!(
            detect_model(&teleocr, &[]).unwrap(),
            AnyPageParserModel::TeleOcr
        );
    }

    #[test]
    fn vision_model_type_separates_the_qwen3_5_pair() {
        let pair = |vision: &str| {
            json!({
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "vision_config": { "model_type": vision },
                "text_config": { "model_type": "qwen3_5_text" },
            })
        };
        assert_eq!(
            detect_model(&pair("qwen3_5"), &[]).unwrap(),
            AnyPageParserModel::OvisOcr2
        );
        assert_eq!(
            detect_model(&pair("qwen3_5_vision"), &[]).unwrap(),
            AnyPageParserModel::XiaomiOcr
        );
    }

    #[test]
    fn generic_backbones_without_their_marker_are_rejected() {
        // Plain backbone configs a stock checkpoint would also produce.
        for config in [
            json!({ "architectures": ["InternVLChatModel"] }),
            json!({ "architectures": ["Qwen2VLForConditionalGeneration"] }),
            json!({ "architectures": ["Qwen3VLForConditionalGeneration"] }),
            json!({ "architectures": ["Qwen2_5_VLForConditionalGeneration"] }),
        ] {
            let error = detect_model(&config, &[]).unwrap_err().to_string();
            assert!(error.contains("generic backbone"), "{error}");
            assert!(error.contains("with_model"), "{error}");
        }
    }

    #[test]
    fn unknown_or_ambiguous_configs_error_with_the_supported_list() {
        // Unknown: the error names what was found and what is supported.
        let error = detect_model(&json!({ "architectures": ["LlamaForCausalLM"] }), &[])
            .unwrap_err()
            .to_string();
        assert!(error.contains("\"LlamaForCausalLM\""), "{error}");
        assert!(error.contains("supported architectures"), "{error}");
        assert!(
            error.contains("PaddleOCRVLForConditionalGeneration"),
            "{error}"
        );
        // Ambiguous: the error names every matched model instead of picking one.
        let error = detect_model(
            &json!({
                "architectures": [
                    "HunYuanVLForConditionalGeneration",
                    "GlmOcrForConditionalGeneration"
                ]
            }),
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("ambiguous"), "{error}");
        assert!(error.contains("HunyuanOCR"), "{error}");
        assert!(error.contains("GLM-OCR"), "{error}");
    }
}
