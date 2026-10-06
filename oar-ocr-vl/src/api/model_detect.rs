//! Model detection from a checkpoint directory's `config.json`.

use crate::api::error::Error;
use serde_json::Value;

/// The supported model a configuration identifies. One variant per
/// [`crate::AnyPageParser`] variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectedModel {
    HpdParsing,
    HunyuanOcr,
    JinaOcr,
    MinerU,
    MinerUDiffusion,
    MonkeyOcrV2,
    OvisOcr2,
    WeVisDoc,
    XiaomiOcr,
    PaddleOcrVl,
    GlmOcr,
    TeleOcr,
}

impl DetectedModel {
    /// Human-readable model name used in detection and loading errors.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::HpdParsing => "HPD-Parsing",
            Self::HunyuanOcr => "HunyuanOCR",
            Self::JinaOcr => "jina-ocr-v1",
            Self::MinerU => "MinerU2.5",
            Self::MinerUDiffusion => "MinerU-Diffusion",
            Self::MonkeyOcrV2 => "MonkeyOCRv2",
            Self::OvisOcr2 => "OvisOCR2",
            Self::WeVisDoc => "WeVisDoc",
            Self::XiaomiOcr => "Xiaomi-OCR-0",
            Self::PaddleOcrVl => "PaddleOCR-VL",
            Self::GlmOcr => "GLM-OCR",
            Self::TeleOcr => "TeleOCR",
        }
    }
}

/// `config.json` `architectures` entries that identify exactly one supported
/// model.
const SINGLE_ARCHITECTURE_RULES: &[(&str, DetectedModel)] = &[
    ("InternVLChatModel", DetectedModel::HpdParsing),
    (
        "HunYuanVLForConditionalGeneration",
        DetectedModel::HunyuanOcr,
    ),
    ("DeepseekOCRForCausalLM", DetectedModel::JinaOcr),
    ("Qwen2VLForConditionalGeneration", DetectedModel::MinerU),
    (
        "MinerUDiffusionForConditionalGeneration",
        DetectedModel::MinerUDiffusion,
    ),
    ("MonkeyOCRv2ForCausalLM", DetectedModel::MonkeyOcrV2),
    ("Qwen3VLForConditionalGeneration", DetectedModel::WeVisDoc),
    (
        "PaddleOCRVLForConditionalGeneration",
        DetectedModel::PaddleOcrVl,
    ),
    ("GlmOcrForConditionalGeneration", DetectedModel::GlmOcr),
    ("Qwen2_5_VLForConditionalGeneration", DetectedModel::TeleOcr),
];

/// OvisOCR2 and Xiaomi-OCR-0 both carry `Qwen3_5ForConditionalGeneration`
/// (they share a Qwen3.5 text tower), so their different vision towers
/// disambiguate them through `vision_config.model_type`.
const QWEN3_5_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const OVIS_VISION_MODEL_TYPE: &str = "qwen3_5";
const XIAOMI_VISION_MODEL_TYPE: &str = "qwen3_5_vision";

/// Detects the supported model described by a parsed `config.json`.
///
/// Detection keys on the `architectures` entries; every error names what the
/// config contains and what this crate supports, so an unsupported or
/// ambiguous checkpoint never falls back silently.
pub(crate) fn detect_model(config: &Value) -> Result<DetectedModel, Error> {
    let architectures = config_architectures(config);
    let mut candidates: Vec<DetectedModel> = Vec::new();
    for architecture in &architectures {
        let candidate = if *architecture == QWEN3_5_ARCHITECTURE {
            Some(qwen3_5_model(config)?)
        } else {
            SINGLE_ARCHITECTURE_RULES
                .iter()
                .find(|(arch, _)| arch == architecture)
                .map(|(_, model)| *model)
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

/// Resolves the shared Qwen3.5 architecture by its vision tower.
fn qwen3_5_model(config: &Value) -> Result<DetectedModel, Error> {
    let vision_model_type = config
        .pointer("/vision_config/model_type")
        .and_then(Value::as_str);
    match vision_model_type {
        Some(value) if value == OVIS_VISION_MODEL_TYPE => Ok(DetectedModel::OvisOcr2),
        Some(value) if value == XIAOMI_VISION_MODEL_TYPE => Ok(DetectedModel::XiaomiOcr),
        _ => Err(Error::config(format!(
            "unsupported {QWEN3_5_ARCHITECTURE} config (vision_config.model_type={vision_model_type:?}); \
             expected {OVIS_VISION_MODEL_TYPE:?} (OvisOCR2) or {XIAOMI_VISION_MODEL_TYPE:?} (Xiaomi-OCR-0)"
        ))),
    }
}

/// One-line summary of every supported architecture, for error messages.
fn supported_architectures() -> String {
    let mut entries: Vec<String> = SINGLE_ARCHITECTURE_RULES
        .iter()
        .map(|(architecture, model)| format!("{architecture} ({})", model.name()))
        .collect();
    entries.push(format!(
        "{QWEN3_5_ARCHITECTURE} (OvisOCR2 with vision_config.model_type={OVIS_VISION_MODEL_TYPE}, \
         or Xiaomi-OCR-0 with vision_config.model_type={XIAOMI_VISION_MODEL_TYPE})"
    ));
    entries.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal(architecture: &str) -> Value {
        json!({ "architectures": [architecture] })
    }

    #[test]
    fn detects_each_supported_model() {
        let cases = [
            ("InternVLChatModel", DetectedModel::HpdParsing),
            (
                "HunYuanVLForConditionalGeneration",
                DetectedModel::HunyuanOcr,
            ),
            ("DeepseekOCRForCausalLM", DetectedModel::JinaOcr),
            ("Qwen2VLForConditionalGeneration", DetectedModel::MinerU),
            (
                "MinerUDiffusionForConditionalGeneration",
                DetectedModel::MinerUDiffusion,
            ),
            ("MonkeyOCRv2ForCausalLM", DetectedModel::MonkeyOcrV2),
            ("Qwen3VLForConditionalGeneration", DetectedModel::WeVisDoc),
            (
                "PaddleOCRVLForConditionalGeneration",
                DetectedModel::PaddleOcrVl,
            ),
            ("GlmOcrForConditionalGeneration", DetectedModel::GlmOcr),
            ("Qwen2_5_VLForConditionalGeneration", DetectedModel::TeleOcr),
        ];
        for (architecture, expected) in cases {
            assert_eq!(detect_model(&minimal(architecture)).unwrap(), expected);
        }
        // The model_type field is informational only.
        let mut with_model_type = minimal("InternVLChatModel");
        with_model_type["model_type"] = json!("internvl_chat");
        assert_eq!(
            detect_model(&with_model_type).unwrap(),
            DetectedModel::HpdParsing
        );
        // Repeated entries still identify one model.
        let repeated = json!({ "architectures": ["InternVLChatModel", "InternVLChatModel"] });
        assert_eq!(detect_model(&repeated).unwrap(), DetectedModel::HpdParsing);
    }

    #[test]
    fn vision_model_type_separates_the_qwen3_5_pair() {
        let ovis = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vision_config": { "model_type": "qwen3_5" },
            "text_config": { "model_type": "qwen3_5_text" },
        });
        assert_eq!(detect_model(&ovis).unwrap(), DetectedModel::OvisOcr2);
        let xiaomi = json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vision_config": { "model_type": "qwen3_5_vision" },
            "text_config": { "model_type": "qwen3_5_text" },
        });
        assert_eq!(detect_model(&xiaomi).unwrap(), DetectedModel::XiaomiOcr);
    }

    #[test]
    fn unknown_qwen3_5_vision_tower_names_the_expected_values() {
        let error = detect_model(&json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_config": { "model_type": "siglip" },
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("vision_config.model_type"), "{error}");
        assert!(error.contains("\"qwen3_5\" (OvisOCR2)"), "{error}");
        assert!(
            error.contains("\"qwen3_5_vision\" (Xiaomi-OCR-0)"),
            "{error}"
        );
        let missing = detect_model(&minimal("Qwen3_5ForConditionalGeneration"))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("vision_config.model_type"), "{missing}");
    }

    #[test]
    fn unknown_config_names_what_was_found_and_what_is_supported() {
        let error = detect_model(&json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("\"LlamaForCausalLM\""), "{error}");
        assert!(error.contains("model_type=\"llama\""), "{error}");
        for (architecture, _) in SINGLE_ARCHITECTURE_RULES {
            assert!(error.contains(architecture), "{error}");
        }
        assert!(error.contains("OvisOCR2"), "{error}");
        assert!(error.contains("Xiaomi-OCR-0"), "{error}");
    }

    #[test]
    fn config_without_architectures_is_unsupported() {
        let error = detect_model(&json!({ "model_type": "llama" }))
            .unwrap_err()
            .to_string();
        assert!(error.contains("architectures=[]"), "{error}");
        assert!(error.contains("model_type=\"llama\""), "{error}");

        let missing_model_type = detect_model(&json!({})).unwrap_err().to_string();
        assert!(
            missing_model_type.contains("model_type=\"<missing>\""),
            "{missing_model_type}"
        );
    }

    #[test]
    fn several_supported_models_in_one_config_is_ambiguous() {
        let error = detect_model(&json!({
            "architectures": ["InternVLChatModel", "GlmOcrForConditionalGeneration"],
        }))
        .unwrap_err()
        .to_string();
        assert!(error.contains("ambiguous"), "{error}");
        assert!(error.contains("HPD-Parsing"), "{error}");
        assert!(error.contains("GLM-OCR"), "{error}");
    }
}
