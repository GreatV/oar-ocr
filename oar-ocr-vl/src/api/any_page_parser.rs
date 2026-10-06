//! Model-agnostic complete-page parsing and model-directory loading.

use crate::api::error::Error;
use crate::api::model_detect::{DetectedModel, detect_model};
use crate::api::page_parser::PageParser;
use crate::api::recognition::RecognitionBackend;
use crate::document::page::PageDocument;
use crate::glmocr::GlmOcr;
use crate::hpd_parsing::{HpdGenerationConfig, HpdParsing};
use crate::hunyuanocr::{HunyuanOcr, HunyuanOcrParseOptions};
use crate::jina_ocr::{JinaOcr, JinaOcrParseOptions};
use crate::layout::LayoutSource;
use crate::mineru::{MinerU, MinerUParseOptions};
use crate::mineru_diffusion::{MinerUDiffusion, MinerUDiffusionParseOptions};
use crate::monkeyocrv2::{MonkeyOcrV2, MonkeyOcrV2ParseOptions};
use crate::ovisocr2::{OvisOcr2, OvisOcr2ParseOptions};
use crate::paddleocr_vl::PaddleOcrVl;
use crate::pipeline::page_parser::{LayoutPageParser, LayoutPageParserOptions};
use crate::pp_doclayout::PpDocLayout;
use crate::runtime::checkpoint::load_json_config;
use crate::teleocr::TeleOcr;
use crate::wevisdoc::{WeVisDoc, WeVisDocParseOptions};
use crate::xiaomi_ocr::{XiaomiOcr, XiaomiOcrParseOptions};
use candle_core::Device;
use image::RgbImage;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Per-page knobs shared by every parser behind [`AnyPageParser`].
///
/// Each knob is optional: `None` keeps the wrapped parser's own default, and
/// `Some` overrides only that knob, leaving every model-specific option at its
/// default.
///
/// - `max_new_tokens` maps to each model's generation budget:
///   `HpdGenerationConfig::max_new_tokens`, the `max_new_tokens` fields of the
///   model-native `*ParseOptions`, `MinerUParseOptions::max_tokens`, the
///   block-diffusion analog `MinerUDiffusionParseOptions::generation.gen_length`,
///   and `DocParserConfig::max_tokens` for the layout-composed models.
/// - `region_batch_size` overrides same-task region batching where the parser
///   has it (`MinerUParseOptions::region_batch_size` and
///   [`LayoutPageParserOptions::region_batch_size`], zero treated as one) and
///   is ignored by parsers without region batching.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AnyPageParserOptions {
    /// Maximum number of tokens generated per page or region. `None` keeps the
    /// model default.
    pub max_new_tokens: Option<usize>,
    /// Maximum same-task region batch size. `None` keeps the model default;
    /// parsers without region batching ignore this knob.
    pub region_batch_size: Option<usize>,
}

impl AnyPageParserOptions {
    /// Override the generation budget for this page.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = Some(max_new_tokens);
        self
    }

    /// Override the same-task region batch size. Zero is treated as one.
    pub fn with_region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = Some(size);
        self
    }
}

/// Directory-loading options for [`AnyPageParser::from_dir_with_options`].
///
/// The layout-composed models (PaddleOCR-VL, GLM-OCR, TeleOCR) combine their
/// recognition backbone with an external PP-DocLayout detector, so loading
/// them needs a PP-DocLayout checkpoint directory in `layout_dir`. Every
/// other model ignores it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AnyPageParserLoadOptions {
    /// PP-DocLayout checkpoint directory used by the layout-composed models.
    pub layout_dir: Option<PathBuf>,
}

impl AnyPageParserLoadOptions {
    /// Set the PP-DocLayout directory used by the layout-composed models.
    pub fn with_layout_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.layout_dir = Some(dir.into());
        self
    }
}

/// A complete-page parser that dispatches to any of the crate's parsers.
///
/// Every supported model parses through the same [`PageParser`] contract
/// behind this enum, so callers can pick a parser at runtime — from a config
/// file, CLI flag, or benchmark manifest — without writing per-model dispatch.
/// Convert an already-loaded model with [`From`], or load one from its model
/// directory with [`from_dir`](Self::from_dir), which detects the model from
/// its `config.json`. `AnyPageParserOptions` carries the knobs the parsers
/// share and leaves every other model-specific option at its default.
///
/// ```no_run
/// use oar_ocr_vl::{
///     AnyPageParser, AnyPageParserOptions, LayoutPageParser, PageParser,
///     PaddleOcrVl, PpDocLayout,
/// };
/// use oar_ocr_vl::utils::{image::load_image, parse_device};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = parse_device("cpu")?;
/// let layout = PpDocLayout::from_dir("PaddlePaddle/PP-DocLayoutV3_safetensors", device.clone())?;
/// let backend = PaddleOcrVl::from_dir("PaddlePaddle/PaddleOCR-VL-1.5", device)?;
/// let parser = AnyPageParser::from(LayoutPageParser::new(layout, backend));
/// let image = load_image("document.jpg")?;
/// let page = parser.parse_page(&image, &AnyPageParserOptions::default())?;
/// # let _ = page;
/// # Ok(())
/// # }
/// ```
#[non_exhaustive]
pub enum AnyPageParser {
    /// HPD-Parsing model-native hierarchical full-page parsing.
    HpdParsing(HpdParsing),
    /// HunyuanOCR model-native prompt-driven parsing.
    HunyuanOcr(HunyuanOcr),
    /// jina-ocr-v1 end-to-end page parsing.
    JinaOcr(JinaOcr),
    /// MinerU2.5 / MinerU2.5-Pro two-step parsing.
    MinerU(MinerU),
    /// MinerU-Diffusion block-diffusion two-step parsing.
    MinerUDiffusion(MinerUDiffusion),
    /// MonkeyOCRv2 model-native parsing.
    MonkeyOcrV2(MonkeyOcrV2),
    /// OvisOCR2 end-to-end page parsing.
    OvisOcr2(OvisOcr2),
    /// WeVisDoc end-to-end page parsing.
    WeVisDoc(WeVisDoc),
    /// Xiaomi-OCR-0 end-to-end page parsing.
    XiaomiOcr(XiaomiOcr),
    /// PaddleOCR-VL over a native PP-DocLayout layout source.
    PaddleOcrVl(LayoutPageParser<PpDocLayout, PaddleOcrVl>),
    /// GLM-OCR over a native PP-DocLayout layout source.
    GlmOcr(LayoutPageParser<PpDocLayout, GlmOcr>),
    /// TeleOCR over a native PP-DocLayout layout source.
    TeleOcr(LayoutPageParser<PpDocLayout, TeleOcr>),
}

impl From<HpdParsing> for AnyPageParser {
    fn from(model: HpdParsing) -> Self {
        Self::HpdParsing(model)
    }
}

impl From<HunyuanOcr> for AnyPageParser {
    fn from(model: HunyuanOcr) -> Self {
        Self::HunyuanOcr(model)
    }
}

impl From<JinaOcr> for AnyPageParser {
    fn from(model: JinaOcr) -> Self {
        Self::JinaOcr(model)
    }
}

impl From<MinerU> for AnyPageParser {
    fn from(model: MinerU) -> Self {
        Self::MinerU(model)
    }
}

impl From<MinerUDiffusion> for AnyPageParser {
    fn from(model: MinerUDiffusion) -> Self {
        Self::MinerUDiffusion(model)
    }
}

impl From<MonkeyOcrV2> for AnyPageParser {
    fn from(model: MonkeyOcrV2) -> Self {
        Self::MonkeyOcrV2(model)
    }
}

impl From<OvisOcr2> for AnyPageParser {
    fn from(model: OvisOcr2) -> Self {
        Self::OvisOcr2(model)
    }
}

impl From<WeVisDoc> for AnyPageParser {
    fn from(model: WeVisDoc) -> Self {
        Self::WeVisDoc(model)
    }
}

impl From<XiaomiOcr> for AnyPageParser {
    fn from(model: XiaomiOcr) -> Self {
        Self::XiaomiOcr(model)
    }
}

impl From<LayoutPageParser<PpDocLayout, PaddleOcrVl>> for AnyPageParser {
    fn from(model: LayoutPageParser<PpDocLayout, PaddleOcrVl>) -> Self {
        Self::PaddleOcrVl(model)
    }
}

impl From<LayoutPageParser<PpDocLayout, GlmOcr>> for AnyPageParser {
    fn from(model: LayoutPageParser<PpDocLayout, GlmOcr>) -> Self {
        Self::GlmOcr(model)
    }
}

impl From<LayoutPageParser<PpDocLayout, TeleOcr>> for AnyPageParser {
    fn from(model: LayoutPageParser<PpDocLayout, TeleOcr>) -> Self {
        Self::TeleOcr(model)
    }
}

impl AnyPageParser {
    /// Detects and loads the parser stored in a model directory.
    ///
    /// Detection reads the directory's `config.json`: the `architectures`
    /// entries identify the model, with `vision_config.model_type` separating
    /// the two Qwen3.5-based models (OvisOCR2 and Xiaomi-OCR-0 share a text
    /// tower). The matching parser then loads through its own `from_dir` with
    /// its own defaults. Unsupported or ambiguous configurations return an
    /// error naming what was found and the supported architectures; there is
    /// no fallback. Layout-composed models (PaddleOCR-VL, GLM-OCR, TeleOCR)
    /// additionally need a PP-DocLayout directory, which
    /// [`from_dir_with_options`](Self::from_dir_with_options) accepts.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_options(model_dir, device, &AnyPageParserLoadOptions::default())
    }

    /// Detects and loads a parser from a model directory with loading options.
    ///
    /// See [`from_dir`](Self::from_dir) for how the model is detected; the
    /// options only add the PP-DocLayout directory required by the
    /// layout-composed models.
    ///
    /// ```no_run
    /// use candle_core::Device;
    /// use oar_ocr_vl::{AnyPageParser, AnyPageParserLoadOptions};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let options = AnyPageParserLoadOptions::default()
    ///     .with_layout_dir("PaddlePaddle/PP-DocLayoutV3_safetensors");
    /// let parser =
    ///     AnyPageParser::from_dir_with_options("PaddlePaddle/PaddleOCR-VL-1.5", Device::Cpu, &options)?;
    /// # let _ = parser;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_dir_with_options(
        model_dir: impl AsRef<Path>,
        device: Device,
        options: &AnyPageParserLoadOptions,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        if !config_path.is_file() {
            return Err(Error::config(format!(
                "model directory {} has no config.json to detect a model from",
                model_dir.display()
            )));
        }
        let config: Value = load_json_config(&config_path, "AnyPageParser", "config.json")
            .map_err(|error| {
                Error::config(format!("{error}; model directory: {}", model_dir.display()))
            })?;
        let detected = detect_model(&config)?;
        Ok(match detected {
            DetectedModel::HpdParsing => HpdParsing::from_dir(model_dir, device)?.into(),
            DetectedModel::HunyuanOcr => HunyuanOcr::from_dir(model_dir, device)?.into(),
            DetectedModel::JinaOcr => JinaOcr::from_dir(model_dir, device)?.into(),
            DetectedModel::MinerU => MinerU::from_dir(model_dir, device)?.into(),
            DetectedModel::MinerUDiffusion => MinerUDiffusion::from_dir(model_dir, device)?.into(),
            DetectedModel::MonkeyOcrV2 => MonkeyOcrV2::from_dir(model_dir, device)?.into(),
            DetectedModel::OvisOcr2 => OvisOcr2::from_dir(model_dir, device)?.into(),
            DetectedModel::WeVisDoc => WeVisDoc::from_dir(model_dir, device)?.into(),
            DetectedModel::XiaomiOcr => XiaomiOcr::from_dir(model_dir, device)?.into(),
            DetectedModel::PaddleOcrVl => LayoutPageParser::new(
                PpDocLayout::from_dir(required_layout_dir(options, detected)?, device.clone())?,
                PaddleOcrVl::from_dir(model_dir, device)?,
            )
            .into(),
            DetectedModel::GlmOcr => LayoutPageParser::new(
                PpDocLayout::from_dir(required_layout_dir(options, detected)?, device.clone())?,
                GlmOcr::from_dir(model_dir, device)?,
            )
            .into(),
            DetectedModel::TeleOcr => LayoutPageParser::new(
                PpDocLayout::from_dir(required_layout_dir(options, detected)?, device.clone())?,
                TeleOcr::from_dir(model_dir, device)?,
            )
            .into(),
        })
    }
}

/// Returns the configured layout directory for a layout-composed model,
/// before any weights load, or explains how to provide one.
fn required_layout_dir(
    options: &AnyPageParserLoadOptions,
    detected: DetectedModel,
) -> Result<&Path, Error> {
    options.layout_dir.as_deref().ok_or_else(|| {
        Error::config(format!(
            "{} parses with an external layout detector; pass a PP-DocLayout directory \
             with AnyPageParserLoadOptions::with_layout_dir",
            detected.name()
        ))
    })
}

impl PageParser for AnyPageParser {
    type Options = AnyPageParserOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        match self {
            Self::HpdParsing(model) => model.parse_page(image, &hpd_options(options)),
            Self::HunyuanOcr(model) => model.parse_page(image, &hunyuan_options(options)),
            Self::JinaOcr(model) => model.parse_page(image, &jina_options(options)),
            Self::MinerU(model) => model.parse_page(image, &mineru_options(options)),
            Self::MinerUDiffusion(model) => {
                model.parse_page(image, &mineru_diffusion_options(options))
            }
            Self::MonkeyOcrV2(model) => model.parse_page(image, &monkeyocrv2_options(options)),
            Self::OvisOcr2(model) => model.parse_page(image, &ovisocr2_options(options)),
            Self::WeVisDoc(model) => model.parse_page(image, &wevisdoc_options(options)),
            Self::XiaomiOcr(model) => model.parse_page(image, &xiaomi_options(options)),
            Self::PaddleOcrVl(model) => model.parse_page(image, &layout_options(model, options)),
            Self::GlmOcr(model) => model.parse_page(image, &layout_options(model, options)),
            Self::TeleOcr(model) => model.parse_page(image, &layout_options(model, options)),
        }
    }
}

fn hpd_options(options: &AnyPageParserOptions) -> HpdGenerationConfig {
    let mut model_options = HpdGenerationConfig::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

fn hunyuan_options(options: &AnyPageParserOptions) -> HunyuanOcrParseOptions {
    let mut model_options = HunyuanOcrParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

fn jina_options(options: &AnyPageParserOptions) -> JinaOcrParseOptions {
    let mut model_options = JinaOcrParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

fn mineru_options(options: &AnyPageParserOptions) -> MinerUParseOptions {
    let mut model_options = MinerUParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_tokens = max_new_tokens;
    }
    if let Some(region_batch_size) = options.region_batch_size {
        model_options.region_batch_size = region_batch_size;
    }
    model_options
}

fn mineru_diffusion_options(options: &AnyPageParserOptions) -> MinerUDiffusionParseOptions {
    let mut model_options = MinerUDiffusionParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.generation.gen_length = max_new_tokens;
    }
    model_options
}

fn monkeyocrv2_options(options: &AnyPageParserOptions) -> MonkeyOcrV2ParseOptions {
    let mut model_options = MonkeyOcrV2ParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

fn ovisocr2_options(options: &AnyPageParserOptions) -> OvisOcr2ParseOptions {
    let mut model_options = OvisOcr2ParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

fn wevisdoc_options(options: &AnyPageParserOptions) -> WeVisDocParseOptions {
    let mut model_options = WeVisDocParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

fn xiaomi_options(options: &AnyPageParserOptions) -> XiaomiOcrParseOptions {
    let mut model_options = XiaomiOcrParseOptions::default();
    if let Some(max_new_tokens) = options.max_new_tokens {
        model_options.max_new_tokens = max_new_tokens;
    }
    model_options
}

/// Starts from the parser's own configuration so `None` knobs keep the
/// settings chosen at construction time, then applies the unified overrides.
fn layout_options<L: LayoutSource, B: RecognitionBackend>(
    parser: &LayoutPageParser<L, B>,
    options: &AnyPageParserOptions,
) -> LayoutPageParserOptions {
    let mut config = parser.config().clone();
    if let Some(max_new_tokens) = options.max_new_tokens {
        config.max_tokens = max_new_tokens;
    }
    let mut model_options = LayoutPageParserOptions::default().with_config(config);
    if let Some(region_batch_size) = options.region_batch_size {
        model_options = model_options.with_region_batch_size(region_batch_size);
    }
    model_options
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::generation::GenerationOptions;
    use crate::api::recognition::RecognitionTask;
    use crate::doc_parser::DocParserConfig;
    use crate::layout::LayoutDetections;
    use crate::monkeyocrv2::MonkeyOcrV2Task;
    use std::cell::Cell;

    #[test]
    fn load_options_carry_an_optional_layout_directory() {
        let options = AnyPageParserLoadOptions::default();
        assert!(options.layout_dir.is_none());
        let options = options.with_layout_dir("some/layout_dir");
        assert_eq!(
            options.layout_dir.as_deref(),
            Some(Path::new("some/layout_dir"))
        );
    }

    fn model_dir_with(config: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config).unwrap();
        dir
    }

    #[test]
    fn from_dir_reports_a_missing_layout_directory_before_loading() {
        let dir = model_dir_with(
            r#"{"architectures": ["PaddleOCRVLForConditionalGeneration"], "model_type": "paddleocr_vl"}"#,
        );
        let error = AnyPageParser::from_dir(dir.path(), Device::Cpu)
            .err()
            .expect("detection should fail")
            .to_string();
        assert!(error.contains("PaddleOCR-VL"), "{error}");
        assert!(error.contains("PP-DocLayout"), "{error}");
        assert!(error.contains("with_layout_dir"), "{error}");
    }

    #[test]
    fn from_dir_names_the_directory_when_config_json_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let error = AnyPageParser::from_dir(dir.path(), Device::Cpu)
            .err()
            .expect("detection should fail")
            .to_string();
        assert!(error.contains("no config.json"), "{error}");
        assert!(error.contains(&dir.path().display().to_string()), "{error}");
    }

    #[test]
    fn from_dir_rejects_unknown_configs_without_fallback() {
        let dir = model_dir_with(r#"{"architectures": ["LlamaForCausalLM"]}"#);
        let error = AnyPageParser::from_dir(dir.path(), Device::Cpu)
            .err()
            .expect("detection should fail")
            .to_string();
        assert!(error.contains("LlamaForCausalLM"), "{error}");
        assert!(error.contains("supported architectures"), "{error}");
        assert!(
            error.contains("PaddleOCRVLForConditionalGeneration"),
            "{error}"
        );
    }

    #[test]
    fn from_dir_loads_a_real_checkpoint_when_available() {
        let Some(model_dir) = std::env::var_os("ANY_PAGE_PARSER_MODEL_DIR") else {
            eprintln!("skipping: ANY_PAGE_PARSER_MODEL_DIR is not set");
            return;
        };
        let mut options = AnyPageParserLoadOptions::default();
        if let Some(layout_dir) = std::env::var_os("ANY_PAGE_PARSER_LAYOUT_DIR") {
            options = options.with_layout_dir(layout_dir);
        }
        let parser =
            AnyPageParser::from_dir_with_options(model_dir, Device::Cpu, &options).unwrap();
        let _ = parser;
    }

    #[test]
    fn none_knobs_reuse_each_models_default() {
        let unified = AnyPageParserOptions::default();
        assert!(unified.max_new_tokens.is_none());
        assert!(unified.region_batch_size.is_none());

        let produced = hpd_options(&unified);
        let default = HpdGenerationConfig::default();
        assert_eq!(produced.max_new_tokens, default.max_new_tokens);
        assert_eq!(produced.use_mtp, default.use_mtp);
        assert_eq!(
            produced.num_speculative_tokens,
            default.num_speculative_tokens
        );
        assert_eq!(produced.max_active_branches, default.max_active_branches);

        let produced = hunyuan_options(&unified);
        let default = HunyuanOcrParseOptions::default();
        assert_eq!(produced.max_new_tokens, default.max_new_tokens);
        assert_eq!(produced.prompt, default.prompt);

        let produced = jina_options(&unified);
        assert_eq!(
            produced.max_new_tokens,
            JinaOcrParseOptions::default().max_new_tokens
        );

        let produced = mineru_options(&unified);
        let default = MinerUParseOptions::default();
        assert_eq!(produced.max_tokens, default.max_tokens);
        assert_eq!(produced.region_batch_size, default.region_batch_size);
        assert_eq!(produced.min_image_edge, default.min_image_edge);
        assert_eq!(produced.max_image_edge_ratio, default.max_image_edge_ratio);

        let produced = mineru_diffusion_options(&unified);
        let default = MinerUDiffusionParseOptions::default();
        assert_eq!(
            produced.generation.gen_length,
            default.generation.gen_length
        );
        assert_eq!(
            produced.generation.block_length,
            default.generation.block_length
        );
        assert_eq!(
            produced.generation.denoising_steps,
            default.generation.denoising_steps
        );
        assert_eq!(produced.min_image_edge, default.min_image_edge);
        assert_eq!(produced.max_image_edge_ratio, default.max_image_edge_ratio);

        let produced = monkeyocrv2_options(&unified);
        let default = MonkeyOcrV2ParseOptions::default();
        assert_eq!(produced.max_new_tokens, default.max_new_tokens);
        assert_eq!(produced.task, default.task);

        let produced = ovisocr2_options(&unified);
        let default = OvisOcr2ParseOptions::default();
        assert_eq!(produced.max_new_tokens, default.max_new_tokens);
        assert_eq!(produced.keep_image_tags, default.keep_image_tags);

        let produced = wevisdoc_options(&unified);
        assert_eq!(
            produced.max_new_tokens,
            WeVisDocParseOptions::default().max_new_tokens
        );

        let produced = xiaomi_options(&unified);
        assert_eq!(
            produced.max_new_tokens,
            XiaomiOcrParseOptions::default().max_new_tokens
        );
    }

    #[test]
    fn knobs_override_only_the_mapped_fields() {
        let unified = AnyPageParserOptions::default()
            .with_max_new_tokens(777)
            .with_region_batch_size(5);

        let produced = hpd_options(&unified);
        let default = HpdGenerationConfig::default();
        assert_eq!(produced.max_new_tokens, 777);
        assert_eq!(produced.use_mtp, default.use_mtp);
        assert_eq!(
            produced.num_speculative_tokens,
            default.num_speculative_tokens
        );
        assert_eq!(produced.max_active_branches, default.max_active_branches);

        let produced = hunyuan_options(&unified);
        assert_eq!(produced.max_new_tokens, 777);
        assert_eq!(produced.prompt, HunyuanOcrParseOptions::default().prompt);

        assert_eq!(jina_options(&unified).max_new_tokens, 777);

        let produced = mineru_options(&unified);
        let default = MinerUParseOptions::default();
        assert_eq!(produced.max_tokens, 777);
        assert_eq!(produced.region_batch_size, 5);
        assert_eq!(produced.min_image_edge, default.min_image_edge);
        assert_eq!(produced.max_image_edge_ratio, default.max_image_edge_ratio);

        let produced = mineru_diffusion_options(&unified);
        let default = MinerUDiffusionParseOptions::default();
        assert_eq!(produced.generation.gen_length, 777);
        assert_eq!(
            produced.generation.block_length,
            default.generation.block_length
        );
        assert_eq!(
            produced.generation.denoising_steps,
            default.generation.denoising_steps
        );
        assert_eq!(produced.min_image_edge, default.min_image_edge);
        assert_eq!(produced.max_image_edge_ratio, default.max_image_edge_ratio);

        let produced = monkeyocrv2_options(&unified);
        assert_eq!(produced.max_new_tokens, 777);
        assert_eq!(produced.task, MonkeyOcrV2Task::EndToEnd);

        let produced = ovisocr2_options(&unified);
        assert_eq!(produced.max_new_tokens, 777);
        assert!(!produced.keep_image_tags);

        assert_eq!(wevisdoc_options(&unified).max_new_tokens, 777);
        assert_eq!(xiaomi_options(&unified).max_new_tokens, 777);
    }

    struct EmptyLayout;

    impl LayoutSource for EmptyLayout {
        fn detect(&self, _image: &RgbImage) -> Result<LayoutDetections, Error> {
            Ok(LayoutDetections::new(Vec::new()))
        }
    }

    #[derive(Default)]
    struct RecordingBackend {
        max_new_tokens: Cell<usize>,
    }

    impl RecognitionBackend for RecordingBackend {
        fn recognize(
            &self,
            _image: RgbImage,
            _task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            Ok(String::new())
        }

        fn recognize_with_options(
            &self,
            _image: RgbImage,
            _task: RecognitionTask,
            options: &GenerationOptions,
        ) -> Result<String, Error> {
            self.max_new_tokens.set(options.max_new_tokens);
            Ok("whole page".to_string())
        }
    }

    fn recording_parser() -> LayoutPageParser<EmptyLayout, RecordingBackend> {
        LayoutPageParser::with_config(
            EmptyLayout,
            RecordingBackend::default(),
            DocParserConfig {
                max_tokens: 1234,
                ..Default::default()
            },
        )
        .with_region_batch_size(3)
    }

    #[test]
    fn layout_options_keep_the_parser_configuration_by_default() {
        let parser = recording_parser();
        let produced = layout_options(&parser, &AnyPageParserOptions::default());
        assert_eq!(produced.config.as_ref().unwrap().max_tokens, 1234);
        assert_eq!(produced.config.as_ref().unwrap().crop_pad_ratio, 0.0);
        assert!(produced.region_batch_size.is_none());

        let produced = layout_options(
            &parser,
            &AnyPageParserOptions::default()
                .with_max_new_tokens(777)
                .with_region_batch_size(5),
        );
        assert_eq!(produced.config.as_ref().unwrap().max_tokens, 777);
        // The parser's other configuration settings survive the override.
        assert_eq!(produced.config.as_ref().unwrap().crop_pad_ratio, 0.0);
        assert_eq!(produced.region_batch_size, Some(5));
    }

    #[test]
    fn layout_options_reach_a_real_parse() {
        let parser = recording_parser();
        let image = RgbImage::new(100, 100);
        let page = parser
            .parse_page(
                &image,
                &layout_options(&parser, &AnyPageParserOptions::default()),
            )
            .unwrap();
        assert_eq!(page.blocks.len(), 1);
        assert_eq!(parser.backend().max_new_tokens.get(), 1234);

        parser
            .parse_page(
                &image,
                &layout_options(
                    &parser,
                    &AnyPageParserOptions::default().with_max_new_tokens(555),
                ),
            )
            .unwrap();
        assert_eq!(parser.backend().max_new_tokens.get(), 555);
    }
}
