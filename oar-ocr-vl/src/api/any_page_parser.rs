//! Model-agnostic complete-page parsing and model-directory loading.

#[cfg(feature = "auto-download")]
use crate::api::download::AnyPageParserPretrainedOptions;
use crate::api::error::Error;
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
use crate::teleocr::TeleOcr;
use crate::wevisdoc::{WeVisDoc, WeVisDocParseOptions};
use crate::xiaomi_ocr::{XiaomiOcr, XiaomiOcrParseOptions};
use candle_core::Device;
use image::RgbImage;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

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

/// The checkpoint repository a model directory holds, one variant per
/// supported repo.
///
/// The [`as_str`](Self::as_str) IDs are the Hugging Face repo IDs exactly as
/// published; a new protocol version gets a new ID only when it ships as a
/// new repo, and the same IDs will later drive auto-download. Loading from a
/// directory always names the model explicitly, because most of these models
/// are fine-tunes whose checkpoint configs match their public base models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnyPageParserModel {
    /// PaddlePaddle/HPD-Parsing.
    HpdParsing,
    /// tencent/HunyuanOCR (1.5 and 1.0 revisions).
    HunyuanOcr,
    /// jinaai/jina-ocr-v1.
    JinaOcr,
    /// opendatalab/MinerU2.5-2509-1.2B.
    MinerU2509,
    /// opendatalab/MinerU2.5-Pro-2605-1.2B.
    MinerUPro,
    /// opendatalab/MinerU-Diffusion-V1-0320-2.5B.
    MinerUDiffusion,
    /// zenosai/MonkeyOCRv2-S-Parsing.
    MonkeyOcrV2S,
    /// zenosai/MonkeyOCRv2-B-Parsing.
    MonkeyOcrV2B,
    /// ATH-MaaS/OvisOCR2.
    OvisOcr2,
    /// tencent/WeVisDoc-2B.
    WeVisDoc2B,
    /// tencent/WeVisDoc-4B.
    WeVisDoc4B,
    /// SeerRay-Lab/Xiaomi-OCR-0.
    XiaomiOcr,
    /// PaddlePaddle/PaddleOCR-VL.
    PaddleOcrVl,
    /// PaddlePaddle/PaddleOCR-VL-1.5.
    PaddleOcrVl1_5,
    /// PaddlePaddle/PaddleOCR-VL-1.6.
    PaddleOcrVl1_6,
    /// zai-org/GLM-OCR.
    GlmOcr,
    /// XingChen-AGI/TeleOCR.
    TeleOcr,
}

/// The canonical ID table backing [`AnyPageParserModel`] string conversions.
const MODEL_IDS: &[(&str, AnyPageParserModel)] = &[
    ("PaddlePaddle/HPD-Parsing", AnyPageParserModel::HpdParsing),
    ("tencent/HunyuanOCR", AnyPageParserModel::HunyuanOcr),
    ("jinaai/jina-ocr-v1", AnyPageParserModel::JinaOcr),
    (
        "opendatalab/MinerU2.5-2509-1.2B",
        AnyPageParserModel::MinerU2509,
    ),
    (
        "opendatalab/MinerU2.5-Pro-2605-1.2B",
        AnyPageParserModel::MinerUPro,
    ),
    (
        "opendatalab/MinerU-Diffusion-V1-0320-2.5B",
        AnyPageParserModel::MinerUDiffusion,
    ),
    (
        "zenosai/MonkeyOCRv2-S-Parsing",
        AnyPageParserModel::MonkeyOcrV2S,
    ),
    (
        "zenosai/MonkeyOCRv2-B-Parsing",
        AnyPageParserModel::MonkeyOcrV2B,
    ),
    ("ATH-MaaS/OvisOCR2", AnyPageParserModel::OvisOcr2),
    ("tencent/WeVisDoc-2B", AnyPageParserModel::WeVisDoc2B),
    ("tencent/WeVisDoc-4B", AnyPageParserModel::WeVisDoc4B),
    ("SeerRay-Lab/Xiaomi-OCR-0", AnyPageParserModel::XiaomiOcr),
    ("PaddlePaddle/PaddleOCR-VL", AnyPageParserModel::PaddleOcrVl),
    (
        "PaddlePaddle/PaddleOCR-VL-1.5",
        AnyPageParserModel::PaddleOcrVl1_5,
    ),
    (
        "PaddlePaddle/PaddleOCR-VL-1.6",
        AnyPageParserModel::PaddleOcrVl1_6,
    ),
    ("zai-org/GLM-OCR", AnyPageParserModel::GlmOcr),
    ("XingChen-AGI/TeleOCR", AnyPageParserModel::TeleOcr),
];

impl AnyPageParserModel {
    /// The canonical Hugging Face repo ID, for manifests and CLIs.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HpdParsing => "PaddlePaddle/HPD-Parsing",
            Self::HunyuanOcr => "tencent/HunyuanOCR",
            Self::JinaOcr => "jinaai/jina-ocr-v1",
            Self::MinerU2509 => "opendatalab/MinerU2.5-2509-1.2B",
            Self::MinerUPro => "opendatalab/MinerU2.5-Pro-2605-1.2B",
            Self::MinerUDiffusion => "opendatalab/MinerU-Diffusion-V1-0320-2.5B",
            Self::MonkeyOcrV2S => "zenosai/MonkeyOCRv2-S-Parsing",
            Self::MonkeyOcrV2B => "zenosai/MonkeyOCRv2-B-Parsing",
            Self::OvisOcr2 => "ATH-MaaS/OvisOCR2",
            Self::WeVisDoc2B => "tencent/WeVisDoc-2B",
            Self::WeVisDoc4B => "tencent/WeVisDoc-4B",
            Self::XiaomiOcr => "SeerRay-Lab/Xiaomi-OCR-0",
            Self::PaddleOcrVl => "PaddlePaddle/PaddleOCR-VL",
            Self::PaddleOcrVl1_5 => "PaddlePaddle/PaddleOCR-VL-1.5",
            Self::PaddleOcrVl1_6 => "PaddlePaddle/PaddleOCR-VL-1.6",
            Self::GlmOcr => "zai-org/GLM-OCR",
            Self::TeleOcr => "XingChen-AGI/TeleOCR",
        }
    }
}

impl fmt::Display for AnyPageParserModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AnyPageParserModel {
    type Err = Error;

    /// Parses a Hugging Face repo ID. Hub IDs are case-insensitive, so the
    /// comparison is too; an unknown ID lists every supported one.
    fn from_str(id: &str) -> Result<Self, Self::Err> {
        MODEL_IDS
            .iter()
            .find(|(known, _)| known.eq_ignore_ascii_case(id))
            .map(|(_, model)| *model)
            .ok_or_else(|| {
                let valid = MODEL_IDS
                    .iter()
                    .map(|(known, _)| *known)
                    .collect::<Vec<_>>()
                    .join(", ");
                Error::config(format!("unknown model id {id:?}; valid ids: {valid}"))
            })
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
    HpdParsing(Box<HpdParsing>),
    /// HunyuanOCR model-native prompt-driven parsing.
    HunyuanOcr(Box<HunyuanOcr>),
    /// jina-ocr-v1 end-to-end page parsing.
    JinaOcr(Box<JinaOcr>),
    /// MinerU2.5 / MinerU2.5-Pro two-step parsing.
    MinerU(Box<MinerU>),
    /// MinerU-Diffusion block-diffusion two-step parsing.
    MinerUDiffusion(Box<MinerUDiffusion>),
    /// MonkeyOCRv2 model-native parsing.
    MonkeyOcrV2(Box<MonkeyOcrV2>),
    /// OvisOCR2 end-to-end page parsing.
    OvisOcr2(Box<OvisOcr2>),
    /// WeVisDoc end-to-end page parsing.
    WeVisDoc(Box<WeVisDoc>),
    /// Xiaomi-OCR-0 end-to-end page parsing.
    XiaomiOcr(Box<XiaomiOcr>),
    /// PaddleOCR-VL over a native PP-DocLayout layout source.
    PaddleOcrVl(Box<LayoutPageParser<PpDocLayout, PaddleOcrVl>>),
    /// GLM-OCR over a native PP-DocLayout layout source.
    GlmOcr(Box<LayoutPageParser<PpDocLayout, GlmOcr>>),
    /// TeleOCR over a native PP-DocLayout layout source.
    TeleOcr(Box<LayoutPageParser<PpDocLayout, TeleOcr>>),
}

impl From<HpdParsing> for AnyPageParser {
    fn from(model: HpdParsing) -> Self {
        Self::HpdParsing(Box::new(model))
    }
}

impl From<HunyuanOcr> for AnyPageParser {
    fn from(model: HunyuanOcr) -> Self {
        Self::HunyuanOcr(Box::new(model))
    }
}

impl From<JinaOcr> for AnyPageParser {
    fn from(model: JinaOcr) -> Self {
        Self::JinaOcr(Box::new(model))
    }
}

impl From<MinerU> for AnyPageParser {
    fn from(model: MinerU) -> Self {
        Self::MinerU(Box::new(model))
    }
}

impl From<MinerUDiffusion> for AnyPageParser {
    fn from(model: MinerUDiffusion) -> Self {
        Self::MinerUDiffusion(Box::new(model))
    }
}

impl From<MonkeyOcrV2> for AnyPageParser {
    fn from(model: MonkeyOcrV2) -> Self {
        Self::MonkeyOcrV2(Box::new(model))
    }
}

impl From<OvisOcr2> for AnyPageParser {
    fn from(model: OvisOcr2) -> Self {
        Self::OvisOcr2(Box::new(model))
    }
}

impl From<WeVisDoc> for AnyPageParser {
    fn from(model: WeVisDoc) -> Self {
        Self::WeVisDoc(Box::new(model))
    }
}

impl From<XiaomiOcr> for AnyPageParser {
    fn from(model: XiaomiOcr) -> Self {
        Self::XiaomiOcr(Box::new(model))
    }
}

impl From<LayoutPageParser<PpDocLayout, PaddleOcrVl>> for AnyPageParser {
    fn from(model: LayoutPageParser<PpDocLayout, PaddleOcrVl>) -> Self {
        Self::PaddleOcrVl(Box::new(model))
    }
}

impl From<LayoutPageParser<PpDocLayout, GlmOcr>> for AnyPageParser {
    fn from(model: LayoutPageParser<PpDocLayout, GlmOcr>) -> Self {
        Self::GlmOcr(Box::new(model))
    }
}

impl From<LayoutPageParser<PpDocLayout, TeleOcr>> for AnyPageParser {
    fn from(model: LayoutPageParser<PpDocLayout, TeleOcr>) -> Self {
        Self::TeleOcr(Box::new(model))
    }
}

impl AnyPageParser {
    /// Loads the named parser from a model directory.
    ///
    /// The model is always named explicitly by its Hugging Face repo ID (see
    /// [`AnyPageParserModel`]) — most supported models are fine-tunes whose
    /// checkpoint configs match their public base models, so the directory
    /// cannot be identified reliably on its own. The parser loads through its own
    /// `from_dir` with its own defaults. Layout-composed models (PaddleOCR-VL,
    /// GLM-OCR, TeleOCR) additionally need a PP-DocLayout directory, which
    /// [`from_dir_with_options`](Self::from_dir_with_options) accepts.
    pub fn from_dir(
        model: AnyPageParserModel,
        model_dir: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self, Error> {
        Self::from_dir_with_options(
            model,
            model_dir,
            device,
            &AnyPageParserLoadOptions::default(),
        )
    }

    /// Loads the named parser from a model directory with loading options.
    ///
    /// See [`from_dir`](Self::from_dir); the options carry the PP-DocLayout
    /// directory required by the layout-composed models.
    ///
    /// ```no_run
    /// use candle_core::Device;
    /// use oar_ocr_vl::{AnyPageParser, AnyPageParserLoadOptions, AnyPageParserModel};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // PaddleOCR-VL composes an external PP-DocLayout detector.
    /// let options = AnyPageParserLoadOptions::default()
    ///     .with_layout_dir("PaddlePaddle/PP-DocLayoutV3_safetensors");
    /// let parser = AnyPageParser::from_dir_with_options(
    ///     AnyPageParserModel::PaddleOcrVl1_5,
    ///     "PaddlePaddle/PaddleOCR-VL-1.5",
    ///     Device::Cpu,
    ///     &options,
    /// )?;
    /// # let _ = parser;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_dir_with_options(
        model: AnyPageParserModel,
        model_dir: impl AsRef<Path>,
        device: Device,
        options: &AnyPageParserLoadOptions,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        // Sibling variants of one family share its loader; the loaders tell
        // revisions apart from the directory's own config structure, which the
        // ID has already vouched for.
        Ok(match model {
            AnyPageParserModel::HpdParsing => HpdParsing::from_dir(model_dir, device)?.into(),
            AnyPageParserModel::HunyuanOcr => HunyuanOcr::from_dir(model_dir, device)?.into(),
            AnyPageParserModel::JinaOcr => JinaOcr::from_dir(model_dir, device)?.into(),
            AnyPageParserModel::MinerU2509 | AnyPageParserModel::MinerUPro => {
                MinerU::from_dir(model_dir, device)?.into()
            }
            AnyPageParserModel::MinerUDiffusion => {
                MinerUDiffusion::from_dir(model_dir, device)?.into()
            }
            AnyPageParserModel::MonkeyOcrV2S | AnyPageParserModel::MonkeyOcrV2B => {
                MonkeyOcrV2::from_dir(model_dir, device)?.into()
            }
            AnyPageParserModel::OvisOcr2 => OvisOcr2::from_dir(model_dir, device)?.into(),
            AnyPageParserModel::WeVisDoc2B | AnyPageParserModel::WeVisDoc4B => {
                WeVisDoc::from_dir(model_dir, device)?.into()
            }
            AnyPageParserModel::XiaomiOcr => XiaomiOcr::from_dir(model_dir, device)?.into(),
            AnyPageParserModel::PaddleOcrVl
            | AnyPageParserModel::PaddleOcrVl1_5
            | AnyPageParserModel::PaddleOcrVl1_6 => LayoutPageParser::new(
                PpDocLayout::from_dir(required_layout_dir(options, model)?, device.clone())?,
                PaddleOcrVl::from_dir(model_dir, device)?,
            )
            .into(),
            AnyPageParserModel::GlmOcr => LayoutPageParser::new(
                PpDocLayout::from_dir(required_layout_dir(options, model)?, device.clone())?,
                GlmOcr::from_dir(model_dir, device)?,
            )
            .into(),
            AnyPageParserModel::TeleOcr => LayoutPageParser::new(
                PpDocLayout::from_dir(required_layout_dir(options, model)?, device.clone())?,
                TeleOcr::from_dir(model_dir, device)?,
            )
            .into(),
        })
    }
}

#[cfg(feature = "auto-download")]
impl AnyPageParser {
    /// Downloads the named model's checkpoint and loads it.
    ///
    /// The checkpoint repo — named by the model's Hugging Face ID — is
    /// downloaded into the cache under `$OAR_HOME/models/<org>/<name>/<commit>`
    /// (default `~/.oar`) when it is not already there: the requested
    /// revision resolves to an immutable commit, every file is verified
    /// against the hashes the source API publishes, and the snapshot is
    /// published only once complete. The layout checkpoint uses its source's
    /// default revision.
    /// ModelScope is the default source; Hugging Face is selectable. The
    /// layout-composed models (PaddleOCR-VL, GLM-OCR, TeleOCR) also download
    /// a PP-DocLayout checkpoint (DEFAULT_LAYOUT_REPO by default) unless the
    /// options point at a local layout directory. Loading then goes
    /// through [`from_dir`](Self::from_dir) with each model's own defaults.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use candle_core::Device;
    /// use oar_ocr_vl::{AnyPageParser, AnyPageParserModel, AnyPageParserPretrainedOptions};
    ///
    /// let parser = AnyPageParser::from_pretrained(
    ///     AnyPageParserModel::PaddleOcrVl1_5,
    ///     Device::Cpu,
    ///     &AnyPageParserPretrainedOptions::default(),
    /// )?;
    /// # let _ = parser;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "auto-download")]
    pub fn from_pretrained(
        model: AnyPageParserModel,
        device: Device,
        options: &AnyPageParserPretrainedOptions,
    ) -> Result<Self, Error> {
        let source = options.source();
        let model_dir =
            crate::api::download::snapshot(source, model.as_str(), options.revision.as_deref())?;
        let mut load_options = AnyPageParserLoadOptions::default();
        if Self::needs_layout_dir(model) {
            // The layout checkpoint always uses its source's default
            // revision; a pinned model revision never applies to it.
            let layout_dir = match &options.layout_dir {
                Some(dir) => dir.clone(),
                None => crate::api::download::snapshot(source, options.layout(), None)?,
            };
            load_options = load_options.with_layout_dir(layout_dir);
        }
        Self::from_dir_with_options(model, &model_dir, device, &load_options)
    }

    /// Whether the model composes an external PP-DocLayout detector and so
    /// needs a layout directory to load.
    fn needs_layout_dir(model: AnyPageParserModel) -> bool {
        matches!(
            model,
            AnyPageParserModel::PaddleOcrVl
                | AnyPageParserModel::PaddleOcrVl1_5
                | AnyPageParserModel::PaddleOcrVl1_6
                | AnyPageParserModel::GlmOcr
                | AnyPageParserModel::TeleOcr
        )
    }
}

/// Returns the configured layout directory for a layout-composed model,
/// before any weights load, or explains how to provide one.
fn required_layout_dir(
    options: &AnyPageParserLoadOptions,
    model: AnyPageParserModel,
) -> Result<&Path, Error> {
    options.layout_dir.as_deref().ok_or_else(|| {
        Error::config(format!(
            "{model} parses with an external layout detector; pass a PP-DocLayout directory \
             with AnyPageParserLoadOptions::with_layout_dir"
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
    fn from_dir_reports_a_missing_layout_directory_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let error =
            AnyPageParser::from_dir(AnyPageParserModel::PaddleOcrVl1_5, dir.path(), Device::Cpu)
                .err()
                .expect("the layout directory should be required")
                .to_string();
        assert!(error.contains("PaddlePaddle/PaddleOCR-VL-1.5"), "{error}");
        assert!(error.contains("PP-DocLayout"), "{error}");
        assert!(error.contains("with_layout_dir"), "{error}");
    }

    #[test]
    fn model_ids_round_trip_and_unknown_ids_list_the_valid_ones() {
        for (id, model) in MODEL_IDS {
            assert_eq!(model.as_str(), *id);
            assert_eq!(model.to_string(), *id);
            assert_eq!(&id.parse::<AnyPageParserModel>().unwrap(), model);
        }
        // Hub IDs are case-insensitive.
        assert_eq!(
            "tencent/wevisdoc-4b".parse::<AnyPageParserModel>().unwrap(),
            AnyPageParserModel::WeVisDoc4B
        );
        let error = "a/b".parse::<AnyPageParserModel>().unwrap_err().to_string();
        assert!(error.contains("unknown model id \"a/b\""), "{error}");
        for (id, _) in MODEL_IDS {
            assert!(error.contains(id), "{error}");
        }
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
