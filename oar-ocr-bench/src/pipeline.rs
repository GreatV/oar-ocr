use crate::manifest::{Case, Kind, model_source};
use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use oar_ocr::{
    core::config::{OrtExecutionProvider, OrtSessionConfig},
    oarocr::{OAROCR, OAROCRBuilder, OARStructure, OARStructureBuilder},
};
use oar_ocr_vl::{
    DocParserConfig, GlmOcr, HpdGenerationConfig, HpdParsing, HunyuanOcr, HunyuanOcrParseOptions,
    JinaOcr, JinaOcrParseOptions, LayoutPageParser, LayoutPageParserOptions, MinerU,
    MinerUDiffusion, MinerUDiffusionParseOptions, MinerUParseOptions, MonkeyOcrV2,
    MonkeyOcrV2ParseOptions, OvisOcr2, OvisOcr2ParseOptions, PaddleOcrVl, PageDocument, PageParser,
    PpDocLayout, TeleOcr, WeVisDoc, WeVisDocParseOptions, XiaomiOcr, XiaomiOcrParseOptions,
};
use std::path::Path;

/// Text used for the VL output-rate metric: Markdown, else block content, else
/// the raw model output that some native parsers return exclusively.
fn page_text(page: PageDocument) -> String {
    if let Some(markdown) = page.markdown {
        return markdown;
    }
    let blocks: Vec<_> = page
        .blocks
        .into_iter()
        .filter_map(|block| block.content)
        .collect();
    if blocks.is_empty() {
        page.raw_output.unwrap_or_default()
    } else {
        blocks.join("\n\n")
    }
}

// Per-model dispatch until the VL crate offers a unified parser.
enum VlModel {
    Hpd(Box<HpdParsing>),
    Hunyuan(Box<HunyuanOcr>),
    Jina(Box<JinaOcr>),
    MinerU(Box<MinerU>),
    Diffusion(Box<MinerUDiffusion>),
    Monkey(Box<MonkeyOcrV2>),
    Ovis(Box<OvisOcr2>),
    WeVis(Box<WeVisDoc>),
    Xiaomi(Box<XiaomiOcr>),
    Paddle(Box<LayoutPageParser<PpDocLayout, PaddleOcrVl>>),
    Glm(Box<LayoutPageParser<PpDocLayout, GlmOcr>>),
    Tele(Box<LayoutPageParser<PpDocLayout, TeleOcr>>),
}
impl VlModel {
    fn load(root: &Path, case: &Case, device: &candle_core::Device) -> Result<Self> {
        let path = root.join(
            case.model_path
                .as_deref()
                .context("VL requires model_path")?,
        );
        let layout = || -> Result<PpDocLayout> {
            Ok(PpDocLayout::from_dir(
                root.join(
                    case.layout_path
                        .as_deref()
                        .context("external layout path missing")?,
                ),
                device.clone(),
            )?)
        };
        let config = || DocParserConfig {
            max_tokens: case.options.max_tokens.unwrap_or(4096),
            ..Default::default()
        };
        let batch = case.options.region_batch_size.unwrap_or(1);
        Ok(match case.model.as_deref().unwrap_or_default() {
            "hpd-parsing" => Self::Hpd(Box::new(HpdParsing::from_dir(path, device.clone())?)),
            "hunyuanocr" => Self::Hunyuan(Box::new(HunyuanOcr::from_dir(path, device.clone())?)),
            "jina-ocr" => Self::Jina(Box::new(JinaOcr::from_dir(path, device.clone())?)),
            "mineru" => Self::MinerU(Box::new(MinerU::from_dir(path, device.clone())?)),
            "mineru-diffusion" => {
                Self::Diffusion(Box::new(MinerUDiffusion::from_dir(path, device.clone())?))
            }
            "monkeyocrv2" => Self::Monkey(Box::new(MonkeyOcrV2::from_dir(path, device.clone())?)),
            "ovisocr2" => Self::Ovis(Box::new(OvisOcr2::from_dir(path, device.clone())?)),
            "wevisdoc" => Self::WeVis(Box::new(WeVisDoc::from_dir(path, device.clone())?)),
            "xiaomi-ocr-0" => Self::Xiaomi(Box::new(XiaomiOcr::from_dir(path, device.clone())?)),
            "paddleocr-vl" => Self::Paddle(Box::new(
                LayoutPageParser::with_config(
                    layout()?,
                    PaddleOcrVl::from_dir(path, device.clone())?,
                    config(),
                )
                .with_region_batch_size(batch),
            )),
            "glmocr" => Self::Glm(Box::new(
                LayoutPageParser::with_config(
                    layout()?,
                    GlmOcr::from_dir(path, device.clone())?,
                    config(),
                )
                .with_region_batch_size(batch),
            )),
            "teleocr" => Self::Tele(Box::new(
                LayoutPageParser::with_config(
                    layout()?,
                    TeleOcr::from_dir(path, device.clone())?,
                    config(),
                )
                .with_region_batch_size(batch),
            )),
            model => bail!("unsupported VL model {model}"),
        })
    }

    fn parse(&self, image: &RgbImage, case: &Case) -> Result<PageDocument> {
        let tokens = case.options.max_tokens;
        Ok(match self {
            Self::Hpd(model) => {
                let defaults = HpdGenerationConfig::default();
                model.parse_page(
                    image,
                    &HpdGenerationConfig {
                        max_new_tokens: tokens.unwrap_or(defaults.max_new_tokens),
                        ..defaults
                    },
                )?
            }
            Self::Hunyuan(model) => model.parse_page(
                image,
                &HunyuanOcrParseOptions {
                    max_new_tokens: tokens
                        .unwrap_or(HunyuanOcrParseOptions::default().max_new_tokens),
                    ..Default::default()
                },
            )?,
            Self::Jina(model) => model.parse_page(
                image,
                &JinaOcrParseOptions {
                    max_new_tokens: tokens.unwrap_or(JinaOcrParseOptions::default().max_new_tokens),
                },
            )?,
            Self::MinerU(model) => {
                let defaults = MinerUParseOptions::default();
                model.parse_page(
                    image,
                    &MinerUParseOptions {
                        max_tokens: tokens.unwrap_or(defaults.max_tokens),
                        region_batch_size: case
                            .options
                            .region_batch_size
                            .unwrap_or(defaults.region_batch_size),
                        ..defaults
                    },
                )?
            }
            Self::Diffusion(model) => {
                let defaults = MinerUDiffusionParseOptions::default();
                model.parse_page(
                    image,
                    &MinerUDiffusionParseOptions {
                        generation: oar_ocr_vl::DiffusionGenerationConfig {
                            gen_length: tokens.unwrap_or(defaults.generation.gen_length),
                            ..defaults.generation
                        },
                        ..defaults
                    },
                )?
            }
            Self::Monkey(model) => model.parse_page(
                image,
                &MonkeyOcrV2ParseOptions {
                    max_new_tokens: tokens
                        .unwrap_or(MonkeyOcrV2ParseOptions::default().max_new_tokens),
                    ..Default::default()
                },
            )?,
            Self::Ovis(model) => model.parse_page(
                image,
                &OvisOcr2ParseOptions {
                    max_new_tokens: tokens
                        .unwrap_or(OvisOcr2ParseOptions::default().max_new_tokens),
                    ..Default::default()
                },
            )?,
            Self::WeVis(model) => model.parse_page(
                image,
                &WeVisDocParseOptions {
                    max_new_tokens: tokens
                        .unwrap_or(WeVisDocParseOptions::default().max_new_tokens),
                },
            )?,
            Self::Xiaomi(model) => model.parse_page(
                image,
                &XiaomiOcrParseOptions {
                    max_new_tokens: tokens
                        .unwrap_or(XiaomiOcrParseOptions::default().max_new_tokens),
                },
            )?,
            Self::Paddle(model) => model.parse_page(image, &LayoutPageParserOptions::default())?,
            Self::Glm(model) => model.parse_page(image, &LayoutPageParserOptions::default())?,
            Self::Tele(model) => model.parse_page(image, &LayoutPageParserOptions::default())?,
        })
    }
}

pub(crate) enum Pipeline {
    Ocr(Box<OAROCR>),
    Structure(Box<OARStructure>),
    Vl(Box<VlPipeline>),
}

pub(crate) struct VlPipeline {
    model: VlModel,
    // The shared device provides an explicit synchronization boundary for timing.
    device: candle_core::Device,
}

pub(crate) enum DeviceSelection {
    Classic(OrtSessionConfig),
    Vl(candle_core::Device),
}

impl DeviceSelection {
    pub(crate) fn resolve(case: &Case) -> Result<Self> {
        if case.kind == Kind::Vl {
            let device = if case.device == "auto" {
                oar_ocr_vl::auto_device()
            } else {
                oar_ocr_vl::utils::parse_device(&case.device)?
            };
            return Ok(Self::Vl(device));
        }
        Ok(Self::Classic(ort_config(case)?))
    }

    pub(crate) fn name(&self) -> Result<String> {
        match self {
            Self::Classic(config) => provider_device(config),
            Self::Vl(device) => Ok(match device.location() {
                candle_core::DeviceLocation::Cpu => "cpu".to_string(),
                candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
                candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
            }),
        }
    }
}

fn provider_device(config: &OrtSessionConfig) -> Result<String> {
    Ok(match config.get_execution_providers().first() {
        None | Some(OrtExecutionProvider::CPU) => "cpu".into(),
        Some(OrtExecutionProvider::CUDA { device_id, .. }) => {
            format!("cuda:{}", device_id.unwrap_or(0))
        }
        Some(OrtExecutionProvider::CoreML { .. }) => "coreml".into(),
        Some(OrtExecutionProvider::DirectML { device_id }) => {
            format!("directml:{}", device_id.unwrap_or(0))
        }
        Some(provider) => bail!("unexpected automatic execution provider {provider:?}"),
    })
}

impl Pipeline {
    pub(crate) fn load(root: &Path, case: &Case, device: DeviceSelection) -> Result<Self> {
        if let DeviceSelection::Vl(device) = device {
            let model = VlModel::load(root, case, &device)?;
            device.synchronize()?;
            return Ok(Self::Vl(Box::new(VlPipeline { model, device })));
        }
        let DeviceSelection::Classic(config) = device else {
            unreachable!()
        };
        let source = |name: &str| model_source(root, name);
        let batch = case.options.batch_size();
        let m = &case.models;
        match case.kind {
            Kind::Ocr => {
                let mut builder = OAROCRBuilder::new(
                    source(m.detector.as_deref().context("missing detector")?),
                    source(m.recognizer.as_deref().context("missing recognizer")?),
                    source(m.dictionary.as_deref().context("missing dictionary")?),
                )
                .ort_session(config)
                .image_batch_size(batch);
                if let Some(size) = case.options.region_batch_size {
                    builder = builder.region_batch_size(size);
                }
                Ok(Self::Ocr(Box::new(builder.build()?)))
            }
            Kind::Structure => {
                let mut builder = OARStructureBuilder::new(source(
                    m.layout.as_deref().context("missing layout")?,
                ))
                .ort_session(config)
                .image_batch_size(batch);
                if let Some(name) = &m.layout_name {
                    builder = builder.layout_model_name(name);
                }
                if let Some(size) = case.options.region_batch_size {
                    builder = builder.region_batch_size(size);
                }
                if let Some(detector) = &m.detector {
                    builder = builder.with_ocr(
                        source(detector),
                        source(m.recognizer.as_deref().context("missing recognizer")?),
                        source(m.dictionary.as_deref().context("missing dictionary")?),
                    );
                }
                if let Some(path) = &m.table_dictionary {
                    builder = builder.table_structure_dict_path(source(path));
                }
                if let Some(path) = &m.table_classifier {
                    builder = builder.with_table_classification(source(path));
                }
                if let Some(path) = &m.wired_table_structure {
                    builder = builder.with_wired_table_structure(source(path));
                }
                if let Some(path) = &m.wireless_table_structure {
                    builder = builder.with_wireless_table_structure(source(path));
                }
                if let Some(path) = &m.wired_table_cells {
                    builder = builder.with_wired_table_cell_detection(source(path));
                }
                if let Some(path) = &m.wireless_table_cells {
                    builder = builder.with_wireless_table_cell_detection(source(path));
                }
                Ok(Self::Structure(Box::new(builder.build()?)))
            }
            Kind::Vl => unreachable!(),
        }
    }

    pub(crate) fn infer(&self, images: &[&RgbImage], case: &Case) -> Result<Vec<String>> {
        let owned = || images.iter().map(|image| (*image).clone()).collect();
        match self {
            Self::Ocr(model) => Ok(model
                .predict(owned())?
                .into_iter()
                .map(|page| page.concatenated_text("\n"))
                .collect()),
            Self::Structure(model) => model
                .predict_images(owned())
                .into_iter()
                .map(|page| Ok(page?.to_markdown()))
                .collect(),
            Self::Vl(pipeline) => {
                ensure!(images.len() == 1, "VL PageParser requires a single page");
                let text = page_text(pipeline.model.parse(images[0], case)?);
                pipeline.device.synchronize()?;
                Ok(vec![text])
            }
        }
    }
}

fn ort_config(case: &Case) -> Result<OrtSessionConfig> {
    if case.device == "auto" {
        return Ok(OrtSessionConfig::auto()
            .with_intra_threads(case.options.cpu_threads())
            .resolve_auto());
    }
    let config = OrtSessionConfig::new().with_intra_threads(case.options.cpu_threads());
    if case.device == "cpu" {
        return Ok(config.with_execution_providers(vec![OrtExecutionProvider::CPU]));
    }
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    {
        let (config, strict) = match case.device.as_str() {
            #[cfg(feature = "cuda")]
            value if value.starts_with("cuda:") => {
                let ordinal = value[5..].parse::<i32>()?;
                let config = config.with_execution_providers(vec![
                    OrtExecutionProvider::CUDA {
                        device_id: Some(ordinal),
                        gpu_mem_limit: None,
                        arena_extend_strategy: None,
                        cudnn_conv_algo_search: None,
                        cudnn_conv_use_max_workspace: None,
                    },
                    OrtExecutionProvider::CPU,
                ]);
                (
                    config,
                    ort::ep::CUDA::default()
                        .with_device_id(ordinal)
                        .build()
                        .error_on_failure(),
                )
            }
            #[cfg(all(feature = "metal", target_os = "macos"))]
            "metal" => {
                let config = config.with_execution_providers(vec![
                    OrtExecutionProvider::CoreML {
                        ane_only: None,
                        subgraphs: None,
                    },
                    OrtExecutionProvider::CPU,
                ]);
                (
                    config,
                    ort::ep::CoreML::default().build().error_on_failure(),
                )
            }
            value => bail!("device {value} requires its matching accelerator feature and platform"),
        };
        // A benchmark must not silently label a CPU fallback as an accelerator run.
        let builder = ort::session::Session::builder()?
            .with_intra_threads(1)
            .map_err(ort::Error::<()>::from)?;
        builder
            .with_execution_providers([strict])
            .map_err(ort::Error::<()>::from)?;
        Ok(config)
    }
    #[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
    bail!(
        "device {} requires its matching accelerator feature and platform",
        case.device
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn provider_names_preserve_cuda_ordinals() {
        let config = OrtSessionConfig::new().with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(3),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU,
        ]);
        assert_eq!(provider_device(&config).unwrap(), "cuda:3");
        assert_eq!(provider_device(&OrtSessionConfig::new()).unwrap(), "cpu");
    }
    #[cfg(not(any(feature = "cuda", all(feature = "metal", target_os = "macos"))))]
    #[test]
    fn automatic_devices_resolve_without_loading_weights() {
        let manifest = crate::manifest::Manifest::parse(
            include_str!("../manifests/default.toml"),
            Some("auto"),
            None,
        )
        .unwrap();
        let classic = &manifest.cases[0];
        let vl = manifest
            .cases
            .iter()
            .find(|case| case.kind == Kind::Vl)
            .unwrap();
        assert_eq!(
            DeviceSelection::resolve(classic).unwrap().name().unwrap(),
            "cpu"
        );
        assert_eq!(DeviceSelection::resolve(vl).unwrap().name().unwrap(), "cpu");
        let mut explicit = classic.clone();
        explicit.device = "cuda:0".into();
        assert!(DeviceSelection::resolve(&explicit).is_err());
    }
}
