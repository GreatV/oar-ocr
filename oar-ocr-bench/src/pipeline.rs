use crate::manifest::{Case, Kind, model_source};
use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use oar_ocr::{
    core::{
        config::{OrtExecutionProvider, OrtSessionConfig},
        inference::initialize_ort_environment,
    },
    oarocr::{OAROCR, OAROCRBuilder, OARStructure, OARStructureBuilder},
};
use oar_ocr_vl::{
    AnyPageParser, AnyPageParserLoadOptions, AnyPageParserModel, AnyPageParserOptions,
    PageDocument, PageParser,
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

/// One of the VL page parsers, detected from the case's model directory and
/// driven through the crate's unified [`AnyPageParser`] contract.
struct VlModel(AnyPageParser);

impl VlModel {
    fn load(root: &Path, case: &Case, device: &candle_core::Device) -> Result<Self> {
        let path = root.join(
            case.model_path
                .as_deref()
                .context("VL requires model_path")?,
        );
        let model = case
            .model
            .as_deref()
            .context("VL requires model")?
            .parse::<AnyPageParserModel>()?;
        let mut options = AnyPageParserLoadOptions::default();
        if let Some(layout_path) = &case.layout_path {
            options = options.with_layout_dir(root.join(layout_path));
        }
        Ok(Self(AnyPageParser::from_dir_with_options(
            model,
            path,
            device.clone(),
            &options,
        )?))
    }

    fn parse(&self, image: &RgbImage, case: &Case) -> Result<PageDocument> {
        let mut options = AnyPageParserOptions::default();
        if let Some(max_new_tokens) = case.options.max_tokens {
            options = options.with_max_new_tokens(max_new_tokens);
        }
        if let Some(region_batch_size) = case.options.region_batch_size {
            options = options.with_region_batch_size(region_batch_size);
        }
        Ok(self.0.parse_page(image, &options)?)
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
        let m = &case.models;
        match case.kind {
            Kind::Ocr => {
                let mut builder = OAROCRBuilder::new(
                    source(m.detector.as_deref().context("missing detector")?),
                    source(m.recognizer.as_deref().context("missing recognizer")?),
                    source(m.dictionary.as_deref().context("missing dictionary")?),
                )
                .ort_session(config);
                if let Some(bytes) = case.options.gpu_memory_budget {
                    builder = builder.gpu_memory_budget(bytes);
                }
                if let Some(size) = case.options.region_batch_size {
                    builder = builder.region_batch_size(size);
                }
                Ok(Self::Ocr(Box::new(builder.build()?)))
            }
            Kind::Structure => {
                let mut builder = OARStructureBuilder::new(source(
                    m.layout.as_deref().context("missing layout")?,
                ))
                .ort_session(config);
                if let Some(bytes) = case.options.gpu_memory_budget {
                    builder = builder.gpu_memory_budget(bytes);
                }
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
                let page = pipeline.model.parse(images[0], case)?;
                pipeline.device.synchronize()?;
                // A truncated or partially recognized page would look faster.
                ensure!(
                    page.diagnostics.is_empty(),
                    "parser reported diagnostics: {:?}",
                    page.diagnostics
                );
                let text = page_text(page);
                Ok(vec![text])
            }
        }
    }
}

fn ort_config(case: &Case) -> Result<OrtSessionConfig> {
    initialize_ort_environment()?;
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
