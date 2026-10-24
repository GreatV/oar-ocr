#[cfg(feature = "auto-download")]
use anyhow::Context;
use anyhow::{Result, ensure};
use image::RgbImage;
use oar_ocr::{
    core::config::{OrtExecutionProvider, OrtSessionConfig},
    indra::IndraOcr,
};
use std::path::{Path, PathBuf};

use crate::{Document, args::IndraFormat};

// datalab-to/indra-ocr revision validated by the CPU/CUDA parity run.
#[cfg(feature = "auto-download")]
const MODEL_REVISION: &str = "5993a956d7ea1d43c0d40d409962bb0b033ac1e8";

const REQUIRED: &[&str] = &["meta.json", "det.onnx", "rec.onnx", "order.onnx"];
const MATH: &[&str] = &[
    "math_meta.json",
    "mathcls.onnx",
    "math_enc.onnx",
    "math_dec_init.onnx",
    "math_dec_step.onnx",
];

pub(crate) fn devices(
    devices: &[String],
    workers: usize,
    verbose: bool,
) -> Result<Vec<OrtSessionConfig>> {
    let configs = devices
        .iter()
        .map(|device| {
            let mut config = crate::ort_device(device, verbose)?;
            for provider in config.execution_providers.iter_mut().flatten() {
                if let OrtExecutionProvider::CUDA {
                    arena_extend_strategy,
                    cudnn_conv_algo_search,
                    ..
                } = provider
                {
                    *arena_extend_strategy = Some("SameAsRequested".into());
                    *cudnn_conv_algo_search = Some("DEFAULT".into());
                }
            }
            Ok(config)
        })
        .collect::<Result<Vec<_>>>()?;
    validate_workers(workers, &configs)?;
    Ok(configs)
}

fn validate_workers(workers: usize, configs: &[OrtSessionConfig]) -> Result<()> {
    ensure!(workers > 0, "--workers must be a positive integer");
    ensure!(
        workers == 1
            || !configs
                .iter()
                .any(OrtSessionConfig::has_accelerator_provider),
        "Indra accelerator devices require --workers 1; use --device cpu for CPU workers"
    );
    Ok(())
}

pub(crate) fn model_dir(local: Option<&Path>, math: bool) -> Result<PathBuf> {
    resolve_dir(local, math, || {
        #[cfg(feature = "auto-download")]
        {
            oar_ocr_vl::download_snapshot(
                oar_ocr_vl::DownloadSource::HuggingFace,
                "datalab-to/indra-ocr",
                Some(MODEL_REVISION),
            )
            .context("could not download Indra models from Hugging Face; check connectivity or provide --model-dir DIR")
        }
        #[cfg(not(feature = "auto-download"))]
        anyhow::bail!(
            "model downloads are disabled; provide --model-dir DIR or reinstall with --features auto-download"
        )
    })
}

fn resolve_dir(
    local: Option<&Path>,
    math: bool,
    download: impl FnOnce() -> Result<PathBuf>,
) -> Result<PathBuf> {
    let dir = match local {
        Some(dir) => dir.to_path_buf(),
        None => download()?,
    };
    ensure!(
        dir.is_dir(),
        "Indra model directory {} does not exist; provide --model-dir DIR or omit it to download the models",
        dir.display()
    );
    let missing = REQUIRED
        .iter()
        .chain(if math { MATH } else { &[] }.iter())
        .filter(|name| !dir.join(name).is_file())
        .copied()
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "Indra model directory {} is missing {}; provide a complete --model-dir or omit it to download the models",
        dir.display(),
        missing.join(", ")
    );
    Ok(dir)
}

pub(crate) fn recognize(
    model: &IndraOcr,
    images: Vec<RgbImage>,
    format: IndraFormat,
) -> Result<Vec<Document>> {
    Ok(model
        .predict(images)?
        .into_iter()
        .map(|page| {
            let json = page.to_json();
            let text = if format == IndraFormat::Text {
                page.lines
                    .iter()
                    .map(|line| line.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                page.markdown
            };
            Document { text, json }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn validates_workers_and_devices_without_loading_models() {
        let cpu = OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);
        let gpu = OrtSessionConfig::new().with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU,
        ]);
        assert!(validate_workers(3, &[cpu.clone(), cpu.clone()]).is_ok());
        assert!(validate_workers(1, std::slice::from_ref(&gpu)).is_ok());
        assert!(validate_workers(3, &[cpu.clone(), gpu]).is_err());
        assert!(validate_workers(0, &[cpu]).is_err());
        for flag in ["--workers", "--batch-size"] {
            assert!(
                crate::args::Cli::try_parse_from(["oar", "indra", "page.png", flag, "0"]).is_err()
            );
        }
    }

    #[test]
    fn resolves_local_models_without_downloading_or_requiring_optional_heads() {
        let dir = std::env::temp_dir().join(format!("oar-indra-cli-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let local = |math| resolve_dir(Some(&dir), math, || panic!("local path must win"));
        assert!(local(false).unwrap_err().to_string().contains("det.onnx"));
        for file in REQUIRED {
            std::fs::write(dir.join(file), []).unwrap();
        }
        assert_eq!(local(false).unwrap(), dir);
        assert!(
            local(true)
                .unwrap_err()
                .to_string()
                .contains("math_enc.onnx")
        );
        for file in MATH {
            std::fs::write(dir.join(file), []).unwrap();
        }
        assert_eq!(local(true).unwrap(), dir);
        assert_eq!(resolve_dir(None, true, || Ok(dir.clone())).unwrap(), dir);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
