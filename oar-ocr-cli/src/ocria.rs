//! Locate the OCRia-nano-1 model files for the `ocria-nano` presets.
use anyhow::Result;
use std::path::{Path, PathBuf};

/// The pinned Hugging Face revision of `llamaindex/OCRia-nano-1`.
#[cfg(feature = "auto-download")]
const MODEL_REVISION: &str = "53d45fe6803a568deedb081550c2fdb97cc99f97";

/// The local directory, or the snapshot LlamaIndex publishes on Hugging Face.
///
/// The files are never mirrored: their license covers redistribution, and downloading them from the publisher is the user accepting it.
pub(crate) fn model_dir(local: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = local {
        return Ok(dir.to_path_buf());
    }
    #[cfg(feature = "auto-download")]
    {
        use anyhow::Context;
        tracing::warn!(
            "OCRia-nano-1 is downloaded from llamaindex/OCRia-nano-1 under the OCRia-nano-1 Model License, which restricts commercial use: https://huggingface.co/llamaindex/OCRia-nano-1/blob/main/LICENSE"
        );
        oar_ocr_vl::download_snapshot(
            oar_ocr_vl::DownloadSource::HuggingFace,
            "llamaindex/OCRia-nano-1",
            Some(MODEL_REVISION),
        )
        .context("could not download OCRia-nano-1 from Hugging Face; check connectivity or provide --model-dir DIR")
    }
    #[cfg(not(feature = "auto-download"))]
    anyhow::bail!(
        "model downloads are disabled; provide --model-dir DIR or reinstall with --features auto-download"
    )
}
