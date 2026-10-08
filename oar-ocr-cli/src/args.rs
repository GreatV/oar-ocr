use clap::{Args, Parser, Subcommand, ValueEnum};
use oar_ocr::oarocr::PpOcrV6Size;
use oar_ocr_vl::AnyPageParserModel;
use std::path::PathBuf;

use crate::pdf::PageRanges;

#[derive(Parser)]
#[command(
    name = "oar",
    version,
    about = "OCR and document parsing with automatic model downloads"
)]
pub(crate) struct Cli {
    #[command(flatten)]
    pub(crate) common: Common,
    /// Show model loading and inference progress
    #[arg(short, long, global = true)]
    pub(crate) verbose: bool,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Args)]
pub(crate) struct Common {
    /// auto, cpu, cuda:N, metal, or for ocr/structure a comma list like
    /// cuda:0,cuda:1 to run one pipeline replica per device (duplicates allowed)
    #[arg(
        long,
        global = true,
        default_value = "auto",
        value_parser = device_list
    )]
    pub(crate) device: String,
    /// Write one .md or .json per image or PDF page; use a fresh directory
    #[arg(short, long, global = true, value_name = "DIR")]
    pub(crate) output: Option<PathBuf>,
    /// PDF pages to process, e.g. 1-3,5 (1-based; default: all)
    #[arg(long, global = true, value_name = "RANGES")]
    pub(crate) pages: Option<PageRanges>,
    /// PDF rendering resolution in dots per inch
    #[arg(long, global = true, default_value = "144", value_parser = dpi)]
    pub(crate) dpi: f32,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Recognize text with a PP-OCRv6 preset
    Ocr(Ocr),
    /// Parse layout, text, and tables with the PP-StructureV3 preset
    Structure(Structure),
    /// Parse pages with a supported vision-language model
    Parse(Parse),
}

#[derive(Args)]
pub(crate) struct ClassicModels {
    /// Custom detection model; also requires --rec and --dict
    #[arg(long, requires_all = ["rec", "dict"])]
    pub(crate) det: Option<PathBuf>,
    /// Custom recognition model; also requires --det and --dict
    #[arg(long, requires_all = ["det", "dict"])]
    pub(crate) rec: Option<PathBuf>,
    /// Custom character dictionary; also requires --det and --rec
    #[arg(long, requires_all = ["det", "rec"])]
    pub(crate) dict: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OcrFormat {
    Text,
    Json,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum PageFormat {
    Markdown,
    Json,
}

#[derive(Args)]
pub(crate) struct Ocr {
    /// PP-OCRv6 preset size
    #[arg(long, value_enum, default_value = "tiny", conflicts_with_all = ["det", "rec", "dict"])]
    pub(crate) size: OcrSize,
    #[command(flatten)]
    pub(crate) models: ClassicModels,
    #[arg(long, value_enum, default_value = "text")]
    pub(crate) format: OcrFormat,
    #[arg(required = true, value_name = "INPUTS", num_args = 1..)]
    pub(crate) images: Vec<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum OcrSize {
    Tiny,
    Small,
    Medium,
}

impl From<OcrSize> for PpOcrV6Size {
    fn from(size: OcrSize) -> Self {
        match size {
            OcrSize::Tiny => Self::Tiny,
            OcrSize::Small => Self::Small,
            OcrSize::Medium => Self::Medium,
        }
    }
}

#[derive(Args)]
pub(crate) struct Structure {
    #[command(flatten)]
    pub(crate) models: ClassicModels,
    #[arg(long, value_enum, default_value = "markdown")]
    pub(crate) format: PageFormat,
    #[arg(required = true, value_name = "INPUTS", num_args = 1..)]
    pub(crate) images: Vec<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum Source {
    Modelscope,
    Huggingface,
}

#[derive(Args)]
pub(crate) struct Parse {
    /// Explicit Hugging Face model ID; see --list-models
    #[arg(long, required_unless_present = "list_models", value_parser = model)]
    pub(crate) model: Option<AnyPageParserModel>,
    /// Load the model from a local checkpoint directory
    #[arg(long, value_name = "DIR")]
    pub(crate) model_dir: Option<PathBuf>,
    /// Load PP-DocLayout from a local directory for layout-composed models
    #[arg(long, value_name = "DIR")]
    pub(crate) layout_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "modelscope")]
    pub(crate) source: Source,
    /// Maximum generated tokens per page or region
    #[arg(long, value_parser = positive)]
    pub(crate) max_tokens: Option<usize>,
    /// Print supported model IDs without loading models
    #[arg(long, conflicts_with_all = ["model", "images", "model_dir", "layout_dir", "max_tokens"])]
    pub(crate) list_models: bool,
    #[arg(long, value_enum, default_value = "markdown")]
    pub(crate) format: PageFormat,
    #[arg(required_unless_present = "list_models", value_name = "INPUTS", num_args = 1..)]
    pub(crate) images: Vec<PathBuf>,
}

fn positive(value: &str) -> Result<usize, String> {
    value
        .parse()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| "must be a positive integer".into())
}

fn dpi(value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .ok()
        .filter(|dpi| dpi.is_finite() && *dpi > 0.0)
        .ok_or_else(|| "DPI must be a finite positive number".into())
}

fn model(value: &str) -> Result<AnyPageParserModel, String> {
    value.parse().map_err(|_| {
        format!("unknown model ID {value:?}; run `oar parse --list-models` to see supported IDs")
    })
}

fn device(value: &str) -> Result<String, String> {
    let normalized = value.to_ascii_lowercase();
    if matches!(normalized.as_str(), "auto" | "cpu" | "metal") {
        return Ok(normalized);
    }
    if let Some(index) = normalized.strip_prefix("cuda:")
        && let Ok(index) = index.parse::<i32>()
        && index >= 0
    {
        return Ok(format!("cuda:{index}"));
    }
    Err("device must be auto, cpu, cuda:N (non-negative N), or metal".into())
}

/// Parse a comma-separated device list into validated device specs.
/// Duplicates are allowed (`cuda:0,cuda:0` runs two replicas on one GPU);
/// mixing `auto` with explicit devices or `metal` with anything is not.
pub(crate) fn device_list(value: &str) -> Result<String, String> {
    let parts: Vec<&str> = value.split(',').map(str::trim).collect();
    if parts.len() > 1 && parts.contains(&"auto") {
        return Err("auto cannot be combined with other devices".into());
    }
    if parts.len() > 1 && parts.contains(&"metal") {
        return Err("metal cannot be combined with other devices".into());
    }
    let normalized: Vec<String> = parts
        .iter()
        .map(|part| device(part))
        .collect::<Result<Vec<_>, _>>()?;
    if normalized.len() > 1
        && !normalized
            .iter()
            .all(|d| d.starts_with("cuda:") || d == "cpu")
    {
        return Err("multi-device lists may only contain cuda:N or cpu entries".into());
    }
    Ok(normalized.join(","))
}

/// Split a validated device list into individual device specs.
pub(crate) fn split_devices(devices: &str) -> Vec<String> {
    devices.split(',').map(str::to_string).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_formats_overrides_and_offline_model_listing() {
        let cli = Cli::try_parse_from([
            "oar",
            "ocr",
            "page.png",
            "--format",
            "json",
            "--device",
            "CUDA:2",
            "--rec",
            "custom.onnx",
            "--det",
            "det.onnx",
            "--dict",
            "dict.txt",
            "-o",
            "outputs",
        ])
        .unwrap();
        assert_eq!(cli.common.device, "cuda:2");
        assert_eq!(cli.common.output, Some("outputs".into()));
        let Command::Ocr(args) = cli.command else {
            panic!("expected OCR")
        };
        assert!(args.format == OcrFormat::Json);
        assert_eq!(args.models.rec, Some(PathBuf::from("custom.onnx")));
        // Device lists: duplicates allowed, mixing auto/metal rejected.
        assert_eq!(device_list("cuda:0,cuda:1").unwrap(), "cuda:0,cuda:1");
        assert_eq!(device_list("CUDA:0, cuda:0").unwrap(), "cuda:0,cuda:0");
        assert_eq!(device_list("cpu,cuda:0").unwrap(), "cpu,cuda:0");
        assert!(device_list("auto,cuda:0").is_err());
        assert!(device_list("cuda:0,metal").is_err());
        assert!(device_list("cuda:0,gpu").is_err());
        let listed =
            Cli::try_parse_from(["oar", "ocr", "page.png", "--device", "cuda:0,cuda:0"]).unwrap();
        assert_eq!(listed.common.device, "cuda:0,cuda:0");
        assert_eq!(split_devices(&listed.common.device).len(), 2);
        for (extra, expected) in [
            (vec![], PpOcrV6Size::Tiny),
            (vec!["--size", "small"], PpOcrV6Size::Small),
            (vec!["--size", "medium"], PpOcrV6Size::Medium),
        ] {
            let cli =
                Cli::try_parse_from(["oar", "ocr", "page.png"].into_iter().chain(extra)).unwrap();
            let Command::Ocr(args) = cli.command else {
                panic!("expected OCR")
            };
            assert_eq!(PpOcrV6Size::from(args.size), expected);
        }
        let cli = Cli::try_parse_from(["oar", "parse", "--list-models"]).unwrap();
        let Command::Parse(args) = cli.command else {
            panic!("expected parsing")
        };
        assert!(args.list_models && args.images.is_empty() && args.model.is_none());
        assert_eq!(cli.common.device, "auto");
    }

    #[test]
    fn rejects_invalid_devices_formats_and_missing_parse_inputs() {
        for device in ["cuda:-1", "cuda:2147483648", "gpu", "cuda:x"] {
            assert!(Cli::try_parse_from(["oar", "ocr", "page.png", "--device", device]).is_err());
        }
        assert!(Cli::try_parse_from(["oar", "ocr", "page.png", "--format", "yaml"]).is_err());
        assert!(Cli::try_parse_from(["oar", "ocr", "page.png", "--size", "large"]).is_err());
        assert!(Cli::try_parse_from(["oar", "ocr", "page.png", "--det", "det.onnx"]).is_err());
        assert!(Cli::try_parse_from(["oar", "parse", "page.png"]).is_err());
        assert!(
            Cli::try_parse_from(["oar", "parse", "--model", "unknown/model", "page.png"]).is_err()
        );
        assert!(
            Cli::try_parse_from([
                "oar",
                "parse",
                "--model",
                "jinaai/jina-ocr-v1",
                "page.png",
                "--max-tokens",
                "0"
            ])
            .is_err()
        );
    }
}
