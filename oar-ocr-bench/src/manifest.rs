use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Kind {
    Ocr,
    Structure,
    Vl,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Inputs {
    pub(crate) images: Vec<String>,
    pub(crate) image_dirs: Vec<String>,
    pub(crate) pdfs: Vec<String>,
    pub(crate) pdf_scale: Option<f32>,
    pub(crate) max_pages: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Options {
    pub(crate) batch_size: Option<usize>,
    pub(crate) cpu_threads: Option<usize>,
    pub(crate) region_batch_size: Option<usize>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) use_mtp: Option<bool>,
    pub(crate) diffusion_seed: Option<u64>,
    pub(crate) nvml_device: Option<String>,
    pub(crate) nvml_interval_ms: Option<u64>,
}

impl Options {
    fn inherit(&self, other: &Self) -> Self {
        Self {
            batch_size: self.batch_size.or(other.batch_size),
            cpu_threads: self.cpu_threads.or(other.cpu_threads),
            region_batch_size: self.region_batch_size.or(other.region_batch_size),
            max_tokens: self.max_tokens.or(other.max_tokens),
            use_mtp: self.use_mtp.or(other.use_mtp),
            diffusion_seed: self.diffusion_seed.or(other.diffusion_seed),
            nvml_device: self
                .nvml_device
                .clone()
                .or_else(|| other.nvml_device.clone()),
            nvml_interval_ms: self.nvml_interval_ms.or(other.nvml_interval_ms),
        }
    }
    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size.unwrap_or(1)
    }
    pub(crate) fn cpu_threads(&self) -> usize {
        self.cpu_threads.unwrap_or(4)
    }
    pub(crate) fn interval_ms(&self) -> u64 {
        self.nvml_interval_ms.unwrap_or(10)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Models {
    pub(crate) detector: Option<String>,
    pub(crate) recognizer: Option<String>,
    pub(crate) dictionary: Option<String>,
    pub(crate) layout: Option<String>,
    pub(crate) layout_name: Option<String>,
    pub(crate) table_dictionary: Option<String>,
    pub(crate) table_classifier: Option<String>,
    pub(crate) wired_table_structure: Option<String>,
    pub(crate) wireless_table_structure: Option<String>,
    pub(crate) wired_table_cells: Option<String>,
    pub(crate) wireless_table_cells: Option<String>,
    pub(crate) formula: Option<String>,
    pub(crate) formula_tokenizer: Option<String>,
    pub(crate) formula_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Defaults {
    device: String,
    warmup: usize,
    repetitions: usize,
    options: Options,
}
impl Default for Defaults {
    fn default() -> Self {
        Self {
            device: "cpu".into(),
            warmup: 1,
            repetitions: 3,
            options: Options::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCase {
    name: String,
    kind: Kind,
    device: Option<String>,
    warmup: Option<usize>,
    repetitions: Option<usize>,
    model: Option<String>,
    model_path: Option<String>,
    layout_path: Option<String>,
    #[serde(default)]
    models: Models,
    #[serde(default)]
    options: Options,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Case {
    pub(crate) name: String,
    pub(crate) kind: Kind,
    pub(crate) device: String,
    pub(crate) warmup: usize,
    pub(crate) repetitions: usize,
    pub(crate) model: Option<String>,
    pub(crate) model_path: Option<String>,
    pub(crate) layout_path: Option<String>,
    pub(crate) models: Models,
    pub(crate) options: Options,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawManifest {
    #[serde(default)]
    inputs: Inputs,
    #[serde(default)]
    defaults: Defaults,
    cases: Vec<RawCase>,
}

pub(crate) struct Manifest {
    pub(crate) inputs: Inputs,
    pub(crate) cases: Vec<Case>,
}

pub(crate) fn parse_device(value: &str) -> Result<(String, Option<u32>)> {
    let device = value.to_ascii_lowercase();
    if device == "auto" || device == "cpu" || device == "metal" {
        return Ok((device, None));
    }
    if let Some(ordinal) = device.strip_prefix("cuda:") {
        let ordinal = ordinal
            .parse::<u32>()
            .context("device must be auto, cpu, cuda:N, or metal")?;
        ensure!(ordinal <= i32::MAX as u32, "CUDA ordinal is too large");
        return Ok((format!("cuda:{ordinal}"), Some(ordinal)));
    }
    bail!("device must be auto, cpu, cuda:N, or metal (got {value:?})")
}

impl Manifest {
    pub(crate) fn parse(text: &str, device: Option<&str>, inputs: Option<Inputs>) -> Result<Self> {
        let mut raw: RawManifest = toml::from_str(text).context("invalid benchmark manifest")?;
        if let Some(inputs) = inputs {
            raw.inputs = inputs;
        }
        ensure!(!raw.cases.is_empty(), "manifest has no cases");
        ensure!(
            !raw.inputs.images.is_empty()
                || !raw.inputs.image_dirs.is_empty()
                || !raw.inputs.pdfs.is_empty(),
            "manifest has no inputs"
        );
        if let Some(scale) = raw.inputs.pdf_scale {
            ensure!(
                scale.is_finite() && scale > 0.0 && scale <= 4.0,
                "pdf_scale must be in (0, 4]"
            );
        }
        ensure!(
            raw.inputs.max_pages != Some(0),
            "max_pages must be positive"
        );
        let mut names = BTreeSet::new();
        let mut cases = Vec::new();
        for row in raw.cases {
            let mut case = Case {
                name: row.name,
                kind: row.kind,
                device: device
                    .unwrap_or(row.device.as_deref().unwrap_or(&raw.defaults.device))
                    .into(),
                warmup: row.warmup.unwrap_or(raw.defaults.warmup),
                repetitions: row.repetitions.unwrap_or(raw.defaults.repetitions),
                model: row.model,
                model_path: row.model_path,
                layout_path: row.layout_path,
                models: row.models,
                options: row.options.inherit(&raw.defaults.options),
            };
            case.device = parse_device(&case.device)?.0;
            case.validate()?;
            ensure!(
                names.insert(case.name.clone()),
                "duplicate case name {}",
                case.name
            );
            cases.push(case);
        }
        Ok(Self {
            inputs: raw.inputs,
            cases,
        })
    }
}

impl Case {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(!self.name.trim().is_empty(), "case name cannot be empty");
        parse_device(&self.device)?;
        ensure!(
            self.repetitions > 0,
            "{}: repetitions must be positive",
            self.name
        );
        ensure!(
            self.options.batch_size() > 0
                && self.options.cpu_threads() > 0
                && self.options.interval_ms() > 0,
            "{}: batch size, CPU threads, and interval must be positive",
            self.name
        );
        ensure!(
            self.options.region_batch_size != Some(0) && self.options.max_tokens != Some(0),
            "{}: region batch size and max tokens must be positive",
            self.name
        );
        if let Some(selector) = &self.options.nvml_device {
            ensure!(
                selector.starts_with("GPU-")
                    || selector
                        .strip_prefix("index:")
                        .is_some_and(|n| n.parse::<u32>().is_ok()),
                "nvml_device must be a GPU UUID or index:N"
            );
        }
        match self.kind {
            Kind::Ocr => ensure!(
                self.models.detector.is_some()
                    && self.models.recognizer.is_some()
                    && self.models.dictionary.is_some(),
                "{}: OCR requires detector, recognizer, and dictionary",
                self.name
            ),
            Kind::Structure => {
                ensure!(
                    self.models.layout.is_some(),
                    "{}: structure requires layout",
                    self.name
                );
                let ocr = [
                    self.models.detector.is_some(),
                    self.models.recognizer.is_some(),
                    self.models.dictionary.is_some(),
                ];
                ensure!(
                    ocr.iter().all(|v| *v) || ocr.iter().all(|v| !*v),
                    "{}: structure OCR requires all three OCR models",
                    self.name
                );
                if self.models.wired_table_structure.is_some()
                    || self.models.wireless_table_structure.is_some()
                {
                    ensure!(
                        self.models.table_dictionary.is_some(),
                        "table structure requires table_dictionary"
                    );
                }
                if self.models.formula.is_some() {
                    ensure!(
                        self.models.formula_tokenizer.is_some()
                            && self.models.formula_type.is_some(),
                        "formula needs tokenizer and type"
                    );
                }
            }
            Kind::Vl => {
                let model = self.model.as_deref().context("VL requires model")?;
                ensure!(self.model_path.is_some(), "VL requires model_path");
                ensure!(
                    self.options.batch_size() == 1,
                    "PageParser supports one page at a time; use region_batch_size for VL"
                );
                ensure!(is_supported_model(model), "unsupported VL model {model}");
                if is_external_model(model) {
                    ensure!(self.layout_path.is_some(), "{model} requires layout_path");
                }
                let batched = is_external_model(model) || model == "mineru";
                ensure!(
                    batched
                        || self.options.region_batch_size.is_none()
                        || self.options.region_batch_size == Some(1),
                    "{model} PageParser does not expose region batching"
                );
                ensure!(
                    self.options.use_mtp.is_none() || model == "hpd-parsing",
                    "use_mtp is specific to HPD-Parsing"
                );
                ensure!(
                    self.options.diffusion_seed.is_none() || model == "mineru-diffusion",
                    "diffusion_seed is specific to MinerU-Diffusion"
                );
            }
        }
        Ok(())
    }
    pub(crate) fn model_path(&self) -> &str {
        self.model_path.as_deref().unwrap_or_default()
    }
}

pub(crate) fn is_external_model(model: &str) -> bool {
    matches!(
        model,
        "paddleocr-vl" | "paddleocr-vl-1.5" | "paddleocr-vl-1.6" | "glmocr" | "teleocr"
    )
}
pub(crate) fn is_supported_model(model: &str) -> bool {
    is_external_model(model)
        || matches!(
            model,
            "hpd-parsing"
                | "hunyuanocr"
                | "jina-ocr"
                | "mineru"
                | "mineru-diffusion"
                | "monkeyocrv2"
                | "ovisocr2"
                | "wevisdoc"
                | "xiaomi-ocr-0"
        )
}
pub(crate) fn model_source(root: &Path, value: &str) -> std::path::PathBuf {
    let path = Path::new(value);
    if path.components().count() == 1 || path.is_absolute() {
        path.to_owned()
    } else {
        root.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SAMPLE: &str = "[inputs]\nimages=['page.png']\n[defaults]\nwarmup=2\n[defaults.options]\ncpu_threads=2\n[[cases]]\nname='tiny'\nkind='ocr'\n[cases.models]\ndetector='det.onnx'\nrecognizer='rec.onnx'\ndictionary='dict.txt'\n";
    #[test]
    fn cli_inputs_replace_the_entire_block_and_allow_missing_manifest_inputs() {
        let inputs = Inputs {
            images: vec!["a.png".into(), "b.png".into(), "c.png".into()],
            ..Default::default()
        };
        let limited = SAMPLE.replace(
            "images=['page.png']",
            "images=['page.png']\nmax_pages=1\npdf_scale=3.0",
        );
        let manifest = Manifest::parse(&limited, None, Some(inputs.clone())).unwrap();
        assert_eq!(manifest.inputs, inputs);
        let no_inputs = SAMPLE
            .strip_prefix("[inputs]\nimages=['page.png']\n")
            .unwrap();
        assert!(Manifest::parse(no_inputs, None, None).is_err());
        assert_eq!(
            Manifest::parse(no_inputs, None, Some(inputs.clone()))
                .unwrap()
                .inputs,
            inputs
        );
    }
    #[test]
    fn parses_defaults_and_cli_override() {
        let manifest = Manifest::parse(SAMPLE, Some("cuda:2"), None).unwrap();
        assert_eq!(manifest.cases[0].warmup, 2);
        assert_eq!(manifest.cases[0].device, "cuda:2");
        assert_eq!(manifest.cases[0].options.cpu_threads(), 2);
    }
    #[test]
    fn rejects_invalid_cases_and_typos() {
        assert!(Manifest::parse(&SAMPLE.replace("warmup=2", "warmupp=2"), None, None).is_err());
        assert!(
            Manifest::parse(&SAMPLE.replace("kind='ocr'", "kind='invalid'"), None, None).is_err()
        );
        assert!(Manifest::parse(&SAMPLE.replace("warmup=2", "repetitions=0"), None, None).is_err());
        assert_eq!(
            Manifest::parse(SAMPLE, Some("AUTO"), None).unwrap().cases[0].device,
            "auto"
        );
        assert!(Manifest::parse(SAMPLE, Some("cuda:-1"), None).is_err());
    }
    #[test]
    fn detects_duplicate_names() {
        let second = SAMPLE.split("[[cases]]").nth(1).unwrap();
        assert!(Manifest::parse(&format!("{SAMPLE}\n[[cases]]{second}"), None, None).is_err());
    }
    #[test]
    fn default_manifest_is_valid() {
        let manifest =
            Manifest::parse(include_str!("../manifests/default.toml"), None, None).unwrap();
        assert!(manifest.cases.iter().filter(|c| c.kind == Kind::Vl).count() >= 12);
    }
}
