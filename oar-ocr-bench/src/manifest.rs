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
    pub(crate) max_pages: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Options {
    pub(crate) batch_size: Option<usize>,
    pub(crate) cpu_threads: Option<usize>,
    pub(crate) region_batch_size: Option<usize>,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) gpu_memory_budget: Option<usize>,
}

impl Options {
    /// Fills unset options from `other`, then records the effective batch size
    /// and thread count so equivalent manifests produce equal cases. A default
    /// `max_tokens` only reaches VL cases.
    fn inherit(&self, other: &Self, kind: Kind) -> Self {
        Self {
            batch_size: Some(self.batch_size.or(other.batch_size).unwrap_or(1)),
            cpu_threads: Some(self.cpu_threads.or(other.cpu_threads).unwrap_or(4)),
            region_batch_size: self.region_batch_size.or(other.region_batch_size),
            max_tokens: match kind {
                Kind::Vl => self.max_tokens.or(other.max_tokens),
                _ => self.max_tokens,
            },
            gpu_memory_budget: self.gpu_memory_budget.or_else(|| {
                (kind != Kind::Vl)
                    .then_some(other.gpu_memory_budget)
                    .flatten()
            }),
        }
    }
    pub(crate) fn batch_size(&self) -> usize {
        self.batch_size.unwrap_or(1)
    }
    pub(crate) fn cpu_threads(&self) -> usize {
        self.cpu_threads.unwrap_or(4)
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
}

#[derive(Debug, Clone, Deserialize)]
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
    /// Parser family loaded explicitly instead of detected (vl cases only).
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

fn normalize_device(value: &str) -> Result<String> {
    let device = value.to_ascii_lowercase();
    if matches!(device.as_str(), "auto" | "cpu" | "metal") {
        return Ok(device);
    }
    if let Some(ordinal) = device.strip_prefix("cuda:") {
        let ordinal: i32 = ordinal
            .parse()
            .ok()
            .filter(|n| *n >= 0)
            .context("device must be auto, cpu, cuda:N, or metal")?;
        return Ok(format!("cuda:{ordinal}"));
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
            !raw.inputs.images.is_empty() || !raw.inputs.image_dirs.is_empty(),
            "manifest has no inputs"
        );
        ensure!(
            raw.inputs.max_pages != Some(0),
            "max_pages must be positive"
        );
        let mut names = BTreeSet::new();
        let mut cases = Vec::new();
        for row in raw.cases {
            let case = Case {
                device: normalize_device(
                    device.unwrap_or(row.device.as_deref().unwrap_or(&raw.defaults.device)),
                )?,
                name: row.name,
                kind: row.kind,
                warmup: row.warmup.unwrap_or(raw.defaults.warmup),
                repetitions: row.repetitions.unwrap_or(raw.defaults.repetitions),
                model: row.model,
                model_path: row.model_path,
                layout_path: row.layout_path,
                models: row.models,
                options: row.options.inherit(&raw.defaults.options, row.kind),
            };
            case.validate()?;
            ensure!(
                // Case names become output directories, which may be case-insensitive.
                names.insert(case.name.to_lowercase()),
                "duplicate case name {} (names are compared ignoring case)",
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
    fn validate(&self) -> Result<()> {
        let name = &self.name;
        ensure!(
            self.options.gpu_memory_budget != Some(0),
            "{name}: gpu_memory_budget must be positive"
        );
        ensure!(
            self.kind != Kind::Vl || self.options.gpu_memory_budget.is_none(),
            "{name}: gpu_memory_budget applies only to classic cases"
        );
        ensure!(!name.trim().is_empty(), "case name cannot be empty");
        ensure!(self.repetitions > 0, "{name}: repetitions must be positive");
        ensure!(
            self.options.batch_size() > 0 && self.options.cpu_threads() > 0,
            "{name}: batch size and CPU threads must be positive"
        );
        ensure!(
            self.options.region_batch_size != Some(0) && self.options.max_tokens != Some(0),
            "{name}: region batch size and max tokens must be positive"
        );
        ensure!(
            self.kind == Kind::Vl || self.options.max_tokens.is_none(),
            "{name}: max_tokens applies only to VL cases"
        );
        ensure!(
            self.kind == Kind::Vl || self.model.is_none(),
            "{name}: model applies only to VL cases"
        );
        let m = &self.models;
        match self.kind {
            Kind::Ocr => ensure!(
                m.detector.is_some() && m.recognizer.is_some() && m.dictionary.is_some(),
                "{name}: OCR requires detector, recognizer, and dictionary"
            ),
            Kind::Structure => {
                ensure!(m.layout.is_some(), "{name}: structure requires layout");
                let ocr = [&m.detector, &m.recognizer, &m.dictionary].map(Option::is_some);
                ensure!(
                    ocr.iter().all(|v| *v) || ocr.iter().all(|v| !*v),
                    "{name}: structure OCR requires all three OCR models"
                );
                if m.wired_table_structure.is_some() || m.wireless_table_structure.is_some() {
                    ensure!(
                        m.table_dictionary.is_some(),
                        "{name}: table structure requires table_dictionary"
                    );
                }
            }
            Kind::Vl => {
                ensure!(self.model.is_some(), "{name}: VL requires model");
                ensure!(self.model_path.is_some(), "{name}: VL requires model_path");
                ensure!(
                    self.options.batch_size() == 1,
                    "{name}: PageParser handles one page at a time"
                );
            }
        }
        Ok(())
    }
}

/// Bare file names stay registry names for auto-download; paths resolve
/// against the benchmark root.
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
    fn parses_defaults_and_overrides() {
        let manifest = Manifest::parse(SAMPLE, Some("CUDA:2"), None).unwrap();
        assert_eq!(manifest.cases[0].warmup, 2);
        assert_eq!(manifest.cases[0].device, "cuda:2");
        assert_eq!(manifest.cases[0].options.cpu_threads(), 2);
        let budget = (4 * 1024 * 1024 * 1024u64).min(usize::MAX as u64) as usize;
        let budgeted = SAMPLE.replace(
            "cpu_threads=2",
            &format!("cpu_threads=2\ngpu_memory_budget={budget}"),
        );
        assert_eq!(
            Manifest::parse(&budgeted, None, None).unwrap().cases[0]
                .options
                .gpu_memory_budget,
            Some(budget)
        );
        assert!(Manifest::parse(&budgeted.replace(&budget.to_string(), "0"), None, None).is_err());
        // Omitted options and their explicit defaults produce the same case.
        let explicit = SAMPLE.replace("cpu_threads=2", "cpu_threads=2\nbatch_size=1");
        assert_eq!(
            Manifest::parse(&explicit, None, None).unwrap().cases,
            Manifest::parse(SAMPLE, None, None).unwrap().cases
        );
        // A default max_tokens is meant for VL cases; set on an OCR case it is an error.
        let default_tokens = SAMPLE.replace("cpu_threads=2", "max_tokens=8");
        let manifest = Manifest::parse(&default_tokens, None, None).unwrap();
        assert_eq!(manifest.cases[0].options.max_tokens, None);
        let case_tokens = format!("{SAMPLE}[cases.options]\nmax_tokens=8\n");
        assert!(Manifest::parse(&case_tokens, None, None).is_err());
        let inputs = Inputs {
            images: vec!["a.png".into()],
            ..Default::default()
        };
        let manifest = Manifest::parse(SAMPLE, None, Some(inputs.clone())).unwrap();
        assert_eq!(manifest.inputs, inputs);
    }

    #[test]
    fn rejects_invalid_manifests() {
        assert!(Manifest::parse(&SAMPLE.replace("warmup=2", "warmupp=2"), None, None).is_err());
        assert!(Manifest::parse(&SAMPLE.replace("warmup=2", "repetitions=0"), None, None).is_err());
        assert!(Manifest::parse(SAMPLE, Some("cuda:-1"), None).is_err());
        let no_inputs = SAMPLE.replace("images=['page.png']", "");
        assert!(Manifest::parse(&no_inputs, None, None).is_err());
        let second = SAMPLE.split("[[cases]]").nth(1).unwrap();
        assert!(Manifest::parse(&format!("{SAMPLE}\n[[cases]]{second}"), None, None).is_err());
        let upper = second.replace("name='tiny'", "name='TINY'");
        assert!(Manifest::parse(&format!("{SAMPLE}\n[[cases]]{upper}"), None, None).is_err());
    }

    #[test]
    fn default_manifest_is_valid() {
        let manifest =
            Manifest::parse(include_str!("../manifests/default.toml"), None, None).unwrap();
        assert!(manifest.cases.iter().any(|c| c.kind == Kind::Vl));
    }
}
