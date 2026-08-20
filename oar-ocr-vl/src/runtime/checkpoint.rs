//! Checkpoint discovery, JSON loading, and shared configuration validation.

use crate::api::error::Error;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

pub fn collect_safetensors(model_dir: &Path, model_name: &str) -> Result<Vec<PathBuf>, Error> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: SafetensorsIndex =
            load_json_config(&index_path, model_name, "model.safetensors.index.json")?;
        let files: BTreeSet<_> = index
            .weight_map
            .into_values()
            .map(|name| model_dir.join(name))
            .collect();
        if files.is_empty() {
            return Err(Error::config(format!(
                "{model_name}: safetensors index contains no weight shards"
            )));
        }
        for file in &files {
            if !file.is_file() {
                return Err(Error::config(format!(
                    "{model_name}: safetensors index references missing shard {}",
                    file.display()
                )));
            }
        }
        return Ok(files.into_iter().collect());
    }
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let entries = std::fs::read_dir(model_dir).map_err(|error| {
        Error::config(format!(
            "{model_name}: cannot read model dir {}: {error}",
            model_dir.display()
        ))
    })?;
    let mut shards = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| {
                Error::config(format!(
                    "{model_name}: error reading entry in model dir {}: {error}",
                    model_dir.display()
                ))
            })?
            .path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        {
            shards.push(path);
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(Error::config(format!(
            "{model_name}: no model.safetensors or model-*.safetensors found in {}",
            model_dir.display()
        )));
    }
    Ok(shards)
}

pub fn load_json_config<T: serde::de::DeserializeOwned>(
    path: impl AsRef<Path>,
    model_name: &str,
    file_name: &str,
) -> Result<T, Error> {
    let contents = std::fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|error| {
        Error::config(format!("failed to parse {model_name} {file_name}: {error}"))
    })
}

/// Load an optional JSON sidecar. Absence is allowed; malformed or unreadable
/// files are reported instead of silently changing inference behavior.
pub fn load_optional_json_config<T: serde::de::DeserializeOwned>(
    path: impl AsRef<Path>,
    model_name: &str,
    file_name: &str,
) -> Result<Option<T>, Error> {
    let path = path.as_ref();
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Error::Io(error)),
    };
    serde_json::from_str(&contents).map(Some).map_err(|error| {
        Error::config(format!("failed to parse {model_name} {file_name}: {error}"))
    })
}

pub const fn default_true() -> bool {
    true
}

pub const fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

pub fn validate_image_mean_std(
    model_name: &str,
    image_mean: &[f32],
    image_std: &[f32],
) -> Result<(), Error> {
    if image_mean.len() != 3 || image_std.len() != 3 {
        return Err(Error::config(format!(
            "{model_name} image_mean/std must have length 3, got mean={} std={}",
            image_mean.len(),
            image_std.len()
        )));
    }
    Ok(())
}

pub fn validate_patch_merge_temporal(
    model_name: &str,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
) -> Result<(), Error> {
    if patch_size == 0 || merge_size == 0 || temporal_patch_size == 0 {
        return Err(Error::config(format!(
            "{model_name} patch_size/merge_size/temporal_patch_size must be > 0"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn index_is_authoritative_and_deduplicates_shards() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("model-00001.safetensors"), []).unwrap();
        fs::write(directory.path().join("model-00002.safetensors"), []).unwrap();
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00002.safetensors","b":"model-00001.safetensors","c":"model-00002.safetensors"}}"#,
        )
        .unwrap();
        let files = collect_safetensors(directory.path(), "test").unwrap();
        assert_eq!(files.len(), 2);
        assert!(files[0].ends_with("model-00001.safetensors"));
        assert!(files[1].ends_with("model-00002.safetensors"));
    }

    #[test]
    fn index_rejects_missing_shard() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"missing.safetensors"}}"#,
        )
        .unwrap();
        assert!(collect_safetensors(directory.path(), "test").is_err());
    }

    #[test]
    fn optional_json_distinguishes_absent_from_malformed() {
        let directory = tempfile::tempdir().unwrap();
        let missing: Option<serde_json::Value> = load_optional_json_config(
            directory.path().join("missing.json"),
            "test",
            "missing.json",
        )
        .unwrap();
        assert!(missing.is_none());
        fs::write(directory.path().join("broken.json"), "{").unwrap();
        assert!(
            load_optional_json_config::<serde_json::Value>(
                directory.path().join("broken.json"),
                "test",
                "broken.json"
            )
            .is_err()
        );
    }
}
