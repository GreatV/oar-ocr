//! Configuration file loading utilities for the OCR pipeline.
//!
//! This module provides utilities for loading OCR pipeline configuration
//! from various file formats including TOML and JSON.

use crate::core::OCRError;
use crate::pipeline::OAROCRConfig;
use std::path::Path;

/// Configuration file format
#[derive(Debug, Clone, Copy)]
pub enum ConfigFormat {
    /// TOML format
    Toml,
    /// JSON format
    Json,
}

impl ConfigFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()? {
            "toml" => Some(Self::Toml),
            "json" => Some(Self::Json),
            _ => None,
        }
    }
}

/// Configuration loader for OCR pipeline
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from a file, auto-detecting the format from the extension
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the configuration file
    ///
    /// # Returns
    ///
    /// A Result containing the loaded OAROCRConfig or an OCRError
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::ConfigLoader;
    /// use std::path::Path;
    ///
    /// let config = ConfigLoader::load_from_file(Path::new("config.toml"))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load_from_file(path: &Path) -> Result<OAROCRConfig, OCRError> {
        let format = ConfigFormat::from_extension(path).ok_or_else(|| OCRError::ConfigError {
            message: format!("Unsupported config file extension: {:?}", path.extension()),
        })?;

        let content = std::fs::read_to_string(path).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to read config file {}: {}", path.display(), e),
        })?;

        Self::load_from_string(&content, format)
    }

    /// Load configuration from a string with specified format
    ///
    /// # Arguments
    ///
    /// * `content` - Configuration content as string
    /// * `format` - Configuration format
    ///
    /// # Returns
    ///
    /// A Result containing the loaded OAROCRConfig or an OCRError
    pub fn load_from_string(content: &str, format: ConfigFormat) -> Result<OAROCRConfig, OCRError> {
        match format {
            ConfigFormat::Toml => Self::load_from_toml(content),
            ConfigFormat::Json => Self::load_from_json(content),
        }
    }

    /// Load configuration from TOML string
    pub fn load_from_toml(content: &str) -> Result<OAROCRConfig, OCRError> {
        toml::from_str(content).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to parse TOML config: {e}"),
        })
    }

    /// Load configuration from JSON string
    pub fn load_from_json(content: &str) -> Result<OAROCRConfig, OCRError> {
        serde_json::from_str(content).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to parse JSON config: {e}"),
        })
    }

    /// Save configuration to a file, auto-detecting the format from the extension
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to save
    /// * `path` - Path to save the configuration file
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError
    pub fn save_to_file(config: &OAROCRConfig, path: &Path) -> Result<(), OCRError> {
        let format = ConfigFormat::from_extension(path).ok_or_else(|| OCRError::ConfigError {
            message: format!("Unsupported config file extension: {:?}", path.extension()),
        })?;

        let content = Self::save_to_string(config, format)?;

        std::fs::write(path, content).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to write config file {}: {}", path.display(), e),
        })
    }

    /// Save configuration to string with specified format
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to save
    /// * `format` - Configuration format
    ///
    /// # Returns
    ///
    /// A Result containing the configuration string or an OCRError
    pub fn save_to_string(config: &OAROCRConfig, format: ConfigFormat) -> Result<String, OCRError> {
        match format {
            ConfigFormat::Toml => Self::save_to_toml(config),
            ConfigFormat::Json => Self::save_to_json(config),
        }
    }

    /// Save configuration to TOML string
    pub fn save_to_toml(config: &OAROCRConfig) -> Result<String, OCRError> {
        toml::to_string_pretty(config).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to serialize config to TOML: {e}"),
        })
    }

    /// Save configuration to JSON string
    pub fn save_to_json(config: &OAROCRConfig) -> Result<String, OCRError> {
        serde_json::to_string_pretty(config).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to serialize config to JSON: {e}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_config_format_detection() {
        assert!(matches!(
            ConfigFormat::from_extension(Path::new("config.toml")),
            Some(ConfigFormat::Toml)
        ));
        assert!(matches!(
            ConfigFormat::from_extension(Path::new("config.json")),
            Some(ConfigFormat::Json)
        ));
        assert!(ConfigFormat::from_extension(Path::new("config.txt")).is_none());
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = OAROCRConfig::new(
            PathBuf::from("detection.onnx"),
            PathBuf::from("recognition.onnx"),
            PathBuf::from("dict.txt"),
        );

        let toml_str = ConfigLoader::save_to_toml(&config).unwrap();
        let loaded_config = ConfigLoader::load_from_toml(&toml_str).unwrap();

        assert_eq!(
            config.character_dict_path,
            loaded_config.character_dict_path
        );
    }

    #[test]
    fn test_json_roundtrip() {
        let config = OAROCRConfig::new(
            PathBuf::from("detection.onnx"),
            PathBuf::from("recognition.onnx"),
            PathBuf::from("dict.txt"),
        );

        let json_str = ConfigLoader::save_to_json(&config).unwrap();
        let loaded_config = ConfigLoader::load_from_json(&json_str).unwrap();

        assert_eq!(
            config.character_dict_path,
            loaded_config.character_dict_path
        );
        assert_eq!(
            config.recognition.model_input_shape,
            loaded_config.recognition.model_input_shape
        );
    }
}
