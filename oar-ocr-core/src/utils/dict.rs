//! Dictionary and tokenizer loading utilities.
//!
//! This module provides helper functions for loading character dictionaries
//! and tokenizer files used throughout the OCR pipeline.

use crate::core::OCRError;
use std::path::Path;

/// Reads a character dictionary file and returns a vector of strings.
///
/// Each line in the file becomes one entry in the resulting vector.
/// Empty lines are preserved.
///
/// # Arguments
///
/// * `path` - Path to the dictionary file
///
/// # Returns
///
/// A vector of strings, one per line in the file.
///
/// # Errors
///
/// Returns an `OCRError::InvalidInput` if the file cannot be read.
///
/// # Example
///
/// ```rust,no_run
/// use oar_ocr_core::utils::read_character_dict;
/// use std::path::Path;
///
/// let dict = read_character_dict(Path::new("path/to/dict.txt"))?;
/// # Ok::<(), oar_ocr_core::core::OCRError>(())
/// ```
pub fn read_character_dict(path: &Path) -> Result<Vec<String>, OCRError> {
    let content = std::fs::read_to_string(path).map_err(|e| OCRError::InvalidInput {
        message: format!(
            "Failed to read character dictionary from '{}': {}",
            path.display(),
            e
        ),
    })?;
    Ok(content.lines().map(|s| s.to_string()).collect())
}

/// Reads a character dictionary file and returns the raw content string.
///
/// This is useful when you need the raw content before processing.
///
/// # Arguments
///
/// * `path` - Path to the dictionary file
///
/// # Returns
///
/// The raw file content as a string.
///
/// # Errors
///
/// Returns an `OCRError::InvalidInput` if the file cannot be read.
pub fn read_dict_content(path: &Path) -> Result<String, OCRError> {
    std::fs::read_to_string(path).map_err(|e| OCRError::InvalidInput {
        message: format!("Failed to read dictionary from '{}': {}", path.display(), e),
    })
}

/// Validates that a required path option is present and returns the path.
///
/// This is a helper for builder patterns where a path is required but stored
/// as an `Option<PathBuf>`.
///
/// # Arguments
///
/// * `path` - Optional path to validate
/// * `component` - Component name for error message (e.g., "text_recognition")
/// * `description` - Human-readable description of what the path is for
///
/// # Returns
///
/// The path if present.
///
/// # Errors
///
/// Returns an `OCRError::ConfigError` if the path is None.
///
/// # Example
///
/// ```rust,no_run
/// use oar_ocr_core::utils::require_path;
/// use std::path::PathBuf;
///
/// let path: Option<PathBuf> = Some(PathBuf::from("/path/to/dict.txt"));
/// let validated = require_path(path, "text_recognition", "character dictionary path")?;
/// # Ok::<(), oar_ocr_core::core::OCRError>(())
/// ```
pub fn require_path<P: AsRef<Path> + Clone>(
    path: Option<P>,
    component: &str,
    description: &str,
) -> Result<P, OCRError> {
    path.ok_or_else(|| {
        OCRError::config_error_detailed(
            component,
            format!("{} is required for {}", description, component),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_character_dict() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "a").unwrap();
        writeln!(file, "b").unwrap();
        writeln!(file, "c").unwrap();

        let dict = read_character_dict(file.path()).unwrap();
        assert_eq!(dict, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_read_dict_content() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "hello\nworld").unwrap();

        let content = read_dict_content(file.path()).unwrap();
        assert_eq!(content, "hello\nworld");
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_character_dict(Path::new("/nonexistent/path/dict.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn test_require_path_some() {
        let path = Some(std::path::PathBuf::from("/some/path"));
        let result = require_path(path, "test", "test path");
        assert!(result.is_ok());
    }

    #[test]
    fn test_require_path_none() {
        let path: Option<std::path::PathBuf> = None;
        let result = require_path(path, "test_component", "test path");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("test_component"));
    }
}
