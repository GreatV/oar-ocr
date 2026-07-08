//! Model source abstraction: filesystem path or in-memory bytes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Where to load an ONNX model from.
///
/// Every model-accepting builder takes `impl Into<ModelSource>`, so existing
/// path-based call sites keep working while callers can also pass raw model
/// bytes (e.g. from `include_bytes!` or decrypted at runtime):
///
/// ```
/// use oar_ocr_core::core::ModelSource;
///
/// let from_path: ModelSource = "models/det.onnx".into();
/// let from_bytes: ModelSource = vec![0u8; 4].into();
/// assert!(from_path.as_path().is_some());
/// assert!(from_bytes.as_path().is_none());
/// ```
#[derive(Clone)]
pub enum ModelSource {
    /// Load from a file path (resolved through the auto-download cache where
    /// that feature applies).
    Path(PathBuf),
    /// Load from in-memory ONNX bytes. `Arc` keeps clones cheap; models whose
    /// weights live in external-data sidecar files cannot be loaded this way.
    Memory(Arc<[u8]>),
}

impl ModelSource {
    /// The path for `Path` sources, `None` for in-memory sources.
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            Self::Path(p) => Some(p),
            Self::Memory(_) => None,
        }
    }

    /// A path usable for logging and error messages.
    pub fn display_path(&self) -> PathBuf {
        match self {
            Self::Path(p) => p.clone(),
            Self::Memory(bytes) => PathBuf::from(format!("<in-memory: {} bytes>", bytes.len())),
        }
    }
}

impl std::fmt::Debug for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path(p) => f.debug_tuple("Path").field(p).finish(),
            Self::Memory(bytes) => f.debug_struct("Memory").field("len", &bytes.len()).finish(),
        }
    }
}

impl From<&ModelSource> for ModelSource {
    fn from(source: &ModelSource) -> Self {
        source.clone()
    }
}

impl From<PathBuf> for ModelSource {
    fn from(p: PathBuf) -> Self {
        Self::Path(p)
    }
}

impl From<&PathBuf> for ModelSource {
    fn from(p: &PathBuf) -> Self {
        Self::Path(p.clone())
    }
}

impl From<&Path> for ModelSource {
    fn from(p: &Path) -> Self {
        Self::Path(p.to_path_buf())
    }
}

impl From<String> for ModelSource {
    fn from(p: String) -> Self {
        Self::Path(PathBuf::from(p))
    }
}

impl From<&str> for ModelSource {
    fn from(p: &str) -> Self {
        Self::Path(PathBuf::from(p))
    }
}

impl From<Vec<u8>> for ModelSource {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Memory(bytes.into())
    }
}

impl From<Arc<[u8]>> for ModelSource {
    fn from(bytes: Arc<[u8]>) -> Self {
        Self::Memory(bytes)
    }
}

impl From<&'static [u8]> for ModelSource {
    fn from(bytes: &'static [u8]) -> Self {
        Self::Memory(bytes.into())
    }
}

impl<const N: usize> From<&'static [u8; N]> for ModelSource {
    fn from(bytes: &'static [u8; N]) -> Self {
        Self::Memory(bytes.as_slice().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_like_inputs_become_path_sources() {
        for source in [
            ModelSource::from("a/b.onnx"),
            ModelSource::from(String::from("a/b.onnx")),
            ModelSource::from(PathBuf::from("a/b.onnx")),
            ModelSource::from(Path::new("a/b.onnx")),
        ] {
            assert_eq!(source.as_path(), Some(Path::new("a/b.onnx")));
        }
    }

    #[test]
    fn byte_inputs_become_memory_sources() {
        static EMBEDDED: [u8; 4] = [1, 2, 3, 4];
        for source in [
            ModelSource::from(vec![1u8, 2, 3, 4]),
            ModelSource::from(&EMBEDDED),
            ModelSource::from(EMBEDDED.as_slice()),
        ] {
            assert!(source.as_path().is_none());
            let ModelSource::Memory(bytes) = source else {
                panic!("expected memory source");
            };
            assert_eq!(&bytes[..], &[1, 2, 3, 4]);
        }
    }
}
