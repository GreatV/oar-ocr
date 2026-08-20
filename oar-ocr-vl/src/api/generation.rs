//! Common generation controls shared by high-level APIs.

/// Backend-neutral autoregressive generation controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GenerationOptions {
    /// Maximum number of tokens generated after the prompt.
    pub max_new_tokens: usize,
}

impl GenerationOptions {
    pub const fn new(max_new_tokens: usize) -> Self {
        Self { max_new_tokens }
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self::new(4096)
    }
}
