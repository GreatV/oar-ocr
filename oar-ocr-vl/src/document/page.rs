//! Model-independent page document returned by high-level parsing pipelines.

use serde::{Deserialize, Serialize};

/// One normalized page block and its optional recognized content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBlock {
    /// Model-native semantic label.
    #[serde(rename = "type")]
    pub block_type: String,
    /// Normalized `[x1, y1, x2, y2]` coordinates in `[0, 1]`.
    pub bbox: [f32; 4],
    /// Clockwise rotation applied before recognition.
    pub angle: Option<u16>,
    /// Recognized block content.
    pub content: Option<String>,
}

/// Non-fatal issue produced while parsing a page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseDiagnostic {
    /// Block index when the issue is block-specific.
    pub block_index: Option<usize>,
    /// Pipeline stage that produced the issue.
    pub stage: String,
    /// Human-readable diagnostic.
    pub message: String,
}

/// Unified high-level output for a parsed page.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PageDocument {
    /// Structured page blocks, when the model exposes layout.
    pub blocks: Vec<DocumentBlock>,
    /// Ready-to-render Markdown, when produced natively or by a renderer.
    pub markdown: Option<String>,
    /// Raw model protocol output retained for lossless diagnostics.
    pub raw_output: Option<String>,
    /// Non-fatal block or post-processing failures.
    pub diagnostics: Vec<ParseDiagnostic>,
}
