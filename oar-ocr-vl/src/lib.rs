//! # OAR OCR VL
//!
//! Vision-Language models for document understanding, integrating with oar-ocr-core.
//!
//! ## Modules
//!
//! - `paddleocr_vl` - PaddleOCR-VL, 1.5, and 1.6 for OCR, table, formula,
//!   chart, spotting, and seal recognition
//! - `hunyuanocr` - HunyuanOCR 1.5 / 1.0 OCR expert VLM
//! - `hpd_parsing` - HPD-Parsing hierarchical parallel document parser
//! - `glmocr` - GLM-OCR OCR expert VLM
//! - `mineru` - MinerU2.5 and MinerU2.5-Pro document parsing VLMs (Qwen2-VL
//!   backbone)
//! - `mineru_diffusion` - MinerU-Diffusion-V1 block-diffusion document OCR
//!   (Qwen2-VL vision + SDAR decoder)
//! - `monkeyocrv2` - MonkeyOCRv2-S/B-Parsing full-page and region parsing
//! - `navidc_ocr` - NaviDC-OCR document parsing VLM (Qwen2.5-VL backbone
//!   with a windowed vision tower)
//! - `ovisocr2` - OvisOCR2 end-to-end page-to-Markdown parser (Qwen3.5)
//! - `doc_parser` - Unified document parsing with pluggable recognition backends
//! - `pp_doclayout` - Native PP-DocLayoutV2/V3 layout detection and reading
//!   order
//! - `layout` - Backend-agnostic [`LayoutSource`] trait feeding `doc_parser`
//! - `api` - Stable recognition, page parsing, generation, and runtime contracts
//! - `document` - Standalone VL page and structure types
//! - `pipeline` - Layout-first and model-native parsing orchestration
//! - `render` - Markdown, text, and table output normalization
//! - `attention` - Compatibility path for shared runtime attention
//!
//! [`LayoutSource`]: layout::LayoutSource
//!
//! ## Candle only
//!
//! Everything here runs on Candle, including layout detection via
//! [`PpDocLayout`], so `ort` never enters the dependency tree. To source layout
//! from somewhere else, implement [`LayoutSource`].
//!
//! GPU acceleration is gated behind the `cuda` feature. Parse device strings
//! with [`utils::parse_device`]:
//!
//! ```no_run
//! use oar_ocr_vl::utils::parse_device;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = parse_device("cuda:0")?;
//! # let _ = device;
//! # Ok(())
//! # }
//! ```

// Stable architectural layers.
pub mod api;
pub(crate) mod backbones;
pub mod document;
pub mod pipeline;
pub mod render;
pub(crate) mod runtime;

// Core model modules
pub mod crop;
pub mod doc_parser;
pub mod error;
pub mod geometry;
#[path = "models/glmocr/mod.rs"]
pub mod glmocr;
#[path = "models/hpd_parsing/mod.rs"]
pub mod hpd_parsing;
#[path = "models/hunyuanocr/mod.rs"]
pub mod hunyuanocr;
pub mod layout;
#[path = "models/mineru/mod.rs"]
pub mod mineru;
#[path = "models/mineru_diffusion/mod.rs"]
pub mod mineru_diffusion;
#[path = "models/monkeyocrv2/mod.rs"]
pub mod monkeyocrv2;
#[path = "models/navidc_ocr/mod.rs"]
pub mod navidc_ocr;
#[path = "models/ovisocr2/mod.rs"]
pub mod ovisocr2;
#[path = "models/paddleocr_vl/mod.rs"]
pub mod paddleocr_vl;
#[path = "models/pp_doclayout/mod.rs"]
pub mod pp_doclayout;
pub mod structure;
pub mod utils;

// Backwards-compatible shared attention path.
pub mod attention;

// Re-exports for convenience
pub use paddleocr_vl::{
    PaddleOcrVl, PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig, PaddleOcrVlTask,
};

pub use glmocr::GlmOcr;
pub use hpd_parsing::{HpdGenerationConfig, HpdOutput, HpdParsing, HpdRuntimeStats};
pub use hunyuanocr::{
    DFlashConfig, DFlashTargetConfig, HunyuanOcr, HunyuanOcrParseOptions, HunyuanOcrVersion,
};
pub use mineru::{MinerU, MinerUParseOptions};
pub use mineru_diffusion::{
    DiffusionGenerationConfig, MinerUDiffusion, MinerUDiffusionParseOptions,
};
pub use monkeyocrv2::{MonkeyOcrV2, MonkeyOcrV2ParseOptions, MonkeyOcrV2Task};
pub use navidc_ocr::{NaviDcOcr, NaviDcTask};
pub use ovisocr2::{OvisOcr2, OvisOcr2ParseOptions};

pub use api::generation::GenerationOptions;
pub use api::page_parser::PageParser;
pub use api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
pub use api::runtime::{DTypePolicy, RuntimeConfig};
pub use doc_parser::{DocParser, DocParserConfig};
pub use document::page::{DocumentBlock, PageDocument, ParseDiagnostic};
pub use error::{BatchResult, Error, ProcessingStage, Result};
pub use geometry::{BoundingBox, Point};
pub use layout::{LayoutDetectionElement, LayoutDetections, LayoutSource, StaticLayout};
pub use pipeline::page_parser::LayoutFirstPageParser;
pub use pp_doclayout::{PpDocLayout, PpDocLayoutVersion};
pub use structure::{LayoutElement, LayoutElementType, StructureResult, TableResult, TableType};
