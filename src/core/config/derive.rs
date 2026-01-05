//! Utilities for ConfigValidator trait implementation.
//!
//! Use `#[derive(ConfigValidator)]` from the `oar-ocr-derive` crate:
//!
//! ```rust,ignore
//! use oar_ocr::ConfigValidator;
//!
//! #[derive(ConfigValidator)]
//! pub struct MyConfig {
//!     #[validate(range(0.0, 1.0))]
//!     pub score_threshold: f32,
//!
//!     #[validate(min(1))]
//!     pub batch_size: usize,
//! }
//! ```
