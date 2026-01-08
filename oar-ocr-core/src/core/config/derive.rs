//! Utilities for ConfigValidator trait implementation.
//!
//! The `impl_config_validator!` macro has been replaced by the `#[derive(ConfigValidator)]`
//! procedural derive macro from the `oar-ocr-derive` crate.
//!
//! # Migration
//!
//! Before:
//! ```rust,ignore
//! impl_config_validator!(MyConfig {
//!     score_threshold: range(0.0, 1.0),
//!     batch_size: min(1),
//! });
//! ```
//!
//! After:
//! ```rust,ignore
//! use oar_ocr_core::ConfigValidator;
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
