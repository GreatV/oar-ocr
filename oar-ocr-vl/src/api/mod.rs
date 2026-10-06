//! Stable request, result, and error contracts for the VL crate.

pub mod any_page_parser;
#[cfg(feature = "auto-download")]
pub mod download;
pub mod error;
pub mod generation;
pub mod page_parser;
pub mod recognition;
pub mod runtime;
