//! Error types for the RKNN backend.
//!
//! Codes mirror the `RKNN_*` macros in `rknn_api.h`. The `code_to_msg` helper
//! turns a numeric return code into a short human-readable string.

#![cfg(all(target_arch = "aarch64", feature = "rknpu"))]

use thiserror::Error;

/// All errors that can be produced by the safe RKNN wrapper.
#[derive(Debug, Error)]
pub enum RknnError {
    /// A call to the underlying librknnrt API returned a non-zero status.
    #[error("rknn API call '{api}' returned {code} ({msg})")]
    ApiError {
        api: &'static str,
        code: i32,
        msg: &'static str,
    },
    /// The caller passed an invalid argument before any FFI call was made.
    #[error("invalid argument: {0}")]
    InvalidArg(String),
    /// I/O failure (e.g. reading a model file from disk).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Map a librknnrt error code to a short human-readable message.
///
/// Codes are taken straight from `rknn_api.h`. Unknown codes return
/// `"unknown error"`.
pub(crate) fn code_to_msg(code: i32) -> &'static str {
    match code {
        0 => "success",
        -1 => "execute failed",
        -2 => "execute timeout",
        -3 => "device unavailable",
        -4 => "memory malloc fail",
        -5 => "parameter is invalid",
        -6 => "model is invalid",
        -7 => "context is invalid",
        -8 => "input is invalid",
        -9 => "output is invalid",
        -10 => "device unmatch (update sdk and npu driver/firmware)",
        -11 => "incompatible pre-compile model",
        -12 => "incompatible optimization level version",
        -13 => "target platform unmatch",
        _ => "unknown error",
    }
}
