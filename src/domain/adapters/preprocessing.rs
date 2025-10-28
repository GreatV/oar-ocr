//! Shared preprocessing helpers for model adapters.
//!
//! Centralizes creation of common preprocess configurations so adapters can
//! reuse well-tested defaults instead of duplicating boilerplate.

use crate::models::classification::PPLCNetPreprocessConfig;
use crate::models::detection::db::DBPreprocessConfig;

/// Construct a PP-LCNet preprocessing config with a custom input shape.
///
/// Leaves other fields at their `Default` values so adapters can override only
/// what they need (e.g., normalization statistics).
pub fn pp_lcnet_preprocess(input_shape: (u32, u32)) -> PPLCNetPreprocessConfig {
    PPLCNetPreprocessConfig {
        input_shape,
        ..Default::default()
    }
}

/// Construct a PP-LCNet preprocessing config with custom normalization stats.
///
/// Useful for adapters that expect zero-centered inputs but otherwise rely on
/// the standard defaults.
pub fn pp_lcnet_preprocess_with_norm(
    input_shape: (u32, u32),
    mean: [f32; 3],
    std: [f32; 3],
) -> PPLCNetPreprocessConfig {
    let mut config = pp_lcnet_preprocess(input_shape);
    config.normalize_mean = mean.to_vec();
    config.normalize_std = std.to_vec();
    config
}

/// Construct a DB preprocessing config that limits images by side length.
pub fn db_preprocess_with_limit_side_len(limit_side_len: u32) -> DBPreprocessConfig {
    DBPreprocessConfig {
        limit_side_len: Some(limit_side_len),
        ..Default::default()
    }
}

/// Construct a DB preprocessing config that resizes by long edge.
pub fn db_preprocess_with_resize_long(resize_long: u32) -> DBPreprocessConfig {
    DBPreprocessConfig {
        resize_long: Some(resize_long),
        ..Default::default()
    }
}
