//! Classification model implementations.
//!
//! This module contains pure model implementations for classification tasks.

pub mod pp_lcnet;

pub use pp_lcnet::{
    PPLCNetModel, PPLCNetModelBuilder, PPLCNetModelOutput, PPLCNetPostprocessConfig,
    PPLCNetPreprocessConfig,
};
