//! Classification model implementations.
//!
//! This module contains pure model implementations for classification tasks.

pub mod line_router;
pub mod pp_lcnet;

pub use line_router::{LineRouterModel, LineRouting};
pub use pp_lcnet::{
    PPLCNetModel, PPLCNetModelBuilder, PPLCNetModelOutput, PPLCNetPostprocessConfig,
    PPLCNetPreprocessConfig,
};
