//! Detection model implementations.
//!
//! This module contains pure model implementations for detection tasks.
//! Models handle preprocessing, inference, and postprocessing independently of tasks.

pub mod db;
pub mod picodet;
pub mod pp_doclayout;
pub mod rtdetr;

pub use db::{DBModel, DBModelBuilder, DBPostprocessConfig, DBPreprocessConfig};
pub use picodet::{
    PicoDetModel, PicoDetModelBuilder, PicoDetModelOutput, PicoDetPostprocessConfig,
    PicoDetPreprocessConfig,
};
pub use pp_doclayout::{
    PPDocLayoutModel, PPDocLayoutModelBuilder, PPDocLayoutModelOutput,
    PPDocLayoutPostprocessConfig, PPDocLayoutPreprocessConfig,
};
pub use rtdetr::{
    RTDetrModel, RTDetrModelBuilder, RTDetrModelOutput, RTDetrPostprocessConfig,
    RTDetrPreprocessConfig,
};
