//! Recognition model implementations.
//!
//! This module contains pure model implementations for recognition tasks.

pub mod crnn;
pub mod pp_formulanet;
pub mod unimernet;

pub use crnn::{CRNNModel, CRNNModelBuilder, CRNNModelOutput, CRNNPreprocessConfig};
pub use pp_formulanet::{
    PPFormulaNetModel, PPFormulaNetModelBuilder, PPFormulaNetModelOutput,
    PPFormulaNetPostprocessConfig, PPFormulaNetPreprocessConfig,
};
pub use unimernet::{
    UniMERNetModel, UniMERNetModelBuilder, UniMERNetModelOutput, UniMERNetPostprocessConfig,
    UniMERNetPreprocessConfig,
};
