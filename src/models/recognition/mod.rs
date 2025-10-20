//! Recognition model adapters.
//!
//! This module contains adapters for text and formula recognition models.

pub mod crnn_adapter;
mod paddle_formula_base;
mod pp_formulanet_adapter;
mod unimernet_adapter;

pub use crnn_adapter::{CRNNTextRecognitionAdapter, CRNNTextRecognitionAdapterBuilder};
pub use pp_formulanet_adapter::{PPFormulaNetAdapter, PPFormulaNetAdapterBuilder};
pub use unimernet_adapter::{UniMERNetFormulaAdapter, UniMERNetFormulaAdapterBuilder};
