//! Shared Candle execution machinery used by model implementations.

pub mod attention;
pub(crate) mod cache;
pub(crate) mod checkpoint;
pub(crate) mod decoder_graph;
pub(crate) mod device;
pub(crate) mod errors;
pub(crate) mod image;
pub(crate) mod tensor;

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
