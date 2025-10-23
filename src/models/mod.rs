//! Model implementations.
//!
//! This module contains pure model implementations that handle preprocessing,
//! inference, and postprocessing. Models are independent of tasks and can be
//! reused across different task adapters.

pub mod classification;
pub mod detection;
pub mod recognition;
pub mod rectification;
