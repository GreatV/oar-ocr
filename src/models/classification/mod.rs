//! Classification model adapters.
//!
//! This module contains adapters for classification models (orientation, etc.).

pub mod pp_lcnet_adapter;

pub use pp_lcnet_adapter::{
    DocOrientationAdapter, DocOrientationAdapterBuilder, PPLCNetAdapter, PPLCNetAdapterBuilder,
    TextLineOrientationAdapter, TextLineOrientationAdapterBuilder,
};
