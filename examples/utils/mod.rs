//! Common utilities for examples.

pub mod device_config;
pub mod image_loader;

pub use device_config::parse_device_config;
#[allow(unused_imports)]
pub use image_loader::load_rgb_image;
