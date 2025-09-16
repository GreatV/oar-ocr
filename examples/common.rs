//! Common utilities for examples.
//!
//! This module contains shared functionality used across multiple examples
//! to reduce code duplication and improve maintainability.

use oar_ocr::core::config::onnx::OrtExecutionProvider;
use tracing::warn;

/// Parses a device string and returns the corresponding execution providers.
///
/// This function handles the common pattern of parsing device strings like "cpu",
/// "cuda", or "cuda:0" and converting them to ONNX Runtime execution providers.
///
/// # Arguments
///
/// * `device` - Device string (e.g., "cpu", "cuda", "cuda:0")
///
/// # Returns
///
/// Vector of execution providers in order of preference
///
/// # Errors
///
/// Returns an error if the device string is not recognized or if CUDA is requested
/// but the cuda feature is not enabled.
pub fn parse_device(device: &str) -> Result<Vec<OrtExecutionProvider>, Box<dyn std::error::Error>> {
    let device = device.to_lowercase();

    if device == "cpu" {
        Ok(vec![OrtExecutionProvider::CPU])
    } else if device == "cuda" {
        #[cfg(feature = "cuda")]
        {
            Ok(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(0),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    do_copy_in_default_stream: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU, // Fallback to CPU
            ])
        }
        #[cfg(not(feature = "cuda"))]
        {
            warn!("CUDA requested but cuda feature not enabled. Falling back to CPU.");
            Ok(vec![OrtExecutionProvider::CPU])
        }
    } else if device.starts_with("cuda:") {
        #[cfg(feature = "cuda")]
        {
            let device_id_str = device.strip_prefix("cuda:").unwrap();
            let device_id: i32 = device_id_str.parse()?;
            Ok(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(device_id),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    do_copy_in_default_stream: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU, // Fallback to CPU
            ])
        }
        #[cfg(not(feature = "cuda"))]
        {
            warn!("CUDA requested but cuda feature not enabled. Falling back to CPU.");
            Ok(vec![OrtExecutionProvider::CPU])
        }
    } else {
        Err(format!("Unsupported device: {}. Supported devices: cpu, cuda, cuda:N", device).into())
    }
}

/// Common argument validation for example applications.
///
/// This function performs common validation checks that are used across
/// multiple examples, such as checking for valid image files.
///
/// # Arguments
///
/// * `image_paths` - Vector of image file paths to validate
///
/// # Returns
///
/// Vector of valid image paths
///
/// # Errors
///
/// Returns an error if no valid image files are found.
pub fn validate_image_paths(image_paths: &[String]) -> Result<Vec<&String>, Box<dyn std::error::Error>> {
    let valid_images: Vec<&String> = image_paths
        .iter()
        .filter(|path| {
            let path = std::path::Path::new(path);
            path.exists() && path.is_file()
        })
        .collect();

    if valid_images.is_empty() {
        return Err("No valid image files found".into());
    }

    Ok(valid_images)
}