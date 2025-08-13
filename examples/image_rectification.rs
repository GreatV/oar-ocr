//! Image Rectification Example
//!
//! This example demonstrates how to use the DocTr image rectification functionality
//! to correct perspective distortions in document images.
//!
//! # Usage
//! ```bash
//! cargo run --example image_rectification -- \
//!   --model-path /path/to/model.onnx \
//!   --output-dir /path/to/output \
//!   /path/to/image1.jpg [/path/to/image2.jpg ...]
//! ```
//!
//! # Arguments
//! * `--model-path` - Path to the DocTr rectification model file
//! * `--output-dir` - Directory to save rectified images
//! * `images` - Paths to input images to rectify

use clap::Parser;
use oar_ocr::core::config::onnx::{OrtExecutionProvider, OrtSessionConfig};
use oar_ocr::core::traits::StandardPredictor;
use oar_ocr::predictor::DoctrRectifierPredictorBuilder;
use oar_ocr::predictor::doctr_rectifier::DoctrRectifierResult;
use oar_ocr::utils::init_tracing;
use oar_ocr::utils::load_image;
use std::path::Path;

use tracing::{error, info};

/// Command-line arguments for the image rectification example
#[derive(Parser)]
#[command(name = "image_rectification")]
#[command(about = "Image Rectification Example - rectifies document images")]
struct Args {
    /// Path to the DocTr rectification model file
    #[arg(short, long)]
    model_path: String,

    /// Paths to input images to rectify
    #[arg(required = true)]
    images: Vec<String>,

    /// Directory to save rectified images
    #[arg(short, long)]
    output_dir: String,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(short, long, default_value = "cpu")]
    device: String,
}

use image::{Rgb, RgbImage};

/// Creates a comparison image showing the original and rectified images side by side
///
/// # Arguments
/// * `original` - The original image
/// * `rectified` - The rectified image
/// * `output_path` - Path where to save the comparison image
///
/// # Returns
/// * `Ok(())` if the image was successfully created and saved
/// * `Err` if there was an error during image processing or saving
fn create_comparison_image(
    original: &RgbImage,
    rectified: &RgbImage,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate dimensions for the comparison image
    let (orig_width, orig_height) = original.dimensions();
    let (rect_width, rect_height) = rectified.dimensions();

    // Determine the size of the comparison image
    let max_height = orig_height.max(rect_height);
    let total_width = orig_width + rect_width + 20; // 20 pixels spacing between images

    // Create a white background for the comparison image
    let mut comparison = RgbImage::new(total_width, max_height);
    for pixel in comparison.pixels_mut() {
        *pixel = Rgb([255, 255, 255]); // White background
    }

    // Copy the original image to the left side of the comparison
    for y in 0..orig_height {
        for x in 0..orig_width {
            let pixel = original.get_pixel(x, y);
            comparison.put_pixel(x, y, *pixel);
        }
    }

    // Copy the rectified image to the right side of the comparison
    let rect_start_x = orig_width + 20; // Add spacing between images
    for y in 0..rect_height {
        for x in 0..rect_width {
            let pixel = rectified.get_pixel(x, y);
            comparison.put_pixel(rect_start_x + x, y, *pixel);
        }
    }

    // Save the comparison image
    comparison.save(output_path)?;
    info!("Comparison image saved to: {}", output_path);

    Ok(())
}

/// Visualizes the rectification results by saving the rectified image and creating a comparison
///
/// # Arguments
/// * `result` - The rectification result containing input and rectified images
/// * `output_path` - Base path for saving the output images
///
/// # Returns
/// * `Ok(())` if visualization was successful
/// * `Err` if there was an error during visualization
fn visualize_rectification_results(
    result: &DoctrRectifierResult,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if we have images to visualize
    if result.input_img.is_empty() || result.rectified_img.is_empty() {
        return Err("No images to visualize".into());
    }

    // Save the rectified image
    let rectified_img = &result.rectified_img[0];
    rectified_img.save(output_path)?;
    info!("Rectified image saved to: {}", output_path);

    // Create a comparison image with original and rectified images
    let original_img = &result.input_img[0];
    let comparison_path = output_path.replace(".jpg", "_comparison.jpg");
    create_comparison_image(original_img, rectified_img, &comparison_path)?;

    // Log image dimensions for information
    let (orig_w, orig_h) = original_img.dimensions();
    let (rect_w, rect_h) = rectified_img.dimensions();
    info!("Original dimensions: {}x{}", orig_w, orig_h);
    info!("Rectified dimensions: {}x{}", rect_w, rect_h);

    Ok(())
}

/// Main function for the image rectification example
///
/// This function:
/// 1. Parses command-line arguments
/// 2. Initializes the DocTr rectifier predictor
/// 3. Processes input images (either in batch or individual mode)
/// 4. Saves rectified images and comparison images
///
/// # Returns
/// * `Ok(())` if the example completed successfully
/// * `Err` if there was an error during execution
///
/// Parse device string and create appropriate ONNX execution provider
///
/// # Arguments
///
/// * `device` - Device string (e.g., "cpu", "cuda", "cuda:0")
///
/// # Returns
///
/// Vector of execution providers in order of preference
fn parse_device(device: &str) -> Result<Vec<OrtExecutionProvider>, Box<dyn std::error::Error>> {
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
                OrtExecutionProvider::CPU,
            ])
        }
        #[cfg(not(feature = "cuda"))]
        {
            error!("CUDA support not compiled in. Falling back to CPU.");
            Ok(vec![OrtExecutionProvider::CPU])
        }
    } else if device.starts_with("cuda:") {
        #[cfg(feature = "cuda")]
        {
            let device_id_str = device.strip_prefix("cuda:").unwrap();
            let device_id: i32 = device_id_str
                .parse()
                .map_err(|_| format!("Invalid CUDA device ID: {}", device_id_str))?;

            Ok(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(device_id),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    do_copy_in_default_stream: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ])
        }
        #[cfg(not(feature = "cuda"))]
        {
            error!("CUDA support not compiled in. Falling back to CPU.");
            Ok(vec![OrtExecutionProvider::CPU])
        }
    } else {
        Err(format!(
            "Unsupported device: {}. Supported devices: cpu, cuda, cuda:N",
            device
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Image Rectification Example");

    // Validate model path
    let model_path = &args.model_path;
    if !Path::new(model_path).exists() {
        error!("Model file not found: {}", model_path);
        return Err("Model file not found".into());
    }

    // Filter out non-existent image files
    let existing_images: Vec<String> = args
        .images
        .iter()
        .filter(|path| {
            let exists = Path::new(path).exists();
            if !exists {
                error!("Image file not found: {}", path);
            }
            exists
        })
        .cloned()
        .collect();

    // Check if we have any valid images to process
    if existing_images.is_empty() {
        error!("No valid image files found");
        return Err("No valid image files found".into());
    }

    // Parse device configuration
    let execution_providers = parse_device(&args.device)?;
    info!(
        "Using device: {} with providers: {:?}",
        args.device, execution_providers
    );

    // Create ONNX session configuration with device settings
    let ort_config = OrtSessionConfig::new().with_execution_providers(execution_providers);

    // Initialize the DocTr rectifier predictor
    let predictor = DoctrRectifierPredictorBuilder::new()
        .model_name("DocTr_Image_Rectification".to_string())
        .ort_session(ort_config) // Set device configuration
        .build(Path::new(model_path))?;

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(Path::new(image_path)) {
            Ok(img) => {
                images.push(img);
            }
            Err(e) => {
                error!("Failed to load image {}: {}", image_path, e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Perform rectification using the predict API (handles both single and batch automatically)
    let result = match predictor.predict(images, None) {
        Ok(res) => res,
        Err(e) => {
            error!("Rectification failed: {}", e);
            return Err("Rectification failed".into());
        }
    };

    info!(
        "Processing completed: {} rectified images",
        result.rectified_img.len()
    );

    // Generate output files for each rectified image
    for (i, _) in existing_images.iter().enumerate() {
        if i < result.rectified_img.len() {
            let output_filename = format!("rectified_result_{}.jpg", i + 1);
            let output_path = Path::new(&args.output_dir).join(&output_filename);

            // Create a single-image result for visualization
            let single_result = DoctrRectifierResult {
                input_path: vec![result.input_path[i].clone()],
                index: vec![result.index[i]],
                input_img: vec![result.input_img[i].clone()],
                rectified_img: vec![result.rectified_img[i].clone()],
            };

            // Visualize the rectification results
            if let Err(e) =
                visualize_rectification_results(&single_result, output_path.to_str().unwrap())
            {
                error!("Visualization failed for image {}: {}", i + 1, e);
            }
        }
    }

    info!("Example completed!");
    Ok(())
}
