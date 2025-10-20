//! Formula Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize mathematical formulas
//! in images and convert them to LaTeX strings. It supports various formula recognition models
//! from PaddleOCR, including UniMERNet and PP-FormulaNet.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example formula_recognition -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the formula recognition model file (ONNX)
//! * `-t, --tokenizer-path` - Path to the tokenizer file (tokenizer.json)
//! * `-o, --output-dir` - Directory to save visualization results (optional)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `--model-name` - Model name (automatically infers model type). Supported names:
//!   - UniMERNet
//!   - PP-FormulaNet-S, PP-FormulaNet-L
//!   - PP-FormulaNet_plus-S, PP-FormulaNet_plus-M, PP-FormulaNet_plus-L
//! * `--score-thresh` - Score threshold for recognition (default: 0.0)
//! * `--session-pool-size` - Session pool size for concurrent inference (default: 1)
//! * `--target-width` - Target image width (default: auto)
//! * `--target-height` - Target image height (default: auto)
//! * `--max-length` - Maximum formula length in tokens (default: 1536)
//! * `-v, --verbose` - Enable verbose output
//! * `<IMAGES>...` - Paths to input formula images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example formula_recognition -- \
//!     -m models/PP-FormulaNet_plus-M/inference.onnx \
//!     -t models/PP-FormulaNet_plus-M/tokenizer.json \
//!     --model-name "PP-FormulaNet_plus-M" \
//!     formula1.jpg formula2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::traits::adapter::{AdapterBuilder, ModelAdapter};
use oar_ocr::core::traits::task::{ImageTaskInput, Task};
use oar_ocr::domain::tasks::formula_recognition::{
    FormulaRecognitionConfig, FormulaRecognitionTask,
};
use oar_ocr::models::recognition::{PPFormulaNetAdapterBuilder, UniMERNetFormulaAdapterBuilder};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};

#[cfg(feature = "visualization")]
use image::RgbImage;

#[derive(Clone, Debug)]
enum FormulaModelKind {
    UniMERNet,
    PPFormulaNet,
}

impl FormulaModelKind {
    /// Infer model kind from model name
    ///
    /// Supported model names:
    /// - UniMERNet
    /// - PP-FormulaNet-S, PP-FormulaNet-L
    /// - PP-FormulaNet_plus-S, PP-FormulaNet_plus-M, PP-FormulaNet_plus-L
    fn from_model_name(name: &str) -> Self {
        match name {
            "UniMERNet" => FormulaModelKind::UniMERNet,
            "PP-FormulaNet-S"
            | "PP-FormulaNet-L"
            | "PP-FormulaNet_plus-S"
            | "PP-FormulaNet_plus-M"
            | "PP-FormulaNet_plus-L" => FormulaModelKind::PPFormulaNet,
            _ => {
                // Fallback: try to infer from name pattern
                let name_lower = name.to_lowercase();
                if name_lower.contains("unimernet") {
                    FormulaModelKind::UniMERNet
                } else if name_lower.contains("pp-formulanet")
                    || name_lower.contains("ppformulanet")
                {
                    FormulaModelKind::PPFormulaNet
                } else {
                    // Default to UniMERNet
                    FormulaModelKind::UniMERNet
                }
            }
        }
    }
}

/// Command-line arguments for the formula recognition example
#[derive(Parser)]
#[command(name = "formula_recognition")]
#[command(about = "Formula Recognition Example - recognizes mathematical formulas in images")]
struct Args {
    /// Path to the formula recognition model file (ONNX)
    #[arg(short, long)]
    model_path: PathBuf,

    /// Path to the tokenizer file (tokenizer.json)
    #[arg(short, long)]
    tokenizer_path: Option<PathBuf>,

    /// Paths to input formula images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save visualization results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Score threshold for recognition (default: 0.0)
    #[arg(long, default_value = "0.0")]
    score_thresh: f32,

    /// Session pool size for concurrent inference (default: 1)
    #[arg(long, default_value = "1")]
    session_pool_size: usize,

    /// Target image width (default: auto)
    #[arg(long, default_value = "0")]
    target_width: u32,

    /// Target image height (default: auto)
    #[arg(long, default_value = "0")]
    target_height: u32,

    /// Maximum formula length in tokens (default: 1536)
    #[arg(long, default_value = "1536")]
    max_length: usize,

    /// Model name (automatically infers model type)
    /// Supported: UniMERNet, PP-FormulaNet-S/L, PP-FormulaNet_plus-S/M/L
    #[arg(long, default_value = "FormulaRecognition")]
    model_name: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    oar_ocr::utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Formula Recognition Example");

    // Verify that the model file exists
    if !args.model_path.exists() {
        error!("Model file not found: {}", args.model_path.display());
        return Err("Model file not found".into());
    }

    // Filter out non-existent image files and log errors for missing files
    let existing_images: Vec<PathBuf> = args
        .images
        .iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                error!("Image file not found: {}", path.display());
            }
            exists
        })
        .cloned()
        .collect();

    // Exit early if no valid images were provided
    if existing_images.is_empty() {
        error!("No valid image files found");
        return Err("No valid image files found".into());
    }

    // Log device configuration
    info!("Using device: {}", args.device);
    if args.device != "cpu" {
        warn!("GPU support not yet implemented in new architecture. Using CPU.");
    }

    // Create formula recognition configuration
    let config = FormulaRecognitionConfig {
        score_threshold: args.score_thresh,
        max_length: args.max_length,
    };

    // Infer model type from model name
    let model_type = FormulaModelKind::from_model_name(&args.model_name);
    let target_width = args.target_width;
    let target_height = args.target_height;
    let tokenizer_path = args.tokenizer_path.clone();

    if args.verbose {
        info!("Formula Recognition Configuration:");
        info!("  Score threshold: {}", config.score_threshold);
        info!("  Max formula length: {}", config.max_length);
        if target_width > 0 && target_height > 0 {
            info!("  Target size override: {}x{}", target_width, target_height);
        } else {
            info!("  Target size: auto-detect from model input");
        }
    }

    // Build the formula recognition adapter
    if args.verbose {
        info!("Building formula recognition adapter...");
        info!("  Model: {}", args.model_path.display());
        info!("  Session pool size: {}", args.session_pool_size);
        if let Some(ref tokenizer) = tokenizer_path {
            info!("  Tokenizer: {}", tokenizer.display());
        }
        info!("  Inferred model type: {:?}", model_type);
    }

    let adapter: Box<dyn ModelAdapter<Task = FormulaRecognitionTask>> = match model_type {
        FormulaModelKind::UniMERNet => {
            let mut builder = UniMERNetFormulaAdapterBuilder::new()
                .with_config(config.clone())
                .session_pool_size(args.session_pool_size)
                .model_name(&args.model_name);

            if target_width > 0 && target_height > 0 {
                builder = builder.target_size(target_width, target_height);
            }

            if let Some(ref tokenizer) = tokenizer_path {
                builder = builder.tokenizer_path(tokenizer.clone());
            }

            Box::new(builder.build(&args.model_path)?)
        }
        FormulaModelKind::PPFormulaNet => {
            let mut builder = PPFormulaNetAdapterBuilder::new()
                .with_config(config.clone())
                .session_pool_size(args.session_pool_size)
                .model_name(&args.model_name);

            if target_width > 0 && target_height > 0 {
                builder = builder.target_size(target_width, target_height);
            }

            if let Some(ref tokenizer) = tokenizer_path {
                builder = builder.tokenizer_path(tokenizer.clone());
            }

            Box::new(builder.build(&args.model_path)?)
        }
    };

    info!("Formula recognition adapter built successfully");
    if args.verbose {
        let adapter_info = adapter.info();
        info!("  Task type: {:?}", adapter_info.task_type);
        info!("  Model name: {}", adapter_info.model_name);
        info!("  Version: {}", adapter_info.version);
    }

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match image::open(image_path) {
            Ok(img) => {
                let rgb_img = img.to_rgb8();
                if args.verbose {
                    info!(
                        "Loaded image: {} ({}x{})",
                        image_path.display(),
                        rgb_img.width(),
                        rgb_img.height()
                    );
                }
                images.push(rgb_img);
            }
            Err(e) => {
                error!("Failed to load image {}: {}", image_path.display(), e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Create task input
    let input = ImageTaskInput::new(images.clone());

    // Create task for validation
    let task = FormulaRecognitionTask::new(config.clone());
    task.validate_input(&input)?;

    // Run formula recognition
    info!("Running formula recognition...");
    let start = Instant::now();
    let output = adapter.execute(input, Some(&config))?;
    let duration = start.elapsed();

    info!(
        "Recognition completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Formula Recognition Results ===");
    for (idx, (image_path, formula, score)) in existing_images
        .iter()
        .zip(output.formulas.iter())
        .zip(output.scores.iter())
        .map(|((path, formula), score)| (path, formula, score))
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());
        if formula.is_empty() {
            warn!("  No formula recognized (below threshold or invalid)");
        } else {
            info!("  LaTeX: {}", formula);
            if let Some(confidence) = score {
                info!("  Confidence: {:.2}%", confidence * 100.0);
            } else {
                info!("  Confidence: N/A");
            }
        }
    }

    // Save visualization if output directory is provided
    #[cfg(feature = "visualization")]
    if let Some(output_dir) = args.output_dir {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        for (image_path, rgb_img, formula, score) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.formulas.iter())
            .zip(output.scores.iter())
            .map(|(((path, img), formula), score)| (path, img, formula, score))
        {
            if !formula.is_empty() {
                let input_filename = image_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let output_filename = format!("{}_formula.jpg", input_filename);
                let output_path = output_dir.join(&output_filename);

                let visualized = visualize_formula(rgb_img, formula, *score);
                visualized.save(&output_path)?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no formula recognized)",
                    image_path.display()
                );
            }
        }
    }

    Ok(())
}

/// Visualizes recognized formula by drawing the LaTeX string on the image
#[cfg(feature = "visualization")]
fn visualize_formula(img: &RgbImage, formula: &str, score: Option<f32>) -> RgbImage {
    use image::Rgb;
    use imageproc::drawing::draw_text_mut;

    let mut output = img.clone();
    let text_color = Rgb([255u8, 0u8, 0u8]); // Red for text

    // Try to load a font for text rendering
    let font = load_font();

    if let Some(ref font) = font {
        // Draw the recognized formula at the top
        let label = if let Some(s) = score {
            format!("{} ({:.1}%)", formula, s * 100.0)
        } else {
            formula.to_string()
        };
        let text_x = 10;
        let text_y = 10;

        draw_text_mut(&mut output, text_color, text_x, text_y, 16.0, font, &label);
    }

    output
}

#[cfg(feature = "visualization")]
fn load_font() -> Option<ab_glyph::FontVec> {
    use ab_glyph::FontVec;

    // Try common font paths
    let font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ];

    for path in &font_paths {
        if let Ok(font_data) = std::fs::read(path)
            && let Ok(font) = FontVec::try_from_vec(font_data)
        {
            return Some(font);
        }
    }

    None
}
