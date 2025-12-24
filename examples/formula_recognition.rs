//! Formula Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize mathematical formulas
//! in images and convert them to LaTeX strings. It supports various formula recognition models
//! from formula recognition models, including UniMERNet and PP-FormulaNet.
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
//! * `--model-name` - Model name to explicitly specify the model type (required for correct model detection).
//!   Supported names:
//!   - `UniMERNet` - UniMERNet formula recognition model
//!   - `PP-FormulaNet-S` - PP-FormulaNet Small variant
//!   - `PP-FormulaNet-L` - PP-FormulaNet Large variant
//!   - `PP-FormulaNet_plus-S` - PP-FormulaNet Plus Small variant
//!   - `PP-FormulaNet_plus-M` - PP-FormulaNet Plus Medium variant
//!   - `PP-FormulaNet_plus-L` - PP-FormulaNet Plus Large variant
//! * `--score-thresh` - Score threshold for recognition (default: 0.0)
//! * `--target-width` - Target image width (default: auto)
//! * `--target-height` - Target image height (default: auto)
//! * `--max-length` - Maximum formula length in tokens (default: 1536)
//! * `-v, --verbose` - Enable verbose output
//! * `<IMAGES>...` - Paths to input formula images to process
//!
//! # Examples
//!
//! Basic usage:
//! ```bash
//! cargo run --example formula_recognition -- \
//!     -m models/PP-FormulaNet_plus-M/inference.onnx \
//!     -t models/PP-FormulaNet_plus-M/tokenizer.json \
//!     --model-name "PP-FormulaNet_plus-M" \
//!     formula1.jpg formula2.jpg
//! ```
//!
//! With visualization (requires `pdflatex` and `convert` from ImageMagick):
//! ```bash
//! cargo run --release --features=visualization --example formula_recognition -- \
//!     -m models/unimernet.onnx \
//!     -t models/unimernet_tokenizer.json \
//!     --model-name UniMERNet \
//!     -o output/ \
//!     formula1.jpg formula2.jpg
//! ```
//!
//! # Visualization
//!
//! When the `visualization` feature is enabled and `--output-dir` is specified,
//! the example will create visualizations with:
//! - Top panel: Original formula image
//! - Bottom panel: Rendered LaTeX formula (requires external tools)
//!
//! ## Requirements for LaTeX Rendering
//!
//! To render LaTeX formulas in the visualization, you need:
//! - `pdflatex` (from TeX Live, MiKTeX, or similar)
//! - `convert` (from ImageMagick)
//!
//! Install on Ubuntu/Debian:
//! ```bash
//! sudo apt-get install texlive-latex-base texlive-latex-extra imagemagick
//! ```
//!
//! Install on macOS:
//! ```bash
//! brew install --cask mactex
//! brew install imagemagick
//! ```
//!
//! If these tools are not available, the visualization will fall back to
//! displaying the LaTeX formula as text.

mod utils;

use clap::Parser;
use oar_ocr::predictors::FormulaRecognitionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

#[cfg(feature = "visualization")]
use image::RgbImage;

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
    tokenizer_path: PathBuf,

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

    /// Target image width (default: auto)
    #[arg(long, default_value = "0")]
    target_width: u32,

    /// Target image height (default: auto)
    #[arg(long, default_value = "0")]
    target_height: u32,

    /// Maximum formula length in tokens (default: 1536)
    #[arg(long, default_value = "1536")]
    max_length: usize,

    /// Model name to explicitly specify the model type (required for correct model detection).
    /// Supported: UniMERNet, PP-FormulaNet-S, PP-FormulaNet-L, PP-FormulaNet_plus-S, PP-FormulaNet_plus-M, PP-FormulaNet_plus-L
    #[arg(long, default_value = "FormulaRecognition")]
    model_name: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

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
    let ort_config = parse_device_config(&args.device)?.unwrap_or_default();

    if ort_config.execution_providers.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    if args.verbose {
        info!("Formula Recognition Configuration:");
        info!("  Score threshold: {}", args.score_thresh);
        info!("  Max formula length: {}", args.max_length);
        if args.target_width > 0 && args.target_height > 0 {
            info!(
                "  Target size override: {}x{}",
                args.target_width, args.target_height
            );
        } else {
            info!("  Target size: auto-detect from model input");
        }
    }

    // Build the formula recognition predictor
    if args.verbose {
        info!("Building formula recognition predictor...");
        info!("  Model: {}", args.model_path.display());
        info!("  Tokenizer: {}", args.tokenizer_path.display());
    }

    let predictor = FormulaRecognitionPredictor::builder()
        .score_threshold(args.score_thresh)
        .model_name(&args.model_name)
        .tokenizer_path(&args.tokenizer_path)
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Formula recognition predictor built successfully");

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(rgb_img) => {
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

    // Run formula recognition
    info!("Running formula recognition...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Recognition completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Formula Recognition Results ===");
    for (idx, (image_path, formula, _score)) in existing_images
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
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

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

/// Visualizes recognized formula by creating a two-panel layout:
/// - Top panel: Original image
/// - Separator: Light gray background
/// - Bottom panel: Rendered LaTeX formula with light gray background
#[cfg(feature = "visualization")]
fn visualize_formula(img: &RgbImage, formula: &str, _score: Option<f32>) -> RgbImage {
    use image::{Rgb, RgbImage};

    let original_width = img.width();
    let original_height = img.height();

    // Try to render LaTeX formula to image
    let rendered_formula = match render_latex_to_image(formula, original_width) {
        Ok(rendered) => rendered,
        Err(e) => {
            warn!(
                "Failed to render LaTeX formula: {}. Using fallback text rendering.",
                e
            );
            return create_fallback_visualization(img, formula);
        }
    };

    let rendered_height = rendered_formula.height();
    let separator_height = 20; // Height for the separator area
    let total_height = original_height + separator_height + rendered_height;

    // Create output image with original on top and rendered formula on bottom
    let mut output = RgbImage::from_pixel(original_width, total_height, Rgb([255u8, 255u8, 255u8]));

    // Copy original image to top panel
    for y in 0..original_height {
        for x in 0..original_width {
            output.put_pixel(x, y, *img.get_pixel(x, y));
        }
    }

    // Draw separator with dashed line
    draw_separator(
        &mut output,
        original_height,
        original_width,
        separator_height,
    );

    // Copy rendered formula to bottom panel
    let y_offset = original_height + separator_height;
    for y in 0..rendered_height {
        for x in 0..original_width.min(rendered_formula.width()) {
            output.put_pixel(x, y_offset + y, *rendered_formula.get_pixel(x, y));
        }
    }

    output
}

/// Draws a separator with light gray background
#[cfg(feature = "visualization")]
fn draw_separator(img: &mut RgbImage, y_start: u32, width: u32, height: u32) {
    use image::Rgb;

    let bg_color = Rgb([240u8, 240u8, 240u8]); // Light gray background

    // Fill separator background
    for y in y_start..(y_start + height) {
        for x in 0..width {
            img.put_pixel(x, y, bg_color);
        }
    }
}

/// Renders LaTeX formula to an image using external tools
#[cfg(feature = "visualization")]
fn render_latex_to_image(
    formula: &str,
    target_width: u32,
) -> Result<RgbImage, Box<dyn std::error::Error>> {
    use image::Rgb;
    use std::fs;
    use std::process::Command;

    // Create temporary directory for LaTeX rendering
    let temp_dir = std::env::temp_dir().join(format!("oar_latex_{}", std::process::id()));
    fs::create_dir_all(&temp_dir)?;

    let tex_file = temp_dir.join("formula.tex");
    let pdf_file = temp_dir.join("formula.pdf");
    let png_file = temp_dir.join("formula.png");

    // Create LaTeX document
    let latex_content = format!(
        r#"\documentclass[border=2pt]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}
\begin{{document}}
${}$
\end{{document}}"#,
        formula
    );

    fs::write(&tex_file, latex_content)?;

    // Compile LaTeX to PDF
    let latex_output = Command::new("pdflatex")
        .arg("-interaction=nonstopmode")
        .arg("-output-directory")
        .arg(&temp_dir)
        .arg(&tex_file)
        .output();

    if latex_output.is_err() || !pdf_file.exists() {
        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
        return Err("pdflatex failed or not installed".into());
    }

    // Convert PDF to PNG using ImageMagick with light gray background
    let convert_output = Command::new("convert")
        .arg("-density")
        .arg("300")
        .arg("-quality")
        .arg("100")
        .arg(&pdf_file)
        .arg("-background")
        .arg("rgb(240,240,240)") // Light gray background
        .arg("-alpha")
        .arg("remove")
        .arg("-alpha")
        .arg("off")
        .arg(&png_file)
        .output();

    if convert_output.is_err() || !png_file.exists() {
        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
        return Err("ImageMagick convert failed or not installed".into());
    }

    // Load the rendered image
    let rendered_img = match load_image(&png_file) {
        Ok(img) => img,
        Err(e) => {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(
                format!("Failed to load rendered PNG {}: {}", png_file.display(), e).into(),
            );
        }
    };

    // Clean up temporary files
    let _ = fs::remove_dir_all(&temp_dir);

    // Resize if necessary to match target width
    let resized = if rendered_img.width() > target_width {
        let scale = target_width as f32 / rendered_img.width() as f32;
        let new_height = (rendered_img.height() as f32 * scale) as u32;
        image::imageops::resize(
            &rendered_img,
            target_width,
            new_height,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        // Center the image on light gray background
        let mut centered = RgbImage::from_pixel(
            target_width,
            rendered_img.height(),
            Rgb([240u8, 240u8, 240u8]),
        );
        let x_offset = (target_width - rendered_img.width()) / 2;
        for y in 0..rendered_img.height() {
            for x in 0..rendered_img.width() {
                centered.put_pixel(x + x_offset, y, *rendered_img.get_pixel(x, y));
            }
        }
        centered
    };

    Ok(resized)
}

/// Fallback visualization when LaTeX rendering is not available
#[cfg(feature = "visualization")]
fn create_fallback_visualization(img: &RgbImage, formula: &str) -> RgbImage {
    use image::{Rgb, RgbImage};
    use imageproc::drawing::draw_text_mut;

    let original_width = img.width();
    let original_height = img.height();

    let margin = 15;
    let line_height = 28;
    let separator_height = 20;
    let max_chars_per_line = ((original_width - 2 * margin) / 7) as usize;
    let formula_lines = wrap_text(formula, max_chars_per_line);
    let num_lines = formula_lines.len().min(15);

    let bottom_panel_height = (num_lines as u32 * (line_height + 2)) + margin * 2;
    let total_height = original_height + separator_height + bottom_panel_height;

    let mut output = RgbImage::from_pixel(original_width, total_height, Rgb([255u8, 255u8, 255u8]));

    // Copy original image
    for y in 0..original_height {
        for x in 0..original_width {
            output.put_pixel(x, y, *img.get_pixel(x, y));
        }
    }

    // Draw separator
    draw_separator(
        &mut output,
        original_height,
        original_width,
        separator_height,
    );

    // Draw formula text
    let font = load_font();
    if let Some(ref font) = font {
        let formula_color = Rgb([0u8, 0u8, 0u8]);
        let mut y_offset = original_height + separator_height + margin;

        for line in formula_lines.iter().take(15) {
            if y_offset + line_height < total_height {
                draw_text_mut(
                    &mut output,
                    formula_color,
                    margin as i32,
                    y_offset as i32,
                    16.0,
                    font,
                    line,
                );
            }
            y_offset += line_height + 2;
        }
    }

    output
}

/// Wraps text to fit within a maximum line length
#[cfg(feature = "visualization")]
fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + word.len() < max_chars {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(text.to_string());
    }

    lines
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
