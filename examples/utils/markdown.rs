//! Markdown export utilities with image extraction for examples
//!
//! This module provides I/O functions for exporting markdown with extracted images.
//! The library crate provides pure transformations (e.g., `StructureResult::to_markdown()`),
//! while these examples utilities handle the file system operations.

use oar_ocr::domain::structure::{LayoutElementType, StructureResult};
use std::path::Path;

/// Exports markdown with extracted images saved to disk.
///
/// This function generates markdown content and extracts image/chart regions
/// from the source image, saving them to the output directory.
///
/// # Arguments
///
/// * `result` - Structure result containing layout elements and rectified image
/// * `output_dir` - Directory to save extracted images (an `imgs` subdirectory will be created)
///
/// # Returns
///
/// A markdown string with relative paths to the saved images
pub fn export_markdown_with_images(
    result: &StructureResult,
    output_dir: impl AsRef<Path>,
) -> std::io::Result<String> {
    let output_dir = output_dir.as_ref();
    let imgs_dir = output_dir.join("imgs");

    // Create imgs directory if it doesn't exist
    if !imgs_dir.exists() {
        std::fs::create_dir_all(&imgs_dir)?;
    }

    // Extract and save images for Image/Chart elements
    for element in &result.layout_elements {
        if matches!(
            element.element_type,
            LayoutElementType::Image | LayoutElementType::Chart
        ) {
            let type_name = if element.element_type == LayoutElementType::Chart {
                "chart"
            } else {
                "image"
            };

            // Generate image filename matching StructureResult::to_markdown() placeholder
            let img_name = format!(
                "img_in_{}_box_{:.0}_{:.0}_{:.0}_{:.0}.jpg",
                type_name,
                element.bbox.x_min(),
                element.bbox.y_min(),
                element.bbox.x_max(),
                element.bbox.y_max()
            );
            let img_path = imgs_dir.join(&img_name);

            // Extract and save image region if we have the source image
            if let Some(ref img) = result.rectified_img {
                let x = element.bbox.x_min().max(0.0) as u32;
                let y = element.bbox.y_min().max(0.0) as u32;
                let width = ((element.bbox.x_max() - element.bbox.x_min()) as u32)
                    .min(img.width().saturating_sub(x));
                let height = ((element.bbox.y_max() - element.bbox.y_min()) as u32)
                    .min(img.height().saturating_sub(y));

                if width > 0 && height > 0 {
                    let cropped =
                        image::imageops::crop_imm(img.as_ref(), x, y, width, height).to_image();
                    // Save as JPEG to match extension in markdown
                    if let Err(e) = cropped.save(&img_path) {
                        tracing::warn!("Failed to save image {}: {}", img_path.display(), e);
                    }
                }
            }
        }
    }

    // Use core library markdown generation (already implements PaddleX rules)
    Ok(result.to_markdown())
}

/// Exports concatenated markdown from multiple pages with images and post-processing.
pub fn export_concatenated_markdown_with_images(
    results: &[StructureResult],
    output_dir: impl AsRef<Path>,
) -> std::io::Result<String> {
    let output_dir = output_dir.as_ref();

    if results.is_empty() {
        return Ok(String::new());
    }

    // First, save all images from all pages
    for result in results {
        export_markdown_with_images(result, output_dir)?;
    }

    // Use core library concatenation logic (handles paragraph continuity and CJK spacing)
    let raw_markdown = oar_ocr::domain::structure::concatenate_markdown_pages(results);

    // Apply advanced PaddleX post-processing (dehyphenation, word merging fixes, deduplication)
    let processed_markdown = oar_ocr::domain::structure::postprocess_markdown(&raw_markdown);

    Ok(processed_markdown)
}

// Remove redundant local helpers that are now handled by core library
