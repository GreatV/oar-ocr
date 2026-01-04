//! Visualization utilities for OCR and document structure analysis results.
//!
//! This module provides functions for creating visual representations of:
//! - OCR results with bounding boxes and detected text
//! - Document structure analysis results with layout elements, tables, and formulas
//!
//! # Features
//!
//! - Visualization of complete OCR results with original and processed images side-by-side
//! - Visualization of text detection results with bounding boxes
//! - Document structure visualization with type-specific colors (PP-StructureV3 style)
//! - Table cell boundaries and formula bounding boxes
//! - Labels with confidence scores and reading order indices
//! - Configurable fonts, colors, and styling
//! - Support for both horizontal and vertical text layouts
//!
//! # Examples
//!
//! ```rust
//! use oar_ocr::utils::visualization::{create_ocr_visualization, VisualizationConfig};
//! // Assuming you have an OAROCRResult
//! // let result = oar_ocr_result;
//! // let config = VisualizationConfig::with_system_font();
//! // let visualization = create_ocr_visualization(&result, &config);
//! ```

use crate::core::errors::OCRError;
use crate::core::OcrResult;
use crate::domain::structure::{LayoutElement, LayoutElementType, StructureResult, TableResult};
use crate::oarocr::OAROCRResult;
use crate::processors::BoundingBox;

use ab_glyph::FontVec;
use image::{Rgb, RgbImage, Rgba, RgbaImage, imageops};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::path::Path;
use tracing::{debug, info, warn};

const BBOX_COLOR: Rgb<u8> = Rgb([0, 255, 0]);
const WORD_BBOX_COLOR: Rgb<u8> = Rgb([255, 165, 0]); // Orange color for word boxes

const TEXT_COLOR: Rgb<u8> = Rgb([0, 0, 0]);

const BACKGROUND_COLOR: Rgb<u8> = Rgb([255, 255, 255]);

/// Color palette following standard colormap (RGB format).
/// These colors are designed to be visually distinct for different element types.
const COLOR_PALETTE: [[u8; 3]; 20] = [
    [255, 0, 0],   // 0: Red
    [204, 255, 0], // 1: Yellow-Green
    [0, 255, 102], // 2: Green
    [0, 102, 255], // 3: Blue
    [204, 0, 255], // 4: Magenta
    [255, 77, 0],  // 5: Orange
    [128, 255, 0], // 6: Lime
    [0, 255, 178], // 7: Cyan-Green
    [0, 26, 255],  // 8: Deep Blue
    [255, 0, 229], // 9: Pink
    [255, 153, 0], // 10: Orange-Yellow
    [51, 255, 0],  // 11: Bright Green
    [0, 255, 255], // 12: Cyan
    [51, 0, 255],  // 13: Violet
    [255, 0, 153], // 14: Hot Pink
    [255, 229, 0], // 15: Yellow
    [0, 255, 26],  // 16: Spring Green
    [0, 178, 255], // 17: Sky Blue
    [128, 0, 128], // 18: Purple
    [255, 0, 77],  // 19: Crimson
];

/// Dark font color for light backgrounds.
const FONT_COLOR_DARK: Rgba<u8> = Rgba([20, 14, 53, 255]);

/// Light font color for dark backgrounds.
const FONT_COLOR_LIGHT: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// Table cell border color (red).
const TABLE_CELL_COLOR: Rgba<u8> = Rgba([255, 0, 0, 255]);

/// Represents the layout of text for visualization purposes.
///
/// Text can be laid out either horizontally or vertically depending on the aspect ratio
/// of the bounding box. Vertical layout is used when the height of the bounding box is
/// more than 1.2 times its width.
enum TextLayout {
    /// Horizontal text layout with position, scale, and text content
    Horizontal {
        pos: (i32, i32),
        scale: f32,
        text: String,
    },

    /// Vertical text layout with start position, scale, line height, and individual characters
    Vertical {
        start_pos: (i32, i32),
        scale: f32,
        line_height: f32,
        chars: Vec<char>,
    },
}

/// Configuration for OCR visualization.
///
/// This struct holds settings that control how OCR results are visualized,
/// including font settings and bounding box styling. You can customize these
/// settings to change the appearance of the visualization output.
pub struct VisualizationConfig {
    /// The font to use for text rendering. If None, text rendering is skipped.
    pub font: Option<FontVec>,

    /// The scale factor for the font. Defaults to 16.0.
    pub font_scale: f32,

    /// The thickness of bounding box lines. Defaults to 2.
    pub bbox_thickness: i32,
}

impl Default for VisualizationConfig {
    /// Creates a default VisualizationConfig with no font, font scale of 16.0, and bbox thickness of 2.
    fn default() -> Self {
        Self {
            font: None,
            font_scale: 16.0,
            bbox_thickness: 2,
        }
    }
}

impl VisualizationConfig {
    /// Creates a VisualizationConfig with a font loaded from the specified path.
    ///
    /// # Arguments
    ///
    /// * `font_path` - Path to the font file to load
    ///
    /// # Returns
    ///
    /// A Result containing the VisualizationConfig if successful, or an error if the font could not be loaded.
    pub fn with_font_path(font_path: &Path) -> OcrResult<Self> {
        let font_data = std::fs::read(font_path)?;
        let font = FontVec::try_from_vec(font_data).map_err(|_| OCRError::InvalidInput {
            message: format!("Failed to parse font file: {}", font_path.display()),
        })?;

        Ok(Self {
            font: Some(font),
            font_scale: 16.0,
            bbox_thickness: 2,
        })
    }

    /// Creates a VisualizationConfig with a system font.
    ///
    /// This function attempts to load a system font from common locations.
    /// If no system font is found, it falls back to the default configuration.
    ///
    /// # Returns
    ///
    /// A VisualizationConfig with a system font if found, otherwise with default settings.
    pub fn with_system_font() -> Self {
        // Try a mix of common Latin and Chinese fonts
        let font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", // Good Chinese coverage on many Linux distros
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ];

        for path in &font_paths {
            if let Ok(font_data) = std::fs::read(path)
                && let Ok(font) = FontVec::try_from_vec(font_data)
            {
                info!("Loaded system font: {}", path);
                return Self {
                    font: Some(font),
                    font_scale: 16.0,
                    bbox_thickness: 2,
                };
            }
        }

        debug!("No system font found, text rendering will be skipped");
        Self::default()
    }
}

/// Creates an OCR visualization image by combining the original image with detected text and bounding boxes.
///
/// This function generates a visualization that shows the original image on the left and the processed
/// image with text detection results on the right. The right side includes:
/// - Bounding boxes around detected text regions
/// - Recognized text overlaid on the image
///
/// # Arguments
///
/// * `result` - The OAROCRResult containing the OCR results to visualize
/// * `config` - The VisualizationConfig controlling how the visualization is rendered
///
/// # Returns
///
/// A Result containing the RgbImage with the visualization if successful, or an error if visualization failed.
pub fn create_ocr_visualization(
    result: &OAROCRResult,
    config: &VisualizationConfig,
) -> OcrResult<RgbImage> {
    let original_img = &*result.input_img;
    let (width, height) = (original_img.width(), original_img.height());

    let mut vis_img = RgbImage::new(width * 2, height);

    imageops::overlay(&mut vis_img, original_img, 0, 0);

    let fill_rect = Rect::at(width as i32, 0).of_size(width, height);
    draw_filled_rect_mut(&mut vis_img, fill_rect, BACKGROUND_COLOR);

    draw_detection_results(&mut vis_img, result, config, width as i32)?;

    Ok(vis_img)
}

/// Draws detection results (bounding boxes and text) onto an image.
///
/// This function iterates through all detected text boxes and draws both the bounding boxes
/// and the recognized text on the image according to the provided configuration.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `result` - The OCR results containing text boxes and recognized text
/// * `config` - Visualization configuration controlling how elements are drawn
/// * `x_offset` - Horizontal offset for positioning (used when drawing on the right side of a split view)
///
/// # Returns
///
/// A Result indicating success or failure of the drawing operations.
fn draw_detection_results(
    img: &mut RgbImage,
    result: &OAROCRResult,
    config: &VisualizationConfig,
    x_offset: i32,
) -> OcrResult<()> {
    // Get image dimensions for bounds checking
    let img_bounds = (img.width() as i32, img.height() as i32);

    for region in result.text_regions.iter() {
        draw_bounding_box(
            img,
            &region.bounding_box,
            config,
            x_offset,
            img_bounds,
            BBOX_COLOR,
        );

        if let Some(word_boxes) = &region.word_boxes {
            if let Some(text) = &region.text {
                let chars: Vec<char> = text.chars().collect();
                for (i, word_bbox) in word_boxes.iter().enumerate() {
                    // Draw word bounding box
                    draw_bounding_box(
                        img,
                        word_bbox,
                        config,
                        x_offset,
                        img_bounds,
                        WORD_BBOX_COLOR,
                    );

                    // Draw individual character if available
                    if let Some(char_to_draw) = chars.get(i) {
                        draw_text_for_single_char(
                            img,
                            word_bbox,
                            &char_to_draw.to_string(),
                            config,
                            x_offset,
                            img_bounds,
                        );
                    }
                }
            }
        } else {
            // Only draw text for the whole region if word boxes are not present
            draw_text_for_region(img, region, config, x_offset, img_bounds);
        }
    }

    Ok(())
}

/// Draws a bounding box on an image with the specified configuration.
///
/// This function converts a BoundingBox to a Rect and draws it on the image
/// with the specified thickness. It also performs bounds checking to ensure
/// the box is within the image boundaries.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `bbox` - The bounding box to draw
/// * `config` - Visualization configuration controlling line thickness
/// * `x_offset` - Horizontal offset for positioning
/// * `img_bounds` - Image dimensions as (width, height) for bounds checking
fn draw_bounding_box(
    img: &mut RgbImage,
    bbox: &BoundingBox,
    config: &VisualizationConfig,
    x_offset: i32,
    img_bounds: (i32, i32),
    color: Rgb<u8>,
) {
    // Convert the bounding box to a rectangle for easier drawing
    let Some(rect) = bbox_to_rect(bbox, x_offset) else {
        return;
    };
    let (img_width, img_height) = img_bounds;

    if !is_rect_in_bounds(&rect, img_width, img_height) {
        return;
    }

    for thickness in 0..config.bbox_thickness {
        let thick_rect = Rect::at(rect.left() - thickness, rect.top() - thickness).of_size(
            rect.width() + (2 * thickness) as u32,
            rect.height() + (2 * thickness) as u32,
        );

        if is_rect_in_bounds(&thick_rect, img_width, img_height) {
            draw_hollow_rect_mut(img, thick_rect, color);
        }
    }
}

/// Draws recognized text for a single character within its bounding box on an image.
///
/// This function draws a single character from a word box.
/// It attempts to center the character within its small bounding box.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `bbox` - The bounding box for the single character
/// * `char_str` - The character string to draw
/// * `config` - Visualization configuration including font settings
/// * `x_offset` - Horizontal offset for positioning
/// * `img_bounds` - Image dimensions as (width, height) for bounds checking
fn draw_text_for_single_char(
    img: &mut RgbImage,
    bbox: &BoundingBox,
    char_str: &str,
    config: &VisualizationConfig,
    x_offset: i32,
    img_bounds: (i32, i32),
) {
    let Some(ref font) = config.font else { return };
    let Some(bbox_rect) = bbox_to_rect(bbox, x_offset) else {
        return;
    };

    let (img_width, img_height) = img_bounds;

    // Calculate dynamic font scale based on box dimensions
    let box_height = bbox_rect.height() as f32;
    let box_width = bbox_rect.width() as f32;

    // Start with a scale based on height (e.g., 80% of box height)
    // This matches the logic in calculate_horizontal_text_layout (0.7 there)
    // We use 0.8 here to fill the box a bit more for single characters
    let mut font_scale = (box_height * 0.8).max(8.0);

    // Ensure the character fits within the box width
    if let Some(text_width) = measure_text_width(char_str, font, font_scale) {
        // If wider than 90% of box width, scale down
        if text_width > box_width * 0.9 {
            let scale_factor = (box_width * 0.9) / text_width;
            font_scale *= scale_factor;
        }
    }

    let text_width_px = measure_text_width(char_str, font, font_scale)
        .unwrap_or(font_scale)
        .round() as i32;
    let text_height_px = font_scale.round() as i32;

    // Calculate position to center the character in its bbox
    // Note: draw_text_mut draws from top-left of the glyph bounding box
    let text_x = bbox_rect.left() + (bbox_rect.width() as i32 - text_width_px) / 2;

    // Vertical centering:
    // bbox_rect.top() + half box height - half text height
    let text_y = bbox_rect.top() + (bbox_rect.height() as i32 - text_height_px) / 2;

    if text_x >= 0
        && text_y >= 0
        && text_x + text_width_px <= img_width
        && text_y + text_height_px <= img_height
    {
        draw_text_mut(img, TEXT_COLOR, text_x, text_y, font_scale, font, char_str);
    }
}

/// Draws recognized text within a text region on an image.
///
/// This function draws the recognized text from a TextRegion on the image.
/// It handles both horizontal and vertical text layouts based on the
/// bounding box dimensions.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `region` - The text region containing the text and bounding box
/// * `config` - Visualization configuration including font settings
/// * `x_offset` - Horizontal offset for positioning
/// * `img_bounds` - Image dimensions as (width, height) for bounds checking
fn draw_text_for_region(
    img: &mut RgbImage,
    region: &crate::prelude::TextRegion,
    config: &VisualizationConfig,
    x_offset: i32,
    img_bounds: (i32, i32),
) {
    // Check if the text is available (not filtered out)
    let Some(text) = &region.text else {
        return;
    };
    let Some(ref font) = config.font else { return };

    let Some(layout) = calculate_text_layout(&region.bounding_box, x_offset, text, font) else {
        return;
    };

    let (img_width, img_height) = img_bounds;

    match layout {
        TextLayout::Horizontal { pos, scale, text } => {
            if pos.0 >= 0 && pos.1 >= 0 && pos.0 < img_width && pos.1 < img_height {
                draw_text_mut(img, TEXT_COLOR, pos.0, pos.1, scale, font, &text);
            }
        }
        TextLayout::Vertical {
            start_pos,
            scale,
            line_height,
            chars,
        } => {
            let mut current_y = start_pos.1;
            for ch in chars {
                let char_str = ch.to_string();

                let char_width = measure_text_width(&char_str, font, scale).unwrap_or(scale);
                let char_x = start_pos.0 - (char_width / 2.0) as i32;

                if char_x >= 0 && current_y >= 0 && char_x < img_width && current_y < img_height {
                    draw_text_mut(img, TEXT_COLOR, char_x, current_y, scale, font, &char_str);
                }
                current_y += line_height as i32;
            }
        }
    }
}

/// Checks if a rectangle is within the bounds of an image.
///
/// This function verifies that all sides of a rectangle are within the specified
/// image dimensions, ensuring that drawing operations won't go outside the image boundaries.
///
/// # Arguments
///
/// * `rect` - The rectangle to check
/// * `img_width` - The width of the image
/// * `img_height` - The height of the image
///
/// # Returns
///
/// `true` if the rectangle is completely within the image bounds, `false` otherwise.
fn is_rect_in_bounds(rect: &Rect, img_width: i32, img_height: i32) -> bool {
    rect.left() >= 0 && rect.top() >= 0 && rect.right() < img_width && rect.bottom() < img_height
}

/// Converts a BoundingBox to a Rect for easier drawing operations.
///
/// This function calculates the bounding rectangle of a polygon by finding
/// the minimum and maximum x and y coordinates of all points in the bounding box.
///
/// # Arguments
///
/// * `bbox` - The bounding box to convert
/// * `x_offset` - Horizontal offset to apply to the resulting rectangle
///
/// # Returns
///
/// An Option containing the calculated Rect, or None if the bounding box is empty
/// or has invalid dimensions.
fn bbox_to_rect(bbox: &BoundingBox, x_offset: i32) -> Option<Rect> {
    // Return None for empty bounding boxes
    if bbox.points.is_empty() {
        return None;
    }

    let (min_x, max_x, min_y, max_y) = bbox.points.iter().fold(
        (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ),
        |(min_x, max_x, min_y, max_y), p| {
            (
                min_x.min(p.x),
                max_x.max(p.x),
                min_y.min(p.y),
                max_y.max(p.y),
            )
        },
    );

    let left = min_x as i32 + x_offset;
    let top = min_y as i32;
    let width = (max_x - min_x).max(0.0).round() as u32;
    let height = (max_y - min_y).max(0.0).round() as u32;

    (width > 0 && height > 0).then(|| Rect::at(left, top).of_size(width, height))
}

/// Calculates the appropriate text layout (horizontal or vertical) based on the bounding box dimensions.
///
/// This function determines whether text should be laid out horizontally or vertically
/// based on the aspect ratio of the bounding box. If the height is more than 1.2 times
/// the width, vertical layout is used; otherwise, horizontal layout is used.
///
/// # Arguments
///
/// * `bbox` - The bounding box for the text
/// * `x_offset` - The x-axis offset for positioning
/// * `text` - The text to be laid out
/// * `font` - The font to be used for text measurement
///
/// # Returns
///
/// An Option containing the calculated TextLayout, or None if layout could not be determined.
fn calculate_text_layout(
    bbox: &BoundingBox,
    x_offset: i32,
    text: &str,
    font: &FontVec,
) -> Option<TextLayout> {
    // Return None if bbox or text is empty
    if bbox.points.is_empty() || text.is_empty() {
        return None;
    }

    // Convert bbox to rect for easier manipulation
    let bbox_rect = bbox_to_rect(bbox, x_offset)?;
    let bbox_width = bbox_rect.width() as f32;
    let bbox_height = bbox_rect.height() as f32;

    // Return None if bbox dimensions are invalid
    if bbox_width <= 0.0 || bbox_height <= 0.0 {
        return None;
    }

    // Choose layout based on aspect ratio
    // If height is more than 1.2 times the width, use vertical layout
    if bbox_height > bbox_width * 1.2 {
        calculate_vertical_text_layout(text, font, &bbox_rect)
    } else {
        calculate_horizontal_text_layout(text, font, &bbox_rect)
    }
}

/// Calculates horizontal text layout parameters for a given bounding box.
///
/// This function determines the appropriate font size and position for horizontally
/// laid out text within a bounding box, taking into account available space and
/// text length.
///
/// # Arguments
///
/// * `text` - The text to be laid out
/// * `font` - The font to be used for text measurement
/// * `bbox_rect` - The bounding rectangle for the text
///
/// # Returns
///
/// An Option containing the calculated TextLayout, or None if layout could not be determined.
fn calculate_horizontal_text_layout(
    text: &str,
    font: &FontVec,
    bbox_rect: &Rect,
) -> Option<TextLayout> {
    // Define padding and minimum font size
    const PADDING: f32 = 4.0;
    const MIN_FONT_SIZE: f32 = 8.0;

    let available_width = bbox_rect.width() as f32 - PADDING;
    let available_height = bbox_rect.height() as f32;

    let mut font_scale = (available_height * 0.7).max(MIN_FONT_SIZE);

    if let Some(actual_width) = measure_text_width(text, font, font_scale)
        && actual_width > available_width
    {
        let scale_factor = available_width / actual_width;
        font_scale = (font_scale * scale_factor).max(MIN_FONT_SIZE);
    }

    let display_text = text.to_string();

    let text_x = bbox_rect.left() + (PADDING / 2.0) as i32;
    let text_y = bbox_rect.top() + (available_height / 2.0) as i32 - (font_scale / 2.0) as i32;

    Some(TextLayout::Horizontal {
        pos: (text_x, text_y),
        scale: font_scale,
        text: display_text,
    })
}

/// Calculates vertical text layout parameters for a given bounding box.
///
/// This function determines the appropriate font size, line height, and position for vertically
/// laid out text within a bounding box. Each character is positioned on a separate line.
/// The font is used to measure character widths for proper scaling.
///
/// # Arguments
///
/// * `text` - The text to be laid out vertically
/// * `font` - The font to be used for measuring character dimensions
/// * `bbox_rect` - The bounding rectangle for the text
///
/// # Returns
///
/// An Option containing the calculated TextLayout, or None if layout could not be determined.
fn calculate_vertical_text_layout(
    text: &str,
    font: &FontVec,
    bbox_rect: &Rect,
) -> Option<TextLayout> {
    // Define padding and minimum font size
    const PADDING: f32 = 4.0;
    const MIN_FONT_SIZE: f32 = 8.0;

    let available_width = bbox_rect.width() as f32 - PADDING;
    let available_height = bbox_rect.height() as f32 - PADDING;

    let mut font_scale = (available_width * 0.8).max(MIN_FONT_SIZE);
    let mut line_height = font_scale * 1.1;

    // Check if characters fit within the available width at the current scale
    let display_chars: Vec<char> = text.chars().collect();
    if !display_chars.is_empty() {
        // Find the widest character to ensure all characters fit
        let max_char_width = display_chars
            .iter()
            .filter_map(|&ch| measure_text_width(&ch.to_string(), font, font_scale))
            .fold(0.0, f32::max);

        // Scale down if the widest character doesn't fit
        if max_char_width > available_width {
            let scale_factor = available_width / max_char_width;
            font_scale = (font_scale * scale_factor).max(MIN_FONT_SIZE);
            line_height = font_scale * 1.1;
        }
    }

    if line_height <= 0.0 {
        return None;
    }

    let char_count = display_chars.len();

    if char_count == 0 {
        return None;
    }

    let required_height = char_count as f32 * line_height;

    if required_height > available_height {
        let scale_factor = available_height / required_height;
        font_scale = (font_scale * scale_factor).max(MIN_FONT_SIZE);
        line_height = font_scale * 1.1;
    }

    let total_text_height = display_chars.len() as f32 * line_height;
    let start_y = bbox_rect.top()
        + ((available_height - total_text_height) / 2.0).max(0.0) as i32
        + (PADDING / 2.0) as i32;

    let start_x = bbox_rect.left() + (bbox_rect.width() as f32 / 2.0) as i32;

    Some(TextLayout::Vertical {
        start_pos: (start_x, start_y),
        scale: font_scale,
        line_height,
        chars: display_chars,
    })
}

/// Measures the width of text when rendered with a specific font and scale.
///
/// This function calculates the total width of a text string by summing the advance
/// widths of each character when rendered with the specified font and scale.
///
/// # Arguments
///
/// * `text` - The text to measure
/// * `font` - The font to use for measurement
/// * `scale` - The scale at which the font will be rendered
///
/// # Returns
///
/// An Option containing the calculated width, or None if measurement failed.
fn measure_text_width(text: &str, font: &FontVec, scale: f32) -> Option<f32> {
    use ab_glyph::{Font, ScaleFont};

    let scaled_font = font.as_scaled(scale);
    let mut width = 0.0;

    for ch in text.chars() {
        let glyph = scaled_font.scaled_glyph(ch);
        width += scaled_font.h_advance(glyph.id);
    }

    Some(width)
}

/// Creates an OCR visualization and saves it to a file.
///
/// This function generates an OCR visualization image and saves it to the specified output path.
/// It can optionally use a custom font for text rendering.
///
/// # Arguments
///
/// * `result` - The OAROCRResult containing the OCR results to visualize
/// * `output_path` - The path where the visualization image will be saved
/// * `font_path` - Optional path to a custom font file for text rendering
///
/// # Returns
///
/// A Result indicating success or failure of the visualization process.
pub fn visualize_ocr_results(
    result: &OAROCRResult,
    output_path: &Path,
    font_path: Option<&Path>,
) -> OcrResult<()> {
    info!("Creating OCR visualization for: {}", result.input_path);

    let config = create_visualization_config(font_path);
    let vis_img = create_ocr_visualization(result, &config)?;
    vis_img.save(output_path)?;

    info!("Visualization saved to: {}", output_path.display());
    Ok(())
}

/// Creates a VisualizationConfig with appropriate font settings.
///
/// This function attempts to create a VisualizationConfig with a custom font if specified,
/// falling back to system fonts or default settings if the custom font cannot be loaded.
///
/// # Arguments
///
/// * `font_path` - Optional path to a custom font file
///
/// # Returns
///
/// A VisualizationConfig with the appropriate font settings.
fn create_visualization_config(font_path: Option<&Path>) -> VisualizationConfig {
    match font_path {
        // If a custom font path is provided
        Some(path) => VisualizationConfig::with_font_path(path)
            // Log success if custom font is loaded
            .inspect(|_| info!("Using custom font: {}", path.display()))
            // Log error and fall back if custom font fails to load
            .inspect_err(|e| {
                debug!(
                    "Failed to load custom font {}: {}. Falling back to system font.",
                    path.display(),
                    e
                )
            })
            // Use system font as fallback if custom font fails
            .unwrap_or_else(|_| {
                info!("Falling back to system font");
                VisualizationConfig::with_system_font()
            }),
        // If no custom font is specified, use system font
        None => {
            info!("No custom font specified, using system font");
            VisualizationConfig::with_system_font()
        }
    }
}

/// Configuration for document structure visualization.
pub struct StructureVisualizationConfig {
    /// Font for rendering labels.
    pub font: Option<FontVec>,
    /// Font size for labels.
    pub font_size: f32,
    /// Bounding box line thickness.
    pub line_thickness: i32,
    /// Whether to show element labels.
    pub show_labels: bool,
    /// Whether to show confidence scores.
    pub show_scores: bool,
    /// Whether to show reading order indices.
    pub show_order: bool,
    /// Whether to show table cells.
    pub show_table_cells: bool,
}

impl Default for StructureVisualizationConfig {
    fn default() -> Self {
        Self {
            font: None,
            font_size: 14.0,
            line_thickness: 2,
            show_labels: true,
            show_scores: true,
            show_order: true,
            show_table_cells: true,
        }
    }
}

impl StructureVisualizationConfig {
    /// Creates a config with a system font loaded.
    pub fn with_system_font() -> Self {
        let font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ];

        for path in &font_paths {
            if let Ok(font_data) = std::fs::read(path)
                && let Ok(font) = FontVec::try_from_vec(font_data)
            {
                info!("Structure visualization: Loaded font from {}", path);
                return Self {
                    font: Some(font),
                    ..Default::default()
                };
            }
        }

        debug!("No system font found for structure visualization, labels will be skipped");
        Self::default()
    }

    /// Creates a config with a custom font path.
    pub fn with_font_path(font_path: impl AsRef<Path>) -> OcrResult<Self> {
        let font_data = std::fs::read(font_path.as_ref())?;
        let font = FontVec::try_from_vec(font_data).map_err(|_| OCRError::InvalidInput {
            message: format!(
                "Failed to parse font file: {}",
                font_path.as_ref().display()
            ),
        })?;

        Ok(Self {
            font: Some(font),
            ..Default::default()
        })
    }
}

/// Gets the color for a layout element type.
///
/// Colors are assigned based on semantic categories to maintain consistency
/// and visual distinction between different types of elements.
pub fn get_element_color(element_type: &LayoutElementType) -> Rgba<u8> {
    let idx = match element_type {
        // Document structure - warm colors
        LayoutElementType::DocTitle => 0,       // Red
        LayoutElementType::ParagraphTitle => 5, // Orange
        LayoutElementType::Text => 3,           // Blue
        LayoutElementType::Content => 17,       // Sky Blue
        LayoutElementType::Abstract => 8,       // Deep Blue

        // Visual elements - green spectrum
        LayoutElementType::Image => 6, // Lime
        LayoutElementType::Table => 2, // Green
        LayoutElementType::Chart => 7, // Cyan-Green

        // Formulas - purple spectrum
        LayoutElementType::Formula => 18,       // Purple
        LayoutElementType::FormulaNumber => 13, // Violet

        // Captions - yellow spectrum
        LayoutElementType::FigureTitle => 10, // Orange-Yellow
        LayoutElementType::TableTitle => 15,  // Yellow
        LayoutElementType::ChartTitle => 1,   // Yellow-Green
        LayoutElementType::FigureTableChartTitle => 10, // Orange-Yellow

        // Page structure - gray spectrum (using muted colors)
        LayoutElementType::Header => 17,      // Sky Blue
        LayoutElementType::HeaderImage => 17, // Sky Blue
        LayoutElementType::Footer => 12,      // Cyan
        LayoutElementType::FooterImage => 12, // Cyan
        LayoutElementType::Footnote => 12,    // Cyan

        // Special elements
        LayoutElementType::Seal => 14,            // Hot Pink
        LayoutElementType::Number => 9,           // Pink
        LayoutElementType::Reference => 4,        // Magenta
        LayoutElementType::ReferenceContent => 4, // Magenta
        LayoutElementType::Algorithm => 13,       // Violet
        LayoutElementType::AsideText => 11,       // Bright Green
        LayoutElementType::List => 16,            // Spring Green

        // Region blocks
        LayoutElementType::Region => 19, // Crimson

        // Fallback
        LayoutElementType::Other => 19, // Crimson
    };

    let [r, g, b] = COLOR_PALETTE[idx % COLOR_PALETTE.len()];
    Rgba([r, g, b, 255])
}

/// Gets the appropriate font color based on the background color.
/// Returns light font color for dark backgrounds and vice versa.
fn get_font_color(bg_color: Rgba<u8>) -> Rgba<u8> {
    // Calculate relative luminance using sRGB coefficients
    let luminance =
        0.299 * bg_color[0] as f32 + 0.587 * bg_color[1] as f32 + 0.114 * bg_color[2] as f32;

    if luminance > 128.0 {
        FONT_COLOR_DARK
    } else {
        FONT_COLOR_LIGHT
    }
}

/// Creates a visualization image from a StructureResult.
///
/// This function generates a visualization that shows:
/// - Layout element bounding boxes with type-specific colors
/// - Labels with confidence scores
/// - Table cell boundaries
/// - Reading order indices
pub fn create_structure_visualization(
    result: &StructureResult,
    config: &StructureVisualizationConfig,
) -> OcrResult<RgbaImage> {
    // Load the base image
    let base_img = if let Some(ref rectified) = result.rectified_img {
        image::DynamicImage::ImageRgb8((**rectified).clone()).to_rgba8()
    } else {
        image::open(Path::new(result.input_path.as_ref()))?.to_rgba8()
    };

    let mut img = base_img;

    // Draw layout elements
    for element in &result.layout_elements {
        draw_layout_element(&mut img, element, config);
    }

    // Draw table cells if enabled
    if config.show_table_cells {
        for table in &result.tables {
            draw_table_cells(&mut img, table, config);
        }
    }

    // Draw formula bounding boxes
    for formula in &result.formulas {
        let color = get_element_color(&LayoutElementType::Formula);
        draw_structure_bbox(&mut img, &formula.bbox, color, config.line_thickness);

        // Draw label if font is available
        if config.show_labels
            && let Some(ref font) = config.font
        {
            let label = if config.show_scores {
                format!("formula {:.0}%", formula.confidence * 100.0)
            } else {
                "formula".to_string()
            };
            draw_structure_label(
                &mut img,
                &formula.bbox,
                &label,
                color,
                font,
                config.font_size,
            );
        }
    }

    Ok(img)
}

/// Draws a layout element on the image.
fn draw_layout_element(
    img: &mut RgbaImage,
    element: &LayoutElement,
    config: &StructureVisualizationConfig,
) {
    let color = get_element_color(&element.element_type);

    // Draw bounding box
    draw_structure_bbox(img, &element.bbox, color, config.line_thickness);

    // Draw label if enabled and font is available
    if config.show_labels
        && let Some(ref font) = config.font
    {
        let label = build_element_label(element, config);
        draw_structure_label(img, &element.bbox, &label, color, font, config.font_size);
    }
}

/// Builds the label string for an element.
fn build_element_label(element: &LayoutElement, config: &StructureVisualizationConfig) -> String {
    let type_name = element
        .label
        .as_deref()
        .unwrap_or_else(|| element.element_type.as_str());

    let mut label = String::new();

    // Add reading order index if available and enabled
    if config.show_order
        && let Some(order) = element.order_index
    {
        label.push_str(&format!("[{}] ", order));
    }

    // Add type name
    label.push_str(type_name);

    // Add confidence score if enabled
    if config.show_scores {
        label.push_str(&format!(" {:.0}%", element.confidence * 100.0));
    }

    label
}

/// Draws table cells on the image.
fn draw_table_cells(
    img: &mut RgbaImage,
    table: &TableResult,
    config: &StructureVisualizationConfig,
) {
    for cell in &table.cells {
        draw_structure_bbox(img, &cell.bbox, TABLE_CELL_COLOR, config.line_thickness);
    }
}

/// Draws a bounding box on an RGBA image.
fn draw_structure_bbox(img: &mut RgbaImage, bbox: &BoundingBox, color: Rgba<u8>, thickness: i32) {
    if bbox.points.is_empty() {
        return;
    }

    let min_x = bbox.x_min().max(0.0) as i32;
    let min_y = bbox.y_min().max(0.0) as i32;
    let max_x = (bbox.x_max() as i32).min(img.width() as i32 - 1);
    let max_y = (bbox.y_max() as i32).min(img.height() as i32 - 1);

    if max_x <= min_x || max_y <= min_y {
        return;
    }

    let width = (max_x - min_x) as u32;
    let height = (max_y - min_y) as u32;

    // Draw multiple rectangles for thickness effect
    for t in 0..thickness {
        let left = (min_x - t).max(0);
        let top = (min_y - t).max(0);
        let w = (width + 2 * t as u32).min(img.width().saturating_sub(left as u32));
        let h = (height + 2 * t as u32).min(img.height().saturating_sub(top as u32));

        if w > 0 && h > 0 {
            let rect = Rect::at(left, top).of_size(w, h);
            draw_hollow_rect_mut(img, rect, color);
        }
    }
}

/// Draws a label near the bounding box.
fn draw_structure_label(
    img: &mut RgbaImage,
    bbox: &BoundingBox,
    label: &str,
    bg_color: Rgba<u8>,
    font: &FontVec,
    font_size: f32,
) {
    if label.is_empty() || bbox.points.is_empty() {
        return;
    }

    let min_x = bbox.x_min() as i32;
    let min_y = bbox.y_min() as i32;

    // Calculate text dimensions
    let text_width =
        measure_text_width(label, font, font_size).unwrap_or(label.len() as f32 * font_size * 0.6);
    let text_height = font_size;

    // Padding around text
    let padding = 2;
    let rect_width = (text_width as i32 + padding * 2).min(img.width() as i32 - min_x);
    let rect_height = (text_height as i32 + padding * 2).max(1);

    // Position label above the box if possible, otherwise below
    let (rect_x, rect_y) = if min_y > rect_height {
        (min_x, min_y - rect_height)
    } else {
        (min_x, min_y)
    };

    // Ensure we're within bounds
    let rect_x = rect_x.max(0);
    let rect_y = rect_y.max(0);

    if rect_x >= img.width() as i32 || rect_y >= img.height() as i32 {
        return;
    }

    let rect_width = rect_width.min(img.width() as i32 - rect_x) as u32;
    let rect_height = rect_height.min(img.height() as i32 - rect_y) as u32;

    if rect_width == 0 || rect_height == 0 {
        return;
    }

    // Draw background rectangle
    let label_rect = Rect::at(rect_x, rect_y).of_size(rect_width, rect_height);
    draw_filled_rect_mut(img, label_rect, bg_color);

    // Draw text
    let font_color = get_font_color(bg_color);
    let text_x = rect_x + padding;
    let text_y = rect_y + padding;

    if text_x >= 0 && text_y >= 0 && text_x < img.width() as i32 && text_y < img.height() as i32 {
        draw_text_mut(img, font_color, text_x, text_y, font_size, font, label);
    }
}

/// Creates a structure visualization and saves it to a file.
pub fn visualize_structure_results(
    result: &StructureResult,
    output_path: impl AsRef<Path>,
    font_path: Option<&Path>,
) -> OcrResult<()> {
    info!(
        "Creating structure visualization for: {}",
        result.input_path
    );

    let config = match font_path {
        Some(path) => StructureVisualizationConfig::with_font_path(path).unwrap_or_else(|e| {
            warn!("Failed to load custom font: {}, using system font", e);
            StructureVisualizationConfig::with_system_font()
        }),
        None => StructureVisualizationConfig::with_system_font(),
    };

    let vis_img = create_structure_visualization(result, &config)?;
    let out_path = output_path.as_ref();

    // JPEG doesn't support alpha; convert to RGB to avoid encoder errors
    let ext = out_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if ext == "jpg" || ext == "jpeg" {
        image::DynamicImage::ImageRgba8(vis_img)
            .to_rgb8()
            .save(out_path)?;
    } else {
        vis_img.save(out_path)?;
    }

    info!(
        "Structure visualization saved to: {}",
        output_path.as_ref().display()
    );
    Ok(())
}
