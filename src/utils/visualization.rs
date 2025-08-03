//! Visualization utilities for OCR results.
//!
//! This module provides functions for creating visual representations of OCR results,
//! including bounding boxes, detected text, and confidence indicators. It supports
//! both full OCR pipeline results and text detection results.
//!
//! # Features
//!
//! - Visualization of complete OCR results with original and processed images side-by-side
//! - Visualization of text detection results with confidence indicators
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

use crate::pipeline::OAROCRResult;
use crate::predictor::db_detector::TextDetResult;
use crate::processors::BoundingBox;

use ab_glyph::FontVec;
use image::{Rgb, RgbImage, imageops};
use imageproc::drawing::{
    draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut,
};
use imageproc::rect::Rect;
use std::path::Path;
use tracing::{debug, info};

const BBOX_COLOR: Rgb<u8> = Rgb([0, 255, 0]);

const TEXT_COLOR: Rgb<u8> = Rgb([0, 0, 0]);

const BACKGROUND_COLOR: Rgb<u8> = Rgb([255, 255, 255]);

const HIGH_CONFIDENCE_COLOR: Rgb<u8> = Rgb([0, 255, 0]);

const MEDIUM_CONFIDENCE_COLOR: Rgb<u8> = Rgb([255, 165, 0]);

const LOW_CONFIDENCE_COLOR: Rgb<u8> = Rgb([255, 0, 0]);

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
    pub fn with_font_path(font_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let font_data = std::fs::read(font_path)?;
        let font = FontVec::try_from_vec(font_data).map_err(|_| "Failed to parse font file")?;

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
        let font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ];

        for path in &font_paths {
            if let Ok(font_data) = std::fs::read(path) {
                if let Ok(font) = FontVec::try_from_vec(font_data) {
                    info!("Loaded system font: {}", path);
                    return Self {
                        font: Some(font),
                        font_scale: 16.0,
                        bbox_thickness: 2,
                    };
                }
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
/// - Different colors for different confidence levels (if applicable)
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
) -> Result<RgbImage, Box<dyn std::error::Error>> {
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
) -> Result<(), Box<dyn std::error::Error>> {
    // Get image dimensions for bounds checking
    let img_bounds = (img.width() as i32, img.height() as i32);

    for (i, bbox) in result.text_boxes.iter().enumerate() {
        draw_bounding_box(img, bbox, config, x_offset, img_bounds);

        draw_text_for_bbox(img, result, bbox, config, i, x_offset, img_bounds);
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
            draw_hollow_rect_mut(img, thick_rect, BBOX_COLOR);
        }
    }
}

/// Draws recognized text within a bounding box on an image.
///
/// This function retrieves the recognized text for a given bounding box and draws it
/// on the image. It handles both horizontal and vertical text layouts based on the
/// bounding box dimensions.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `result` - The OCR results containing recognized text
/// * `bbox` - The bounding box for positioning the text
/// * `config` - Visualization configuration including font settings
/// * `index` - Index of the text in the results to draw
/// * `x_offset` - Horizontal offset for positioning
/// * `img_bounds` - Image dimensions as (width, height) for bounds checking
fn draw_text_for_bbox(
    img: &mut RgbImage,
    result: &OAROCRResult,
    bbox: &BoundingBox,
    config: &VisualizationConfig,
    index: usize,
    x_offset: i32,
    img_bounds: (i32, i32),
) {
    // Check if we have text for this bounding box
    if index >= result.rec_texts.len() {
        return;
    }

    let text = &result.rec_texts[index];
    let _score = result.rec_scores.get(index).unwrap_or(&0.0);
    let Some(ref font) = config.font else { return };

    let Some(layout) = calculate_text_layout(bbox, x_offset, text, font) else {
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

    if let Some(actual_width) = measure_text_width(text, font, font_scale) {
        if actual_width > available_width {
            let scale_factor = available_width / actual_width;
            font_scale = (font_scale * scale_factor).max(MIN_FONT_SIZE);
        }
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
///
/// # Arguments
///
/// * `text` - The text to be laid out vertically
/// * `_font` - The font to be used (currently unused in this implementation)
/// * `bbox_rect` - The bounding rectangle for the text
///
/// # Returns
///
/// An Option containing the calculated TextLayout, or None if layout could not be determined.
fn calculate_vertical_text_layout(
    text: &str,
    _font: &FontVec,
    bbox_rect: &Rect,
) -> Option<TextLayout> {
    // Define padding and minimum font size
    const PADDING: f32 = 4.0;
    const MIN_FONT_SIZE: f32 = 8.0;

    let available_width = bbox_rect.width() as f32 - PADDING;
    let available_height = bbox_rect.height() as f32 - PADDING;

    let mut font_scale = (available_width * 0.8).max(MIN_FONT_SIZE);
    let mut line_height = font_scale * 1.1;

    if line_height <= 0.0 {
        return None;
    }

    let display_chars: Vec<char> = text.chars().collect();
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
) -> Result<(), Box<dyn std::error::Error>> {
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

/// Creates a visualization of text detection results and saves it to a file.
///
/// This function generates a visualization that shows the detected text regions
/// with bounding boxes and confidence indicators. Different colors are used
/// to represent different confidence levels:
/// - Green for high confidence (> 0.8)
/// - Orange for medium confidence (> 0.5)
/// - Red for low confidence (≤ 0.5)
///
/// The visualization also includes:
/// - Corner points of each detected polygon
/// - Confidence indicators in the top-left corner of each detection
///
/// # Arguments
///
/// * `image` - The original image on which to draw the detection results
/// * `result` - The TextDetResult containing the detection results to visualize
/// * `output_path` - The path where the visualization image will be saved
///
/// # Returns
///
/// A Result indicating success or failure of the visualization process.
pub fn visualize_detection_results(
    image: &RgbImage,
    result: &TextDetResult,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut vis_image = image.clone();
    let img_bounds = (vis_image.width() as i32, vis_image.height() as i32);

    for (polys, scores) in result.dt_polys.iter().zip(result.dt_scores.iter()) {
        for (poly, &score) in polys.iter().zip(scores.iter()) {
            if poly.points.len() >= 4 {
                let color = get_confidence_color(score);
                draw_detection_polygon(&mut vis_image, poly, color, img_bounds);
            }
        }
    }

    vis_image.save(output_path)?;
    info!("Visualization saved to: {}", output_path);

    Ok(())
}

fn get_confidence_color(score: f32) -> Rgb<u8> {
    match score {
        s if s > 0.8 => HIGH_CONFIDENCE_COLOR,
        s if s > 0.5 => MEDIUM_CONFIDENCE_COLOR,
        _ => LOW_CONFIDENCE_COLOR,
    }
}

/// Draws a detection polygon with confidence indicators on the image.
///
/// This function draws the bounding box of the detected text region with a thick rectangle,
/// marks the corner points of the polygon, and adds a confidence indicator.
///
/// # Arguments
///
/// * `img` - The image on which to draw
/// * `poly` - The bounding box polygon to draw
/// * `color` - The color to use for drawing, based on confidence level
/// * `img_bounds` - The dimensions of the image as (width, height)
fn draw_detection_polygon(
    img: &mut RgbImage,
    poly: &BoundingBox,
    color: Rgb<u8>,
    img_bounds: (i32, i32),
) {
    // Convert polygon to rectangle for easier drawing
    let Some(rect) = bbox_to_rect(poly, 0) else {
        return;
    };

    // Draw thick rectangle around the detection
    draw_thick_rectangle(
        img,
        (rect.left(), rect.top(), rect.width(), rect.height()),
        color,
        2,
        img_bounds,
    );

    // Draw corner points of the polygon
    draw_corner_points(img, &poly.points, color, img_bounds);

    // Draw confidence indicator in the top-left corner
    draw_confidence_indicator(img, rect.left() - 15, rect.top() - 15, color);
}

/// Draws a thick rectangle on an image.
///
/// This function draws a rectangle with the specified thickness by drawing
/// multiple hollow rectangles with increasing offsets.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `rect` - The rectangle to draw (x, y, width, height)
/// * `color` - The color to draw with
/// * `thickness` - The thickness of the rectangle lines
/// * `img_bounds` - Image dimensions as (width, height) for bounds checking
fn draw_thick_rectangle(
    img: &mut RgbImage,
    rect: (i32, i32, u32, u32),
    color: Rgb<u8>,
    thickness: i32,
    img_bounds: (i32, i32),
) {
    let (x, y, width, height) = rect;
    let (img_width, img_height) = img_bounds;

    for t in 0..thickness {
        let rect = Rect::at(x - t, y - t).of_size(width + (2 * t) as u32, height + (2 * t) as u32);

        if is_rect_in_bounds(&rect, img_width, img_height) {
            draw_hollow_rect_mut(img, rect, color);
        }
    }
}

/// Draws corner points of a polygon on an image.
///
/// This function draws small filled circles at each corner point of a polygon
/// to highlight the exact detection points.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `points` - The corner points to draw
/// * `color` - The color to draw with
/// * `img_bounds` - Image dimensions as (width, height) for bounds checking
fn draw_corner_points(
    img: &mut RgbImage,
    points: &[crate::processors::Point],
    color: Rgb<u8>,
    img_bounds: (i32, i32),
) {
    let (img_width, img_height) = img_bounds;

    for point in points {
        let x = point.x as i32;
        let y = point.y as i32;

        if is_point_in_bounds(x, y, img_width, img_height) {
            draw_filled_circle_mut(img, (x, y), 3, color);
        }
    }
}

/// Draws a confidence indicator (colored square) on an image.
///
/// This function draws a small colored square that indicates the confidence level
/// of a detection. The color corresponds to the confidence level.
///
/// # Arguments
///
/// * `img` - The image to draw on
/// * `x` - The x-coordinate of the indicator's top-left corner
/// * `y` - The y-coordinate of the indicator's top-left corner
/// * `color` - The color of the indicator (based on confidence level)
fn draw_confidence_indicator(img: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    // Define the size of the confidence indicator
    const INDICATOR_SIZE: u32 = 10;
    let indicator_rect = Rect::at(x, y).of_size(INDICATOR_SIZE, INDICATOR_SIZE);

    draw_filled_rect_mut(img, indicator_rect, color);
}

/// Checks if a point is within the bounds of an image.
///
/// This function verifies that a point's coordinates are within the specified
/// image dimensions, ensuring that drawing operations won't go outside the image boundaries.
///
/// # Arguments
///
/// * `x` - The x-coordinate of the point
/// * `y` - The y-coordinate of the point
/// * `img_width` - The width of the image
/// * `img_height` - The height of the image
///
/// # Returns
///
/// `true` if the point is within the image bounds, `false` otherwise.
fn is_point_in_bounds(x: i32, y: i32, img_width: i32, img_height: i32) -> bool {
    x >= 0 && y >= 0 && x < img_width && y < img_height
}
