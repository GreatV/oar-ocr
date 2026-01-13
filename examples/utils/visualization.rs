//! Visualization utilities for OCR examples.
//!
//! This module provides unified visualization functions for all examples:
//! - Detection visualization (bounding boxes with labels)
//! - Classification visualization (header with result)
//! - Layout visualization (colored boxes by element type)
//! - OCR visualization (text regions with recognized text)
//! - Structure visualization (layout elements, tables, formulas)

use ab_glyph::FontVec;
use image::{Rgb, RgbImage, Rgba, RgbaImage, imageops};
use imageproc::drawing::{
    draw_filled_circle_mut, draw_filled_rect_mut, draw_hollow_rect_mut, draw_line_segment_mut,
    draw_text_mut,
};
use imageproc::rect::Rect;
use oar_ocr::oarocr::{OAROCRResult, TextRegion};
use oar_ocr_core::core::OcrResult;
use oar_ocr_core::core::errors::OCRError;
use oar_ocr_core::domain::structure::{
    LayoutElement, LayoutElementType, StructureResult, TableResult,
};
use oar_ocr_core::processors::BoundingBox;
use std::path::Path;
use tracing::{debug, info, warn};

/// Load a system font for text rendering.
pub fn load_system_font() -> Option<FontVec> {
    let font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ];

    for path in &font_paths {
        if let Ok(font_data) = std::fs::read(path)
            && let Ok(font) = FontVec::try_from_vec(font_data)
        {
            debug!("Loaded font from {}", path);
            return Some(font);
        }
    }

    debug!("No system font found");
    None
}

/// Save an image to a file, handling JPEG alpha channel conversion.
pub fn save_image(img: &RgbaImage, output_path: &Path) -> Result<(), String> {
    let ext = output_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if ext == "jpg" || ext == "jpeg" {
        image::DynamicImage::ImageRgba8(img.clone())
            .to_rgb8()
            .save(output_path)
            .map_err(|e| format!("Failed to save image: {}", e))
    } else {
        img.save(output_path)
            .map_err(|e| format!("Failed to save image: {}", e))
    }
}

/// Save an RGB image to a file.
pub fn save_rgb_image(img: &RgbImage, output_path: &Path) -> Result<(), String> {
    img.save(output_path)
        .map_err(|e| format!("Failed to save image: {}", e))
}

/// Configuration for detection visualization.
pub struct DetectionVisConfig {
    /// Box color
    pub box_color: Rgb<u8>,
    /// Label color
    pub label_color: Rgb<u8>,
    /// Font size for labels
    pub font_size: f32,
    /// Line thickness
    pub thickness: i32,
    /// Whether to draw corner points
    pub draw_corners: bool,
    /// Whether to draw polygon lines (for curved text)
    pub draw_polygon: bool,
}

impl Default for DetectionVisConfig {
    fn default() -> Self {
        Self {
            box_color: Rgb([0, 255, 0]),   // Green
            label_color: Rgb([255, 0, 0]), // Red
            font_size: 20.0,
            thickness: 2,
            draw_corners: true,
            draw_polygon: false,
        }
    }
}

impl DetectionVisConfig {
    pub fn with_box_color(mut self, color: Rgb<u8>) -> Self {
        self.box_color = color;
        self
    }

    pub fn with_label_color(mut self, color: Rgb<u8>) -> Self {
        self.label_color = color;
        self
    }

    pub fn with_polygon(mut self, draw_polygon: bool) -> Self {
        self.draw_polygon = draw_polygon;
        self
    }
}

/// A detection result with bounding box, score, and optional label.
pub struct Detection<'a> {
    pub bbox: &'a BoundingBox,
    pub score: f32,
    pub label: Option<&'a str>,
}

impl<'a> Detection<'a> {
    pub fn new(bbox: &'a BoundingBox, score: f32) -> Self {
        Self {
            bbox,
            score,
            label: None,
        }
    }

    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }
}

/// Visualize detection results on an image.
pub fn visualize_detections(
    img: &RgbImage,
    detections: &[Detection],
    config: &DetectionVisConfig,
) -> RgbImage {
    let mut output = img.clone();
    let font = load_system_font();
    let img_bounds = (output.width() as i32, output.height() as i32);

    for (idx, det) in detections.iter().enumerate() {
        let bbox = det.bbox;
        if bbox.points.is_empty() {
            continue;
        }

        if config.draw_polygon && bbox.points.len() > 4 {
            // Draw polygon lines for curved text
            draw_polygon_lines(&mut output, bbox, config.box_color, config.thickness);
        } else {
            // Draw bounding rectangle
            if let Some((x, y, w, h)) = bbox_to_rect_coords(bbox) {
                draw_thick_rect(
                    &mut output,
                    x,
                    y,
                    w,
                    h,
                    config.box_color,
                    config.thickness,
                    img_bounds,
                );
            }
        }

        // Draw corner points
        if config.draw_corners {
            for point in &bbox.points {
                let x = point.x as i32;
                let y = point.y as i32;
                if is_point_in_bounds(x, y, img_bounds) {
                    draw_filled_circle_mut(&mut output, (x, y), 3, config.box_color);
                }
            }
        }

        // Draw label
        if let Some(ref font) = font {
            let label = if let Some(lbl) = det.label {
                format!("{} #{} {:.1}%", lbl, idx + 1, det.score * 100.0)
            } else {
                format!("#{} {:.1}%", idx + 1, det.score * 100.0)
            };

            let (label_x, label_y) = get_label_position(bbox, img_bounds);
            draw_text_mut(
                &mut output,
                config.label_color,
                label_x,
                label_y,
                config.font_size,
                font,
                &label,
            );
        }
    }

    output
}

fn draw_polygon_lines(img: &mut RgbImage, bbox: &BoundingBox, color: Rgb<u8>, thickness: i32) {
    for i in 0..bbox.points.len() {
        let p1 = &bbox.points[i];
        let p2 = &bbox.points[(i + 1) % bbox.points.len()];

        // Draw with thickness
        for t in 0..thickness {
            let offset = t as f32;
            draw_line_segment_mut(img, (p1.x + offset, p1.y), (p2.x + offset, p2.y), color);
            draw_line_segment_mut(img, (p1.x, p1.y + offset), (p2.x, p2.y + offset), color);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_thick_rect(
    img: &mut RgbImage,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    color: Rgb<u8>,
    thickness: i32,
    bounds: (i32, i32),
) {
    for t in 0..thickness {
        let rect = Rect::at(x - t, y - t).of_size(w + (2 * t) as u32, h + (2 * t) as u32);
        if is_rect_in_bounds(&rect, bounds.0, bounds.1) {
            draw_hollow_rect_mut(img, rect, color);
        }
    }
}

fn get_label_position(bbox: &BoundingBox, bounds: (i32, i32)) -> (i32, i32) {
    let min_x = bbox
        .points
        .iter()
        .map(|p| p.x as i32)
        .min()
        .unwrap_or(0)
        .max(0);
    let min_y = bbox.points.iter().map(|p| p.y as i32).min().unwrap_or(0);
    let label_y = (min_y - 22).max(0);
    (min_x.min(bounds.0 - 100), label_y.min(bounds.1 - 20))
}

/// Configuration for classification visualization.
pub struct ClassificationVisConfig {
    /// Header height
    pub header_height: u32,
    /// Background color
    pub bg_color: Rgb<u8>,
    /// Text color
    pub text_color: Rgb<u8>,
    /// Font size
    pub font_size: f32,
}

impl Default for ClassificationVisConfig {
    fn default() -> Self {
        Self {
            header_height: 60,
            bg_color: Rgb([240, 240, 240]),
            text_color: Rgb([0, 0, 0]),
            font_size: 24.0,
        }
    }
}

/// Visualize classification result with a header above the image.
pub fn visualize_classification(
    img: &RgbImage,
    label: &str,
    confidence: f32,
    prefix: &str,
    config: &ClassificationVisConfig,
) -> RgbImage {
    let (width, height) = (img.width(), img.height());
    let total_height = config.header_height + height;
    let mut output = RgbImage::new(width, total_height);

    // Fill header with background color
    for y in 0..config.header_height {
        for x in 0..width {
            output.put_pixel(x, y, config.bg_color);
        }
    }

    // Copy original image below header
    for y in 0..height {
        for x in 0..width {
            output.put_pixel(x, y + config.header_height, *img.get_pixel(x, y));
        }
    }

    // Draw label text
    if let Some(ref font) = load_system_font() {
        let text = format!(
            "{}: {} (Confidence: {:.1}%)",
            prefix,
            label,
            confidence * 100.0
        );
        draw_text_mut(
            &mut output,
            config.text_color,
            10,
            18,
            config.font_size,
            font,
            &text,
        );
    }

    output
}

/// Color palette for layout element types.
const LAYOUT_COLORS: [Rgba<u8>; 10] = [
    Rgba([255, 0, 0, 255]),     // 0: Red - Text
    Rgba([0, 200, 0, 255]),     // 1: Green - Title
    Rgba([0, 0, 255, 255]),     // 2: Blue - List
    Rgba([255, 200, 0, 255]),   // 3: Yellow - Table
    Rgba([255, 0, 255, 255]),   // 4: Magenta - Figure
    Rgba([0, 255, 255, 255]),   // 5: Cyan - Formula
    Rgba([255, 128, 0, 255]),   // 6: Orange - Header
    Rgba([128, 0, 255, 255]),   // 7: Purple - Footer
    Rgba([0, 128, 128, 255]),   // 8: Teal - Chart
    Rgba([128, 128, 128, 255]), // 9: Gray - Other
];

/// Get color for a layout element type.
pub fn get_layout_color(element_type: &str) -> Rgba<u8> {
    match element_type.to_lowercase().as_str() {
        "text" | "content" | "paragraph" => LAYOUT_COLORS[0],
        "title" | "paragraph_title" | "doc_title" => LAYOUT_COLORS[1],
        "list" => LAYOUT_COLORS[2],
        "table" => LAYOUT_COLORS[3],
        "figure" | "image" => LAYOUT_COLORS[4],
        "formula" => LAYOUT_COLORS[5],
        "header" | "header_image" => LAYOUT_COLORS[6],
        "footer" | "footer_image" | "footnote" => LAYOUT_COLORS[7],
        "chart" => LAYOUT_COLORS[8],
        _ => LAYOUT_COLORS[9],
    }
}

/// A layout element with bounding box, type, and score.
pub struct LayoutItem<'a> {
    pub bbox: &'a BoundingBox,
    pub element_type: &'a str,
    pub score: f32,
}

/// Visualize layout detection results.
pub fn visualize_layout(
    img: &RgbImage,
    elements: &[LayoutItem],
    thickness: i32,
    show_labels: bool,
) -> RgbaImage {
    let mut output = image::DynamicImage::ImageRgb8(img.clone()).to_rgba8();
    let font = load_system_font();
    let img_bounds = (output.width() as i32, output.height() as i32);

    for (idx, elem) in elements.iter().enumerate() {
        let color = get_layout_color(elem.element_type);

        if let Some((x, y, w, h)) = bbox_to_rect_coords(elem.bbox) {
            // Draw rectangle with thickness
            for t in 0..thickness {
                let rect = Rect::at(x - t, y - t).of_size(w + (2 * t) as u32, h + (2 * t) as u32);
                if is_rect_in_bounds(&rect, img_bounds.0, img_bounds.1) {
                    draw_hollow_rect_mut(&mut output, rect, color);
                }
            }

            // Draw label
            if show_labels && let Some(ref font) = font {
                let label = format!(
                    "[{}] {} {:.0}%",
                    idx + 1,
                    elem.element_type,
                    elem.score * 100.0
                );
                let label_y = (y - 18).max(0);

                // Draw label background
                let label_width = (label.len() * 8) as u32;
                let bg_rect = Rect::at(x, label_y).of_size(label_width.min(w), 18);
                if is_rect_in_bounds(&bg_rect, img_bounds.0, img_bounds.1) {
                    draw_filled_rect_mut(&mut output, bg_rect, color);
                }

                // Draw text (dark color for readability)
                let text_color = get_contrasting_text_color(color);
                draw_text_mut(
                    &mut output,
                    text_color,
                    x + 2,
                    label_y + 1,
                    14.0,
                    font,
                    &label,
                );
            }
        }
    }

    output
}

fn get_contrasting_text_color(bg: Rgba<u8>) -> Rgba<u8> {
    let luminance = 0.299 * bg[0] as f32 + 0.587 * bg[1] as f32 + 0.114 * bg[2] as f32;
    if luminance > 128.0 {
        Rgba([20, 20, 20, 255])
    } else {
        Rgba([255, 255, 255, 255])
    }
}

fn bbox_to_rect_coords(bbox: &BoundingBox) -> Option<(i32, i32, u32, u32)> {
    if bbox.points.is_empty() {
        return None;
    }

    let min_x = bbox
        .points
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    let max_x = bbox
        .points
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = bbox
        .points
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    let max_y = bbox
        .points
        .iter()
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);

    let x = min_x.max(0.0) as i32;
    let y = min_y.max(0.0) as i32;
    let w = (max_x - min_x).max(1.0) as u32;
    let h = (max_y - min_y).max(1.0) as u32;

    Some((x, y, w, h))
}

fn is_point_in_bounds(x: i32, y: i32, bounds: (i32, i32)) -> bool {
    x >= 0 && y >= 0 && x < bounds.0 && y < bounds.1
}

fn is_rect_in_bounds(rect: &Rect, img_width: i32, img_height: i32) -> bool {
    rect.left() >= 0 && rect.top() >= 0 && rect.right() < img_width && rect.bottom() < img_height
}

/// Background color for the OCR visualization (light gray).
const BACKGROUND_COLOR: Rgb<u8> = Rgb([238, 238, 238]);

/// Bounding box color for OCR (red).
const BBOX_COLOR: Rgb<u8> = Rgb([255, 0, 0]);

/// Word bounding box color (blue).
const WORD_BBOX_COLOR: Rgb<u8> = Rgb([0, 0, 255]);

/// Text color for rendered text (dark gray).
const TEXT_COLOR: Rgb<u8> = Rgb([50, 50, 50]);

/// Color palette for structure visualization (RGB format).
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
enum TextLayout {
    Horizontal {
        pos: (i32, i32),
        scale: f32,
        text: String,
    },
    Vertical {
        start_pos: (i32, i32),
        scale: f32,
        line_height: f32,
        chars: Vec<char>,
    },
}

/// Configuration for OCR visualization.
pub struct VisualizationConfig {
    pub font: Option<FontVec>,
    pub bbox_thickness: i32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            font: None,
            bbox_thickness: 2,
        }
    }
}

impl VisualizationConfig {
    pub fn with_font_path(font_path: &Path) -> OcrResult<Self> {
        let font_data = std::fs::read(font_path)?;
        let font = FontVec::try_from_vec(font_data).map_err(|_| OCRError::InvalidInput {
            message: format!("Failed to parse font file: {}", font_path.display()),
        })?;

        Ok(Self {
            font: Some(font),
            bbox_thickness: 2,
        })
    }

    pub fn with_system_font() -> Self {
        Self {
            font: load_system_font(),
            bbox_thickness: 2,
        }
    }
}

/// Creates an OCR visualization image.
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

fn draw_detection_results(
    img: &mut RgbImage,
    result: &OAROCRResult,
    config: &VisualizationConfig,
    x_offset: i32,
) -> OcrResult<()> {
    let img_bounds = (img.width() as i32, img.height() as i32);

    for region in result.text_regions.iter() {
        draw_ocr_bounding_box(
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
                    draw_ocr_bounding_box(
                        img,
                        word_bbox,
                        config,
                        x_offset,
                        img_bounds,
                        WORD_BBOX_COLOR,
                    );

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
            draw_text_for_region(img, region, config, x_offset, img_bounds);
        }
    }

    Ok(())
}

fn draw_ocr_bounding_box(
    img: &mut RgbImage,
    bbox: &BoundingBox,
    config: &VisualizationConfig,
    x_offset: i32,
    img_bounds: (i32, i32),
    color: Rgb<u8>,
) {
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

    let box_height = bbox_rect.height() as f32;
    let box_width = bbox_rect.width() as f32;

    let mut font_scale = (box_height * 0.8).max(8.0);

    if let Some(text_width) = measure_text_width(char_str, font, font_scale)
        && text_width > box_width * 0.9
    {
        let scale_factor = (box_width * 0.9) / text_width;
        font_scale *= scale_factor;
    }

    let text_width_px = measure_text_width(char_str, font, font_scale)
        .unwrap_or(font_scale)
        .round() as i32;
    let text_height_px = font_scale.round() as i32;

    let text_x = bbox_rect.left() + (bbox_rect.width() as i32 - text_width_px) / 2;
    let text_y = bbox_rect.top() + (bbox_rect.height() as i32 - text_height_px) / 2;

    if text_x >= 0
        && text_y >= 0
        && text_x + text_width_px <= img_width
        && text_y + text_height_px <= img_height
    {
        draw_text_mut(img, TEXT_COLOR, text_x, text_y, font_scale, font, char_str);
    }
}

fn draw_text_for_region(
    img: &mut RgbImage,
    region: &TextRegion,
    config: &VisualizationConfig,
    x_offset: i32,
    img_bounds: (i32, i32),
) {
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

fn bbox_to_rect(bbox: &BoundingBox, x_offset: i32) -> Option<Rect> {
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

fn calculate_text_layout(
    bbox: &BoundingBox,
    x_offset: i32,
    text: &str,
    font: &FontVec,
) -> Option<TextLayout> {
    if bbox.points.is_empty() || text.is_empty() {
        return None;
    }

    let bbox_rect = bbox_to_rect(bbox, x_offset)?;
    let bbox_width = bbox_rect.width() as f32;
    let bbox_height = bbox_rect.height() as f32;

    if bbox_width <= 0.0 || bbox_height <= 0.0 {
        return None;
    }

    if bbox_height > bbox_width * 1.2 {
        calculate_vertical_text_layout(text, font, &bbox_rect)
    } else {
        calculate_horizontal_text_layout(text, font, &bbox_rect)
    }
}

fn calculate_horizontal_text_layout(
    text: &str,
    font: &FontVec,
    bbox_rect: &Rect,
) -> Option<TextLayout> {
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

fn calculate_vertical_text_layout(
    text: &str,
    font: &FontVec,
    bbox_rect: &Rect,
) -> Option<TextLayout> {
    const PADDING: f32 = 4.0;
    const MIN_FONT_SIZE: f32 = 8.0;

    let available_width = bbox_rect.width() as f32 - PADDING;
    let available_height = bbox_rect.height() as f32 - PADDING;

    let mut font_scale = (available_width * 0.8).max(MIN_FONT_SIZE);
    let mut line_height = font_scale * 1.1;

    let display_chars: Vec<char> = text.chars().collect();
    if !display_chars.is_empty() {
        let max_char_width = display_chars
            .iter()
            .filter_map(|&ch| measure_text_width(&ch.to_string(), font, font_scale))
            .fold(0.0, f32::max);

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

/// Configuration for document structure visualization.
pub struct StructureVisualizationConfig {
    pub font: Option<FontVec>,
    pub font_size: f32,
    pub line_thickness: i32,
    pub show_labels: bool,
    pub show_scores: bool,
    pub show_order: bool,
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
    pub fn with_system_font() -> Self {
        Self {
            font: load_system_font(),
            ..Default::default()
        }
    }

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
pub fn get_element_color(element_type: &LayoutElementType) -> Rgba<u8> {
    let idx = match element_type {
        LayoutElementType::DocTitle => 0,
        LayoutElementType::ParagraphTitle => 5,
        LayoutElementType::Text => 3,
        LayoutElementType::Content => 17,
        LayoutElementType::Abstract => 8,
        LayoutElementType::Image => 6,
        LayoutElementType::Table => 2,
        LayoutElementType::Chart => 7,
        LayoutElementType::Formula => 18,
        LayoutElementType::FormulaNumber => 13,
        LayoutElementType::FigureTitle => 10,
        LayoutElementType::TableTitle => 15,
        LayoutElementType::ChartTitle => 1,
        LayoutElementType::FigureTableChartTitle => 10,
        LayoutElementType::Header => 17,
        LayoutElementType::HeaderImage => 17,
        LayoutElementType::Footer => 12,
        LayoutElementType::FooterImage => 12,
        LayoutElementType::Footnote => 12,
        LayoutElementType::Seal => 14,
        LayoutElementType::Number => 9,
        LayoutElementType::Reference => 4,
        LayoutElementType::ReferenceContent => 4,
        LayoutElementType::Algorithm => 13,
        LayoutElementType::AsideText => 11,
        LayoutElementType::List => 16,
        LayoutElementType::Region => 19,
        LayoutElementType::Other => 19,
    };

    let [r, g, b] = COLOR_PALETTE[idx % COLOR_PALETTE.len()];
    Rgba([r, g, b, 255])
}

fn get_font_color(bg_color: Rgba<u8>) -> Rgba<u8> {
    let luminance =
        0.299 * bg_color[0] as f32 + 0.587 * bg_color[1] as f32 + 0.114 * bg_color[2] as f32;

    if luminance > 128.0 {
        FONT_COLOR_DARK
    } else {
        FONT_COLOR_LIGHT
    }
}

/// Creates a visualization image from a StructureResult.
pub fn create_structure_visualization(
    result: &StructureResult,
    config: &StructureVisualizationConfig,
) -> OcrResult<RgbaImage> {
    let base_img = if let Some(ref rectified) = result.rectified_img {
        image::DynamicImage::ImageRgb8((**rectified).clone()).to_rgba8()
    } else {
        image::open(Path::new(result.input_path.as_ref()))?.to_rgba8()
    };

    let mut img = base_img;

    for element in &result.layout_elements {
        draw_layout_element(&mut img, element, config);
    }

    if config.show_table_cells {
        for table in &result.tables {
            draw_table_cells(&mut img, table, config);
        }
    }

    for formula in &result.formulas {
        let color = get_element_color(&LayoutElementType::Formula);
        draw_structure_bbox(&mut img, &formula.bbox, color, config.line_thickness);

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

fn draw_layout_element(
    img: &mut RgbaImage,
    element: &LayoutElement,
    config: &StructureVisualizationConfig,
) {
    let color = get_element_color(&element.element_type);

    draw_structure_bbox(img, &element.bbox, color, config.line_thickness);

    if config.show_labels
        && let Some(ref font) = config.font
    {
        let label = build_element_label(element, config);
        draw_structure_label(img, &element.bbox, &label, color, font, config.font_size);
    }
}

fn build_element_label(element: &LayoutElement, config: &StructureVisualizationConfig) -> String {
    let type_name = element
        .label
        .as_deref()
        .unwrap_or_else(|| element.element_type.as_str());

    let mut label = String::new();

    if config.show_order
        && let Some(order) = element.order_index
    {
        label.push_str(&format!("[{}] ", order));
    }

    label.push_str(type_name);

    if config.show_scores {
        label.push_str(&format!(" {:.0}%", element.confidence * 100.0));
    }

    label
}

fn draw_table_cells(
    img: &mut RgbaImage,
    table: &TableResult,
    config: &StructureVisualizationConfig,
) {
    for cell in &table.cells {
        draw_structure_bbox(img, &cell.bbox, TABLE_CELL_COLOR, config.line_thickness);
    }
}

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

    let text_width =
        measure_text_width(label, font, font_size).unwrap_or(label.len() as f32 * font_size * 0.6);
    let text_height = font_size;

    let padding = 2;
    let rect_width = (text_width as i32 + padding * 2).min(img.width() as i32 - min_x);
    let rect_height = (text_height as i32 + padding * 2).max(1);

    let (rect_x, rect_y) = if min_y > rect_height {
        (min_x, min_y - rect_height)
    } else {
        (min_x, min_y)
    };

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

    let label_rect = Rect::at(rect_x, rect_y).of_size(rect_width, rect_height);
    draw_filled_rect_mut(img, label_rect, bg_color);

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
    output_path: &Path,
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

    let ext = output_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if ext == "jpg" || ext == "jpeg" {
        image::DynamicImage::ImageRgba8(vis_img)
            .to_rgb8()
            .save(output_path)?;
    } else {
        vis_img.save(output_path)?;
    }

    info!(
        "Structure visualization saved to: {}",
        output_path.display()
    );
    Ok(())
}
