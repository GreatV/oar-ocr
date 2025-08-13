//! Orientation angle parsing and validation utilities
//!
//! This module provides robust utilities for parsing orientation angles from
//! classifier outputs, handling various label formats and providing confidence
//! validation. It also includes utilities for applying rotations based on
//! orientation angles.

use image::{RgbImage, imageops};
use std::str::FromStr;
use tracing::warn;

/// Represents an orientation angle with confidence information
#[derive(Debug, Clone, PartialEq)]
pub struct OrientationResult {
    /// The orientation angle in degrees
    pub angle: f32,
    /// The confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Whether the confidence meets the threshold
    pub is_confident: bool,
}

impl OrientationResult {
    /// Creates a new orientation result
    pub fn new(angle: f32, confidence: f32, threshold: Option<f32>) -> Self {
        let is_confident = threshold.is_none_or(|t| confidence >= t);
        Self {
            angle,
            confidence,
            is_confident,
        }
    }

    /// Creates an uncertain result with default angle
    pub fn uncertain(default_angle: f32) -> Self {
        Self {
            angle: default_angle,
            confidence: 0.0,
            is_confident: false,
        }
    }
}

/// Parses orientation angle from various label formats
///
/// This function provides robust parsing that can handle:
/// - String labels: "0", "90", "180", "270"
/// - Numeric strings: "0.0", "90.0", etc.
/// - Direct numeric values when converted to strings
///
/// # Arguments
///
/// * `label` - The label string to parse
/// * `confidence` - The confidence score for this prediction
/// * `threshold` - Optional confidence threshold
/// * `valid_angles` - Set of valid angles to accept
///
/// # Returns
///
/// An `OrientationResult` containing the parsed angle and confidence information
pub fn parse_orientation_angle(
    label: &str,
    confidence: f32,
    threshold: Option<f32>,
    valid_angles: &[f32],
) -> OrientationResult {
    // First try to parse as a direct angle value
    if let Ok(angle) = f32::from_str(label.trim()) {
        // Check if the angle is in our valid set (with small tolerance for floating point)
        let is_valid = valid_angles
            .iter()
            .any(|&valid| (angle - valid).abs() < 0.1);

        if is_valid {
            return OrientationResult::new(angle, confidence, threshold);
        }
    }

    // Try common string patterns
    let normalized = label.trim().to_lowercase();
    let angle = match normalized.as_str() {
        "0" | "0.0" | "0°" | "0deg" | "normal" | "upright" => 0.0,
        "90" | "90.0" | "90°" | "90deg" | "right" | "clockwise" => 90.0,
        "180" | "180.0" | "180°" | "180deg" | "inverted" | "upside_down" => 180.0,
        "270" | "270.0" | "270°" | "270deg" | "left" | "counterclockwise" => 270.0,
        _ => {
            // Unknown label - return uncertain result with 0 degrees as default
            warn!("Unknown orientation label: '{}', defaulting to 0°", label);
            return OrientationResult::uncertain(0.0);
        }
    };

    // Verify the parsed angle is in our valid set
    let is_valid = valid_angles
        .iter()
        .any(|&valid| (angle - valid).abs() < 0.1);

    if is_valid {
        OrientationResult::new(angle, confidence, threshold)
    } else {
        warn!(
            "Parsed angle {}° from label '{}' is not in valid set {:?}, defaulting to 0°",
            angle, label, valid_angles
        );
        OrientationResult::uncertain(0.0)
    }
}

/// Parses document orientation from classifier output
///
/// Handles the standard document orientation angles: 0°, 90°, 180°, 270°
pub fn parse_document_orientation(
    label: &str,
    confidence: f32,
    threshold: Option<f32>,
) -> OrientationResult {
    const VALID_DOC_ANGLES: &[f32] = &[0.0, 90.0, 180.0, 270.0];
    parse_orientation_angle(label, confidence, threshold, VALID_DOC_ANGLES)
}

/// Parses text line orientation from classifier output
///
/// Handles the standard text line orientation angles: 0°, 180°
pub fn parse_text_line_orientation(
    label: &str,
    confidence: f32,
    threshold: Option<f32>,
) -> OrientationResult {
    const VALID_LINE_ANGLES: &[f32] = &[0.0, 180.0];
    parse_orientation_angle(label, confidence, threshold, VALID_LINE_ANGLES)
}

/// Applies document orientation rotation to an image
///
/// This function rotates an image based on the detected orientation angle.
/// It supports rotation by 0°, 90°, 180°, and 270° degrees.
///
/// # Arguments
///
/// * `image` - The input image to rotate
/// * `angle` - The rotation angle in degrees (0, 90, 180, or 270)
///
/// # Returns
///
/// The rotated image
pub fn apply_document_orientation(image: RgbImage, angle: f32) -> RgbImage {
    match angle as i32 {
        0 => image,
        90 => imageops::rotate90(&image),
        180 => imageops::rotate180(&image),
        270 => imageops::rotate270(&image),
        _ => {
            warn!(
                "Unsupported document rotation angle: {:.1}°, using original image",
                angle
            );
            image
        }
    }
}

/// Applies text line orientation rotation to an image
///
/// This function rotates a text line image based on the detected orientation angle.
/// It supports rotation by 0° and 180° degrees.
///
/// # Arguments
///
/// * `image` - The input image to rotate
/// * `angle` - The rotation angle in degrees (0 or 180)
///
/// # Returns
///
/// The rotated image
pub fn apply_text_line_orientation(image: RgbImage, angle: f32) -> RgbImage {
    match angle as i32 {
        0 => image,
        180 => imageops::rotate180(&image),
        _ => {
            warn!(
                "Unsupported text line rotation angle: {:.1}°, using original image",
                angle
            );
            image
        }
    }
}

/// Converts a numeric orientation label to a human-readable degree representation
///
/// # Arguments
///
/// * `label` - The numeric label (e.g., "0", "90", "180", "270")
///
/// # Returns
///
/// A string representation with degree symbol (e.g., "0°", "90°", "180°", "270°")
pub fn format_orientation_label(label: &str) -> String {
    match label {
        "0" => "0°".to_string(),
        "90" => "90°".to_string(),
        "180" => "180°".to_string(),
        "270" => "270°".to_string(),
        _ => label.to_string(),
    }
}

/// Gets the standard document orientation labels
///
/// # Returns
///
/// A vector of standard document orientation labels: ["0", "90", "180", "270"]
pub fn get_document_orientation_labels() -> Vec<String> {
    vec![
        "0".to_string(),
        "90".to_string(),
        "180".to_string(),
        "270".to_string(),
    ]
}

/// Gets the standard text line orientation labels
///
/// # Returns
///
/// A vector of standard text line orientation labels: ["0", "180"]
pub fn get_text_line_orientation_labels() -> Vec<String> {
    vec!["0".to_string(), "180".to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_document_orientation_string_labels() {
        let result = parse_document_orientation("0", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0);
        assert_eq!(result.confidence, 0.9);
        assert!(result.is_confident);

        let result = parse_document_orientation("90", 0.7, Some(0.8));
        assert_eq!(result.angle, 90.0);
        assert!(!result.is_confident); // Below threshold

        let result = parse_document_orientation("180", 0.95, None);
        assert_eq!(result.angle, 180.0);
        assert!(result.is_confident); // No threshold

        let result = parse_document_orientation("270", 0.85, Some(0.8));
        assert_eq!(result.angle, 270.0);
        assert!(result.is_confident);
    }

    #[test]
    fn test_parse_document_orientation_numeric_labels() {
        let result = parse_document_orientation("0.0", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0);
        assert!(result.is_confident);

        let result = parse_document_orientation("90.0", 0.9, Some(0.8));
        assert_eq!(result.angle, 90.0);
        assert!(result.is_confident);
    }

    #[test]
    fn test_parse_document_orientation_alternative_formats() {
        let result = parse_document_orientation("0°", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0);

        let result = parse_document_orientation("normal", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0);

        let result = parse_document_orientation("upside_down", 0.9, Some(0.8));
        assert_eq!(result.angle, 180.0);
    }

    #[test]
    fn test_parse_document_orientation_unknown_label() {
        let result = parse_document_orientation("unknown", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);
    }

    #[test]
    fn test_parse_text_line_orientation() {
        let result = parse_text_line_orientation("0", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0);
        assert!(result.is_confident);

        let result = parse_text_line_orientation("180", 0.9, Some(0.8));
        assert_eq!(result.angle, 180.0);
        assert!(result.is_confident);

        // 90 degrees is not valid for text lines
        let result = parse_text_line_orientation("90", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);
    }

    #[test]
    fn test_format_orientation_label() {
        assert_eq!(format_orientation_label("0"), "0°");
        assert_eq!(format_orientation_label("90"), "90°");
        assert_eq!(format_orientation_label("180"), "180°");
        assert_eq!(format_orientation_label("270"), "270°");
        assert_eq!(format_orientation_label("unknown"), "unknown");
    }

    #[test]
    fn test_get_document_orientation_labels() {
        let labels = get_document_orientation_labels();
        assert_eq!(labels, vec!["0", "90", "180", "270"]);
    }

    #[test]
    fn test_get_text_line_orientation_labels() {
        let labels = get_text_line_orientation_labels();
        assert_eq!(labels, vec!["0", "180"]);
    }

    #[test]
    fn test_apply_document_orientation() {
        use image::{Rgb, RgbImage};

        // Create a simple test image
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red pixel at top-left

        // Test 0 degree rotation (no change)
        let rotated = apply_document_orientation(img.clone(), 0.0);
        assert_eq!(rotated.dimensions(), img.dimensions());
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0]));

        // Test 180 degree rotation
        let rotated = apply_document_orientation(img.clone(), 180.0);
        assert_eq!(rotated.dimensions(), img.dimensions());
        // After 180° rotation, top-left becomes bottom-right
        assert_eq!(rotated.get_pixel(1, 1), &Rgb([255, 0, 0]));
    }

    #[test]
    fn test_apply_text_line_orientation() {
        use image::{Rgb, RgbImage};

        // Create a simple test image
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red pixel at top-left

        // Test 0 degree rotation (no change)
        let rotated = apply_text_line_orientation(img.clone(), 0.0);
        assert_eq!(rotated.dimensions(), img.dimensions());
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0]));

        // Test 180 degree rotation
        let rotated = apply_text_line_orientation(img.clone(), 180.0);
        assert_eq!(rotated.dimensions(), img.dimensions());
        // After 180° rotation, top-left becomes bottom-right
        assert_eq!(rotated.get_pixel(1, 1), &Rgb([255, 0, 0]));

        // Test invalid angle (should return original image)
        let rotated = apply_text_line_orientation(img.clone(), 90.0);
        assert_eq!(rotated.dimensions(), img.dimensions());
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0]));
    }

    #[test]
    fn test_orientation_result_creation() {
        let result = OrientationResult::new(90.0, 0.9, Some(0.8));
        assert_eq!(result.angle, 90.0);
        assert_eq!(result.confidence, 0.9);
        assert!(result.is_confident);

        let result = OrientationResult::new(90.0, 0.7, Some(0.8));
        assert!(!result.is_confident);

        let result = OrientationResult::uncertain(45.0);
        assert_eq!(result.angle, 45.0);
        assert_eq!(result.confidence, 0.0);
        assert!(!result.is_confident);
    }

    #[test]
    fn test_parse_orientation_angle_direct_numeric() {
        let valid_angles = &[0.0, 90.0, 180.0, 270.0];

        let result = parse_orientation_angle("90.0", 0.9, Some(0.8), valid_angles);
        assert_eq!(result.angle, 90.0);
        assert!(result.is_confident);

        let result = parse_orientation_angle("180", 0.7, Some(0.8), valid_angles);
        assert_eq!(result.angle, 180.0);
        assert!(!result.is_confident); // Below threshold
    }

    #[test]
    fn test_parse_orientation_angle_alternative_formats() {
        let valid_angles = &[0.0, 90.0, 180.0, 270.0];

        // Test degree symbols
        let result = parse_orientation_angle("90°", 0.9, None, valid_angles);
        assert_eq!(result.angle, 90.0);
        assert!(result.is_confident);

        // Test descriptive names
        let result = parse_orientation_angle("upright", 0.9, None, valid_angles);
        assert_eq!(result.angle, 0.0);

        let result = parse_orientation_angle("inverted", 0.9, None, valid_angles);
        assert_eq!(result.angle, 180.0);

        let result = parse_orientation_angle("clockwise", 0.9, None, valid_angles);
        assert_eq!(result.angle, 90.0);

        let result = parse_orientation_angle("counterclockwise", 0.9, None, valid_angles);
        assert_eq!(result.angle, 270.0);
    }

    #[test]
    fn test_parse_orientation_angle_invalid_angle() {
        let valid_angles = &[0.0, 90.0, 180.0, 270.0];

        // Test angle not in valid set
        let result = parse_orientation_angle("45.0", 0.9, None, valid_angles);
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);

        // Test completely invalid string
        let result = parse_orientation_angle("invalid", 0.9, None, valid_angles);
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);
    }

    #[test]
    fn test_parse_orientation_angle_edge_cases() {
        let valid_angles = &[0.0, 90.0, 180.0, 270.0];

        // Test empty string
        let result = parse_orientation_angle("", 0.9, None, valid_angles);
        assert_eq!(result.angle, 0.0);
        assert!(!result.is_confident);

        // Test whitespace
        let result = parse_orientation_angle("  90  ", 0.9, None, valid_angles);
        assert_eq!(result.angle, 90.0);
        assert!(result.is_confident);

        // Test case insensitive
        let result = parse_orientation_angle("NORMAL", 0.9, None, valid_angles);
        assert_eq!(result.angle, 0.0);
        assert!(result.is_confident);
    }

    #[test]
    fn test_parse_orientation_angle_floating_point_tolerance() {
        let valid_angles = &[0.0, 90.0, 180.0, 270.0];

        // Test values very close to valid angles (within tolerance)
        let result = parse_orientation_angle("89.95", 0.9, None, valid_angles);
        assert_eq!(result.angle, 89.95);
        assert!(result.is_confident);

        let result = parse_orientation_angle("90.05", 0.9, None, valid_angles);
        assert_eq!(result.angle, 90.05);
        assert!(result.is_confident);

        // Test values outside tolerance
        let result = parse_orientation_angle("89.5", 0.9, None, valid_angles);
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);
    }

    #[test]
    fn test_parse_orientation_angle_custom_valid_set() {
        let custom_angles = &[0.0, 45.0, 90.0, 135.0];

        let result = parse_orientation_angle("45", 0.9, None, custom_angles);
        assert_eq!(result.angle, 45.0);
        assert!(result.is_confident);

        // 180 is not in the custom set
        let result = parse_orientation_angle("180", 0.9, None, custom_angles);
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);
    }

    #[test]
    fn test_parse_text_line_orientation_edge_cases() {
        // Test valid angles for text lines
        let result = parse_text_line_orientation("0", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0);
        assert!(result.is_confident);

        let result = parse_text_line_orientation("180", 0.9, Some(0.8));
        assert_eq!(result.angle, 180.0);
        assert!(result.is_confident);

        // Test invalid angle for text lines (90 degrees)
        let result = parse_text_line_orientation("90", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);

        // Test invalid angle for text lines (270 degrees)
        let result = parse_text_line_orientation("270", 0.9, Some(0.8));
        assert_eq!(result.angle, 0.0); // Default
        assert!(!result.is_confident);

        // Test alternative format
        let result = parse_text_line_orientation("upside_down", 0.9, Some(0.8));
        assert_eq!(result.angle, 180.0);
        assert!(result.is_confident);
    }

    #[test]
    fn test_parse_document_orientation_edge_cases() {
        // Test all valid document angles
        for &angle in &[0.0, 90.0, 180.0, 270.0] {
            let result = parse_document_orientation(&angle.to_string(), 0.9, Some(0.8));
            assert_eq!(result.angle, angle);
            assert!(result.is_confident);
        }

        // Test confidence threshold behavior
        let result = parse_document_orientation("90", 0.7, Some(0.8));
        assert_eq!(result.angle, 90.0);
        assert!(!result.is_confident);

        let result = parse_document_orientation("90", 0.9, Some(0.8));
        assert_eq!(result.angle, 90.0);
        assert!(result.is_confident);

        // Test no threshold
        let result = parse_document_orientation("90", 0.1, None);
        assert_eq!(result.angle, 90.0);
        assert!(result.is_confident); // No threshold means always confident
    }

    #[test]
    fn test_apply_document_orientation_all_angles() {
        use image::{Rgb, RgbImage};

        // Create a 3x3 test image with a distinctive pattern
        let mut img = RgbImage::new(3, 3);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red at top-left
        img.put_pixel(2, 0, Rgb([0, 255, 0])); // Green at top-right
        img.put_pixel(0, 2, Rgb([0, 0, 255])); // Blue at bottom-left
        img.put_pixel(2, 2, Rgb([255, 255, 0])); // Yellow at bottom-right

        // Test 0 degrees (no change)
        let rotated = apply_document_orientation(img.clone(), 0.0);
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0])); // Red still at top-left

        // Test 90 degrees (clockwise)
        let rotated = apply_document_orientation(img.clone(), 90.0);
        assert_eq!(rotated.dimensions(), (3, 3));

        // Test 180 degrees
        let rotated = apply_document_orientation(img.clone(), 180.0);
        assert_eq!(rotated.get_pixel(2, 2), &Rgb([255, 0, 0])); // Red now at bottom-right

        // Test 270 degrees (counter-clockwise)
        let rotated = apply_document_orientation(img.clone(), 270.0);
        assert_eq!(rotated.dimensions(), (3, 3));

        // Test invalid angle (should return original)
        let rotated = apply_document_orientation(img.clone(), 45.0);
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0])); // Should be unchanged
    }

    #[test]
    fn test_apply_text_line_orientation_edge_cases() {
        use image::{Rgb, RgbImage};

        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red at top-left
        img.put_pixel(1, 1, Rgb([0, 255, 0])); // Green at bottom-right

        // Test 0 degrees (no change)
        let rotated = apply_text_line_orientation(img.clone(), 0.0);
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0]));
        assert_eq!(rotated.get_pixel(1, 1), &Rgb([0, 255, 0]));

        // Test 180 degrees
        let rotated = apply_text_line_orientation(img.clone(), 180.0);
        assert_eq!(rotated.get_pixel(1, 1), &Rgb([255, 0, 0])); // Red now at bottom-right
        assert_eq!(rotated.get_pixel(0, 0), &Rgb([0, 255, 0])); // Green now at top-left

        // Test invalid angles (should return original)
        for &invalid_angle in &[45.0, 90.0, 135.0, 225.0, 270.0, 315.0] {
            let rotated = apply_text_line_orientation(img.clone(), invalid_angle);
            assert_eq!(rotated.get_pixel(0, 0), &Rgb([255, 0, 0])); // Should be unchanged
            assert_eq!(rotated.get_pixel(1, 1), &Rgb([0, 255, 0]));
        }
    }
}
