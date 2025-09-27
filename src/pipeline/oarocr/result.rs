//! Result types for the OAROCR pipeline.

use crate::processors::BoundingBox;
use image::RgbImage;
use std::fmt;
use std::sync::Arc;

/// A text region containing detection and recognition results.
///
/// This struct groups together all the information related to a single detected text region,
/// including the bounding box, recognized text, confidence score, and orientation angle.
/// This design eliminates the need for parallel vectors and provides better ergonomics
/// for iterating over text regions.
#[derive(Debug, Clone)]
pub struct TextRegion {
    /// The bounding box of the detected text region.
    pub bounding_box: BoundingBox,
    /// The recognized text, if recognition was successful.
    /// None indicates that recognition failed or was filtered out due to low confidence.
    pub text: Option<Arc<str>>,
    /// The confidence score for the recognized text.
    /// None indicates that recognition failed or was filtered out due to low confidence.
    pub confidence: Option<f32>,
    /// The text line orientation angle, if orientation classification was performed.
    /// None indicates that orientation classification was not performed or failed.
    pub orientation_angle: Option<f32>,
}

impl TextRegion {
    /// Creates a new TextRegion with the given bounding box.
    ///
    /// The text, confidence, and orientation_angle are initially set to None.
    pub fn new(bounding_box: BoundingBox) -> Self {
        Self {
            bounding_box,
            text: None,
            confidence: None,
            orientation_angle: None,
        }
    }

    /// Creates a new TextRegion with detection and recognition results.
    pub fn with_recognition(
        bounding_box: BoundingBox,
        text: Option<Arc<str>>,
        confidence: Option<f32>,
    ) -> Self {
        Self {
            bounding_box,
            text,
            confidence,
            orientation_angle: None,
        }
    }

    /// Creates a new TextRegion with all fields specified.
    pub fn with_all(
        bounding_box: BoundingBox,
        text: Option<Arc<str>>,
        confidence: Option<f32>,
        orientation_angle: Option<f32>,
    ) -> Self {
        Self {
            bounding_box,
            text,
            confidence,
            orientation_angle,
        }
    }

    /// Returns true if this text region has recognized text.
    pub fn has_text(&self) -> bool {
        self.text.is_some()
    }

    /// Returns true if this text region has a confidence score.
    pub fn has_confidence(&self) -> bool {
        self.confidence.is_some()
    }

    /// Returns true if this text region has an orientation angle.
    pub fn has_orientation(&self) -> bool {
        self.orientation_angle.is_some()
    }

    /// Returns the text and confidence as a tuple if both are available.
    pub fn text_with_confidence(&self) -> Option<(&str, f32)> {
        match (&self.text, self.confidence) {
            (Some(text), Some(confidence)) => Some((text, confidence)),
            _ => None,
        }
    }
}

/// Result of the OAROCR pipeline execution.
///
/// This struct contains all the results from processing an image through
/// the OCR pipeline, including detected text boxes, recognized text, and
/// any intermediate processing results.
#[derive(Debug, Clone)]
pub struct OAROCRResult {
    /// Path to the input image file.
    pub input_path: Arc<str>,
    /// Index of the image in a batch (0 for single image processing).
    pub index: usize,
    /// The input image.
    pub input_img: Arc<RgbImage>,
    /// Structured text regions containing detection and recognition results.
    /// This is the modern, preferred way to access OCR results as it groups related data together.
    pub text_regions: Vec<TextRegion>,
    /// Document orientation angle (if orientation classification was used).
    pub orientation_angle: Option<f32>,
    /// Rectified image (if document unwarping was used).
    pub rectified_img: Option<Arc<RgbImage>>,
    /// Error metrics for data quality monitoring.
    pub error_metrics: ErrorMetrics,
}

impl OAROCRResult {
    /// Creates text regions from parallel vectors.
    ///
    /// This is a helper method used internally during result construction.
    pub(crate) fn create_text_regions_from_vectors(
        text_boxes: &[BoundingBox],
        rec_texts: &[Option<Arc<str>>],
        rec_scores: &[Option<f32>],
        text_line_orientation_angles: &[Option<f32>],
    ) -> Vec<TextRegion> {
        text_boxes
            .iter()
            .enumerate()
            .map(|(i, bbox)| {
                let text = rec_texts.get(i).and_then(|t| t.clone());
                let confidence = rec_scores.get(i).and_then(|s| *s);
                let orientation_angle = text_line_orientation_angles.get(i).and_then(|a| *a);

                TextRegion::with_all(bbox.clone(), text, confidence, orientation_angle)
            })
            .collect()
    }

    /// Returns an iterator over text regions that have recognized text.
    pub fn recognized_text_regions(&self) -> impl Iterator<Item = &TextRegion> {
        self.text_regions.iter().filter(|region| region.has_text())
    }

    /// Returns an iterator over text regions with both text and confidence scores.
    pub fn confident_text_regions(&self) -> impl Iterator<Item = &TextRegion> {
        self.text_regions
            .iter()
            .filter(|region| region.has_confidence())
    }

    /// Returns all recognized text as a vector of strings.
    pub fn all_text(&self) -> Vec<&str> {
        self.text_regions
            .iter()
            .filter_map(|region| region.text.as_ref().map(|s| s.as_ref()))
            .collect()
    }

    /// Returns all recognized text concatenated with the specified separator.
    pub fn concatenated_text(&self, separator: &str) -> String {
        self.all_text().join(separator)
    }

    /// Returns the number of text regions that have recognized text.
    pub fn recognized_text_count(&self) -> usize {
        self.text_regions
            .iter()
            .filter(|region| region.has_text())
            .count()
    }

    /// Returns the average confidence score of all recognized text regions.
    pub fn average_confidence(&self) -> Option<f32> {
        let confident_regions: Vec<_> = self.confident_text_regions().collect();
        if confident_regions.is_empty() {
            None
        } else {
            let sum: f32 = confident_regions
                .iter()
                .filter_map(|region| region.confidence)
                .sum();
            Some(sum / confident_regions.len() as f32)
        }
    }
}

/// Error metrics for monitoring data quality and model performance issues.
#[derive(Debug, Clone, Default)]
pub struct ErrorMetrics {
    /// Number of text boxes that failed to crop.
    pub failed_crops: usize,
    /// Number of text recognition failures.
    pub failed_recognitions: usize,
    /// Number of text line orientation classification failures.
    pub failed_orientations: usize,
    /// Total number of text boxes detected.
    pub total_text_boxes: usize,
}

impl ErrorMetrics {
    /// Creates a new ErrorMetrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the success rate for cropping operations (0.0 to 1.0).
    pub fn crop_success_rate(&self) -> f32 {
        if self.total_text_boxes == 0 {
            1.0
        } else {
            self.total_text_boxes.saturating_sub(self.failed_crops) as f32
                / self.total_text_boxes as f32
        }
    }

    /// Returns the success rate for recognition operations (0.0 to 1.0).
    pub fn recognition_success_rate(&self) -> f32 {
        let successful_crops = self.total_text_boxes.saturating_sub(self.failed_crops);
        if successful_crops == 0 {
            1.0
        } else {
            successful_crops.saturating_sub(self.failed_recognitions) as f32
                / successful_crops as f32
        }
    }

    /// Returns the success rate for orientation classification (0.0 to 1.0).
    pub fn orientation_success_rate(&self) -> f32 {
        let successful_crops = self.total_text_boxes.saturating_sub(self.failed_crops);
        if successful_crops == 0 {
            1.0
        } else {
            successful_crops.saturating_sub(self.failed_orientations) as f32
                / successful_crops as f32
        }
    }

    /// Returns true if there are any errors that indicate data quality issues.
    pub fn has_quality_issues(&self) -> bool {
        self.failed_crops > 0 || self.failed_recognitions > 0 || self.failed_orientations > 0
    }
}

impl fmt::Display for OAROCRResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Input path: {}", self.input_path)?;
        writeln!(f, "Page index: {}", self.index)?;
        writeln!(
            f,
            "Image dimensions: [{}, {}]",
            self.input_img.width(),
            self.input_img.height()
        )?;

        if let Some(angle) = self.orientation_angle {
            writeln!(f, "Orientation angle: {angle:.1}°")?;
        } else {
            writeln!(f, "Orientation angle: not detected")?;
        }

        writeln!(f, "Total text regions: {}", self.text_regions.len())?;
        writeln!(f, "Recognized texts: {}", self.recognized_text_count())?;

        if !self.text_regions.is_empty() {
            writeln!(f, "Text regions (detection + recognition):")?;

            // Use the new structured text regions for cleaner iteration
            for (region_index, region) in self.text_regions.iter().enumerate() {
                write!(f, "  Region {}: ", region_index + 1)?;

                // Display bounding box
                let bbox = &region.bounding_box;
                if bbox.points.is_empty() {
                    write!(f, "[] (empty)")?;
                } else {
                    write!(f, "[")?;
                    for (j, point) in bbox.points.iter().enumerate() {
                        if j == 0 {
                            write!(f, "[{:.0}, {:.0}]", point.x, point.y)?;
                        } else {
                            write!(f, ", [{:.0}, {:.0}]", point.x, point.y)?;
                        }
                    }
                    write!(f, "]")?;
                }

                // Display recognition result if available
                match (&region.text, region.confidence) {
                    (Some(text), Some(score)) => {
                        let orientation_str = match region.orientation_angle {
                            Some(angle) => format!(" (orientation: {angle:.1}°)"),
                            None => String::new(),
                        };
                        writeln!(f, " -> '{text}' (confidence: {score:.3}){orientation_str}")?;
                    }
                    _ => {
                        writeln!(f, " -> [no text recognized]")?;
                    }
                }
            }
        }

        if let Some(rectified_img) = &self.rectified_img {
            writeln!(
                f,
                "Rectified image: available [{} x {}]",
                rectified_img.width(),
                rectified_img.height()
            )?;
        } else {
            writeln!(
                f,
                "Rectified image: not available (document unwarping not enabled)"
            )?;
        }

        // Display error metrics if there are any quality issues
        if self.error_metrics.has_quality_issues() {
            writeln!(f, "Error metrics:")?;
            writeln!(
                f,
                "  Failed crops: {}/{} ({:.1}% success)",
                self.error_metrics.failed_crops,
                self.error_metrics.total_text_boxes,
                self.error_metrics.crop_success_rate() * 100.0
            )?;
            writeln!(
                f,
                "  Failed recognitions: {} ({:.1}% success)",
                self.error_metrics.failed_recognitions,
                self.error_metrics.recognition_success_rate() * 100.0
            )?;
            writeln!(
                f,
                "  Failed orientations: {} ({:.1}% success)",
                self.error_metrics.failed_orientations,
                self.error_metrics.orientation_success_rate() * 100.0
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_success_rate_zero_total() {
        let metrics = ErrorMetrics {
            failed_crops: 0,
            failed_recognitions: 0,
            failed_orientations: 0,
            total_text_boxes: 0,
        };
        assert_eq!(metrics.crop_success_rate(), 1.0);
    }

    #[test]
    fn test_has_quality_issues_no_issues() {
        let metrics = ErrorMetrics {
            failed_crops: 0,
            failed_recognitions: 0,
            failed_orientations: 0,
            total_text_boxes: 10,
        };
        assert!(!metrics.has_quality_issues());
    }

    #[test]
    fn test_has_quality_issues_with_failures() {
        let metrics = ErrorMetrics {
            failed_crops: 1,
            failed_recognitions: 0,
            failed_orientations: 0,
            total_text_boxes: 10,
        };
        assert!(metrics.has_quality_issues());
    }
}
