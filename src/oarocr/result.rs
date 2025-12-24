//! Result types for the OAROCR pipeline.

use crate::processors::BoundingBox;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// A text region containing detection and recognition results.
///
/// This struct groups together all the information related to a single detected text region,
/// including the bounding box, recognized text, confidence score, orientation angle, and
/// optional word-level boxes for fine-grained text localization.
/// This design eliminates the need for parallel vectors and provides better ergonomics
/// for iterating over text regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    /// The bounding box of the detected text region.
    pub bounding_box: BoundingBox,
    /// Detection polygon (dt_polys in overall OCR).
    /// When available, this preserves the original detection polygon before any
    /// layout-guided refinement. Defaults to the same as `bounding_box`.
    #[serde(default)]
    pub dt_poly: Option<BoundingBox>,
    /// Recognition polygon (rec_polys in overall OCR).
    /// After layout-guided refinement, this may differ from `dt_poly`.
    #[serde(default)]
    pub rec_poly: Option<BoundingBox>,
    /// The recognized text, if recognition was successful.
    /// None indicates that recognition failed or was filtered out due to low confidence.
    pub text: Option<Arc<str>>,
    /// The confidence score for the recognized text.
    /// None indicates that recognition failed or was filtered out due to low confidence.
    pub confidence: Option<f32>,
    /// The text line orientation angle, if orientation classification was performed.
    /// None indicates that orientation classification was not performed or failed.
    pub orientation_angle: Option<f32>,
    /// Word-level bounding boxes within this text region (optional).
    /// Only populated when word-level detection is enabled.
    /// Each box corresponds to a word or character in the recognized text.
    pub word_boxes: Option<Vec<BoundingBox>>,
}

impl TextRegion {
    /// Creates a new TextRegion with the given bounding box.
    ///
    /// The text, confidence, orientation_angle, and word_boxes are initially set to None.
    pub fn new(bounding_box: BoundingBox) -> Self {
        Self {
            bounding_box,
            dt_poly: None,
            rec_poly: None,
            text: None,
            confidence: None,
            orientation_angle: None,
            word_boxes: None,
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
            dt_poly: None,
            rec_poly: None,
            text,
            confidence,
            orientation_angle: None,
            word_boxes: None,
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
            dt_poly: None,
            rec_poly: None,
            text,
            confidence,
            orientation_angle,
            word_boxes: None,
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

    /// Returns true if this text region has word-level boxes.
    pub fn has_word_boxes(&self) -> bool {
        self.word_boxes.is_some()
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
///
/// # Coordinate System
///
/// **Important**: All bounding boxes (`text_regions.bounding_box` and `word_boxes`)
/// are in the **original input image's coordinate system**, even if transformations
/// were applied during processing.
///
/// ## Rotation Correction
/// - If `orientation_angle` is set, the image was rotated during preprocessing (90°/180°/270°)
/// - Bounding boxes have been **automatically transformed back** to the original coordinate system
/// - You can safely overlay boxes on `input_img` for visualization
///
/// ## Rectification (Document Unwarping)
/// - If `rectified_img` is set, neural network-based rectification (UVDoc) was applied
/// - **Limitation**: UVDoc doesn't provide inverse transformations from rectified to distorted coordinates
/// - Bounding boxes are in the **rectified image's coordinate system**, not the original distorted image
/// - **Solution**: Use `rectified_img` for visualization instead of `input_img` when rectification was applied
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAROCRResult {
    /// Path to the input image file.
    pub input_path: Arc<str>,
    /// Index of the image in a batch (0 for single image processing).
    pub index: usize,
    /// The input image.
    #[serde(skip)]
    pub input_img: Arc<RgbImage>,
    /// Structured text regions containing detection and recognition results.
    /// This is the modern, preferred way to access OCR results as it groups related data together.
    pub text_regions: Vec<TextRegion>,
    /// Document orientation angle (if orientation classification was used).
    pub orientation_angle: Option<f32>,
    /// Rectified image (if document unwarping was used).
    #[serde(skip)]
    pub rectified_img: Option<Arc<RgbImage>>,
}

impl OAROCRResult {
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

        Ok(())
    }
}
