//! Text region types for OCR results.

use crate::processors::BoundingBox;
use serde::{Deserialize, Serialize};
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
