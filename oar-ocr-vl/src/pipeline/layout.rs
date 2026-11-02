//! Backend-agnostic layout detection interface used by [`DocParser`].
//!
//! [`DocParser`] only needs a per-page list of boxes, already in reading order.
//! Expressing that as the [`LayoutSource`] trait keeps the parser independent
//! of *how* layout is produced.
//!
//! [`PpDocLayout`](crate::PpDocLayout) is the built-in implementation. To run
//! layout detection somewhere else — an ONNX model, a remote service, a cached
//! run — implement the trait over it:
//!
//! ```no_run
//! use image::RgbImage;
//! use oar_ocr_vl::Error;
//! use oar_ocr_vl::{LayoutDetections, LayoutSource};
//!
//! struct MyDetector;
//!
//! impl LayoutSource for MyDetector {
//!     fn detect(&self, image: &RgbImage) -> Result<LayoutDetections, Error> {
//!         let _ = image;
//!         Ok(LayoutDetections::new(Vec::new()))
//!     }
//! }
//! ```
//!
//! [`DocParser`]: crate::DocParser

use crate::api::error::Error;
use crate::document::geometry::BoundingBox;
use image::RgbImage;
use serde::{Deserialize, Serialize};

pub use super::layout_options::{
    ContainmentMode, ContainmentRule, LayoutPostprocessOptions, LayoutShapeMode, ScoreComparison,
};

/// One region reported by a detector, before it becomes a
/// [`LayoutElement`](crate::LayoutElement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDetectionElement {
    /// Region bounds in page pixels.
    pub bbox: BoundingBox,
    /// Raw label emitted by the model.
    pub element_type: String,
    /// Confidence in `[0, 1]`.
    pub score: f32,
}

/// Layout detections for a single page, in reading order.
#[derive(Debug, Clone, Default)]
pub struct LayoutDetections {
    /// Detected regions, in reading order.
    pub elements: Vec<LayoutDetectionElement>,
}

impl LayoutDetections {
    /// Wraps regions that are already in reading order.
    pub fn new(elements: Vec<LayoutDetectionElement>) -> Self {
        Self { elements }
    }
}

/// Layout regions with aligned crop geometry and parser trace.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LayoutPrediction {
    /// Regions in reading order, with page-pixel rectangles.
    pub detections: LayoutDetections,
    /// Optional page-pixel crop polygons aligned with the regions.
    /// An empty vector selects rectangular geometry for every region.
    pub polygons: Vec<Option<BoundingBox>>,
    /// Raw generation from a generative layout source.
    pub raw_output: Option<String>,
    /// Diagnostics produced during layout parsing.
    pub diagnostics: Vec<crate::document::page::ParseDiagnostic>,
}

impl From<LayoutDetections> for LayoutPrediction {
    fn from(detections: LayoutDetections) -> Self {
        Self {
            detections,
            ..Default::default()
        }
    }
}

/// A source of layout detections for a page image.
///
/// Implement this to plug a custom detector into
/// [`DocParser::parse`](crate::DocParser::parse).
pub trait LayoutSource {
    /// Detects layout regions in a single page image.
    ///
    /// Elements must come back in reading order: [`DocParser`] numbers them as
    /// given and never reorders them. PP-DocLayoutV2/V3 predict reading order
    /// directly; a detector that does not must sort before returning.
    ///
    /// [`DocParser`]: crate::DocParser
    fn detect(&self, image: &RgbImage) -> Result<LayoutDetections, Error>;

    /// Detect regions with configured suppression and crop geometry.
    fn detect_with_options(
        &self,
        image: &RgbImage,
        _options: &LayoutPostprocessOptions,
    ) -> Result<LayoutPrediction, Error> {
        self.detect(image).map(Into::into)
    }
}

impl<T: LayoutSource + ?Sized> LayoutSource for &T {
    fn detect(&self, image: &RgbImage) -> Result<LayoutDetections, Error> {
        (**self).detect(image)
    }

    fn detect_with_options(
        &self,
        image: &RgbImage,
        options: &LayoutPostprocessOptions,
    ) -> Result<LayoutPrediction, Error> {
        (**self).detect_with_options(image, options)
    }
}

/// A [`LayoutSource`] backed by a precomputed detection list.
///
/// Useful for testing and for callers that obtain layout from an external
/// service or a cached run.
#[derive(Debug, Clone, Default)]
pub struct StaticLayout {
    detections: LayoutDetections,
}

impl StaticLayout {
    /// Wraps regions that are already in reading order.
    pub fn new(elements: Vec<LayoutDetectionElement>) -> Self {
        Self {
            detections: LayoutDetections::new(elements),
        }
    }
}

impl LayoutSource for StaticLayout {
    fn detect(&self, _image: &RgbImage) -> Result<LayoutDetections, Error> {
        Ok(self.detections.clone())
    }
}
