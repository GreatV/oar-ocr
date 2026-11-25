pub use oar_ocr_core::models::{
    detection::db::{DbGeometry, DetectionPreprocessing},
    recognition::crnn::{CtcDecoding, RecognitionPreprocessing},
};
/// Selects the line crop policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CropStrategy {
    /// Use the classic cropper.
    #[default]
    Standard,
    /// Sample each quadrilateral using fixed-point cubic interpolation.
    PerspectiveCubic,
    /// Sample each quadrilateral with cubic interpolation and turn crops at least 1.5 times taller than wide a quarter turn counter-clockwise.
    UprightCubic,
}
/// Selects the recognition postprocessing policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextPostprocessing {
    /// Keep the recognized text.
    #[default]
    Standard,
    /// Normalize printed Thai and filter low-confidence and detached mark readings.
    ThaiPrinted,
}
/// Selects the document orientation policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PageOrientation {
    /// Use the configured document preprocessor.
    #[default]
    Standard,
    /// Choose page rotation by comparing primary recognizer confidence.
    RecognitionVote,
    /// Redetect a page a quarter turn counter-clockwise when few of its boxes are horizontal lines, and keep the turn if it finds far more lines.
    ///
    /// The turn can leave the page upside down, which the vote of a [script router](crate::oarocr::OAROCRBuilder::script_router) corrects.
    LineShapes,
}
/// Selects the reading order policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReadingOrder {
    /// Return each region on its own line in geometric detection order.
    #[default]
    Standard,
    /// Find prose columns and group recognized lines into rows.
    TextColumns,
    /// Keep the detector's line order with each region on its own line.
    Detector,
}
/// Selects the secondary recognition routing policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SecondaryRouting {
    /// Reread lines containing Thai and Latin letters in the same word.
    #[default]
    MixedThaiLatin,
}
#[derive(Debug, Default, PartialEq, Clone)]
pub(in crate::oarocr) struct PageStages {
    pub(in crate::oarocr) detection_preprocessing: DetectionPreprocessing,
    pub(in crate::oarocr) db_geometry: DbGeometry,
    pub(in crate::oarocr) recognition_preprocessing: RecognitionPreprocessing,
    pub(in crate::oarocr) ctc_decoding: CtcDecoding,
    pub(in crate::oarocr) crop_strategy: CropStrategy,
    pub(in crate::oarocr) recognize_line_images: bool,
    pub(in crate::oarocr) split_line_gaps: bool,
    pub(in crate::oarocr) recognize_tall_rotations: bool,
    pub(in crate::oarocr) text_postprocessing: TextPostprocessing,
    pub(in crate::oarocr) page_orientation: PageOrientation,
    pub(in crate::oarocr) deskew_threshold: Option<f64>,
    pub(in crate::oarocr) reading_order: ReadingOrder,
    pub(in crate::oarocr) reattach_marks: bool,
    pub(in crate::oarocr) split_column_gutters: bool,
    pub(in crate::oarocr) read_faint_tokens: bool,
}

impl PageStages {
    pub(in crate::oarocr) fn needs_page_loop(&self) -> bool {
        self.page_orientation == PageOrientation::RecognitionVote
            || self.deskew_threshold.is_some()
            || self.reattach_marks
            || self.page_orientation == PageOrientation::LineShapes
            || self.split_column_gutters
            || self.read_faint_tokens
    }
}
