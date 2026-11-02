//! Parameters for layout suppression, containment, and document geometry.

/// Geometry selected from detector masks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayoutShapeMode {
    /// Use the detector rectangle.
    #[default]
    Rect,
    /// Use a minimum-area quadrilateral.
    Quad,
    /// Use the mask polygon.
    Poly,
    /// Select rectangle, quadrilateral, or polygon from mask geometry.
    Auto,
}

/// Comparison used to retain a score at the configured threshold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScoreComparison {
    /// Retain scores equal to the threshold.
    #[default]
    Inclusive,
    /// Retain scores strictly greater than the threshold.
    Strict,
}

/// Selection among nested boxes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContainmentMode {
    /// Keep both boxes.
    #[default]
    Union,
    /// Keep the containing box.
    Large,
    /// Keep the contained box.
    Small,
}

/// A containment policy for one detector class.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ContainmentRule {
    /// Native detector class ID.
    pub class_id: usize,
    /// Selection for boxes of this class.
    pub mode: ContainmentMode,
}

impl ContainmentRule {
    /// Set the native detector class ID.
    pub fn with_class_id(mut self, class_id: usize) -> Self {
        self.class_id = class_id;
        self
    }

    /// Set the containment selection.
    pub fn with_mode(mut self, mode: ContainmentMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Parameters shared by detector and document layout post-processing.
///
/// Default preserves the existing document overlap filter.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayoutPostprocessOptions {
    /// Same-class suppression threshold; None keeps every candidate.
    pub nms_same_class: Option<f64>,
    /// Cross-class suppression threshold; None keeps every candidate.
    pub nms_cross_class: Option<f64>,
    /// Maximum image-region area fraction on landscape pages.
    pub landscape_image_ratio: Option<f64>,
    /// Maximum image-region area fraction on portrait or square pages.
    pub portrait_image_ratio: Option<f64>,
    /// Fraction of the inner box covered by a containing box.
    pub containment_ratio: f64,
    /// Class-specific containment modes.
    pub containment_rules: Vec<ContainmentRule>,
    /// Labels exempted as inner boxes during containment filtering.
    pub preserved_labels: Vec<String>,
    /// Minimum document-region width and height in pixels.
    pub min_size: f64,
    /// Smaller-box overlap threshold for document filtering.
    pub overlap_threshold: Option<f64>,
    /// Overlap threshold for removing inline formula detections.
    pub inline_formula_overlap: Option<f64>,
    /// Polygon overlap required to remove a document region.
    pub polygon_overlap_threshold: Option<f64>,
    /// Visual labels exempted from mixed-label overlap filtering.
    pub visual_labels: Vec<String>,
    /// Restrict the table exemption to pairs of visual labels.
    pub table_visual_exemption: bool,
    /// Labels removed by the document overlap filter.
    pub drop_labels: Vec<String>,
    /// Geometry used for region crops.
    pub shape_mode: LayoutShapeMode,
    /// Comparison used for score thresholding.
    pub score_comparison: ScoreComparison,
    /// Round detector coordinates before suppression and containment.
    pub round_boxes: bool,
    /// Clip detector boxes to the page and truncate coordinates at output.
    pub clip_boxes: bool,
    /// Require polygon overlap to exceed its threshold strictly.
    pub polygon_overlap_strict: bool,
    pub(crate) single_precision_overlap: bool,
}

impl Default for LayoutPostprocessOptions {
    fn default() -> Self {
        Self {
            nms_same_class: None,
            nms_cross_class: None,
            landscape_image_ratio: None,
            portrait_image_ratio: None,
            containment_ratio: 0.9,
            containment_rules: Vec::new(),
            preserved_labels: Vec::new(),
            min_size: 0.,
            overlap_threshold: Some(0.7),
            inline_formula_overlap: None,
            polygon_overlap_threshold: None,
            visual_labels: vec!["image".into()],
            table_visual_exemption: false,
            drop_labels: vec!["reference".into()],
            shape_mode: LayoutShapeMode::Rect,
            score_comparison: ScoreComparison::Inclusive,
            round_boxes: false,
            clip_boxes: false,
            polygon_overlap_strict: false,
            single_precision_overlap: true,
        }
    }
}

impl LayoutPostprocessOptions {
    /// Select PaddleX suppression, containment, and Auto crop geometry.
    pub fn paddlex() -> Self {
        Self {
            nms_same_class: Some(0.6),
            nms_cross_class: Some(0.98),
            landscape_image_ratio: Some(0.82),
            portrait_image_ratio: Some(0.93),
            containment_rules: [3, 5, 6, 15, 17]
                .into_iter()
                .map(|class_id| ContainmentRule {
                    class_id,
                    mode: ContainmentMode::Large,
                })
                .collect(),
            min_size: 6.,
            inline_formula_overlap: Some(0.5),
            polygon_overlap_threshold: Some(0.7),
            visual_labels: ["image", "table", "seal", "chart"]
                .map(str::to_string)
                .into(),
            table_visual_exemption: true,
            shape_mode: LayoutShapeMode::Auto,
            score_comparison: ScoreComparison::Strict,
            round_boxes: true,
            clip_boxes: true,
            single_precision_overlap: false,
            ..Default::default()
        }
    }

    /// Set output clipping and integer coordinates.
    pub fn with_clip_boxes(mut self, value: bool) -> Self {
        self.clip_boxes = value;
        self
    }

    /// Set whether the polygon overlap comparison is strict.
    pub fn with_polygon_overlap_strict(mut self, value: bool) -> Self {
        self.polygon_overlap_strict = value;
        self
    }

    /// Set the same-class suppression threshold; None keeps every candidate.
    pub fn with_nms_same_class(mut self, value: Option<f64>) -> Self {
        self.nms_same_class = value;
        self
    }

    /// Set the cross-class suppression threshold; None keeps every candidate.
    pub fn with_nms_cross_class(mut self, value: Option<f64>) -> Self {
        self.nms_cross_class = value;
        self
    }

    /// Set the maximum image-region area fraction on landscape pages.
    pub fn with_landscape_image_ratio(mut self, value: Option<f64>) -> Self {
        self.landscape_image_ratio = value;
        self
    }

    /// Set the maximum image-region area fraction on portrait or square pages.
    pub fn with_portrait_image_ratio(mut self, value: Option<f64>) -> Self {
        self.portrait_image_ratio = value;
        self
    }

    /// Set the fraction of the inner box covered by a containing box.
    pub fn with_containment_ratio(mut self, value: f64) -> Self {
        self.containment_ratio = value;
        self
    }

    /// Set the class-specific containment modes.
    pub fn with_containment_rules(mut self, value: Vec<ContainmentRule>) -> Self {
        self.containment_rules = value;
        self
    }

    /// Set the labels retained during containment filtering.
    pub fn with_preserved_labels(mut self, value: Vec<String>) -> Self {
        self.preserved_labels = value;
        self
    }

    /// Set the minimum document-region width and height in pixels.
    pub fn with_min_size(mut self, value: f64) -> Self {
        self.min_size = value;
        self
    }

    /// Set the smaller-box overlap threshold for document filtering.
    pub fn with_overlap_threshold(mut self, value: Option<f64>) -> Self {
        self.overlap_threshold = value;
        self
    }

    /// Set the overlap threshold for removing inline formula detections.
    pub fn with_inline_formula_overlap(mut self, value: Option<f64>) -> Self {
        self.inline_formula_overlap = value;
        self
    }

    /// Set the polygon overlap required to remove a document region.
    pub fn with_polygon_overlap_threshold(mut self, value: Option<f64>) -> Self {
        self.polygon_overlap_threshold = value;
        self
    }

    /// Set the visual labels exempted from mixed-label overlap filtering.
    pub fn with_visual_labels(mut self, value: Vec<String>) -> Self {
        self.visual_labels = value;
        self
    }

    /// Set whether to restrict the table exemption to pairs of visual labels.
    pub fn with_table_visual_exemption(mut self, value: bool) -> Self {
        self.table_visual_exemption = value;
        self
    }

    /// Set the labels removed by the document overlap filter.
    pub fn with_drop_labels(mut self, value: Vec<String>) -> Self {
        self.drop_labels = value;
        self
    }

    /// Set the geometry used for region crops.
    pub fn with_shape_mode(mut self, value: LayoutShapeMode) -> Self {
        self.shape_mode = value;
        self
    }

    /// Set the comparison used for score thresholding.
    pub fn with_score_comparison(mut self, value: ScoreComparison) -> Self {
        self.score_comparison = value;
        self
    }

    /// Set whether to round detector coordinates before suppression and containment.
    pub fn with_round_boxes(mut self, value: bool) -> Self {
        self.round_boxes = value;
        self
    }
}
