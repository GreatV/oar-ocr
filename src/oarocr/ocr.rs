//! High-level OCR builder API.
//!
//! This module provides `OAROCRBuilder` for constructing OCR pipelines with a fluent API.
//! It simplifies the process of configuring text detection, recognition, and optional
//! preprocessing components.

use super::builder_utils::{
    apply_gpu_memory_budget, build_optional_adapter, default_cpu_region_batch_size,
    resolve_device_batch_sizes, resolve_model_path, resolve_model_source,
};
use super::reading;
use oar_ocr_core::core::ModelSource;
use oar_ocr_core::core::config::OrtSessionConfig;
use oar_ocr_core::core::errors::OCRError;
use oar_ocr_core::core::traits::OrtConfigurable;
use oar_ocr_core::core::traits::adapter::{AdapterBuilder, ModelAdapter};
use oar_ocr_core::core::traits::task::ImageTaskInput;
use oar_ocr_core::domain::adapters::{
    DocumentOrientationAdapter, DocumentOrientationAdapterBuilder, TextDetectionAdapter,
    TextDetectionAdapterBuilder, TextLineOrientationAdapter, TextLineOrientationAdapterBuilder,
    TextRecognitionAdapter, TextRecognitionAdapterBuilder, UVDocRectifierAdapter,
    UVDocRectifierAdapterBuilder,
};
use oar_ocr_core::domain::tasks::{TextDetectionConfig, TextRecognitionConfig};
use oar_ocr_core::processors::BoundingBox;
use reading::options::PageStages;
pub use reading::options::{
    CropStrategy, CtcDecoding, DbGeometry, DetectionPreprocessing, PageOrientation, ReadingOrder,
    RecognitionPreprocessing, SecondaryRouting, TextPostprocessing,
};
use std::path::PathBuf;
use std::sync::Arc;

/// Internal structure holding the OCR pipeline adapters.
#[derive(Debug)]
pub(super) struct OCRPipeline {
    pub(super) rectification_adapter: Option<UVDocRectifierAdapter>,
    pub(super) document_orientation_adapter: Option<DocumentOrientationAdapter>,
    pub(super) text_detection_adapter: TextDetectionAdapter,
    pub(super) text_line_orientation_adapter: Option<TextLineOrientationAdapter>,
    pub(super) text_recognition_adapter: TextRecognitionAdapter,
    pub(super) secondary_recognizer: Option<(TextRecognitionAdapter, SecondaryRouting)>,
}

/// Builder for constructing OCR pipelines.
///
/// This builder provides a high-level API for configuring text detection and recognition
/// pipelines with optional preprocessing components like orientation classification and
/// image rectification.
///
/// # Example
///
/// ```no_run
/// use oar_ocr::oarocr::OAROCRBuilder;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ocr = OAROCRBuilder::new(
///     "path/to/text_detection.onnx",
///     "path/to/text_recognition.onnx",
///     "path/to/character_dict.txt"
/// )
/// .with_document_image_orientation_classification("path/to/orientation.onnx")
/// .with_text_line_orientation_classification("path/to/line_orientation.onnx")
/// .image_batch_size(4)
/// .region_batch_size(32)
/// .build()?;
/// # let _ = ocr;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct OAROCRBuilder {
    stages: PageStages,
    secondary_recognizer: Option<(ModelSource, SecondaryRouting)>,
    // Required fields
    text_detection_model: ModelSource,
    text_recognition_model: ModelSource,
    character_dict_path: PathBuf,
    character_dict_content: Option<String>,

    // Optional components
    document_orientation_model: Option<ModelSource>,
    text_line_orientation_model: Option<ModelSource>,
    document_rectification_model: Option<ModelSource>,

    // Configuration
    ort_session_config: Option<OrtSessionConfig>,
    text_detection_config: Option<TextDetectionConfig>,
    text_recognition_config: Option<TextRecognitionConfig>,
    image_batch_size: Option<usize>,
    region_batch_size: Option<usize>,
    gpu_memory_budget: Option<usize>,

    // Detection defaults a preset chose; build() consults them only when the
    // caller set no explicit detection config and the text type is general,
    // so text-type defaults (seal, table) still take over.
    preset_text_detection_config: Option<TextDetectionConfig>,

    // Text type and word box options
    text_type: Option<String>,
    return_word_box: bool,
}

/// PP-OCRv6 model sizes selectable through the [`OAROCRBuilder::pp_ocrv6`]
/// preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PpOcrV6Size {
    /// Fastest models and a reduced 6,904-entry dictionary.
    Tiny,
    /// Balanced models over the full 18,708-entry dictionary.
    Small,
    /// Most accurate models over the full 18,708-entry dictionary.
    Medium,
}

impl PpOcrV6Size {
    /// Registry name of the detection model for this size.
    pub(crate) fn detection_model(self) -> &'static str {
        match self {
            Self::Tiny => "pp-ocrv6_tiny_det.onnx",
            Self::Small => "pp-ocrv6_small_det.onnx",
            Self::Medium => "pp-ocrv6_medium_det.onnx",
        }
    }

    /// Registry name of the recognition model for this size.
    pub(crate) fn recognition_model(self) -> &'static str {
        match self {
            Self::Tiny => "pp-ocrv6_tiny_rec.onnx",
            Self::Small => "pp-ocrv6_small_rec.onnx",
            Self::Medium => "pp-ocrv6_medium_rec.onnx",
        }
    }

    /// Registry name of the matching character dictionary; tiny ships its
    /// own reduced dictionary, small and medium share the full one.
    pub(crate) fn character_dict(self) -> &'static str {
        match self {
            Self::Tiny => "ppocrv6_tiny_dict.txt",
            Self::Small | Self::Medium => "ppocrv6_dict.txt",
        }
    }
}

/// The official PaddleOCR detection defaults published for PP-OCRv6
/// (`DBPostProcess`: `thresh` 0.2, `box_thresh` 0.45 — 0.4 for tiny —
/// `unclip_ratio` 1.4, `max_candidates` 3000).
pub(crate) fn pp_ocrv6_detection_config(size: PpOcrV6Size) -> TextDetectionConfig {
    TextDetectionConfig {
        score_threshold: 0.2,
        box_threshold: match size {
            PpOcrV6Size::Tiny => 0.4,
            _ => 0.45,
        },
        unclip_ratio: 1.4,
        max_candidates: 3000,
        ..TextDetectionConfig::default()
    }
}

impl OAROCRBuilder {
    // Guardrail against pathological user input. This is intentionally generous and
    // not a model-tuned throughput limit.
    const MAX_BATCH_SIZE: usize = 4096;

    /// Creates a new OCR builder with required components.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model` - Text detection ONNX model: a path, or raw
    ///   model bytes (e.g. from `include_bytes!`)
    /// * `text_recognition_model` - Text recognition ONNX model: a path or
    ///   raw model bytes
    /// * `character_dict_path` - Path to the character dictionary file (see
    ///   [`Self::character_dict_content`] for the in-memory alternative)
    pub fn new(
        text_detection_model: impl Into<ModelSource>,
        text_recognition_model: impl Into<ModelSource>,
        character_dict_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            text_detection_model: text_detection_model.into(),
            text_recognition_model: text_recognition_model.into(),
            character_dict_path: character_dict_path.into(),
            character_dict_content: None,
            stages: PageStages::default(),
            secondary_recognizer: None,
            document_orientation_model: None,
            text_line_orientation_model: None,
            document_rectification_model: None,
            ort_session_config: None,
            text_detection_config: None,
            text_recognition_config: None,
            image_batch_size: None,
            region_batch_size: None,
            gpu_memory_budget: None,
            preset_text_detection_config: None,
            text_type: None,
            return_word_box: false,
        }
    }

    /// Creates a PP-OCRv6 pipeline for the given model size.
    ///
    /// This fills in the registry model names (`pp-ocrv6_{size}_det.onnx`,
    /// `pp-ocrv6_{size}_rec.onnx`), the matching dictionary (tiny ships its
    /// own reduced one), and the official PP-OCRv6 detection defaults
    /// (score 0.2, box 0.45 — 0.4 for tiny —, unclip 1.4, up to 3000
    /// candidates), which behave as defaults: an explicit
    /// [`text_detection_config`](Self::text_detection_config) or a
    /// [`text_type`](Self::text_type) like seal or table still replaces
    /// them. With the
    /// `auto-download` feature the names resolve through the model registry;
    /// without it they resolve as local paths like any other model source.
    /// The returned builder is ordinary, so every setter can still override
    /// anything.
    ///
    /// ```no_run
    /// use oar_ocr::oarocr::{OAROCRBuilder, PpOcrV6Size};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ocr = OAROCRBuilder::pp_ocrv6(PpOcrV6Size::Small).build()?;
    /// # let _ = ocr;
    /// # Ok(())
    /// # }
    /// ```
    pub fn pp_ocrv6(size: PpOcrV6Size) -> Self {
        Self::new(
            size.detection_model(),
            size.recognition_model(),
            size.character_dict(),
        )
        .with_preset_detection_config(pp_ocrv6_detection_config(size))
    }

    /// Build the OpenThai v2.1 preset from registered detector, recognizers and dictionary names.
    pub fn openthai() -> Self {
        Self::new(
            "pp-ocrv6_tiny_det.onnx",
            "openthai2_rec.onnx",
            "openthai2_dict.txt",
        )
        .with_openthai_defaults("openthai2_rec2.onnx")
    }
    /// Load the OpenThai v2.1 preset from the upstream filenames in a local directory.
    pub fn openthai_from_dir(dir: impl AsRef<std::path::Path>) -> Self {
        let dir = dir.as_ref();
        Self::new(
            dir.join("det.onnx"),
            dir.join("rec.onnx"),
            dir.join("rec_dict.txt"),
        )
        .with_openthai_defaults(dir.join("rec2.onnx"))
    }
    fn with_openthai_defaults(self, secondary: impl Into<ModelSource>) -> Self {
        let mut detection = pp_ocrv6_detection_config(PpOcrV6Size::Tiny);
        detection.limit_side_len = Some(1280);
        self.with_preset_detection_config(detection)
            .detection_preprocessing(DetectionPreprocessing::PaddedLongSide)
            .db_geometry(DbGeometry::OpenCv)
            .recognition_preprocessing(RecognitionPreprocessing::Flatten)
            .region_batch_size(16)
            .text_recognition_config(TextRecognitionConfig {
                score_threshold: 0.8,
            })
            .recognize_line_images(true)
            .ctc_decoding(CtcDecoding::FrameSpans)
            .return_word_box(true)
            .crop_strategy(CropStrategy::PerspectiveCubic)
            .split_line_gaps(true)
            .recognize_tall_rotations(true)
            .text_postprocessing(TextPostprocessing::ThaiPrinted)
            .secondary_recognizer(secondary, SecondaryRouting::MixedThaiLatin)
            .page_orientation(PageOrientation::RecognitionVote)
            .deskew_threshold(Some(2.5))
            .reading_order(ReadingOrder::TextColumns)
            .reattach_marks(true)
    }
    /// Load a secondary recognizer for lines selected by the routing policy.
    pub fn secondary_recognizer(
        mut self,
        model: impl Into<ModelSource>,
        routing: SecondaryRouting,
    ) -> Self {
        self.secondary_recognizer = Some((model.into(), routing));
        self
    }
    /// Disable secondary recognition for all lines.
    pub fn without_secondary_recognizer(mut self) -> Self {
        self.secondary_recognizer = None;
        self
    }
    /// Select the detector input border and resize policy.
    pub fn detection_preprocessing(mut self, value: DetectionPreprocessing) -> Self {
        self.stages.detection_preprocessing = value;
        self
    }
    /// Select the polygon scoring and expansion policy for DB detections.
    pub fn db_geometry(mut self, value: DbGeometry) -> Self {
        self.stages.db_geometry = value;
        self
    }
    /// Recognize a small line-shaped input as one crop when detection returns no boxes.
    pub fn recognize_line_images(mut self, value: bool) -> Self {
        self.stages.recognize_line_images = value;
        self
    }
    /// Select the line background normalization and width policy.
    pub fn recognition_preprocessing(mut self, value: RecognitionPreprocessing) -> Self {
        self.stages.recognition_preprocessing = value;
        self
    }
    /// Select how the recognizer collapses CTC runs and records their positions.
    pub fn ctc_decoding(mut self, value: CtcDecoding) -> Self {
        self.stages.ctc_decoding = value;
        self
    }
    /// Select the resampling policy for detected line crops.
    pub fn crop_strategy(mut self, value: CropStrategy) -> Self {
        self.stages.crop_strategy = value;
        self
    }
    /// Split wide crops at blank gutters and compare the readings with the original line.
    pub fn split_line_gaps(mut self, value: bool) -> Self {
        self.stages.split_line_gaps = value;
        self
    }
    /// Compare the upright and both quarter-turned readings of tall crops.
    pub fn recognize_tall_rotations(mut self, value: bool) -> Self {
        self.stages.recognize_tall_rotations = value;
        self
    }
    /// Select the text normalization and line filtering policy.
    pub fn text_postprocessing(mut self, value: TextPostprocessing) -> Self {
        self.stages.text_postprocessing = value;
        self
    }
    /// Select the document orientation strategy.
    pub fn page_orientation(mut self, value: PageOrientation) -> Self {
        self.stages.page_orientation = value;
        self
    }
    /// Redetect pages after correcting text angles at or above this absolute threshold in degrees.
    pub fn deskew_threshold(mut self, value: Option<f64>) -> Self {
        self.stages.deskew_threshold = value;
        self
    }
    /// Select the ordering and row-grouping strategy.
    pub fn reading_order(mut self, value: ReadingOrder) -> Self {
        self.stages.reading_order = value;
        self
    }
    /// Attach displaced tone marks to accepted lines and reread their expanded crops.
    pub fn reattach_marks(mut self, value: bool) -> Self {
        self.stages.reattach_marks = value;
        self
    }
    fn with_preset_detection_config(mut self, config: TextDetectionConfig) -> Self {
        self.preset_text_detection_config = Some(config);
        self
    }

    /// Sets the character dictionary from an in-memory string (e.g. from
    /// `include_str!`). When set, `character_dict_path` is ignored.
    pub fn character_dict_content(mut self, content: impl Into<String>) -> Self {
        self.character_dict_content = Some(content.into());
        self
    }

    /// Sets the ONNX Runtime session configuration.
    ///
    /// This configuration will be applied to all models in the pipeline.
    pub fn ort_session(mut self, config: OrtSessionConfig) -> Self {
        self.ort_session_config = Some(config);
        self
    }

    /// Sets a GPU memory budget in bytes without selecting a device.
    ///
    /// With `g = floor(bytes / GiB)`, image batches use `clamp(2*g - 6, 1, 8)`
    /// and text-region batches use `clamp(12*g - 28, 4, 64)`, saturating negative
    /// values at zero. Thus 4 GiB gives 2/20, and 8 GiB or more gives 8/64.
    /// Explicit batch sizes take precedence. CUDA sessions cap each arena at half
    /// the budget and default to idle memory recovery and SameAsRequested growth.
    /// Existing smaller limits and explicit arena settings are retained. This is a tuning hint, not a device-wide cap:
    /// other sessions, weights, and driver allocations also consume memory.
    pub fn gpu_memory_budget(mut self, bytes: usize) -> Self {
        self.gpu_memory_budget = Some(bytes);
        self
    }

    /// Sets the text detection model configuration.
    ///
    /// Controls text detection preprocessing and postprocessing.
    pub fn text_detection_config(mut self, config: TextDetectionConfig) -> Self {
        self.text_detection_config = Some(config);
        self
    }

    /// Sets the text recognition model configuration.
    ///
    /// Controls text recognition preprocessing and postprocessing.
    pub fn text_recognition_config(mut self, config: TextRecognitionConfig) -> Self {
        self.text_recognition_config = Some(config);
        self
    }

    /// Sets the batch size for processing input images during text detection.
    ///
    /// This controls how many images are sent to the text detection adapter per call.
    /// If a detector cannot batch the provided images (e.g., mismatched sizes), the
    /// pipeline falls back to per-image detection. Values are validated in `build()`
    /// and must be within `1..=MAX_BATCH_SIZE`. When unset, CPU execution uses `1`;
    /// explicitly configured accelerators use the adapter's throughput default.
    pub fn image_batch_size(mut self, size: usize) -> Self {
        self.image_batch_size = Some(size);
        self
    }

    /// Sets the batch size for processing detected text regions.
    ///
    /// Controls memory usage during text recognition. Smaller values use less memory.
    /// When unset, CPU execution uses `16` for PP-OCRv6 Tiny and `4` for other
    /// models; explicitly configured accelerators use the adapter's throughput
    /// default (`64`). Values are validated in `build()` and must be within
    /// `1..=MAX_BATCH_SIZE`.
    pub fn region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = Some(size);
        self
    }

    /// Adds document image orientation classification to the pipeline.
    ///
    /// This component detects and corrects document orientation before text detection.
    pub fn with_document_image_orientation_classification(
        mut self,
        model_source: impl Into<ModelSource>,
    ) -> Self {
        self.document_orientation_model = Some(model_source.into());
        self
    }

    /// Adds text line orientation classification to the pipeline.
    ///
    /// This component detects and corrects text line orientation after text detection.
    pub fn with_text_line_orientation_classification(
        mut self,
        model_source: impl Into<ModelSource>,
    ) -> Self {
        self.text_line_orientation_model = Some(model_source.into());
        self
    }

    /// Adds document image rectification to the pipeline.
    ///
    /// This component corrects document distortion before text detection.
    pub fn with_document_image_rectification(
        mut self,
        model_source: impl Into<ModelSource>,
    ) -> Self {
        self.document_rectification_model = Some(model_source.into());
        self
    }

    /// Resolves the text detection config from an explicit config, a
    /// preset's defaults, and the text type.
    fn effective_text_detection_config(&self) -> TextDetectionConfig {
        // Align text detection defaults with OCR pipeline.
        // Defaults depend on text_type:
        // - general: limit_side_len=960, limit_type="max", thresh=0.3, box_thresh=0.6, unclip_ratio=2.0
        // - table: limit_side_len=960, limit_type="max", thresh=0.3, box_thresh=0.4, unclip_ratio=2.0
        // - seal: limit_side_len=736, limit_type="min", thresh=0.2, box_thresh=0.6, unclip_ratio=0.5
        // A preset's thresholds seed the general path only: explicit configs
        // win, and the seal/table branches below replace them wholesale.
        // The adapter matches text types case-insensitively, so do the same here.
        let text_type = self
            .text_type
            .as_deref()
            .unwrap_or("general")
            .to_ascii_lowercase();
        let preset_det_defaults = self.preset_text_detection_config.is_some()
            && self.text_detection_config.is_none()
            && !matches!(text_type.as_str(), "table" | "seal");
        let mut effective_det_cfg = match (
            &self.text_detection_config,
            &self.preset_text_detection_config,
        ) {
            (Some(explicit), _) => explicit.clone(),
            (None, Some(preset)) if preset_det_defaults => preset.clone(),
            _ => TextDetectionConfig::default(),
        };
        let has_explicit_det_cfg = self.text_detection_config.is_some();
        if !has_explicit_det_cfg {
            match text_type.as_str() {
                "table" => {
                    effective_det_cfg.score_threshold = 0.3;
                    effective_det_cfg.box_threshold = 0.4;
                    effective_det_cfg.unclip_ratio = 2.0;
                    if effective_det_cfg.limit_side_len.is_none() {
                        effective_det_cfg.limit_side_len = Some(960);
                    }
                    if effective_det_cfg.limit_type.is_none() {
                        effective_det_cfg.limit_type = Some(crate::processors::LimitType::Max);
                    }
                    if effective_det_cfg.max_side_len.is_none() {
                        effective_det_cfg.max_side_len = Some(4000);
                    }
                }
                "seal" => {
                    effective_det_cfg.score_threshold = 0.2;
                    effective_det_cfg.box_threshold = 0.6;
                    effective_det_cfg.unclip_ratio = 0.5;
                    if effective_det_cfg.limit_side_len.is_none() {
                        effective_det_cfg.limit_side_len = Some(736);
                    }
                    if effective_det_cfg.limit_type.is_none() {
                        effective_det_cfg.limit_type = Some(crate::processors::LimitType::Min);
                    }
                    if effective_det_cfg.max_side_len.is_none() {
                        effective_det_cfg.max_side_len = Some(4000);
                    }
                }
                _ => {
                    // The preset already chose its thresholds; only fill the
                    // limits around them.
                    if !preset_det_defaults {
                        effective_det_cfg.score_threshold = 0.3;
                        effective_det_cfg.box_threshold = 0.6;
                        effective_det_cfg.unclip_ratio = 2.0;
                    }
                    if effective_det_cfg.limit_side_len.is_none() {
                        effective_det_cfg.limit_side_len = Some(960);
                    }
                    if effective_det_cfg.limit_type.is_none() {
                        effective_det_cfg.limit_type = Some(crate::processors::LimitType::Max);
                    }
                    if effective_det_cfg.max_side_len.is_none() {
                        effective_det_cfg.max_side_len = Some(4000);
                    }
                }
            }
        }
        effective_det_cfg
    }

    /// Sets the text type for sorting and cropping strategy.
    ///
    /// This matches the text_type parameter:
    /// - "seal": Uses polygon-based sorting/cropping for seal text (circular/curved)
    /// - "table": Uses table-friendly detection defaults (box_threshold=0.4)
    /// - Other values or None: Uses quad-based sorting (default)
    ///
    /// # Arguments
    ///
    /// * `text_type` - Text type identifier ("seal", etc.)
    pub fn text_type(mut self, text_type: impl Into<String>) -> Self {
        self.text_type = Some(text_type.into());
        self
    }

    /// Enables word-level bounding box detection.
    ///
    /// When enabled, the pipeline will attempt to detect individual words
    /// within each text line and populate the `word_boxes` field in `TextRegion`.
    ///
    /// Note: This feature requires word-level detection support in the recognition model.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable word box detection
    pub fn return_word_box(mut self, enable: bool) -> Self {
        self.return_word_box = enable;
        self
    }

    /// Builds the OCR runtime.
    ///
    /// This instantiates all adapters and returns an `OAROCR` instance ready for prediction.
    pub fn build(mut self) -> Result<OAROCR, OCRError> {
        apply_gpu_memory_budget(self.ort_session_config.as_mut(), self.gpu_memory_budget)?;
        if let Some(size) = self.image_batch_size {
            Self::validate_batch_size("image_batch_size", size)?;
        }
        if let Some(size) = self.region_batch_size {
            Self::validate_batch_size("region_batch_size", size)?;
        }

        // Resolve required model paths through the auto-download cache when
        // the feature is enabled. With the feature off these are no-ops.
        let text_detection_model = resolve_model_source(&self.text_detection_model)?;
        let text_recognition_model = resolve_model_source(&self.text_recognition_model)?;

        // CPU operators already fan work out across ORT's intra-op pool. Tiny's
        // much cheaper recognizer benefits from a wider batch, while larger
        // models regress beyond four on Windows. Accelerators retain their
        // throughput-oriented adapter defaults (8 detection / 64 recognition).
        self.ort_session_config = self.ort_session_config.map(OrtSessionConfig::resolve_auto);
        let cpu_region_batch_size =
            default_cpu_region_batch_size(Some(&text_recognition_model), None);
        let (image_batch_size, region_batch_size) = resolve_device_batch_sizes(
            self.ort_session_config.as_ref(),
            self.image_batch_size,
            self.region_batch_size,
            1,
            cpu_region_batch_size,
            self.gpu_memory_budget,
        );

        // Load character dictionary for text recognition
        let char_dict = match &self.character_dict_content {
            Some(content) => content.clone(),
            None => {
                let character_dict_path = resolve_model_path(&self.character_dict_path)?;
                std::fs::read_to_string(&character_dict_path).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!(
                            "Failed to read character dictionary from '{}': {}",
                            character_dict_path.display(),
                            e
                        ),
                    }
                })?
            }
        };

        // Build document rectification adapter if enabled
        let rectification_adapter = build_optional_adapter(
            self.document_rectification_model.as_ref(),
            self.ort_session_config.as_ref(),
            UVDocRectifierAdapterBuilder::new,
        )?;

        // Build document orientation adapter if enabled
        let document_orientation_adapter = build_optional_adapter(
            self.document_orientation_model.as_ref(),
            self.ort_session_config.as_ref(),
            DocumentOrientationAdapterBuilder::new,
        )?;

        // Build text detection adapter (required)
        let mut detection_builder = TextDetectionAdapterBuilder::new()
            .preprocessing(self.stages.detection_preprocessing)
            .geometry(self.stages.db_geometry);

        if let Some(ref ort_config) = self.ort_session_config {
            detection_builder = detection_builder.with_ort_config(ort_config.clone());
        }

        detection_builder = detection_builder.with_config(self.effective_text_detection_config());

        // Pass text_type to detection adapter for proper preprocessing configuration
        if let Some(ref text_type) = self.text_type {
            detection_builder = detection_builder.text_type(text_type.clone());
        }

        let text_detection_adapter = detection_builder.build(text_detection_model)?;

        // Build text line orientation adapter if enabled
        let text_line_orientation_adapter = build_optional_adapter(
            self.text_line_orientation_model.as_ref(),
            self.ort_session_config.as_ref(),
            TextLineOrientationAdapterBuilder::new,
        )?;

        // Build text recognition adapter (required)
        // Parse char_dict into Vec<String> - one character per line
        let char_dict_vec: Vec<String> = char_dict.lines().map(|s| s.to_string()).collect();

        let build_recognizer = |source, dict| {
            let mut builder = TextRecognitionAdapterBuilder::new()
                .character_dict(dict)
                .return_word_box(self.return_word_box);
            if let Some(config) = &self.ort_session_config {
                builder = builder.with_ort_config(config.clone());
            }
            if let Some(config) = &self.text_recognition_config {
                builder = builder.with_config(config.clone());
            }
            builder.build(source)
        };
        let secondary_recognizer = if let Some((model, routing)) = &self.secondary_recognizer {
            Some((
                build_recognizer(resolve_model_source(model)?, char_dict_vec.clone())?,
                *routing,
            ))
        } else {
            None
        };
        let text_recognition_adapter = build_recognizer(text_recognition_model, char_dict_vec)?;
        let pipeline = OCRPipeline {
            rectification_adapter,
            document_orientation_adapter,
            text_detection_adapter,
            text_line_orientation_adapter,
            text_recognition_adapter,
            secondary_recognizer,
        };

        Ok(OAROCR {
            rec_score_threshold: self
                .text_recognition_config
                .as_ref()
                .map_or(0., |c| c.score_threshold),
            stages: self.stages,
            pipeline,
            text_type: self.text_type,
            return_word_box: self.return_word_box,
            image_batch_size,
            region_batch_size,
        })
    }

    fn validate_batch_size(field: &str, size: usize) -> Result<(), OCRError> {
        if size == 0 || size > Self::MAX_BATCH_SIZE {
            return Err(OCRError::validation_error(
                "OAROCRBuilder",
                field,
                &format!("1..={}", Self::MAX_BATCH_SIZE),
                &size.to_string(),
            ));
        }

        Ok(())
    }
}

/// OCR runtime for executing text detection and recognition.
///
/// This struct represents a configured OCR pipeline that can process images
/// to extract text.
#[derive(Debug)]
pub struct OAROCR {
    pub(super) rec_score_threshold: f32,
    pub(super) stages: PageStages,
    pub(super) pipeline: OCRPipeline,
    pub(super) text_type: Option<String>,
    pub(super) return_word_box: bool,
    /// Text detection batch size for `predict(images)`.
    ///
    /// This controls how many preprocessed images are sent to the text detection adapter in a
    /// single call. CPU builders resolve this to `1`; accelerator builders leave it as `None`
    /// so the adapter's `recommended_batch_size()` is used.
    pub(super) image_batch_size: Option<usize>,
    /// Batch size for text region recognition
    pub(super) region_batch_size: Option<usize>,
}

impl OAROCR {
    /// Predicts text from images using the configured OCR pipeline.
    ///
    /// This method orchestrates the execution of all configured tasks in the pipeline,
    /// including optional components like document orientation, rectification, and
    /// text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `images` - Collection of RGB images to process
    ///
    /// # Returns
    ///
    /// A vector of `OAROCRResult` containing the OCR results for each image,
    /// or an error if processing fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oar_ocr::oarocr::ocr::OAROCRBuilder;
    /// use oar_ocr::utils::load_image;
    /// use std::path::Path;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let ocr = OAROCRBuilder::new(
    ///     "det.onnx",
    ///     "rec.onnx",
    ///     "dict.txt",
    /// ).build()?;
    ///
    /// let image = load_image(Path::new("document.jpg"))?;
    /// let results = ocr.predict(vec![image])?;
    ///
    /// for result in results {
    ///     for region in result.text_regions {
    ///         if let Some(text) = region.text {
    ///             println!("Text: {}", text);
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict(
        &self,
        images: Vec<image::RgbImage>,
    ) -> Result<Vec<crate::oarocr::OAROCRResult>, OCRError> {
        Ok(self
            .predict_with_rows(images)?
            .into_iter()
            .map(|prediction| prediction.page)
            .collect())
    }

    /// Return each page with its region-aligned row IDs for joining same-row text with spaces and rows with newlines.
    pub fn predict_with_rows(
        &self,
        images: Vec<image::RgbImage>,
    ) -> Result<Vec<crate::oarocr::OAROCRPageWithRows>, OCRError> {
        use crate::oarocr::preprocess::DocumentPreprocessor;
        use std::sync::Arc;

        if images.is_empty() {
            return Err(OCRError::validation_error(
                "OCR Pipeline",
                "images",
                "non-empty slice",
                "empty slice",
            ));
        }

        let preprocessor = DocumentPreprocessor::new(
            self.pipeline.document_orientation_adapter.as_ref(),
            self.pipeline.rectification_adapter.as_ref(),
        );

        let mut prepared: Vec<(
            Arc<image::RgbImage>,
            crate::oarocr::preprocess::PreprocessResult,
        )> = Vec::with_capacity(images.len());

        for image in images.into_iter() {
            let input_img_arc = Arc::new(image);
            let preprocess = preprocessor.preprocess(Arc::clone(&input_img_arc))?;
            prepared.push((input_img_arc, preprocess));
        }

        let det_batch_size = self
            .image_batch_size
            .unwrap_or_else(|| {
                self.pipeline
                    .text_detection_adapter
                    .recommended_batch_size()
            })
            .max(1);

        let mut all_detection_boxes: Vec<Vec<BoundingBox>> = vec![Vec::new(); prepared.len()];

        let mut start = 0usize;
        while start < prepared.len() {
            let end = (start + det_batch_size).min(prepared.len());

            let batch_images: Vec<Arc<image::RgbImage>> = prepared[start..end]
                .iter()
                .map(|(_, preprocess)| Arc::clone(&preprocess.image))
                .collect();

            match self.detect_text_boxes_batch(batch_images) {
                Ok(batch_boxes) => {
                    for (offset, boxes) in batch_boxes.into_iter().enumerate() {
                        all_detection_boxes[start + offset] = boxes;
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        target: "ocr",
                        error = %err,
                        batch_start = start,
                        batch_end = end,
                        "Batched text detection failed; falling back to per-image detection"
                    );
                    for i in start..end {
                        all_detection_boxes[i] = self.detect_text_boxes(&prepared[i].1.image)?;
                    }
                }
            }

            start = end;
        }

        let inputs = prepared
            .iter()
            .map(|(_, p)| &p.image)
            .zip(all_detection_boxes)
            .map(|(img, boxes)| (img, self.line_image_fallback(img, boxes)));
        let pages = if self.stages.needs_page_loop() {
            inputs
                .map(|(image, boxes)| self.read_working_page(image, boxes))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let inputs: Vec<_> = inputs.collect();
            self.read_batch(&inputs)?
                .into_iter()
                .zip(&inputs)
                .map(|(page, (image, _))| {
                    self.finish_page(image, page, &nalgebra::Matrix3::identity())
                        .map(|(regions, rows)| (regions, rows, 0))
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut results = Vec::with_capacity(prepared.len());
        for (img_idx, ((input_img_arc, preprocess), (mut text_regions, rows, rotation))) in
            prepared.into_iter().zip(pages).enumerate()
        {
            if let Some(rot) = preprocess.rotation {
                Self::rotate_text_regions_back(&mut text_regions, rot);
            }

            results.push(crate::oarocr::OAROCRPageWithRows {
                page: crate::oarocr::OAROCRResult {
                    input_path: Arc::from(format!("image_{}", img_idx)),
                    index: img_idx,
                    input_img: input_img_arc,
                    text_regions,
                    orientation_angle: if self.stages.page_orientation
                        == PageOrientation::RecognitionVote
                    {
                        Some((preprocess.orientation_angle.unwrap_or(0.) + rotation as f32) % 360.)
                    } else {
                        preprocess.orientation_angle
                    },
                    rectified_img: preprocess.rectified_img,
                },
                row_ids: rows,
            });
        }

        Ok(results)
    }

    fn detect_text_boxes_batch(
        &self,
        images: Vec<Arc<image::RgbImage>>,
    ) -> Result<Vec<Vec<BoundingBox>>, OCRError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let input = ImageTaskInput::from_arc_images(images);
        let det = self.pipeline.text_detection_adapter.execute(input, None)?;

        Ok(det
            .detections
            .into_iter()
            .map(|detections| detections.into_iter().map(|d| d.bbox).collect())
            .collect())
    }

    pub(super) fn detect_text_boxes(
        &self,
        image: &Arc<image::RgbImage>,
    ) -> Result<Vec<BoundingBox>, OCRError> {
        Ok(self
            .detect_text_boxes_batch(vec![Arc::clone(image)])?
            .pop()
            .unwrap_or_default())
    }

    pub(super) fn line_orientations(
        &self,
        images: Vec<Arc<image::RgbImage>>,
    ) -> Result<Vec<Option<f32>>, OCRError> {
        let Some(adapter) = &self.pipeline.text_line_orientation_adapter else {
            return Ok(vec![None; images.len()]);
        };
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let n = images.len();
        let orient = adapter.execute(ImageTaskInput::from_arc_images(images), None)?;
        let mut angles: Vec<_> = orient
            .classifications
            .iter()
            .map(|classes| classes.first().map(|c| c.class_id as f32 * 180.))
            .collect();
        angles.resize(n, None);
        Ok(angles)
    }

    fn rotate_text_regions_back(
        regions: &mut [crate::oarocr::TextRegion],
        rot: crate::oarocr::preprocess::OrientationCorrection,
    ) {
        for region in regions {
            region.dt_poly = region.dt_poly.take().map(|poly| {
                poly.rotate_back_to_original(rot.angle, rot.rotated_width, rot.rotated_height)
            });
            region.rec_poly = region.rec_poly.take().map(|poly| {
                poly.rotate_back_to_original(rot.angle, rot.rotated_width, rot.rotated_height)
            });
            region.bounding_box = region.bounding_box.rotate_back_to_original(
                rot.angle,
                rot.rotated_width,
                rot.rotated_height,
            );

            if let Some(ref word_boxes) = region.word_boxes {
                let transformed_word_boxes: Vec<_> = word_boxes
                    .iter()
                    .map(|wb| {
                        wb.rotate_back_to_original(rot.angle, rot.rotated_width, rot.rotated_height)
                    })
                    .collect();
                region.word_boxes = Some(transformed_word_boxes);
            }
        }
    }

    /// Converts CTC column indices to word-level bounding boxes using standard approach.
    ///
    /// This method calculates character-specific widths based on the column indices from CTC decoding,
    /// which provides more accurate word boxes than uniform distribution.
    ///
    /// It aligns with standard logic by distinguishing between CJK and other characters:
    /// - CJK characters use a center-based approach with average character width to avoid being too narrow.
    /// - Other characters use the standard column-based width.
    ///
    /// # Arguments
    ///
    /// * `line_bbox` - The bounding box of the entire text line
    /// * `text` - The recognized text string
    /// * `col_indices` - Column indices (timesteps) for each character from CTC output
    /// * `seq_len` - Total number of columns (sequence length) in the CTC output
    /// * `wh_ratio` - Width/height ratio of this region's crop
    /// * `max_wh_ratio` - Max width/height ratio in the batch (used to undo padding)
    ///
    /// # Returns
    ///
    /// A vector of bounding boxes, one for each character
    pub(super) fn ctc_word_boxes(
        line_bbox: &BoundingBox,
        text: &str,
        col_indices: &[usize],
        seq_len: usize,
        wh_ratio: f32,
        max_wh_ratio: f32,
    ) -> Vec<BoundingBox> {
        if col_indices.is_empty() || seq_len == 0 || text.is_empty() {
            return Vec::new();
        }

        // Scale effective column count using standard logic (handles padding to max width)
        let effective_col_num = (seq_len as f32) * (wh_ratio / max_wh_ratio);
        if effective_col_num <= f32::EPSILON {
            return Vec::new();
        }

        // Get the line bounding box coordinates
        let x_min = line_bbox.x_min();
        let y_min = line_bbox.y_min();
        let x_max = line_bbox.x_max();
        let y_max = line_bbox.y_max();
        let width = x_max - x_min;

        // Calculate cell width (width of each column in the CTC output)
        let cell_width = width / effective_col_num.max(f32::EPSILON);

        let mut word_boxes = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let avg_char_width = width / chars.len().max(1) as f32;

        // Pre-calculate centers for all characters
        let centers: Vec<f32> = col_indices
            .iter()
            .map(|&idx| x_min + (idx as f32 + 0.5) * cell_width)
            .collect();

        for (i, _) in col_indices.iter().enumerate() {
            let ch = chars.get(i).copied().unwrap_or('?');
            let center_x = centers[i];

            if Self::is_cjk(ch) {
                let half_width = avg_char_width / 2.0;
                let char_x_min = (center_x - half_width).max(x_min);
                let char_x_max = (center_x + half_width).min(x_max);
                let char_box = BoundingBox::from_coords(char_x_min, y_min, char_x_max, y_max);
                word_boxes.push(char_box);
            } else {
                // For non-CJK characters, use the midpoint between adjacent character centers
                // to determine boundaries. This provides contiguous boxes that adapt to character density.
                let char_x_min = if i == 0 {
                    x_min
                } else {
                    (centers[i - 1] + center_x) / 2.0
                }
                .max(x_min);

                let char_x_max = if i == col_indices.len() - 1 {
                    x_max
                } else {
                    (center_x + centers[i + 1]) / 2.0
                }
                .min(x_max);

                let char_box = BoundingBox::from_coords(char_x_min, y_min, char_x_max, y_max);
                word_boxes.push(char_box);
            }
        }

        word_boxes
    }

    /// Converts normalized character positions to word-level bounding boxes.
    ///
    /// This is a fallback method that uses uniform character width distribution.
    /// Use `ctc_word_boxes` when CTC column indices are available for better accuracy.
    ///
    /// # Arguments
    ///
    /// * `line_bbox` - The bounding box of the entire text line
    /// * `char_positions` - Normalized x-positions (0.0-1.0) for each character
    /// * `char_count` - Number of characters in the text
    ///
    /// # Returns
    ///
    /// A vector of bounding boxes, one for each character/word
    pub(super) fn char_positions_to_word_boxes(
        line_bbox: &BoundingBox,
        char_positions: &[f32],
        char_count: usize,
    ) -> Vec<BoundingBox> {
        if char_positions.is_empty() || char_count == 0 {
            return Vec::new();
        }

        // Get the line bounding box coordinates
        let x_min = line_bbox.x_min();
        let y_min = line_bbox.y_min();
        let x_max = line_bbox.x_max();
        let y_max = line_bbox.y_max();
        let width = x_max - x_min;

        // Calculate approximate character width
        let char_width = width / char_count as f32;

        // Create a bounding box for each character based on its position
        let mut word_boxes = Vec::new();
        for &pos in char_positions.iter() {
            // Calculate x position (pos is normalized 0.0-1.0)
            let char_x_center = x_min + (pos * width);

            // Estimate character box boundaries
            // Use half character width on each side of the position
            let char_x_min = (char_x_center - char_width / 2.0).max(x_min);
            let char_x_max = (char_x_center + char_width / 2.0).min(x_max);

            // Use the full height of the text line for each character
            let char_box = BoundingBox::from_coords(char_x_min, y_min, char_x_max, y_max);
            word_boxes.push(char_box);
        }

        word_boxes
    }

    /// Detect whether a character is CJK.
    fn is_cjk(c: char) -> bool {
        let u = c as u32;
        (0x4E00..=0x9FFF).contains(&u)
            || (0x3400..=0x4DBF).contains(&u)
            || (0x20000..=0x2A6DF).contains(&u)
            || (0x2A700..=0x2B73F).contains(&u)
            || (0x2B740..=0x2B81F).contains(&u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pp_ocrv6_preset_sets_names_and_official_detection_defaults() {
        for (size, det, rec, dict, box_threshold) in [
            (
                PpOcrV6Size::Tiny,
                "pp-ocrv6_tiny_det.onnx",
                "pp-ocrv6_tiny_rec.onnx",
                "ppocrv6_tiny_dict.txt",
                0.4,
            ),
            (
                PpOcrV6Size::Small,
                "pp-ocrv6_small_det.onnx",
                "pp-ocrv6_small_rec.onnx",
                "ppocrv6_dict.txt",
                0.45,
            ),
            (
                PpOcrV6Size::Medium,
                "pp-ocrv6_medium_det.onnx",
                "pp-ocrv6_medium_rec.onnx",
                "ppocrv6_dict.txt",
                0.45,
            ),
        ] {
            let builder = OAROCRBuilder::pp_ocrv6(size);
            assert_eq!(
                builder.text_detection_model.as_path(),
                Some(std::path::Path::new(det))
            );
            assert_eq!(
                builder.text_recognition_model.as_path(),
                Some(std::path::Path::new(rec))
            );
            assert_eq!(builder.character_dict_path, PathBuf::from(dict));
            // The thresholds are preset defaults, not an explicit config, so
            // text-type defaults and explicit configs still take over.
            assert!(builder.text_detection_config.is_none());
            let config = builder.preset_text_detection_config.as_ref().unwrap();
            assert_eq!(config.score_threshold, 0.2);
            assert_eq!(config.box_threshold, box_threshold);
            assert_eq!(config.unclip_ratio, 1.4);
            assert_eq!(config.max_candidates, 3000);
            // Text types match case-insensitively: seal/table replace the
            // preset thresholds, anything else keeps them.
            let resolve = |text_type: &str| {
                OAROCRBuilder::pp_ocrv6(size)
                    .text_type(text_type)
                    .effective_text_detection_config()
                    .unclip_ratio
            };
            assert_eq!(resolve("Seal"), 0.5);
            assert_eq!(resolve("GENERAL"), 1.4);
        }
    }

    #[test]
    fn test_oarocr_builder_new() {
        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt");

        assert_eq!(
            builder.text_detection_model.as_path(),
            Some(std::path::Path::new("det.onnx"))
        );
        assert_eq!(
            builder.text_recognition_model.as_path(),
            Some(std::path::Path::new("rec.onnx"))
        );
        assert_eq!(builder.character_dict_path, PathBuf::from("dict.txt"));
        assert!(builder.document_orientation_model.is_none());
        assert!(builder.text_line_orientation_model.is_none());
        assert!(builder.document_rectification_model.is_none());
    }

    #[test]
    fn test_oarocr_builder_with_optional_components() {
        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt")
            .with_document_image_orientation_classification("doc_orient.onnx")
            .with_text_line_orientation_classification("line_orient.onnx")
            .with_document_image_rectification("rectify.onnx");

        let Some(source) = builder.document_orientation_model.as_ref() else {
            panic!("expected document_orientation_model to be Some");
        };
        assert_eq!(
            source.as_path(),
            Some(std::path::Path::new("doc_orient.onnx"))
        );
        let Some(source) = builder.text_line_orientation_model.as_ref() else {
            panic!("expected text_line_orientation_model to be Some");
        };
        assert_eq!(
            source.as_path(),
            Some(std::path::Path::new("line_orient.onnx"))
        );
        let Some(source) = builder.document_rectification_model.as_ref() else {
            panic!("expected document_rectification_model to be Some");
        };
        assert_eq!(source.as_path(), Some(std::path::Path::new("rectify.onnx")));
    }

    #[test]
    fn test_oarocr_builder_with_configuration() {
        let det_config = TextDetectionConfig {
            score_threshold: 0.5,
            box_threshold: 0.6,
            unclip_ratio: 1.8,
            max_candidates: 1000,
            limit_side_len: None,
            limit_type: None,
            max_side_len: None,
        };

        let rec_config = TextRecognitionConfig {
            score_threshold: 0.7,
        };

        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt")
            .text_detection_config(det_config.clone())
            .text_recognition_config(rec_config.clone());

        assert!(builder.text_detection_config.is_some());
        assert!(builder.text_recognition_config.is_some());
    }

    #[test]
    fn test_oarocr_builder_with_batch_sizes() {
        let builder = OAROCRBuilder::new("det.onnx", "rec.onnx", "dict.txt")
            .image_batch_size(4)
            .region_batch_size(64);

        assert_eq!(builder.image_batch_size, Some(4));
        assert_eq!(builder.region_batch_size, Some(64));
    }

    #[test]
    fn test_validate_batch_size_accepts_bounds() {
        assert!(OAROCRBuilder::validate_batch_size("image_batch_size", 1).is_ok());
        assert!(
            OAROCRBuilder::validate_batch_size("region_batch_size", OAROCRBuilder::MAX_BATCH_SIZE,)
                .is_ok()
        );
    }

    #[test]
    fn test_validate_batch_size_rejects_zero() {
        let err = OAROCRBuilder::validate_batch_size("image_batch_size", 0).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("image_batch_size"));
        assert!(msg.contains(&format!("1..={}", OAROCRBuilder::MAX_BATCH_SIZE)));
    }

    #[test]
    fn test_validate_batch_size_rejects_values_above_max() {
        let err = OAROCRBuilder::validate_batch_size(
            "region_batch_size",
            OAROCRBuilder::MAX_BATCH_SIZE + 1,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("region_batch_size"));
        assert!(msg.contains(&format!("1..={}", OAROCRBuilder::MAX_BATCH_SIZE)));
    }

    #[test]
    fn test_ctc_word_boxes_logic() {
        let line_bbox = BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0);
        // seq_len=10, wh_ratio=5 (100/20), max_wh_ratio=5 -> effective_col_num = 10
        // cell_width = 100/10 = 10.0

        // Test 1: Non-CJK "ABC"
        // Indices: 1, 4, 7 (approx centers: 15, 45, 75)
        let text = "ABC";
        let col_indices = vec![1, 4, 7];
        let seq_len = 10;
        let wh_ratio = 5.0;
        let max_wh_ratio = 5.0;

        let boxes = OAROCR::ctc_word_boxes(
            &line_bbox,
            text,
            &col_indices,
            seq_len,
            wh_ratio,
            max_wh_ratio,
        );

        assert_eq!(boxes.len(), 3);
        // Center 0: 1.5 * 10 = 15. Center 1: 4.5 * 10 = 45. Center 2: 7.5 * 10 = 75.
        // Box 0: Left=0, Right=(15+45)/2 = 30.
        // Box 1: Left=30, Right=(45+75)/2 = 60.
        // Box 2: Left=60, Right=100.

        assert!((boxes[0].x_min() - 0.0).abs() < 1e-5);
        assert!((boxes[0].x_max() - 30.0).abs() < 1e-5);
        assert!((boxes[1].x_min() - 30.0).abs() < 1e-5);
        assert!((boxes[1].x_max() - 60.0).abs() < 1e-5);
        assert!((boxes[2].x_min() - 60.0).abs() < 1e-5);
        assert!((boxes[2].x_max() - 100.0).abs() < 1e-5);
    }
}
