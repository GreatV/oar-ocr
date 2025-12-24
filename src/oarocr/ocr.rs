//! High-level OCR builder API.
//!
//! This module provides `OAROCRBuilder` for constructing OCR pipelines with a fluent API.
//! It simplifies the process of configuring text detection, recognition, and optional
//! preprocessing components.

use crate::core::config::OrtSessionConfig;
use crate::core::constants::DEFAULT_REC_IMAGE_SHAPE;
use crate::core::errors::OCRError;
use crate::core::registry::{DynModelAdapter, TaskAdapter};
use crate::core::traits::adapter::AdapterBuilder;
use crate::domain::adapters::{
    DocumentOrientationAdapterBuilder, TextDetectionAdapterBuilder,
    TextLineOrientationAdapterBuilder, TextRecognitionAdapterBuilder, UVDocRectifierAdapterBuilder,
};
use crate::domain::tasks::{TextDetectionConfig, TextRecognitionConfig};
use crate::processors::BoundingBox;
use std::path::PathBuf;
use std::sync::Arc;

/// Internal structure holding the OCR pipeline adapters.
#[derive(Debug)]
struct OCRPipeline {
    rectification_adapter: Option<Arc<dyn DynModelAdapter>>,
    document_orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
    text_detection_adapter: Arc<dyn DynModelAdapter>,
    text_line_orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
    text_recognition_adapter: Arc<dyn DynModelAdapter>,
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
/// let ocr = OAROCRBuilder::new(
///     "path/to/text_detection.onnx",
///     "path/to/text_recognition.onnx",
///     "path/to/character_dict.txt"
/// )
/// .with_document_image_orientation_classification("path/to/orientation.onnx")
/// .with_text_line_orientation_classification("path/to/line_orientation.onnx")
/// .image_batch_size(4)
/// .region_batch_size(32)
/// .build()
/// .expect("Failed to build OCR pipeline");
/// ```
#[derive(Debug)]
pub struct OAROCRBuilder {
    // Required fields
    text_detection_model: PathBuf,
    text_recognition_model: PathBuf,
    character_dict_path: PathBuf,

    // Optional components
    document_orientation_model: Option<PathBuf>,
    text_line_orientation_model: Option<PathBuf>,
    document_rectification_model: Option<PathBuf>,

    // Configuration
    ort_session_config: Option<OrtSessionConfig>,
    text_detection_config: Option<TextDetectionConfig>,
    text_recognition_config: Option<TextRecognitionConfig>,
    image_batch_size: Option<usize>,
    region_batch_size: Option<usize>,

    // Text type and word box options
    text_type: Option<String>,
    return_word_box: bool,
}

impl OAROCRBuilder {
    /// Creates a new OCR builder with required components.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model` - Path to the text detection ONNX model
    /// * `text_recognition_model` - Path to the text recognition ONNX model
    /// * `character_dict_path` - Path to the character dictionary file
    pub fn new(
        text_detection_model: impl Into<PathBuf>,
        text_recognition_model: impl Into<PathBuf>,
        character_dict_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            text_detection_model: text_detection_model.into(),
            text_recognition_model: text_recognition_model.into(),
            character_dict_path: character_dict_path.into(),
            document_orientation_model: None,
            text_line_orientation_model: None,
            document_rectification_model: None,
            ort_session_config: None,
            text_detection_config: None,
            text_recognition_config: None,
            image_batch_size: None,
            region_batch_size: None,
            text_type: None,
            return_word_box: false,
        }
    }

    /// Sets the ONNX Runtime session configuration.
    ///
    /// This configuration will be applied to all models in the pipeline.
    pub fn ort_session(mut self, config: OrtSessionConfig) -> Self {
        self.ort_session_config = Some(config);
        self
    }

    /// Sets the text detection model configuration.
    ///
    /// The configuration should be a JSON value containing model-specific settings.
    pub fn text_detection_config(mut self, config: TextDetectionConfig) -> Self {
        self.text_detection_config = Some(config);
        self
    }

    /// Sets the text recognition model configuration.
    ///
    /// The configuration should be a JSON value containing model-specific settings.
    pub fn text_recognition_config(mut self, config: TextRecognitionConfig) -> Self {
        self.text_recognition_config = Some(config);
        self
    }

    /// Sets the batch size for processing input images during text detection.
    ///
    /// This controls how many images are sent to the text detection adapter per call.
    /// If a detector cannot batch the provided images (e.g., mismatched sizes), the
    /// pipeline falls back to per-image detection.
    pub fn image_batch_size(mut self, size: usize) -> Self {
        self.image_batch_size = Some(size);
        self
    }

    /// Sets the batch size for processing detected text regions.
    ///
    /// Controls memory usage during text recognition. Smaller values use less memory.
    /// Recommended: 32 for medium VRAM, 16 for low VRAM/CPU.
    pub fn region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = Some(size);
        self
    }

    /// Adds document image orientation classification to the pipeline.
    ///
    /// This component detects and corrects document orientation before text detection.
    pub fn with_document_image_orientation_classification(
        mut self,
        model_path: impl Into<PathBuf>,
    ) -> Self {
        self.document_orientation_model = Some(model_path.into());
        self
    }

    /// Adds text line orientation classification to the pipeline.
    ///
    /// This component detects and corrects text line orientation after text detection.
    pub fn with_text_line_orientation_classification(
        mut self,
        model_path: impl Into<PathBuf>,
    ) -> Self {
        self.text_line_orientation_model = Some(model_path.into());
        self
    }

    /// Adds document image rectification to the pipeline.
    ///
    /// This component corrects document distortion before text detection.
    pub fn with_document_image_rectification(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.document_rectification_model = Some(model_path.into());
        self
    }

    /// Sets the text type for sorting and cropping strategy.
    ///
    /// This matches the text_type parameter:
    /// - "seal": Uses polygon-based sorting/cropping for seal text (circular/curved)
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
    pub fn build(self) -> Result<OAROCR, OCRError> {
        // Load character dictionary for text recognition
        let char_dict = std::fs::read_to_string(&self.character_dict_path).map_err(|e| {
            OCRError::InvalidInput {
                message: format!(
                    "Failed to read character dictionary from '{}': {}",
                    self.character_dict_path.display(),
                    e
                ),
            }
        })?;

        // Build document rectification adapter if enabled
        let rectification_adapter = if let Some(ref rectification_model) =
            self.document_rectification_model
        {
            let mut builder = UVDocRectifierAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(rectification_model)?;
            Some(Arc::new(TaskAdapter::document_rectification(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build document orientation adapter if enabled
        let document_orientation_adapter = if let Some(ref orientation_model) =
            self.document_orientation_model
        {
            let mut builder = DocumentOrientationAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(orientation_model)?;
            Some(Arc::new(TaskAdapter::document_orientation(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build text detection adapter (required)
        let mut detection_builder = TextDetectionAdapterBuilder::new();

        if let Some(ref ort_config) = self.ort_session_config {
            detection_builder = detection_builder.with_ort_config(ort_config.clone());
        }

        // Align text detection defaults with OCR pipeline.
        // Defaults depend on text_type:
        // - general: limit_side_len=960, limit_type="max", thresh=0.3, box_thresh=0.6, unclip_ratio=2.0
        // - seal: limit_side_len=736, limit_type="min", thresh=0.2, box_thresh=0.6, unclip_ratio=0.5
        let mut effective_det_cfg = self.text_detection_config.clone().unwrap_or_default();
        let has_explicit_det_cfg = self.text_detection_config.is_some();
        if !has_explicit_det_cfg {
            match self.text_type.as_deref().unwrap_or("general") {
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
                    effective_det_cfg.score_threshold = 0.3;
                    effective_det_cfg.box_threshold = 0.6;
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
            }
        }

        detection_builder = detection_builder.with_config(effective_det_cfg);

        // Pass text_type to detection adapter for proper preprocessing configuration
        if let Some(ref text_type) = self.text_type {
            detection_builder = detection_builder.text_type(text_type.clone());
        }

        let text_detection_adapter = Arc::new(TaskAdapter::text_detection(
            detection_builder.build(&self.text_detection_model)?,
        )) as Arc<dyn DynModelAdapter>;

        // Build text line orientation adapter if enabled
        let text_line_orientation_adapter = if let Some(ref line_orientation_model) =
            self.text_line_orientation_model
        {
            let mut builder = TextLineOrientationAdapterBuilder::new();

            if let Some(ref ort_config) = self.ort_session_config {
                builder = builder.with_ort_config(ort_config.clone());
            }

            let adapter = builder.build(line_orientation_model)?;
            Some(Arc::new(TaskAdapter::text_line_orientation(adapter)) as Arc<dyn DynModelAdapter>)
        } else {
            None
        };

        // Build text recognition adapter (required)
        // Parse char_dict into Vec<String> - one character per line
        let char_dict_vec: Vec<String> = char_dict.lines().map(|s| s.to_string()).collect();

        let mut recognition_builder = TextRecognitionAdapterBuilder::new()
            .character_dict(char_dict_vec)
            .return_word_box(self.return_word_box);

        if let Some(ref ort_config) = self.ort_session_config {
            recognition_builder = recognition_builder.with_ort_config(ort_config.clone());
        }

        if let Some(ref rec_config) = self.text_recognition_config {
            recognition_builder = recognition_builder.with_config(rec_config.clone());
        }

        let text_recognition_adapter = Arc::new(TaskAdapter::text_recognition(
            recognition_builder.build(&self.text_recognition_model)?,
        )) as Arc<dyn DynModelAdapter>;

        let pipeline = OCRPipeline {
            rectification_adapter,
            document_orientation_adapter,
            text_detection_adapter,
            text_line_orientation_adapter,
            text_recognition_adapter,
        };

        Ok(OAROCR {
            pipeline,
            text_type: self.text_type,
            return_word_box: self.return_word_box,
            image_batch_size: self.image_batch_size,
            region_batch_size: self.region_batch_size,
        })
    }
}

/// OCR runtime for executing text detection and recognition.
///
/// This struct represents a configured OCR pipeline that can process images
/// to extract text.
#[derive(Debug)]
pub struct OAROCR {
    pipeline: OCRPipeline,
    text_type: Option<String>,
    return_word_box: bool,
    /// Text detection batch size for `predict(images)`.
    ///
    /// This controls how many preprocessed images are sent to the text detection adapter in a
    /// single call. If `None`, the adapter's `recommended_batch_size()` is used.
    image_batch_size: Option<usize>,
    /// Batch size for text region recognition
    region_batch_size: Option<usize>,
}

struct CroppedTextRegion {
    detection_index: usize,
    bbox: BoundingBox,
    image: image::RgbImage,
    wh_ratio: f32,
    line_orientation_angle: Option<f32>,
}

impl std::fmt::Debug for CroppedTextRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CroppedTextRegion")
            .field("detection_index", &self.detection_index)
            .field("bbox", &self.bbox)
            .field(
                "image",
                &format_args!("RgbImage({}x{})", self.image.width(), self.image.height()),
            )
            .field("wh_ratio", &self.wh_ratio)
            .field("line_orientation_angle", &self.line_orientation_angle)
            .finish()
    }
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
    ///     "models/det.onnx",
    ///     "models/rec.onnx",
    ///     "models/dict.txt",
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
            self.pipeline.document_orientation_adapter.clone(),
            self.pipeline.rectification_adapter.clone(),
        );

        let mut prepared: Vec<(
            Arc<image::RgbImage>,
            crate::oarocr::preprocess::PreprocessResult,
        )> = Vec::with_capacity(images.len());

        for image in images.into_iter() {
            let input_img_arc = Arc::new(image.clone());
            let preprocess = preprocessor.preprocess(image)?;
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

            let batch_images: Vec<image::RgbImage> = prepared[start..end]
                .iter()
                .map(|(_, preprocess)| preprocess.image.clone())
                .collect();

            match self.detect_sorted_text_boxes_batch(batch_images) {
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
                        all_detection_boxes[i] =
                            self.detect_sorted_text_boxes(&prepared[i].1.image)?;
                    }
                }
            }

            start = end;
        }

        let mut results = Vec::with_capacity(prepared.len());
        for (img_idx, (input_img_arc, preprocess)) in prepared.into_iter().enumerate() {
            let detection_boxes = all_detection_boxes[img_idx].clone();
            results.push(self.predict_single(
                img_idx,
                input_img_arc,
                preprocess,
                detection_boxes,
            )?);
        }

        Ok(results)
    }

    fn predict_single(
        &self,
        img_idx: usize,
        input_img: std::sync::Arc<image::RgbImage>,
        preprocess: crate::oarocr::preprocess::PreprocessResult,
        detection_boxes: Vec<BoundingBox>,
    ) -> Result<crate::oarocr::OAROCRResult, OCRError> {
        use std::sync::Arc;

        let current_image = preprocess.image;

        let mut cropped_regions = self.crop_text_regions(&current_image, &detection_boxes)?;
        self.classify_line_orientations(&mut cropped_regions)?;

        let recognized = self.recognize_text_regions(detection_boxes.len(), cropped_regions)?;

        // Preserve reading order by emitting in detection-index order.
        let mut text_regions: Vec<crate::oarocr::TextRegion> =
            recognized.into_iter().flatten().collect();

        if let Some(rot) = preprocess.rotation {
            Self::rotate_text_regions_back(&mut text_regions, rot);
        }

        Ok(crate::oarocr::OAROCRResult {
            input_path: Arc::from(format!("image_{}", img_idx)),
            index: img_idx,
            input_img,
            text_regions,
            orientation_angle: preprocess.orientation_angle,
            rectified_img: preprocess.rectified_img,
        })
    }

    fn detect_sorted_text_boxes_batch(
        &self,
        images: Vec<image::RgbImage>,
    ) -> Result<Vec<Vec<BoundingBox>>, OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;

        if images.is_empty() {
            return Ok(Vec::new());
        }

        let input = DynTaskInput::from_images(ImageTaskInput::new(images));
        let output = self.pipeline.text_detection_adapter.execute_dyn(input)?;
        let det = output.into_text_detection()?;

        let mut results: Vec<Vec<BoundingBox>> = Vec::with_capacity(det.detections.len());
        for detections in det.detections.into_iter() {
            let boxes = detections.into_iter().map(|d| d.bbox).collect::<Vec<_>>();
            results.push(self.sort_detection_boxes(&boxes));
        }

        Ok(results)
    }

    fn detect_sorted_text_boxes(
        &self,
        image: &image::RgbImage,
    ) -> Result<Vec<BoundingBox>, OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;

        let input = DynTaskInput::from_images(ImageTaskInput::new(vec![image.clone()]));
        let output = self.pipeline.text_detection_adapter.execute_dyn(input)?;
        let det = output.into_text_detection()?;

        let boxes = det
            .detections
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(|d| d.bbox)
            .collect::<Vec<_>>();

        Ok(self.sort_detection_boxes(&boxes))
    }

    fn sort_detection_boxes(&self, boxes: &[BoundingBox]) -> Vec<BoundingBox> {
        if boxes.is_empty() {
            return Vec::new();
        }

        let is_seal_text = self
            .text_type
            .as_ref()
            .map(|t| t.to_lowercase() == "seal")
            .unwrap_or(false);

        if is_seal_text {
            crate::processors::sort_poly_boxes(boxes)
        } else {
            crate::processors::sort_quad_boxes(boxes)
        }
    }

    fn crop_text_regions(
        &self,
        image: &image::RgbImage,
        detection_boxes: &[BoundingBox],
    ) -> Result<Vec<CroppedTextRegion>, OCRError> {
        use crate::oarocr::EdgeProcessor;
        use crate::oarocr::TextCroppingProcessor;
        use std::sync::Arc;

        if detection_boxes.is_empty() {
            return Ok(Vec::new());
        }

        let processor = TextCroppingProcessor::new(true); // handle_rotation = true
        let cropped = processor.process((Arc::new(image.clone()), detection_boxes.to_vec()))?;

        let mut regions = Vec::new();
        for (idx, crop_result) in cropped.into_iter().enumerate() {
            let Some(img) = crop_result else {
                continue;
            };
            let img = (*img).clone();
            let wh_ratio = img.width() as f32 / img.height().max(1) as f32;
            regions.push(CroppedTextRegion {
                detection_index: idx,
                bbox: detection_boxes
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| BoundingBox::from_coords(0.0, 0.0, 0.0, 0.0)),
                image: img,
                wh_ratio,
                line_orientation_angle: None,
            });
        }

        Ok(regions)
    }

    fn classify_line_orientations(
        &self,
        regions: &mut [CroppedTextRegion],
    ) -> Result<(), OCRError> {
        use crate::core::registry::DynTaskInput;
        use crate::core::traits::task::ImageTaskInput;

        let Some(ref line_orientation_adapter) = self.pipeline.text_line_orientation_adapter else {
            return Ok(());
        };

        if regions.is_empty() {
            return Ok(());
        }

        let input_images = regions.iter().map(|r| r.image.clone()).collect();
        let input = DynTaskInput::from_images(ImageTaskInput::new(input_images));
        let output = line_orientation_adapter.execute_dyn(input)?;
        let orient = output.into_text_line_orientation()?;

        for (idx, classifications) in orient
            .classifications
            .iter()
            .enumerate()
            .take(regions.len())
        {
            let Some(top_class) = classifications.first() else {
                continue;
            };

            // Convert class_id to angle (0=0°, 1=180°)
            let angle = (top_class.class_id as f32) * 180.0;
            regions[idx].line_orientation_angle = Some(angle);

            if top_class.class_id == 1 {
                regions[idx].image = image::imageops::rotate180(&regions[idx].image);
            }
        }

        Ok(())
    }

    fn recognize_text_regions(
        &self,
        detection_count: usize,
        mut regions: Vec<CroppedTextRegion>,
    ) -> Result<Vec<Option<crate::oarocr::TextRegion>>, OCRError> {
        use crate::core::registry::DynTaskInput;

        let mut results: Vec<Option<crate::oarocr::TextRegion>> = vec![None; detection_count];
        if regions.is_empty() {
            return Ok(results);
        }

        regions.sort_by(|a, b| {
            a.wh_ratio
                .partial_cmp(&b.wh_ratio)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let base_rec_ratio = DEFAULT_REC_IMAGE_SHAPE[2] as f32 / DEFAULT_REC_IMAGE_SHAPE[1] as f32;
        let batch_size = self.region_batch_size.unwrap_or(regions.len()).max(1);

        for chunk in regions.chunks(batch_size) {
            let chunk_max_wh_ratio = chunk
                .iter()
                .map(|r| r.wh_ratio)
                .fold(base_rec_ratio, |acc, r| acc.max(r));

            let rec_input = DynTaskInput::from_text_recognition(
                crate::domain::tasks::TextRecognitionInput::new(
                    chunk.iter().map(|r| r.image.clone()).collect(),
                ),
            );

            let rec_output = self
                .pipeline
                .text_recognition_adapter
                .execute_dyn(rec_input)?;
            let rec = rec_output.into_text_recognition()?;

            let n = rec.texts.len().min(chunk.len());
            for (i, region) in chunk.iter().take(n).enumerate() {
                let text = rec.texts.get(i).map(String::as_str).unwrap_or("");
                let score = *rec.scores.get(i).unwrap_or(&0.0);

                let char_positions: &[f32] = rec
                    .char_positions
                    .get(i)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let col_indices: &[usize] = rec
                    .char_col_indices
                    .get(i)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let seq_len = *rec.sequence_lengths.get(i).unwrap_or(&0);

                let bbox = region.bbox.clone();
                let word_boxes = if self.return_word_box && !col_indices.is_empty() && seq_len > 0 {
                    Some(Self::ctc_word_boxes(
                        &bbox,
                        text,
                        col_indices,
                        seq_len,
                        region.wh_ratio,
                        chunk_max_wh_ratio,
                    ))
                } else if self.return_word_box && !char_positions.is_empty() {
                    Some(Self::char_positions_to_word_boxes(
                        &bbox,
                        char_positions,
                        text.chars().count(),
                    ))
                } else {
                    None
                };

                if region.detection_index < results.len() {
                    results[region.detection_index] = Some(crate::oarocr::TextRegion {
                        bounding_box: bbox.clone(),
                        dt_poly: Some(bbox.clone()),
                        rec_poly: Some(bbox),
                        text: Some(std::sync::Arc::from(text)),
                        confidence: Some(score),
                        orientation_angle: region.line_orientation_angle,
                        word_boxes,
                    });
                }
            }
        }

        Ok(results)
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
    /// * `col_indices` - Column indices (timesteps) for each character from CTC output
    /// * `seq_len` - Total number of columns (sequence length) in the CTC output
    /// * `text` - The recognized text string
    ///
    /// # Returns
    ///
    /// A vector of bounding boxes, one for each character
    fn ctc_word_boxes(
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
    /// Use col_indices_to_word_boxes when CTC column indices are available for better accuracy.
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
    fn char_positions_to_word_boxes(
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
    fn test_oarocr_builder_new() {
        let builder = OAROCRBuilder::new("models/det.onnx", "models/rec.onnx", "models/dict.txt");

        assert_eq!(
            builder.text_detection_model,
            PathBuf::from("models/det.onnx")
        );
        assert_eq!(
            builder.text_recognition_model,
            PathBuf::from("models/rec.onnx")
        );
        assert_eq!(
            builder.character_dict_path,
            PathBuf::from("models/dict.txt")
        );
        assert!(builder.document_orientation_model.is_none());
        assert!(builder.text_line_orientation_model.is_none());
        assert!(builder.document_rectification_model.is_none());
    }

    #[test]
    fn test_oarocr_builder_with_optional_components() {
        let builder = OAROCRBuilder::new("models/det.onnx", "models/rec.onnx", "models/dict.txt")
            .with_document_image_orientation_classification("models/doc_orient.onnx")
            .with_text_line_orientation_classification("models/line_orient.onnx")
            .with_document_image_rectification("models/rectify.onnx");

        assert!(builder.document_orientation_model.is_some());
        assert_eq!(
            builder.document_orientation_model.unwrap(),
            PathBuf::from("models/doc_orient.onnx")
        );
        assert!(builder.text_line_orientation_model.is_some());
        assert_eq!(
            builder.text_line_orientation_model.unwrap(),
            PathBuf::from("models/line_orient.onnx")
        );
        assert!(builder.document_rectification_model.is_some());
        assert_eq!(
            builder.document_rectification_model.unwrap(),
            PathBuf::from("models/rectify.onnx")
        );
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
            max_text_length: 128,
        };

        let builder = OAROCRBuilder::new("models/det.onnx", "models/rec.onnx", "models/dict.txt")
            .text_detection_config(det_config.clone())
            .text_recognition_config(rec_config.clone());

        assert!(builder.text_detection_config.is_some());
        assert!(builder.text_recognition_config.is_some());
    }

    #[test]
    fn test_oarocr_builder_with_batch_sizes() {
        let builder = OAROCRBuilder::new("models/det.onnx", "models/rec.onnx", "models/dict.txt")
            .image_batch_size(4)
            .region_batch_size(64);

        assert_eq!(builder.image_batch_size, Some(4));
        assert_eq!(builder.region_batch_size, Some(64));
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
