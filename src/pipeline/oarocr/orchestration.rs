//! Orchestration utilities for reducing duplication in OCR pipeline processing.
//!
//! This module provides abstractions to eliminate the duplicated orchestration logic
//! found across process_single_image, process_single_image_from_memory,
//! process_images_individually, and dynamic batching paths.
//!
//! # Architecture Overview
//!
//! The orchestration system consists of several key components:
//!
//! ## Core Components
//!
//! - **[`ImageProcessingOrchestrator`]**: Main orchestrator that handles batch processing
//!   with configurable parallel/sequential strategies and unified result management.
//!
//! - **[`PipelineExecutor`]**: Executes individual pipeline stages with configurable
//!   entry points, supporting both full pipeline execution and partial execution
//!   for dynamic batching scenarios.
//!
//! - **[`ImageInputSource`]**: Abstraction over different image input sources
//!   (file paths, in-memory images, pre-loaded images with paths).
//!
//! - **[`ProcessingStrategy`]**: Configurable strategy for parallel vs sequential
//!   processing decisions based on item count thresholds.
//!
//! - **[`PipelineStageConfig`]**: Configuration for pipeline execution, allowing
//!   stages to be skipped or execution to start from specific stages.
//!
//! ## Benefits
//!
//! ### Eliminated Duplication
//!
//! Before this refactoring, the following patterns were duplicated across multiple methods:
//!
//! - **Parallel Processing Logic**: Threshold-based decisions between `map()` and `par_iter()`
//! - **Index Management**: `enumerate()`, result collection as `(index, result)` tuples
//! - **Result Sorting**: `sort_by_key(|(index, _)| *index)` and extraction
//! - **Pipeline Stage Execution**: Identical sequences of orientation → rectification → detection → etc.
//! - **Error Handling**: Similar error propagation and logging patterns
//!
//! ### Unified Abstractions
//!
//! The new orchestration system provides:
//!
//! - **Single Source of Truth**: All orchestration logic centralized in one place
//! - **Configurable Execution**: Support for different processing strategies and stage configurations
//! - **Type Safety**: Enums and traits prevent invalid configurations at compile time
//! - **Maintainability**: Changes to orchestration logic only need to be made once
//! - **Testability**: Each component can be tested independently
//!
//! # Usage Examples
//!
//! ## Basic Single Image Processing
//!
//! ```rust,ignore
//! use crate::pipeline::oarocr::{ImageProcessingOrchestrator, ImageInputSource, PipelineStageConfig};
//! use std::path::Path;
//!
//! let orchestrator = ImageProcessingOrchestrator::new(&oarocr);
//! let input_source = ImageInputSource::Path(Path::new("image.jpg"));
//! let config = PipelineStageConfig::default(); // Full pipeline
//!
//! let result = orchestrator.process_single(input_source, 0, config)?;
//! ```
//!
//! ## Batch Processing with Auto Strategy
//!
//! ```rust,ignore
//! use crate::pipeline::oarocr::{ProcessingStrategy, ImageInputSource};
//!
//! let orchestrator = ImageProcessingOrchestrator::new(&oarocr);
//! let inputs: Vec<(usize, &Path)> = image_paths.iter().enumerate().collect();
//! let strategy = ProcessingStrategy::Auto(5); // Parallel if > 5 images
//! let config = PipelineStageConfig::default();
//!
//! let results = orchestrator.process_batch(inputs, strategy, config)?;
//! ```
//!
//! ## Custom Pipeline Configuration
//!
//! ```rust,ignore
//! use crate::pipeline::oarocr::{PipelineStageConfig, PipelineStage};
//! use std::collections::HashSet;
//!
//! let mut config = PipelineStageConfig::default();
//! config.skip_stages.insert(PipelineStage::Recognition); // Skip text recognition
//! config.start_from = PipelineStage::Detection; // Start from detection stage
//!
//! let result = orchestrator.process_single(input_source, 0, config)?;
//! ```
//!
//! # Migration Guide
//!
//! The refactoring maintains backward compatibility at the public API level.
//! Internal methods have been simplified:
//!
//! ## Before (Duplicated Logic)
//!
//! ```rust,ignore
//! // Each method had its own parallel processing logic
//! let results: Result<Vec<_>, OCRError> = if images.len() <= threshold {
//!     images.iter().enumerate().map(|(index, img)| {
//!         let mut result = self.process_single_image_from_memory(img, index)?;
//!         result.index = index;
//!         Ok((index, result))
//!     }).collect()
//! } else {
//!     images.par_iter().enumerate().map(|(index, img)| {
//!         // Same logic but parallel
//!     }).collect()
//! };
//! ```
//!
//! ## After (Unified Orchestration)
//!
//! ```rust,ignore
//! // Single line using orchestration abstraction
//! let orchestrator = ImageProcessingOrchestrator::new(self);
//! let inputs: Vec<(usize, &RgbImage)> = images.iter().enumerate().collect();
//! let strategy = ProcessingStrategy::Auto(threshold);
//! orchestrator.process_batch(inputs, strategy, PipelineStageConfig::default())
//! ```

use crate::core::{OCRError, traits::StandardPredictor};
use crate::pipeline::oarocr::{OAROCRResult, SingleImageProcessingParams};
use image::RgbImage;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

/// Type alias for recognition stage result: (recognized_texts, recognition_scores, failed_recognitions)
type RecognitionStageResult = Result<(Vec<Arc<str>>, Vec<f32>, usize), OCRError>;

/// Represents different sources of image input for processing.
#[derive(Debug)]
pub enum ImageInputSource<'a> {
    /// Load image from file path
    Path(&'a Path),
    /// Use image already in memory
    Memory(&'a RgbImage),
    /// Pre-loaded image with associated path (for dynamic batching)
    LoadedWithPath(Arc<RgbImage>, &'a Path),
}

impl<'a> ImageInputSource<'a> {
    /// Load the image into an Arc<RgbImage>, handling different input sources
    pub fn load_image(&self) -> Result<Arc<RgbImage>, OCRError> {
        match self {
            ImageInputSource::Path(path) => {
                let img = crate::utils::load_image(path)?;
                Ok(Arc::new(img))
            }
            ImageInputSource::Memory(img) => Ok(Arc::new((*img).clone())),
            ImageInputSource::LoadedWithPath(img_arc, _) => Ok(Arc::clone(img_arc)),
        }
    }

    /// Get the associated path if available
    pub fn path(&self) -> Option<&Path> {
        match self {
            ImageInputSource::Path(path) => Some(path),
            ImageInputSource::Memory(_) => None,
            ImageInputSource::LoadedWithPath(_, path) => Some(path),
        }
    }
}

/// Strategy for processing multiple images
#[derive(Debug, Clone)]
pub enum ProcessingStrategy {
    /// Always process sequentially
    Sequential,
    /// Always process in parallel
    Parallel,
    /// Automatically decide based on threshold
    Auto(usize),
}

impl ProcessingStrategy {
    /// Determine if parallel processing should be used for the given item count
    pub fn should_use_parallel(&self, item_count: usize) -> bool {
        match self {
            ProcessingStrategy::Sequential => false,
            ProcessingStrategy::Parallel => true,
            ProcessingStrategy::Auto(threshold) => item_count > *threshold,
        }
    }
}

/// Configuration for pipeline stage execution
#[derive(Debug, Clone)]
pub struct PipelineStageConfig<'a> {
    /// Which stage to start processing from
    pub start_from: PipelineStage,
    /// Stages to skip during processing
    pub skip_stages: HashSet<PipelineStage>,
    /// Custom parameters for continuing from detection stage
    pub custom_params: Option<SingleImageProcessingParams<'a>>,
}

impl<'a> Default for PipelineStageConfig<'a> {
    fn default() -> Self {
        Self {
            start_from: PipelineStage::Orientation,
            skip_stages: HashSet::new(),
            custom_params: None,
        }
    }
}

/// Represents different stages in the OCR pipeline
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PipelineStage {
    Orientation,
    Rectification,
    Detection,
    Cropping,
    TextLineOrientation,
    Recognition,
}

/// Main orchestrator for image processing operations.
///
/// This struct encapsulates the common patterns found across different
/// processing methods, eliminating duplication while maintaining flexibility.
pub struct ImageProcessingOrchestrator<'a> {
    /// Reference to the main OAROCR instance
    oar_ocr: &'a super::OAROCR,
}

/// Parameters for executing remaining pipeline stages
struct RemainingStagesParams<'a> {
    input_img_arc: Arc<RgbImage>,
    current_img: RgbImage,
    text_boxes: Vec<crate::processors::BoundingBox>,
    orientation_angle: Option<f32>,
    rectified_img: Option<Arc<RgbImage>>,
    image_path: Option<&'a Path>,
    index: usize,
    config: &'a PipelineStageConfig<'a>,
}

/// Parameters for building the final OCR result
struct FinalResultParams<'a> {
    input_img_arc: Arc<RgbImage>,
    _current_img: RgbImage,
    text_boxes: Vec<crate::processors::BoundingBox>,
    _cropped_images: Vec<Option<RgbImage>>,
    recognized_texts: Vec<Arc<str>>,
    recognition_scores: Vec<f32>,
    text_line_orientations: Vec<Option<f32>>,
    orientation_angle: Option<f32>,
    rectified_img: Option<Arc<RgbImage>>,
    image_path: Option<&'a Path>,
    index: usize,
    failed_crops: usize,
    failed_recognitions: usize,
}

impl<'a> ImageProcessingOrchestrator<'a> {
    /// Create a new orchestrator with a reference to the OAROCR instance
    pub fn new(oar_ocr: &'a super::OAROCR) -> Self {
        Self { oar_ocr }
    }

    /// Process a batch of images with the specified strategy and configuration.
    ///
    /// This method handles the common orchestration patterns:
    /// - Parallel vs sequential processing decisions
    /// - Index management and result sorting
    /// - Progress logging and error handling
    ///
    /// # Arguments
    ///
    /// * `inputs` - Vector of (index, input_source) pairs to process
    /// * `strategy` - Processing strategy (sequential, parallel, or auto)
    /// * `stage_config` - Configuration for pipeline stage execution
    ///
    /// # Returns
    ///
    /// A Result containing a vector of OAROCRResult ordered by original index
    pub fn process_batch<I>(
        &self,
        inputs: Vec<(usize, I)>,
        strategy: ProcessingStrategy,
        stage_config: PipelineStageConfig<'a>,
    ) -> Result<Vec<OAROCRResult>, OCRError>
    where
        I: Into<ImageInputSource<'a>> + Send + Sync,
    {
        debug!("Processing {} images with orchestrator", inputs.len());

        let use_parallel = strategy.should_use_parallel(inputs.len());

        let results: Result<Vec<_>, OCRError> = if use_parallel {
            debug!("Using parallel processing for {} images", inputs.len());
            inputs
                .into_par_iter()
                .map(|(index, input)| {
                    let input_source = input.into();
                    debug!(
                        "Processing image {} in parallel: {:?}",
                        index + 1,
                        input_source.path().unwrap_or_else(|| Path::new("memory"))
                    );

                    let mut result =
                        self.process_single(input_source, index, stage_config.clone())?;
                    result.index = index;
                    Ok((index, result))
                })
                .collect()
        } else {
            debug!("Using sequential processing for {} images", inputs.len());
            inputs
                .into_iter()
                .map(|(index, input)| {
                    let input_source = input.into();
                    debug!(
                        "Processing image {} sequentially: {:?}",
                        index + 1,
                        input_source.path().unwrap_or_else(|| Path::new("memory"))
                    );

                    let mut result =
                        self.process_single(input_source, index, stage_config.clone())?;
                    result.index = index;
                    Ok((index, result))
                })
                .collect()
        };

        // Sort results by original index and extract final results
        let mut indexed_results = results?;
        indexed_results.sort_by_key(|(index, _)| *index);
        let final_results: Vec<OAROCRResult> = indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect();

        info!(
            "OCR pipeline completed for {} images using orchestrator",
            final_results.len()
        );
        Ok(final_results)
    }

    /// Process a single image with the specified configuration.
    ///
    /// This method provides a unified entry point for single image processing
    /// that can handle different input sources and pipeline configurations.
    ///
    /// # Arguments
    ///
    /// * `input` - The image input source (path, memory, or pre-loaded)
    /// * `index` - Index of this image in the batch (for logging and results)
    /// * `stage_config` - Configuration for pipeline stage execution
    ///
    /// # Returns
    ///
    /// A Result containing the OAROCRResult for this image
    pub fn process_single(
        &self,
        input: ImageInputSource<'a>,
        index: usize,
        stage_config: PipelineStageConfig<'a>,
    ) -> Result<OAROCRResult, OCRError> {
        // Load the image based on input source
        let input_img_arc = input.load_image()?;
        let image_path = input.path();

        // Delegate to the pipeline executor
        let executor = PipelineExecutor::new(self.oar_ocr);
        executor.execute_pipeline(input_img_arc, image_path, index, stage_config)
    }
}

/// Executor for OCR pipeline stages with configurable entry points.
///
/// This struct handles the actual pipeline execution logic, supporting
/// different entry points for dynamic batching scenarios.
pub struct PipelineExecutor<'a> {
    /// Reference to the main OAROCR instance
    oar_ocr: &'a super::OAROCR,
}

impl<'a> PipelineExecutor<'a> {
    /// Create a new pipeline executor
    pub fn new(oar_ocr: &'a super::OAROCR) -> Self {
        Self { oar_ocr }
    }

    /// Execute the full pipeline or a subset based on configuration.
    ///
    /// This method consolidates the duplicated pipeline execution logic
    /// from process_single_image and process_single_image_from_memory.
    pub fn execute_pipeline(
        &self,
        input_img_arc: Arc<RgbImage>,
        image_path: Option<&Path>,
        index: usize,
        config: PipelineStageConfig<'a>,
    ) -> Result<OAROCRResult, OCRError> {
        // Handle custom parameters for detection-onwards processing
        if let Some(params) = config.custom_params {
            return self.execute_from_detection(params);
        }

        // Stage 1: Document orientation classification
        let (orientation_angle, mut current_img) =
            if config.skip_stages.contains(&PipelineStage::Orientation)
                || config.start_from > PipelineStage::Orientation
            {
                (None, input_img_arc.as_ref().clone())
            } else {
                self.execute_orientation_stage(input_img_arc.clone())?
            };

        // Stage 2: Document rectification
        let rectified_img = if config.skip_stages.contains(&PipelineStage::Rectification)
            || config.start_from > PipelineStage::Rectification
        {
            None
        } else {
            self.execute_rectification_stage(&mut current_img)?
        };

        // Stage 3: Text detection
        let text_boxes = if config.skip_stages.contains(&PipelineStage::Detection)
            || config.start_from > PipelineStage::Detection
        {
            Vec::new()
        } else {
            self.execute_detection_stage(&current_img)?
        };

        // Continue with remaining stages using the existing logic pattern
        let params = RemainingStagesParams {
            input_img_arc,
            current_img,
            text_boxes,
            orientation_angle,
            rectified_img,
            image_path,
            index,
            config: &config,
        };
        self.execute_remaining_stages(params)
    }

    /// Execute pipeline from the detection stage onwards.
    ///
    /// This method handles the case where detection has already been performed
    /// in dynamic batching scenarios.
    pub fn execute_from_detection(
        &self,
        params: SingleImageProcessingParams,
    ) -> Result<OAROCRResult, OCRError> {
        // Delegate to the existing method for now
        // This maintains compatibility while we refactor
        self.oar_ocr.process_single_image_from_detection(params)
    }

    /// Execute the document orientation classification stage
    fn execute_orientation_stage(
        &self,
        input_img_arc: Arc<RgbImage>,
    ) -> Result<(Option<f32>, RgbImage), OCRError> {
        use crate::pipeline::stages::OrientationStageProcessor;

        let orientation_config = self.oar_ocr.config.orientation_stage.as_ref().cloned();

        let orientation_stage_result = OrientationStageProcessor::process_single(
            input_img_arc,
            self.oar_ocr.doc_orientation_classifier.as_ref(),
            orientation_config.as_ref(),
        )?;

        let orientation_angle = orientation_stage_result.data.orientation_angle;
        let current_img = orientation_stage_result.data.corrected_image;

        Ok((orientation_angle, current_img))
    }

    /// Execute the document rectification stage
    fn execute_rectification_stage(
        &self,
        current_img: &mut RgbImage,
    ) -> Result<Option<Arc<RgbImage>>, OCRError> {
        if let Some(ref rectifier) = self.oar_ocr.doc_rectifier {
            let result = rectifier.predict(vec![current_img.clone()], None)?;
            if let Some(rectified) = result.rectified_img.first() {
                *current_img = (**rectified).clone();
                Ok(Some(Arc::clone(rectified)))
            } else {
                Ok(Some(Arc::new(current_img.clone())))
            }
        } else {
            Ok(None)
        }
    }

    /// Execute the text detection stage
    fn execute_detection_stage(
        &self,
        current_img: &RgbImage,
    ) -> Result<Vec<crate::processors::BoundingBox>, OCRError> {
        let result = self
            .oar_ocr
            .text_detector
            .predict(vec![current_img.clone()], None)?;
        let text_boxes: Vec<crate::processors::BoundingBox> =
            result.dt_polys.into_iter().flatten().collect();
        Ok(text_boxes)
    }

    /// Execute the remaining pipeline stages (cropping, text line orientation, recognition)
    fn execute_remaining_stages(
        &self,
        params: RemainingStagesParams<'_>,
    ) -> Result<OAROCRResult, OCRError> {
        // Stage 4: Text box cropping
        let (cropped_images, failed_crops) =
            if params.config.skip_stages.contains(&PipelineStage::Cropping)
                || params.config.start_from > PipelineStage::Cropping
            {
                (Vec::new(), 0)
            } else {
                self.execute_cropping_stage(&params.current_img, &params.text_boxes)?
            };

        // Stage 5: Text line orientation classification
        let text_line_orientations = if params
            .config
            .skip_stages
            .contains(&PipelineStage::TextLineOrientation)
            || params.config.start_from > PipelineStage::TextLineOrientation
        {
            Vec::new()
        } else {
            self.execute_text_line_orientation_stage(&cropped_images, &params.text_boxes)?
        };

        // Stage 6: Text recognition
        let (recognized_texts, recognition_scores, failed_recognitions) = if params
            .config
            .skip_stages
            .contains(&PipelineStage::Recognition)
            || params.config.start_from > PipelineStage::Recognition
        {
            (Vec::new(), Vec::new(), 0)
        } else {
            self.execute_recognition_stage(&cropped_images, &text_line_orientations)?
        };

        // Build the final result
        let final_params = FinalResultParams {
            input_img_arc: params.input_img_arc,
            _current_img: params.current_img,
            text_boxes: params.text_boxes,
            _cropped_images: cropped_images,
            recognized_texts,
            recognition_scores,
            text_line_orientations,
            orientation_angle: params.orientation_angle,
            rectified_img: params.rectified_img,
            image_path: params.image_path,
            index: params.index,
            failed_crops,
            failed_recognitions,
        };
        self.build_final_result(final_params)
    }

    /// Execute the text box cropping stage
    fn execute_cropping_stage(
        &self,
        current_img: &RgbImage,
        text_boxes: &[crate::processors::BoundingBox],
    ) -> Result<(Vec<Option<RgbImage>>, usize), OCRError> {
        use crate::pipeline::stages::{CroppingConfig, CroppingStageProcessor};

        let cropping_config = CroppingConfig::default();
        let cropping_stage_result = CroppingStageProcessor::process_single(
            current_img,
            text_boxes,
            Some(&cropping_config),
        )?;

        let cropped_images = cropping_stage_result.data.cropped_images;
        let failed_crops = cropping_stage_result.data.failed_crops;

        Ok((cropped_images, failed_crops))
    }

    /// Execute the text line orientation classification stage
    fn execute_text_line_orientation_stage(
        &self,
        cropped_images: &[Option<RgbImage>],
        text_boxes: &[crate::processors::BoundingBox],
    ) -> Result<Vec<Option<f32>>, OCRError> {
        let mut text_line_orientations: Vec<Option<f32>> = Vec::new();

        if self.oar_ocr.config.use_textline_orientation && !text_boxes.is_empty() {
            if let Some(ref classifier) = self.oar_ocr.text_line_classifier {
                let valid_images: Vec<RgbImage> = cropped_images
                    .iter()
                    .filter_map(|o| o.as_ref().cloned())
                    .collect();

                if !valid_images.is_empty() {
                    match classifier.predict(valid_images, None) {
                        Ok(result) => {
                            let mut result_idx = 0usize;
                            for cropped_img_opt in cropped_images {
                                if cropped_img_opt.is_some() {
                                    if let (Some(labels), Some(score_list)) = (
                                        result.label_names.get(result_idx),
                                        result.scores.get(result_idx),
                                    ) {
                                        if let (Some(label), Some(&score)) =
                                            (labels.first(), score_list.first())
                                        {
                                            let confidence_threshold = self
                                                .oar_ocr
                                                .config
                                                .text_line_orientation_stage
                                                .as_ref()
                                                .and_then(|config| config.confidence_threshold);

                                            let orientation_result =
                                                crate::core::parse_text_line_orientation(
                                                    label.as_ref(),
                                                    score,
                                                    confidence_threshold,
                                                );

                                            text_line_orientations.push(
                                                if orientation_result.is_confident {
                                                    Some(orientation_result.angle)
                                                } else {
                                                    None
                                                },
                                            );
                                        } else {
                                            text_line_orientations.push(None);
                                        }
                                    } else {
                                        text_line_orientations.push(None);
                                    }
                                    result_idx += 1;
                                } else {
                                    text_line_orientations.push(None);
                                }
                            }
                        }
                        Err(_) => {
                            // Fill with None values for failed classification
                            text_line_orientations = vec![None; cropped_images.len()];
                        }
                    }
                } else {
                    text_line_orientations = vec![None; cropped_images.len()];
                }
            } else {
                text_line_orientations = vec![None; cropped_images.len()];
            }
        } else {
            text_line_orientations = vec![None; cropped_images.len()];
        }

        Ok(text_line_orientations)
    }

    /// Execute the text recognition stage
    fn execute_recognition_stage(
        &self,
        cropped_images: &[Option<RgbImage>],
        text_line_orientations: &[Option<f32>],
    ) -> RecognitionStageResult {
        use crate::pipeline::stages::{RecognitionConfig, RecognitionStageProcessor};

        let recognition_config = RecognitionConfig::from_legacy_config(
            self.oar_ocr.config.use_textline_orientation,
            self.oar_ocr.config.aspect_ratio_bucketing.clone(),
        );

        let recognition_stage_result = RecognitionStageProcessor::process_single(
            cropped_images.to_vec(),
            Some(text_line_orientations),
            Some(&self.oar_ocr.text_recognizer),
            Some(&recognition_config),
        )?;

        let recognized_texts = recognition_stage_result.data.rec_texts;
        let recognition_scores = recognition_stage_result.data.rec_scores;
        let failed_recognitions = recognition_stage_result.data.failed_recognitions;

        Ok((recognized_texts, recognition_scores, failed_recognitions))
    }

    /// Build the final OAROCRResult from all pipeline stage results
    fn build_final_result(&self, params: FinalResultParams<'_>) -> Result<OAROCRResult, OCRError> {
        use crate::pipeline::oarocr::ErrorMetrics;

        // Convert recognition results to the format expected by OAROCRResult
        let mut final_texts = Vec::new();
        let mut final_scores = Vec::new();

        for i in 0..params.text_boxes.len() {
            if i < params.recognized_texts.len() {
                final_texts.push(Some(Arc::clone(&params.recognized_texts[i])));
            } else {
                final_texts.push(None);
            }

            if i < params.recognition_scores.len() {
                final_scores.push(Some(params.recognition_scores[i]));
            } else {
                final_scores.push(None);
            }
        }

        // Build text regions using the helper method
        let text_regions = OAROCRResult::create_text_regions_from_vectors(
            &params.text_boxes,
            &final_texts,
            &final_scores,
            &params.text_line_orientations,
        );

        // Calculate error metrics
        let error_metrics = ErrorMetrics {
            failed_crops: params.failed_crops,
            failed_recognitions: params.failed_recognitions,
            failed_orientations: 0, // This would need to be tracked if we implement it
            total_text_boxes: params.text_boxes.len(),
        };

        // Build the final result
        let input_path_str = params
            .image_path
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "memory".to_string());

        Ok(OAROCRResult {
            input_path: Arc::from(input_path_str),
            index: params.index,
            input_img: params.input_img_arc,
            text_regions,
            orientation_angle: params.orientation_angle,
            rectified_img: params.rectified_img,
            error_metrics,
        })
    }
}

// Implement conversion traits for convenience
impl<'a> From<&'a Path> for ImageInputSource<'a> {
    fn from(path: &'a Path) -> Self {
        ImageInputSource::Path(path)
    }
}

impl<'a> From<&'a RgbImage> for ImageInputSource<'a> {
    fn from(image: &'a RgbImage) -> Self {
        ImageInputSource::Memory(image)
    }
}
