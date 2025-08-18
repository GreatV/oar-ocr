//! Builder pattern implementation for the OAROCR pipeline.

use crate::core::{DynamicBatchConfig, ShapeCompatibilityStrategy};
use crate::pipeline::oarocr::config::OAROCRConfig;
use crate::predictor::{
    DocOrientationClassifierConfig, DoctrRectifierPredictorConfig, TextLineClasPredictorConfig,
};
use crate::processors::{AspectRatioBucketingConfig, LimitType};
use crate::{impl_complete_builder, with_nested};
use std::path::PathBuf;
use tracing::warn;

/// Builder for creating OAROCR instances.
///
/// This struct provides a fluent API for configuring and building
/// OAROCR pipeline instances with various options.
#[derive(Debug)]
pub struct OAROCRBuilder {
    config: OAROCRConfig,
}

impl OAROCRBuilder {
    /// Creates a new OAROCRBuilder with the required parameters.
    ///
    /// # Arguments
    ///
    /// * `text_detection_model_path` - Path to the text detection model file
    /// * `text_recognition_model_path` - Path to the text recognition model file
    /// * `text_rec_character_dict_path` - Path to the character dictionary file
    ///
    /// # Returns
    ///
    /// A new OAROCRBuilder instance
    pub fn new(
        text_detection_model_path: String,
        text_recognition_model_path: String,
        text_rec_character_dict_path: String,
    ) -> Self {
        Self {
            config: OAROCRConfig::new(
                text_detection_model_path,
                text_recognition_model_path,
                text_rec_character_dict_path,
            ),
        }
    }

    /// Creates a new OAROCRBuilder from an existing configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The OAROCRConfig to use
    ///
    /// # Returns
    ///
    /// A new OAROCRBuilder instance
    pub fn from_config(config: OAROCRConfig) -> Self {
        Self { config }
    }

    /// Sets the document orientation classification model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_orientation_classify_model_name(mut self, name: String) -> Self {
        with_nested!(self.config.orientation, DocOrientationClassifierConfig, config => {
            config.common.model_name = Some(name);
        });
        self
    }

    /// Sets the document orientation classification model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_orientation_classify_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        with_nested!(self.config.orientation, DocOrientationClassifierConfig, config => {
            config.common.model_path = Some(path.into());
        });
        self
    }

    /// Sets the document orientation confidence threshold.
    ///
    /// Specifies the minimum confidence score required for orientation predictions.
    /// If the confidence is below this threshold, the orientation may be treated
    /// as uncertain and fall back to default behavior.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum confidence threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// The updated builder instance
    /// Sets the confidence threshold for document orientation classification.
    ///
    /// This threshold determines the minimum confidence required for orientation predictions.
    /// If the confidence is below this threshold, the orientation may be treated
    /// as uncertain and fall back to default behavior.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum confidence threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_orientation_threshold(mut self, threshold: f32) -> Self {
        let t = if (0.0..=1.0).contains(&threshold) {
            threshold
        } else {
            warn!("doc_orientation_threshold out of range [{threshold}], clamping to [0.0, 1.0]");
            threshold.clamp(0.0, 1.0)
        };
        if self.config.orientation_stage.is_none() {
            self.config.orientation_stage =
                Some(crate::pipeline::stages::OrientationConfig::default());
        }
        if let Some(ref mut config) = self.config.orientation_stage {
            config.confidence_threshold = Some(t);
        }
        self
    }

    /// Sets the document unwarping model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_unwarping_model_name(mut self, name: String) -> Self {
        with_nested!(self.config.rectification, DoctrRectifierPredictorConfig, config => {
            config.common.model_name = Some(name);
        });
        self
    }

    /// Sets the document unwarping model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn doc_unwarping_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        with_nested!(self.config.rectification, DoctrRectifierPredictorConfig, config => {
            config.common.model_path = Some(path.into());
        });
        self
    }

    /// Sets the text detection model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_model_name(mut self, name: String) -> Self {
        self.config.detection.common.model_name = Some(name);
        self
    }

    /// Sets the text detection model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.detection.common.model_path = Some(path.into());
        self
    }

    /// Sets the text detection batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for inference
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_batch_size(mut self, batch_size: usize) -> Self {
        let bs = if batch_size >= 1 {
            batch_size
        } else {
            warn!("text_detection_batch_size must be >= 1, got {batch_size}; using 1");
            1
        };
        self.config.detection.common.batch_size = Some(bs);
        self
    }

    /// Sets the text recognition model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_model_name(mut self, name: String) -> Self {
        self.config.recognition.common.model_name = Some(name);
        self
    }

    /// Sets the text recognition model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.recognition.common.model_path = Some(path.into());
        self
    }

    /// Sets the text recognition batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for inference
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_batch_size(mut self, batch_size: usize) -> Self {
        let bs = if batch_size >= 1 {
            batch_size
        } else {
            warn!("text_recognition_batch_size must be >= 1, got {batch_size}; using 1");
            1
        };
        self.config.recognition.common.batch_size = Some(bs);
        self
    }

    /// Sets the text line orientation classification model name.
    ///
    /// # Arguments
    ///
    /// * `name` - The model name
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_model_name(mut self, name: String) -> Self {
        with_nested!(self.config.text_line_orientation, TextLineClasPredictorConfig, config => {
            config.common.model_name = Some(name);
        });
        self
    }

    /// Sets the text line orientation classification model path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the model file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        with_nested!(self.config.text_line_orientation, TextLineClasPredictorConfig, config => {
            config.common.model_path = Some(path.into());
        });
        self
    }

    /// Sets the text line orientation classification batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for inference
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_classify_batch_size(mut self, batch_size: usize) -> Self {
        let bs = if batch_size >= 1 {
            batch_size
        } else {
            warn!(
                "textline_orientation_classify_batch_size must be >= 1, got {batch_size}; using 1"
            );
            1
        };
        with_nested!(self.config.text_line_orientation, TextLineClasPredictorConfig, config => {
            config.common.batch_size = Some(bs);
        });
        self
    }

    /// Sets the text line orientation classification input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The input shape as (width, height)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// Sets the text line orientation classifier input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The input shape as (width, height)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_input_shape(mut self, shape: (u32, u32)) -> Self {
        let w = if shape.0 > 0 {
            shape.0
        } else {
            warn!(
                "textline_orientation_input_shape width {} <= 0; using 1",
                shape.0
            );
            1u32
        };
        let h = if shape.1 > 0 {
            shape.1
        } else {
            warn!(
                "textline_orientation_input_shape height {} <= 0; using 1",
                shape.1
            );
            1u32
        };
        with_nested!(self.config.text_line_orientation, TextLineClasPredictorConfig, config => {
            config.input_shape = Some((w, h));
        });
        self
    }

    /// Sets the text line orientation confidence threshold.
    ///
    /// Specifies the minimum confidence score required for text line orientation predictions.
    /// If the confidence is below this threshold, the orientation may be treated
    /// as uncertain and fall back to default behavior.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum confidence threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// Sets the text line orientation confidence threshold.
    ///
    /// Specifies the minimum confidence score required for text line orientation predictions.
    /// If the confidence is below this threshold, the orientation may be treated
    /// as uncertain and fall back to default behavior.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum confidence threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_threshold(mut self, threshold: f32) -> Self {
        let t = if (0.0..=1.0).contains(&threshold) {
            threshold
        } else {
            warn!(
                "textline_orientation_threshold out of range [{threshold}], clamping to [0.0, 1.0]"
            );
            threshold.clamp(0.0, 1.0)
        };
        if self.config.text_line_orientation_stage.is_none() {
            self.config.text_line_orientation_stage =
                Some(crate::pipeline::stages::TextLineOrientationConfig::default());
        }
        if let Some(ref mut config) = self.config.text_line_orientation_stage {
            config.confidence_threshold = Some(t);
        }
        self
    }

    /// Sets whether to use document orientation classification.
    ///
    /// # Arguments
    ///
    /// * `use_it` - Whether to use document orientation classification
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn use_doc_orientation_classify(mut self, use_it: bool) -> Self {
        self.config.use_doc_orientation_classify = use_it;
        self
    }

    /// Sets whether to use document unwarping.
    ///
    /// # Arguments
    ///
    /// * `use_it` - Whether to use document unwarping
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn use_doc_unwarping(mut self, use_it: bool) -> Self {
        self.config.use_doc_unwarping = use_it;
        self
    }

    /// Sets whether to use text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `use_it` - Whether to use text line orientation classification
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn use_textline_orientation(mut self, use_it: bool) -> Self {
        self.config.use_textline_orientation = use_it;
        self
    }

    /// Sets the parallel processing policy for the pipeline.
    ///
    /// # Arguments
    ///
    /// * `policy` - The parallel processing policy configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn parallel_policy(mut self, policy: super::config::ParallelPolicy) -> Self {
        self.config.parallel_policy = policy;
        self
    }

    /// Sets the ONNX Runtime session configuration for text detection.
    ///
    /// # Arguments
    ///
    /// * `config` - The ONNX Runtime session configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_ort_session(
        mut self,
        config: crate::core::config::onnx::OrtSessionConfig,
    ) -> Self {
        self.config.detection.common.ort_session = Some(config);
        self
    }

    /// Sets the ONNX Runtime session configuration for text recognition.
    ///
    /// # Arguments
    ///
    /// * `config` - The ONNX Runtime session configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_ort_session(
        mut self,
        config: crate::core::config::onnx::OrtSessionConfig,
    ) -> Self {
        self.config.recognition.common.ort_session = Some(config);
        self
    }

    /// Sets the ONNX Runtime session configuration for text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `config` - The ONNX Runtime session configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_ort_session(
        mut self,
        config: crate::core::config::onnx::OrtSessionConfig,
    ) -> Self {
        with_nested!(self.config.text_line_orientation, crate::predictor::TextLineClasPredictorConfig, tlo_config => {
            tlo_config.common.ort_session = Some(config);
        });
        self
    }

    /// Sets the ONNX Runtime session configuration for all components.
    ///
    /// This is a convenience method that applies the same ONNX session configuration
    /// to text detection, text recognition, and text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `config` - The ONNX Runtime session configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn global_ort_session(
        mut self,
        config: crate::core::config::onnx::OrtSessionConfig,
    ) -> Self {
        // Apply to text detection
        self.config.detection.common.ort_session = Some(config.clone());

        // Apply to text recognition
        self.config.recognition.common.ort_session = Some(config.clone());

        // Apply to text line orientation (if configured)
        with_nested!(self.config.text_line_orientation, crate::predictor::TextLineClasPredictorConfig, tlo_config => {
            tlo_config.common.ort_session = Some(config.clone());
        });

        self
    }

    /// Convenience method to enable CUDA execution with default settings.
    ///
    /// This configures CUDA execution provider with sensible defaults:
    /// - Uses GPU device 0
    /// - Falls back to CPU if CUDA fails
    /// - Uses default memory and performance settings
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection.onnx".to_string(),
    ///     "recognition.onnx".to_string(),
    ///     "dict.txt".to_string()
    /// )
    /// .with_cuda(); // Simple CUDA setup
    /// ```
    #[cfg(feature = "cuda")]
    pub fn with_cuda(self) -> Self {
        self.with_cuda_device(0)
    }

    /// Convenience method to enable CUDA execution on a specific GPU device.
    ///
    /// This configures CUDA execution provider with:
    /// - Specified GPU device ID
    /// - Falls back to CPU if CUDA fails
    /// - Uses default memory and performance settings
    ///
    /// # Arguments
    ///
    /// * `device_id` - The GPU device ID to use (0, 1, 2, etc.)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection.onnx".to_string(),
    ///     "recognition.onnx".to_string(),
    ///     "dict.txt".to_string()
    /// )
    /// .with_cuda_device(1); // Use GPU 1
    /// ```
    #[cfg(feature = "cuda")]
    pub fn with_cuda_device(self, device_id: u32) -> Self {
        use crate::core::config::{OrtExecutionProvider, OrtSessionConfig};

        let ort_config = OrtSessionConfig::new().with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(device_id as i32),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                do_copy_in_default_stream: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU, // Fallback to CPU
        ]);

        self.global_ort_session(ort_config)
    }

    /// Convenience method to enable high-performance processing configuration.
    ///
    /// This applies optimizations for batch processing:
    /// - Increases parallel processing thresholds
    /// - Optimizes memory usage
    /// - Configures efficient batching strategies
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection.onnx".to_string(),
    ///     "recognition.onnx".to_string(),
    ///     "dict.txt".to_string()
    /// )
    /// .with_high_performance(); // Optimize for batch processing
    /// ```
    pub fn with_high_performance(self) -> Self {
        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4); // Fallback to 4 threads if detection fails

        self.parallel_policy(
            crate::pipeline::oarocr::config::ParallelPolicy::new()
                .with_max_threads(Some(max_threads))
                .with_image_threshold(2)
                .with_text_box_threshold(5)
                .with_batch_threshold(3),
        )
    }

    /// Convenience method for mobile/resource-constrained environments.
    ///
    /// This configures the pipeline for minimal resource usage:
    /// - Reduces parallel processing to avoid overwhelming the system
    /// - Uses conservative memory settings
    /// - Prioritizes stability over speed
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection.onnx".to_string(),
    ///     "recognition.onnx".to_string(),
    ///     "dict.txt".to_string()
    /// )
    /// .with_low_resource(); // Optimize for limited resources
    /// ```
    pub fn with_low_resource(self) -> Self {
        self.parallel_policy(
            crate::pipeline::oarocr::config::ParallelPolicy::new()
                .with_max_threads(Some(2)) // Limit threads
                .with_image_threshold(10) // Higher thresholds for parallel processing
                .with_text_box_threshold(20)
                .with_batch_threshold(10),
        )
    }

    /// Sets the session pool size for text detection.
    ///
    /// # Arguments
    ///
    /// * `size` - The session pool size (minimum 1)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_detection_session_pool_size(mut self, size: usize) -> Self {
        let s = if size >= 1 {
            size
        } else {
            warn!("text_detection_session_pool_size must be >= 1, got {size}; using 1");
            1
        };
        self.config.detection.common.session_pool_size = Some(s);
        self
    }

    /// Sets the session pool size for text recognition.
    ///
    /// # Arguments
    ///
    /// * `size` - The session pool size (minimum 1)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_recognition_session_pool_size(mut self, size: usize) -> Self {
        let s = if size >= 1 {
            size
        } else {
            warn!("text_recognition_session_pool_size must be >= 1, got {size}; using 1");
            1
        };
        self.config.recognition.common.session_pool_size = Some(s);
        self
    }

    /// Sets the session pool size for text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `size` - The session pool size (minimum 1)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn textline_orientation_session_pool_size(mut self, size: usize) -> Self {
        let s = if size >= 1 {
            size
        } else {
            warn!("textline_orientation_session_pool_size must be >= 1, got {size}; using 1");
            1
        };
        with_nested!(self.config.text_line_orientation, crate::predictor::TextLineClasPredictorConfig, tlo_config => {
            tlo_config.common.session_pool_size = Some(s);
        });
        self
    }

    /// Sets the session pool size for all components.
    ///
    /// This is a convenience method that applies the same session pool size
    /// to text detection, text recognition, and text line orientation classification.
    ///
    /// # Arguments
    ///
    /// * `size` - The session pool size (minimum 1)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn global_session_pool_size(mut self, size: usize) -> Self {
        let s = if size >= 1 {
            size
        } else {
            warn!("global_session_pool_size must be >= 1, got {size}; using 1");
            1
        };
        // Apply to text detection
        self.config.detection.common.session_pool_size = Some(s);

        // Apply to text recognition
        self.config.recognition.common.session_pool_size = Some(s);

        // Apply to text line orientation (if configured)
        with_nested!(self.config.text_line_orientation, crate::predictor::TextLineClasPredictorConfig, tlo_config => {
            tlo_config.common.session_pool_size = Some(s);
        });

        self
    }

    /// Sets the text detection limit side length.
    ///
    /// # Arguments
    ///
    /// * `limit` - The maximum side length for resizing
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_limit_side_len(mut self, limit: u32) -> Self {
        let l = if limit >= 1 {
            limit
        } else {
            warn!("text_det_limit_side_len must be >= 1, got {limit}; using 1");
            1
        };
        self.config.detection.limit_side_len = Some(l);
        self
    }

    /// Sets the text detection limit type.
    ///
    /// # Arguments
    ///
    /// * `limit_type` - The type of limit for resizing
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_limit_type(mut self, limit_type: LimitType) -> Self {
        self.config.detection.limit_type = Some(limit_type);
        self
    }

    /// Sets the text detection input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The input shape as (channels, height, width)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_input_shape(mut self, shape: (u32, u32, u32)) -> Self {
        let c = if shape.0 > 0 {
            shape.0
        } else {
            warn!("text_det_input_shape channels {} <= 0; using 1", shape.0);
            1u32
        };
        let h = if shape.1 > 0 {
            shape.1
        } else {
            warn!("text_det_input_shape height {} <= 0; using 1", shape.1);
            1u32
        };
        let w = if shape.2 > 0 {
            shape.2
        } else {
            warn!("text_det_input_shape width {} <= 0; using 1", shape.2);
            1u32
        };
        self.config.detection.input_shape = Some((c, h, w));
        self
    }

    /// Sets the text detection maximum side limit.
    ///
    /// This controls the maximum allowed size for any side of an image during
    /// text detection preprocessing. Images larger than this limit will be resized
    /// to fit within the constraint while maintaining aspect ratio.
    ///
    /// # Arguments
    ///
    /// * `max_side_limit` - The maximum side limit for image processing (default: 4000)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection_model.onnx".to_string(),
    ///     "recognition_model.onnx".to_string(),
    ///     "char_dict.txt".to_string()
    /// )
    /// .text_det_max_side_limit(5000); // Allow larger images
    /// ```
    pub fn text_det_max_side_limit(mut self, max_side_limit: u32) -> Self {
        let m = if max_side_limit >= 1 {
            max_side_limit
        } else {
            warn!("text_det_max_side_limit must be >= 1, got {max_side_limit}; using 1");
            1
        };
        self.config.detection.max_side_limit = Some(m);
        self
    }

    /// Sets the text detection binarization threshold.
    ///
    /// This controls the threshold used for binarizing the detection output.
    /// Lower values may detect more text but with more false positives.
    ///
    /// # Arguments
    ///
    /// * `thresh` - The binarization threshold (default: 0.3)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection_model.onnx".to_string(),
    ///     "recognition_model.onnx".to_string(),
    ///     "char_dict.txt".to_string()
    /// )
    /// .text_det_threshold(0.4); // Higher threshold for more precise detection
    /// ```
    ///
    /// Sets the text detection binarization threshold.
    ///
    /// This controls the threshold used for binarizing the detection output.
    /// Lower values may detect more text but with more false positives.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The binarization threshold (default: 0.3)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_threshold(mut self, threshold: f32) -> Self {
        let t = if (0.0..=1.0).contains(&threshold) {
            threshold
        } else {
            warn!("text_det_threshold out of range [{threshold}], clamping to [0.0, 1.0]");
            threshold.clamp(0.0, 1.0)
        };
        self.config.detection.thresh = Some(t);
        self
    }

    /// Sets the text detection box score threshold.
    ///
    /// This controls the threshold for filtering text boxes based on their confidence scores.
    /// Higher values will filter out more uncertain detections.
    ///
    /// # Arguments
    ///
    /// * `box_thresh` - The box score threshold (default: 0.6)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection_model.onnx".to_string(),
    ///     "recognition_model.onnx".to_string(),
    ///     "char_dict.txt".to_string()
    /// )
    /// .text_det_box_threshold(0.7); // Higher threshold for more confident boxes
    /// ```
    ///
    /// Sets the text detection box score threshold.
    ///
    /// This controls the threshold for filtering text boxes based on their confidence scores.
    /// Higher values will filter out more uncertain detections.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The box score threshold (default: 0.6)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_det_box_threshold(mut self, threshold: f32) -> Self {
        let t = if (0.0..=1.0).contains(&threshold) {
            threshold
        } else {
            warn!("text_det_box_threshold out of range [{threshold}], clamping to [0.0, 1.0]");
            threshold.clamp(0.0, 1.0)
        };
        self.config.detection.box_thresh = Some(t);
        self
    }

    /// Sets the text detection unclip ratio.
    ///
    /// This controls how much to expand detected text boxes. Higher values
    /// will expand boxes more, potentially capturing more complete text.
    ///
    /// # Arguments
    ///
    /// * `unclip_ratio` - The unclip ratio for expanding text boxes (default: 1.5)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use oar_ocr::pipeline::OAROCRBuilder;
    ///
    /// let builder = OAROCRBuilder::new(
    ///     "detection_model.onnx".to_string(),
    ///     "recognition_model.onnx".to_string(),
    ///     "char_dict.txt".to_string()
    /// )
    /// .text_det_unclip_ratio(2.0); // More expansion for better text capture
    /// ```
    pub fn text_det_unclip_ratio(mut self, unclip_ratio: f32) -> Self {
        let r = if unclip_ratio > 0.0 {
            unclip_ratio
        } else {
            warn!("text_det_unclip_ratio must be > 0.0, got {unclip_ratio}; using 1.0");
            1.0
        };
        self.config.detection.unclip_ratio = Some(r);
        self
    }

    /// Sets the text recognition score threshold.
    ///
    /// # Arguments
    ///
    /// * `thresh` - The minimum score threshold for recognition results
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// Sets the text recognition score threshold.
    ///
    /// Results with confidence scores below this threshold will be filtered out.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The minimum score threshold for recognition results
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_score_threshold(mut self, threshold: f32) -> Self {
        let t = if (0.0..=1.0).contains(&threshold) {
            threshold
        } else {
            warn!("text_rec_score_threshold out of range [{threshold}], clamping to [0.0, 1.0]");
            threshold.clamp(0.0, 1.0)
        };
        self.config.recognition.score_thresh = Some(t);
        self
    }

    /// Sets the text recognition model input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The model input shape as (channels, height, width)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// Sets the text recognition model input shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The model input shape as (channels, height, width)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_input_shape(mut self, shape: (u32, u32, u32)) -> Self {
        let c = if shape.0 > 0 {
            shape.0
        } else {
            warn!("text_rec_input_shape channels {} <= 0; using 1", shape.0);
            1u32
        };
        let h = if shape.1 > 0 {
            shape.1
        } else {
            warn!("text_rec_input_shape height {} <= 0; using 1", shape.1);
            1u32
        };
        let w = if shape.2 > 0 {
            shape.2
        } else {
            warn!("text_rec_input_shape width {} <= 0; using 1", shape.2);
            1u32
        };
        self.config.recognition.model_input_shape = Some([c as usize, h as usize, w as usize]);
        self
    }

    /// Sets the text recognition character dictionary path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the character dictionary file
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn text_rec_character_dict_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.character_dict_path = path.into();
        self
    }

    /// Enables aspect ratio bucketing for text recognition with default configuration.
    ///
    /// This enables aspect ratio bucketing which groups images by aspect ratio ranges
    /// instead of exact dimensions, improving batch efficiency.
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// Sets a custom aspect ratio bucketing configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The aspect ratio bucketing configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn aspect_ratio_bucketing_config(mut self, config: AspectRatioBucketingConfig) -> Self {
        self.config.aspect_ratio_bucketing = Some(config);
        self
    }

    /// Disables aspect ratio bucketing (uses exact dimension grouping).
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn disable_aspect_ratio_bucketing(mut self) -> Self {
        self.config.aspect_ratio_bucketing = None;
        self
    }

    /// Enables dynamic batching with default configuration.
    ///
    /// This enables cross-image batching for both detection and recognition,
    /// which can improve performance when processing multiple images with
    /// compatible shapes.
    ///
    /// # Returns
    ///
    /// The updated builder instance
    ///
    /// Sets a custom dynamic batching configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The dynamic batching configuration
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn dynamic_batching_config(mut self, config: DynamicBatchConfig) -> Self {
        self.config.dynamic_batching = Some(config);
        self
    }

    /// Disables dynamic batching (processes images individually).
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn disable_dynamic_batching(mut self) -> Self {
        self.config.dynamic_batching = None;
        self
    }

    /// Sets the maximum batch size for detection.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Maximum number of images to batch for detection
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn max_detection_batch_size(mut self, batch_size: usize) -> Self {
        let bs = if batch_size >= 1 {
            batch_size
        } else {
            warn!("max_detection_batch_size must be >= 1, got {batch_size}; using 1");
            1
        };
        with_nested!(self.config.dynamic_batching, DynamicBatchConfig, config => {
            config.max_detection_batch_size = bs;
        });
        self
    }

    /// Sets the maximum batch size for recognition.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Maximum number of text regions to batch for recognition
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn max_recognition_batch_size(mut self, batch_size: usize) -> Self {
        let bs = if batch_size >= 1 {
            batch_size
        } else {
            warn!("max_recognition_batch_size must be >= 1, got {batch_size}; using 1");
            1
        };
        with_nested!(self.config.dynamic_batching, DynamicBatchConfig, config => {
            config.max_recognition_batch_size = bs;
        });
        self
    }

    /// Sets the minimum batch size to trigger dynamic batching.
    ///
    /// # Arguments
    ///
    /// * `min_size` - Minimum number of items needed to enable batching
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn min_batch_size(mut self, min_size: usize) -> Self {
        let ms = if min_size >= 1 {
            min_size
        } else {
            warn!("min_batch_size must be >= 1, got {min_size}; using 1");
            1
        };
        if self.config.dynamic_batching.is_none() {
            self.config.dynamic_batching = Some(DynamicBatchConfig::default());
        }
        if let Some(ref mut config) = self.config.dynamic_batching {
            config.min_batch_size = ms;
        }
        self
    }

    /// Sets the shape compatibility strategy for dynamic batching.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The shape compatibility strategy to use
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn shape_compatibility_strategy(mut self, strategy: ShapeCompatibilityStrategy) -> Self {
        if self.config.dynamic_batching.is_none() {
            self.config.dynamic_batching = Some(DynamicBatchConfig::default());
        }
        if let Some(ref mut config) = self.config.dynamic_batching {
            config.shape_compatibility = strategy;
        }
        self
    }

    /// Enables only detection batching (disables recognition batching).
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn enable_detection_batching_only(mut self) -> Self {
        if self.config.dynamic_batching.is_none() {
            self.config.dynamic_batching = Some(DynamicBatchConfig::default());
        }
        if let Some(ref mut config) = self.config.dynamic_batching {
            config.enable_detection_batching = true;
            config.enable_recognition_batching = false;
        }
        self
    }

    /// Enables only recognition batching (disables detection batching).
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn enable_recognition_batching_only(mut self) -> Self {
        if self.config.dynamic_batching.is_none() {
            self.config.dynamic_batching = Some(DynamicBatchConfig::default());
        }
        if let Some(ref mut config) = self.config.dynamic_batching {
            config.enable_detection_batching = false;
            config.enable_recognition_batching = true;
        }
        self
    }

    /// Sets aspect ratio tolerance for shape compatibility.
    ///
    /// This is a convenience method that sets the shape compatibility strategy
    /// to AspectRatio with the specified tolerance.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Tolerance for aspect ratio matching (e.g., 0.1 means ±10%)
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn aspect_ratio_tolerance(mut self, tolerance: f32) -> Self {
        let tol = if tolerance >= 0.0 {
            tolerance
        } else {
            warn!("aspect_ratio_tolerance must be >= 0.0, got {tolerance}; using 0.0");
            0.0
        };
        if self.config.dynamic_batching.is_none() {
            self.config.dynamic_batching = Some(DynamicBatchConfig::default());
        }
        if let Some(ref mut config) = self.config.dynamic_batching {
            config.shape_compatibility = ShapeCompatibilityStrategy::AspectRatio { tolerance: tol };
        }
        self
    }

    /// Sets exact shape matching for dynamic batching.
    ///
    /// This requires images to have identical dimensions to be batched together.
    ///
    /// # Returns
    ///
    /// The updated builder instance
    pub fn exact_shape_matching(mut self) -> Self {
        if self.config.dynamic_batching.is_none() {
            self.config.dynamic_batching = Some(DynamicBatchConfig::default());
        }
        if let Some(ref mut config) = self.config.dynamic_batching {
            config.shape_compatibility = ShapeCompatibilityStrategy::Exact;
        }
        self
    }

    /// Builds the OAROCR instance with the configured parameters.
    ///
    /// # Returns
    ///
    /// A Result containing the OAROCR instance or an OCRError
    pub fn build(self) -> crate::core::OcrResult<super::OAROCR> {
        super::OAROCR::new(self.config)
    }

    /// Gets a reference to the configuration for testing purposes.
    ///
    /// # Returns
    ///
    /// A reference to the OAROCRConfig
    #[cfg(test)]
    pub fn get_config(&self) -> &OAROCRConfig {
        &self.config
    }
}

// Demonstration of how the impl_complete_builder! macro could be used
// to generate many of the builder methods automatically.
// This is commented out to avoid conflicts with the existing implementation,
// but shows how the macro could replace much of the manual code.

impl_complete_builder! {
    builder: OAROCRBuilder,
    config_field: config,
    enable_methods: {
        enable_dynamic_batching => dynamic_batching: DynamicBatchConfig => "Enables dynamic batching with default configuration",
        enable_aspect_ratio_bucketing => aspect_ratio_bucketing: AspectRatioBucketingConfig => "Enables aspect ratio bucketing with default configuration",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::onnx::{OrtExecutionProvider, OrtSessionConfig};

    #[test]
    fn test_ort_session_configuration_propagation() {
        // Create an ORT session configuration
        let ort_config =
            OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);

        // Create a builder and set the ORT session configuration
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_detection_ort_session(ort_config.clone())
        .text_recognition_ort_session(ort_config.clone());

        let config = builder.get_config();

        // Verify that the ORT session configuration was properly set
        assert!(config.detection.common.ort_session.is_some());
        assert!(config.recognition.common.ort_session.is_some());

        let detection_ort = config.detection.common.ort_session.as_ref().unwrap();
        let recognition_ort = config.recognition.common.ort_session.as_ref().unwrap();

        assert_eq!(
            detection_ort.execution_providers,
            Some(vec![OrtExecutionProvider::CPU])
        );
        assert_eq!(
            recognition_ort.execution_providers,
            Some(vec![OrtExecutionProvider::CPU])
        );
    }

    #[test]
    fn test_session_pool_size_configuration_propagation() {
        // Create a builder and set session pool sizes
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_detection_session_pool_size(4)
        .text_recognition_session_pool_size(8);

        let config = builder.get_config();

        // Verify that the session pool sizes were properly set
        assert_eq!(config.detection.common.session_pool_size, Some(4));
        assert_eq!(config.recognition.common.session_pool_size, Some(8));
    }

    #[test]
    fn test_global_ort_session_configuration() {
        // Create an ORT session configuration
        let ort_config =
            OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);

        // Create a builder and set the global ORT session configuration
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .global_ort_session(ort_config.clone());

        let config = builder.get_config();

        // Verify that the ORT session configuration was applied to all components
        assert!(config.detection.common.ort_session.is_some());
        assert!(config.recognition.common.ort_session.is_some());

        let detection_ort = config.detection.common.ort_session.as_ref().unwrap();
        let recognition_ort = config.recognition.common.ort_session.as_ref().unwrap();

        assert_eq!(
            detection_ort.execution_providers,
            Some(vec![OrtExecutionProvider::CPU])
        );
        assert_eq!(
            recognition_ort.execution_providers,
            Some(vec![OrtExecutionProvider::CPU])
        );
    }

    #[test]
    fn test_global_session_pool_size_configuration() {
        // Create a builder and set global session pool size
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .global_session_pool_size(6);

        let config = builder.get_config();

        // Verify that the session pool size was applied to all components
        assert_eq!(config.detection.common.session_pool_size, Some(6));
        assert_eq!(config.recognition.common.session_pool_size, Some(6));
    }

    #[test]
    fn test_text_det_max_side_limit_configuration() {
        // Create a builder and set the text detection max side limit
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_det_max_side_limit(5000);

        let config = builder.get_config();

        // Verify that the max side limit was properly set
        assert_eq!(config.detection.max_side_limit, Some(5000));
    }

    #[test]
    fn test_text_det_thresh_configuration() {
        // Create a builder and set the text detection threshold
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_det_threshold(0.4);

        let config = builder.get_config();

        // Verify that the threshold was properly set
        assert_eq!(config.detection.thresh, Some(0.4));
    }

    #[test]
    fn test_text_det_box_thresh_configuration() {
        // Create a builder and set the text detection box threshold
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_det_box_threshold(0.7);

        let config = builder.get_config();

        // Verify that the box threshold was properly set
        assert_eq!(config.detection.box_thresh, Some(0.7));
    }

    #[test]
    fn test_text_det_unclip_ratio_configuration() {
        // Create a builder and set the text detection unclip ratio
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_det_unclip_ratio(2.0);

        let config = builder.get_config();

        // Verify that the unclip ratio was properly set
        assert_eq!(config.detection.unclip_ratio, Some(2.0));
    }

    #[test]
    fn test_text_det_all_thresholds_configuration() {
        // Create a builder and set all text detection thresholds
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .text_det_threshold(0.35)
        .text_det_box_threshold(0.65)
        .text_det_unclip_ratio(1.8);

        let config = builder.get_config();

        // Verify that all thresholds were properly set
        assert_eq!(config.detection.thresh, Some(0.35));
        assert_eq!(config.detection.box_thresh, Some(0.65));
        assert_eq!(config.detection.unclip_ratio, Some(1.8));
    }

    #[test]
    fn test_textline_orientation_configuration_propagation() {
        // Create an ORT session configuration
        let ort_config =
            OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);

        // Create a builder with text line orientation enabled and configure it
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .use_textline_orientation(true)
        .textline_orientation_ort_session(ort_config.clone())
        .textline_orientation_session_pool_size(3);

        let config = builder.get_config();

        // Verify that text line orientation is enabled and configured
        assert!(config.text_line_orientation.is_some());

        let tlo_config = config.text_line_orientation.as_ref().unwrap();
        assert!(tlo_config.common.ort_session.is_some());
        assert_eq!(tlo_config.common.session_pool_size, Some(3));

        let tlo_ort = tlo_config.common.ort_session.as_ref().unwrap();
        assert_eq!(
            tlo_ort.execution_providers,
            Some(vec![OrtExecutionProvider::CPU])
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_with_cuda_convenience_method() {
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .with_cuda();

        let config = builder.get_config();

        // Verify that CUDA configuration was applied to all components
        assert!(config.detection.common.ort_session.is_some());
        assert!(config.recognition.common.ort_session.is_some());

        let det_ort = config.detection.common.ort_session.as_ref().unwrap();
        if let Some(providers) = &det_ort.execution_providers {
            assert!(providers.len() >= 2); // Should have CUDA + CPU fallback
            // First should be CUDA
            if let OrtExecutionProvider::CUDA { device_id, .. } = &providers[0] {
                assert_eq!(*device_id, Some(0)); // Default device 0
            } else {
                panic!("Expected CUDA provider as first execution provider");
            }
            // Second should be CPU fallback
            assert!(matches!(providers[1], OrtExecutionProvider::CPU));
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_with_cuda_device_convenience_method() {
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .with_cuda_device(2);

        let config = builder.get_config();

        let det_ort = config.detection.common.ort_session.as_ref().unwrap();
        if let Some(providers) = &det_ort.execution_providers {
            if let OrtExecutionProvider::CUDA { device_id, .. } = &providers[0] {
                assert_eq!(*device_id, Some(2)); // Should use device 2
            } else {
                panic!("Expected CUDA provider with device 2");
            }
        }
    }

    #[test]
    fn test_with_high_performance_convenience_method() {
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .with_high_performance();

        let config = builder.get_config();

        // Verify that high performance parallel policy was set
        let policy = &config.parallel_policy;

        // Should have reasonable thread count
        assert!(policy.max_threads.is_some());
        let max_threads = policy.max_threads.unwrap();
        assert!((1..=128).contains(&max_threads));

        // Should have low thresholds for more parallel processing
        assert_eq!(policy.image_threshold, 2);
        assert_eq!(policy.text_box_threshold, 5);
        assert_eq!(policy.batch_threshold, 3);
    }

    #[test]
    fn test_with_low_resource_convenience_method() {
        let builder = OAROCRBuilder::new(
            "test_detection_model.onnx".to_string(),
            "test_recognition_model.onnx".to_string(),
            "test_char_dict.txt".to_string(),
        )
        .with_low_resource();

        let config = builder.get_config();

        // Verify that low resource parallel policy was set
        let policy = &config.parallel_policy;

        // Should limit threads for resource-constrained environments
        assert_eq!(policy.max_threads, Some(2));

        // Should have higher thresholds to avoid parallel processing on small workloads
        assert_eq!(policy.image_threshold, 10);
        assert_eq!(policy.text_box_threshold, 20);
        assert_eq!(policy.batch_threshold, 10);
    }
}
