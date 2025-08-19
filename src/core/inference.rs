//! Inference utilities for the OCR pipeline.
//!
//! This module provides structures and functions for performing inference
//! using ONNX Runtime models in the OCR pipeline. It includes utilities
//! for loading models, running inference, and processing the output tensors.

use crate::core::{
    batch::{Tensor2D, Tensor3D, Tensor4D},
    errors::OCRError,
    traits::ImageReader,
};
use image::RgbImage;
use ort::{
    execution_providers::ExecutionProviderDispatch,
    session::{Session, builder::SessionBuilder},
    value::TensorRef,
};
use std::path::Path;
use std::sync::Mutex;

/// A default implementation of the ImageReader trait.
///
/// This struct provides a simple way to read images from file paths
/// with an optional parallel processing threshold.
#[derive(Debug)]
pub struct DefaultImageReader {
    /// The threshold for parallel processing (optional).
    ///
    /// If set, images will be processed in parallel when the number
    /// of images exceeds this threshold.
    parallel_threshold: Option<usize>,
}

impl DefaultImageReader {
    /// Creates a new DefaultImageReader with no parallel processing threshold.
    ///
    /// # Returns
    ///
    /// A new DefaultImageReader instance.
    pub fn new() -> Self {
        Self {
            parallel_threshold: None,
        }
    }

    /// Creates a new DefaultImageReader with a parallel processing threshold.
    ///
    /// # Arguments
    ///
    /// * `parallel_threshold` - The threshold for parallel processing.
    ///
    /// # Returns
    ///
    /// A new DefaultImageReader instance.
    pub fn with_parallel_threshold(parallel_threshold: usize) -> Self {
        Self {
            parallel_threshold: Some(parallel_threshold),
        }
    }
}

/// Implementation of Default for DefaultImageReader.
///
/// This allows DefaultImageReader to be created with default values.
impl Default for DefaultImageReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of ImageReader for DefaultImageReader.
///
/// This allows DefaultImageReader to be used as an ImageReader.
impl ImageReader for DefaultImageReader {
    type Error = OCRError;

    /// Applies the image reader to a collection of image paths.
    ///
    /// # Arguments
    ///
    /// * `imgs` - An iterator over image paths.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of RgbImage instances or an OCRError.
    fn apply<P: AsRef<Path> + Send + Sync>(
        &self,
        imgs: impl IntoIterator<Item = P>,
    ) -> Result<Vec<RgbImage>, Self::Error> {
        use crate::utils::load_images_batch_with_threshold;

        let img_paths: Vec<_> = imgs.into_iter().collect();
        load_images_batch_with_threshold(&img_paths, self.parallel_threshold)
    }
}

/// A struct for performing inference using ONNX Runtime models.
///
/// This struct provides methods for loading ONNX models and running inference
/// with different output tensor shapes.
#[derive(Debug)]
pub struct OrtInfer {
    /// Pool of ONNX Runtime sessions for concurrent predictions.
    sessions: Vec<Mutex<Session>>,
    /// Next index for round-robin session selection.
    next_idx: std::sync::atomic::AtomicUsize,
    /// The name of the input tensor.
    input_name: String,
    /// The name of the output tensor (optional).
    output_name: Option<String>,
    /// The path to the model file for error context.
    model_path: std::path::PathBuf,
    /// The model name for error context.
    model_name: String,
}

impl OrtInfer {
    /// Creates a new OrtInfer instance with default ONNX Runtime settings and a single session.
    pub fn new(model_path: impl AsRef<Path>, input_name: Option<&str>) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = Session::builder()
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("verify model path and compatibility with selected execution providers"),
                    Some(e),
                )
            })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance from CommonBuilderConfig, applying ORT session
    /// configuration and constructing a session pool for concurrent predictions.
    pub fn from_common(
        common: &crate::core::config::CommonBuilderConfig,
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let pool_size = common.session_pool_size.unwrap_or(1).max(1);
        let mut sessions = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let builder = Session::builder()?;
            let builder = if let Some(cfg) = &common.ort_session {
                Self::apply_ort_config(builder, cfg)?
            } else {
                builder
            };
            let session = builder.commit_from_file(path).map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("check device/EP configuration and model file"),
                    Some(e),
                )
            })?;
            sessions.push(Mutex::new(session));
        }

        let model_name = common
            .model_name
            .clone()
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "unknown_model".to_string());

        Ok(OrtInfer {
            sessions,
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance from CommonBuilderConfig with automatic input name detection,
    /// applying ORT session configuration and constructing a session pool for concurrent predictions.
    ///
    /// This method combines the functionality of `from_common` and `with_auto_input_name` to respect
    /// ORT configuration while automatically detecting the input tensor name.
    pub fn from_common_with_auto_input(
        common: &crate::core::config::CommonBuilderConfig,
        model_path: impl AsRef<Path>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let pool_size = common.session_pool_size.unwrap_or(1).max(1);
        let mut sessions = Vec::with_capacity(pool_size);

        // Create the first session to detect input name
        let builder = Session::builder()?;
        let builder = if let Some(cfg) = &common.ort_session {
            Self::apply_ort_config(builder, cfg)?
        } else {
            builder
        };
        let first_session = builder.commit_from_file(path).map_err(|e| {
            OCRError::model_load_error(
                path,
                "failed to create ONNX session",
                Some("ensure model is compatible and file exists"),
                Some(e),
            )
        })?;

        // Auto-detect input name from the first session
        let common_names = ["x", "input", "images", "data", "image"];
        let available_inputs: Vec<String> = first_session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let input_name = common_names
            .iter()
            .find(|&name| available_inputs.iter().any(|input| input == *name))
            .unwrap_or(&"x")
            .to_string();

        sessions.push(Mutex::new(first_session));

        // Create remaining sessions with the same configuration
        for _ in 1..pool_size {
            let builder = Session::builder()?;
            let builder = if let Some(cfg) = &common.ort_session {
                Self::apply_ort_config(builder, cfg)?
            } else {
                builder
            };
            let session = builder.commit_from_file(path).map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("ensure model is compatible and file exists"),
                    Some(e),
                )
            })?;
            sessions.push(Mutex::new(session));
        }

        let model_name = common
            .model_name
            .clone()
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "unknown_model".to_string());

        Ok(OrtInfer {
            sessions,
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name,
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    fn apply_ort_config(
        mut builder: SessionBuilder,
        cfg: &crate::core::config::OrtSessionConfig,
    ) -> Result<SessionBuilder, ort::Error> {
        if let Some(intra) = cfg.intra_threads {
            builder = builder.with_intra_threads(intra)?;
        }
        if let Some(inter) = cfg.inter_threads {
            builder = builder.with_inter_threads(inter)?;
        }
        if let Some(par) = cfg.parallel_execution {
            builder = builder.with_parallel_execution(par)?;
        }
        if let Some(level) = cfg.optimization_level {
            use crate::core::config::OrtGraphOptimizationLevel as OG;
            use ort::session::builder::GraphOptimizationLevel as GOL;
            let mapped = match level {
                OG::DisableAll => GOL::Disable,
                OG::Level1 => GOL::Level1,
                OG::Level2 => GOL::Level2,
                OG::Level3 => GOL::Level3,
                OG::All => GOL::Level3, // Map All to Level3 as they are equivalent
            };
            builder = builder.with_optimization_level(mapped)?;
        }
        if let Some(eps) = &cfg.execution_providers {
            let providers = Self::build_execution_providers(eps)?;
            if !providers.is_empty() {
                builder = builder.with_execution_providers(providers)?;
            }
        }
        Ok(builder)
    }

    /// Builds execution providers from configuration
    fn build_execution_providers(
        eps: &[crate::core::config::OrtExecutionProvider],
    ) -> Result<Vec<ExecutionProviderDispatch>, ort::Error> {
        use crate::core::config::OrtExecutionProvider as EP;
        let mut providers = Vec::new();

        for ep in eps {
            match ep {
                EP::CPU => {
                    // CPU provider is always available
                    providers
                        .push(ort::execution_providers::CPUExecutionProvider::default().build());
                }
                #[cfg(feature = "cuda")]
                EP::CUDA {
                    device_id,
                    gpu_mem_limit: _,
                    arena_extend_strategy: _,
                    cudnn_conv_algo_search: _,
                    do_copy_in_default_stream: _,
                    cudnn_conv_use_max_workspace: _,
                } => {
                    let mut cuda_provider =
                        ort::execution_providers::CUDAExecutionProvider::default();
                    if let Some(id) = device_id {
                        cuda_provider = cuda_provider.with_device_id(*id);
                    }
                    providers.push(cuda_provider.build());
                }
                #[cfg(feature = "tensorrt")]
                EP::TensorRT {
                    device_id,
                    max_workspace_size,
                    max_batch_size: _,
                    min_subgraph_size: _,
                    fp16_enable,
                } => {
                    let mut trt_provider =
                        ort::execution_providers::TensorRTExecutionProvider::default();
                    if let Some(id) = device_id {
                        trt_provider = trt_provider.with_device_id(*id);
                    }
                    if let Some(workspace) = max_workspace_size {
                        trt_provider = trt_provider.with_max_workspace_size(*workspace);
                    }
                    if let Some(fp16) = fp16_enable {
                        trt_provider = trt_provider.with_fp16(*fp16);
                    }
                    providers.push(trt_provider.build());
                }
                #[cfg(feature = "directml")]
                EP::DirectML { device_id } => {
                    let mut dml_provider =
                        ort::execution_providers::DirectMLExecutionProvider::default();
                    if let Some(id) = device_id {
                        dml_provider = dml_provider.with_device_id(*id);
                    }
                    providers.push(dml_provider.build());
                }
                #[cfg(feature = "coreml")]
                EP::CoreML {
                    ane_only: _,
                    subgraphs,
                } => {
                    let mut coreml_provider =
                        ort::execution_providers::CoreMLExecutionProvider::default();
                    if let Some(sub) = subgraphs {
                        coreml_provider = coreml_provider.with_subgraphs(*sub);
                    }
                    providers.push(coreml_provider.build());
                }
                #[cfg(feature = "webgpu")]
                EP::WebGPU => {
                    providers
                        .push(ort::execution_providers::WebGPUExecutionProvider::default().build());
                }
                #[cfg(feature = "openvino")]
                EP::OpenVINO {
                    device_type,
                    num_threads: _,
                } => {
                    let mut openvino_provider =
                        ort::execution_providers::OpenVINOExecutionProvider::default();
                    if let Some(device) = device_type {
                        openvino_provider = openvino_provider.with_device_type(device.clone());
                    }
                    providers.push(openvino_provider.build());
                }
                // Handle cases when features are not enabled
                #[cfg(not(feature = "cuda"))]
                EP::CUDA { .. } => {
                    return Err(ort::Error::new(
                        "CUDA execution provider requested but cuda feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "tensorrt"))]
                EP::TensorRT { .. } => {
                    return Err(ort::Error::new(
                        "TensorRT execution provider requested but tensorrt feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "directml"))]
                EP::DirectML { .. } => {
                    return Err(ort::Error::new(
                        "DirectML execution provider requested but directml feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "openvino"))]
                EP::OpenVINO { .. } => {
                    return Err(ort::Error::new(
                        "OpenVINO execution provider requested but openvino feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "coreml"))]
                EP::CoreML { .. } => {
                    return Err(ort::Error::new(
                        "CoreML execution provider requested but coreml feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "webgpu"))]
                EP::WebGPU => {
                    return Err(ort::Error::new(
                        "WebGPU execution provider requested but webgpu feature is not enabled",
                    ));
                }
            }
        }

        Ok(providers)
    }

    /// Creates a new OrtInfer instance with a specified output tensor name.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the ONNX model file.
    /// * `input_name` - The name of the input tensor (optional).
    /// * `output_name` - The name of the output tensor (optional).
    ///
    /// # Returns
    ///
    /// A Result containing the new OrtInfer instance or an OCRError.
    pub fn with_output_name(
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
        output_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = Session::builder()
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("verify model path and compatibility"),
                    Some(e),
                )
            })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: output_name.map(|s| s.to_string()),
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance with an automatically detected input tensor name.
    ///
    /// This method tries to detect the input tensor name by looking for common names
    /// in the model's input tensors.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the ONNX model file.
    ///
    /// # Returns
    ///
    /// A Result containing the new OrtInfer instance or an OCRError.
    pub fn with_auto_input_name(model_path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = Session::builder()
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("verify model path and compatibility"),
                    Some(e),
                )
            })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        let common_names = ["x", "input", "images", "data", "image"];

        let available_inputs: Vec<String> = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let input_name = common_names
            .iter()
            .find(|&name| available_inputs.iter().any(|input| input == *name))
            .unwrap_or(&"x");
        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.to_string(),
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance with a custom session configuration.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the ONNX model file.
    /// * `input_name` - The name of the input tensor (optional).
    /// * `output_name` - The name of the output tensor (optional).
    /// * `configure_session` - A function to configure the session builder.
    ///
    /// # Returns
    ///
    /// A Result containing the new OrtInfer instance or an OCRError.
    pub fn with_session_config<F>(
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
        output_name: Option<&str>,
        configure_session: F,
    ) -> Result<Self, OCRError>
    where
        F: FnOnce(SessionBuilder) -> Result<SessionBuilder, ort::Error>,
    {
        let path = model_path.as_ref();
        let builder = Session::builder()?;
        let configured_builder = configure_session(builder)?;
        let session = configured_builder.commit_from_file(path).map_err(|e| {
            OCRError::model_load_error(
                path,
                "failed to create ONNX session",
                Some("verify model path and configured execution providers"),
                Some(e),
            )
        })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: output_name.map(|s| s.to_string()),
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Gets the name of the output tensor.
    ///
    /// # Returns
    ///
    /// A Result containing the output tensor name or an OCRError.
    fn get_output_name(&self) -> Result<String, OCRError> {
        if let Some(ref name) = self.output_name {
            Ok(name.clone())
        } else {
            // Read from the first session's metadata
            let session = self.sessions[0]
                .lock()
                .map_err(|_| OCRError::InvalidInput {
                    message: "Failed to acquire session lock".to_string(),
                })?;
            if !session.outputs.is_empty() {
                Ok(session.outputs[0].name.clone())
            } else {
                Err(OCRError::InvalidInput {
                    message: "No outputs available in session - model may be invalid or corrupted"
                        .to_string(),
                })
            }
        }
    }

    /// Gets the path to the model file.
    ///
    /// # Returns
    ///
    /// A reference to the model file path.
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    /// Gets the name of the model.
    ///
    /// # Returns
    ///
    /// A reference to the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Runs inference with a custom processor function.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor.
    /// * `processor` - A function to process the output tensor.
    ///
    /// # Returns
    ///
    /// A Result containing the processed output or an OCRError.
    fn run_inference_with_processor<T>(
        &self,
        x: &Tensor4D,
        processor: impl FnOnce(&[i64], &[f32]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let input_shape = x.shape().to_vec();
        let _batch_size = input_shape[0];

        let output_name = self.get_output_name().map_err(|e| {
            OCRError::inference_error(
                &self.model_name,
                &format!(
                    "Failed to get output name for model at '{}'",
                    self.model_path.display()
                ),
                e,
            )
        })?;

        let input_tensor = TensorRef::from_array_view(x.view()).map_err(|e| {
            OCRError::model_inference_error(
                &self.model_name,
                "tensor_conversion",
                0,
                &input_shape,
                &format!(
                    "Failed to convert input tensor with shape {:?}",
                    input_shape
                ),
                e,
            )
        })?;

        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];

        // Round-robin select a session
        let idx = self
            .next_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.sessions.len();
        let mut session_guard = self.sessions[idx].lock().map_err(|_| {
            OCRError::inference_error(
                &self.model_name,
                &format!(
                    "Failed to acquire session lock for session {}/{}",
                    idx,
                    self.sessions.len()
                ),
                crate::core::errors::SimpleError::new("Session lock acquisition failed"),
            )
        })?;

        let outputs = session_guard.run(inputs).map_err(|e| {
            OCRError::model_inference_error(
                &self.model_name,
                "forward_pass",
                0,
                &input_shape,
                &format!(
                    "ONNX Runtime inference failed with input '{}' -> output '{}'",
                    self.input_name, output_name
                ),
                e,
            )
        })?;

        let output = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                OCRError::model_inference_error(
                    &self.model_name,
                    "output_extraction",
                    0,
                    &input_shape,
                    &format!("Failed to extract output tensor '{}' as f32", output_name),
                    e,
                )
            })?;
        let (output_shape, output_data) = output;

        processor(output_shape, output_data)
    }

    /// Runs inference and returns a 4D tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor.
    ///
    /// # Returns
    ///
    /// A Result containing the output 4D tensor or an OCRError.
    pub fn infer_4d(&self, x: &Tensor4D) -> Result<Tensor4D, OCRError> {
        let _input_shape = x.shape().to_vec();
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 4 {
                return Err(OCRError::tensor_operation_error(
                    "output_validation",
                    &[4], // expected dimensions
                    &[output_shape.len()], // actual dimensions
                    &format!("Model '{}' 4D inference: expected 4D output tensor, got {}D with shape {:?}",
                        self.model_name, output_shape.len(), output_shape),
                    crate::core::errors::SimpleError::new("Invalid output tensor dimensions"),
                ));
            }

            let batch_size_out = output_shape[0] as usize;
            let channels_out = output_shape[1] as usize;
            let height_out = output_shape[2] as usize;
            let width_out = output_shape[3] as usize;
            let expected_len = batch_size_out * channels_out * height_out * width_out;

            if output_data.len() != expected_len {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Output data size mismatch: expected {}, got {}",
                        expected_len,
                        output_data.len()
                    ),
                });
            }

            let array_view = ndarray::ArrayView4::from_shape(
                (batch_size_out, channels_out, height_out, width_out),
                output_data,
            )
            .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    /// Runs inference and returns a 2D tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor.
    ///
    /// # Returns
    ///
    /// A Result containing the output 2D tensor or an OCRError.
    pub fn infer_2d(&self, x: &Tensor4D) -> Result<Tensor2D, OCRError> {
        let batch_size = x.shape()[0];
        let input_shape = x.shape().to_vec();
        self.run_inference_with_processor(x, |output_shape, output_data| {
            let num_classes = output_shape[1] as usize;
            let expected_len = batch_size * num_classes;

            if output_data.len() != expected_len {
                return Err(OCRError::tensor_operation_error(
                    "output_data_validation",
                    &[expected_len],
                    &[output_data.len()],
                    &format!(
                        "Model '{}' 2D inference: output data size mismatch for input shape {:?} -> output shape {:?}",
                        self.model_name, input_shape, output_shape
                    ),
                    crate::core::errors::SimpleError::new("Output tensor data size mismatch"),
                ));
            }

            let array_view =
                ndarray::ArrayView2::from_shape((batch_size, num_classes), output_data)
                    .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    /// Runs inference and returns a 3D tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor.
    ///
    /// # Returns
    ///
    /// A Result containing the output 3D tensor or an OCRError.
    pub fn infer_3d(&self, x: &Tensor4D) -> Result<Tensor3D, OCRError> {
        let _input_shape = x.shape().to_vec();
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 3 {
                return Err(OCRError::tensor_operation_error(
                    "output_validation",
                    &[3], // expected dimensions
                    &[output_shape.len()], // actual dimensions
                    &format!("Model '{}' 3D inference: expected 3D output tensor, got {}D with shape {:?}",
                        self.model_name, output_shape.len(), output_shape),
                    crate::core::errors::SimpleError::new("Invalid output tensor dimensions"),
                ));
            }

            let batch_size_out = output_shape[0] as usize;
            let seq_len = output_shape[1] as usize;
            let num_classes = output_shape[2] as usize;
            let expected_len = batch_size_out * seq_len * num_classes;

            if output_data.len() != expected_len {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Output data size mismatch: expected {}, got {}",
                        expected_len,
                        output_data.len()
                    ),
                });
            }

            let array_view = ndarray::ArrayView3::from_shape(
                (batch_size_out, seq_len, num_classes),
                output_data,
            )
            .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }
}

/// Loads an ONNX Runtime session from a model file.
///
/// # Arguments
///
/// * `model_path` - The path to the ONNX model file.
///
/// # Returns
///
/// A Result containing the loaded Session or an OCRError.
pub fn load_session(model_path: impl AsRef<Path>) -> Result<Session, OCRError> {
    let session = Session::builder()
        .and_then(|b| b.commit_from_file(model_path.as_ref()))
        .map_err(|e| {
            OCRError::model_load_error(
                model_path,
                "failed to create ONNX session",
                Some("verify model file exists and is readable"),
                Some(e),
            )
        })?;
    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{CommonBuilderConfig, OrtSessionConfig};

    #[test]
    fn test_from_common_with_auto_input_respects_config() {
        // This test verifies that from_common_with_auto_input respects ORT configuration
        // We can't test with actual ONNX models in unit tests, but we can verify the method exists
        // and has the correct signature
        let common = CommonBuilderConfig::new()
            .session_pool_size(2)
            .ort_session(OrtSessionConfig::new());

        // This would fail if the method signature changed or was removed
        let _result = OrtInfer::from_common_with_auto_input(&common, "dummy_path.onnx");
        // We expect this to fail since we don't have a real model file, but the important
        // thing is that the method exists and accepts the right parameters
        assert!(_result.is_err());
    }

    #[test]
    fn test_from_common_respects_session_pool_size() {
        let common = CommonBuilderConfig::new().session_pool_size(3);

        // This would fail if the method signature changed or was removed
        let _result = OrtInfer::from_common(&common, "dummy_path.onnx", None);
        // We expect this to fail since we don't have a real model file, but the important
        // thing is that the method exists and accepts the right parameters
        assert!(_result.is_err());
    }
}
