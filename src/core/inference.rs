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
    session::{Session, builder::SessionBuilder},
    value::Value,
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
    /// The ONNX Runtime session.
    session: Mutex<Session>,
    /// The name of the input tensor.
    input_name: String,
    /// The name of the output tensor (optional).
    output_name: Option<String>,
}

impl OrtInfer {
    /// Creates a new OrtInfer instance.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the ONNX model file.
    /// * `input_name` - The name of the input tensor (optional).
    ///
    /// # Returns
    ///
    /// A Result containing the new OrtInfer instance or an OCRError.
    pub fn new(model_path: impl AsRef<Path>, input_name: Option<&str>) -> Result<Self, OCRError> {
        let session = Session::builder()?.commit_from_file(model_path.as_ref())?;
        Ok(OrtInfer {
            session: Mutex::new(session),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: None,
        })
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
        let session = Session::builder()?.commit_from_file(model_path.as_ref())?;
        Ok(OrtInfer {
            session: Mutex::new(session),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: output_name.map(|s| s.to_string()),
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
        let session = Session::builder()?.commit_from_file(model_path.as_ref())?;

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
            session: Mutex::new(session),
            input_name: input_name.to_string(),
            output_name: None,
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
        let builder = Session::builder()?;
        let configured_builder = configure_session(builder)?;
        let session = configured_builder.commit_from_file(model_path.as_ref())?;

        Ok(OrtInfer {
            session: Mutex::new(session),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: output_name.map(|s| s.to_string()),
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
            let session = self.session.lock().map_err(|_| OCRError::InvalidInput {
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
        x: Tensor4D,
        processor: impl FnOnce(&[i64], &[f32]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let output_name = self.get_output_name()?;
        let input_tensor = Value::from_array(x).map_err(OCRError::Session)?;
        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];
        let mut session = self.session.lock().map_err(|_| OCRError::InvalidInput {
            message: "Failed to acquire session lock".to_string(),
        })?;
        let outputs = session.run(inputs).map_err(OCRError::Session)?;
        let output = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(OCRError::Session)?;
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
    pub fn infer_4d(&self, x: Tensor4D) -> Result<Tensor4D, OCRError> {
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 4 {
                return Err(OCRError::InvalidInput {
                    message: format!("Expected 4D output tensor, got {}D", output_shape.len()),
                });
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
    pub fn infer_2d(&self, x: Tensor4D) -> Result<Tensor2D, OCRError> {
        let batch_size = x.shape()[0];
        self.run_inference_with_processor(x, |output_shape, output_data| {
            let num_classes = output_shape[1] as usize;
            let expected_len = batch_size * num_classes;

            if output_data.len() != expected_len {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Output data size mismatch: expected {}, got {}",
                        expected_len,
                        output_data.len()
                    ),
                });
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
    pub fn infer_3d(&self, x: Tensor4D) -> Result<Tensor3D, OCRError> {
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 3 {
                return Err(OCRError::InvalidInput {
                    message: format!("Expected 3D output tensor, got {}D", output_shape.len()),
                });
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
    let session = Session::builder()?.commit_from_file(model_path.as_ref())?;
    Ok(session)
}
