use super::*;
use crate::core::inference::TensorOutput;
use ort::session::SessionInputs;
use ort::value::TensorRef;
use std::borrow::Cow;

/// Represents a tensor input of various dimensions for generic inference.
///
/// This enum allows passing tensors of different dimensions to the inference engine
/// without hardcoding specific tensor types or input semantics.
#[derive(Debug)]
pub enum TensorInput<'a> {
    /// A 2D tensor reference (e.g., scale_factor, im_shape)
    Array2(&'a ndarray::Array2<f32>),
    /// A 3D tensor reference
    Array3(&'a ndarray::Array3<f32>),
    /// A 4D tensor reference (e.g., image batch)
    Array4(&'a ndarray::Array4<f32>),
}

impl<'a> TensorInput<'a> {
    /// Converts the tensor input to an ONNX Runtime TensorRef.
    fn to_tensor_ref(&self) -> Result<TensorRef<'a, f32>, OCRError> {
        match self {
            TensorInput::Array2(arr) => {
                let dims: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();
                let data = arr.as_slice().ok_or_else(|| OCRError::InvalidInput {
                    message: "Array2 tensor is not contiguous in memory".to_string(),
                })?;
                TensorRef::from_array_view((dims, data)).map_err(|e| OCRError::InvalidInput {
                    message: format!("Failed to create Array2 TensorRef: {}", e),
                })
            }
            TensorInput::Array3(arr) => {
                let dims: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();
                let data = arr.as_slice().ok_or_else(|| OCRError::InvalidInput {
                    message: "Array3 tensor is not contiguous in memory".to_string(),
                })?;
                TensorRef::from_array_view((dims, data)).map_err(|e| OCRError::InvalidInput {
                    message: format!("Failed to create Array3 TensorRef: {}", e),
                })
            }
            TensorInput::Array4(arr) => {
                let dims: Vec<i64> = arr.shape().iter().map(|&d| d as i64).collect();
                let data = arr.as_slice().ok_or_else(|| OCRError::InvalidInput {
                    message: "Array4 tensor is not contiguous in memory".to_string(),
                })?;
                TensorRef::from_array_view((dims, data)).map_err(|e| OCRError::InvalidInput {
                    message: format!("Failed to create Array4 TensorRef: {}", e),
                })
            }
        }
    }

    /// Returns the shape of the tensor as a vector.
    fn shape(&self) -> Vec<usize> {
        match self {
            TensorInput::Array2(arr) => arr.shape().to_vec(),
            TensorInput::Array3(arr) => arr.shape().to_vec(),
            TensorInput::Array4(arr) => arr.shape().to_vec(),
        }
    }
}

impl OrtInfer {
    /// Returns the model path associated with this inference engine.
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    /// Returns the model name associated with this inference engine.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Unified inference method: multi-input -> multi-output.
    ///
    /// This is the only public inference method. It accepts arbitrary named inputs
    /// and returns all model outputs without making assumptions about their semantic
    /// meaning, shape, or type.
    ///
    /// # Responsibilities
    ///
    /// This method is responsible for:
    /// - Managing the session pool (round-robin selection)
    /// - Converting inputs to ONNX Runtime format
    /// - Executing inference
    /// - Extracting raw outputs
    ///
    /// This method is NOT responsible for:
    /// - Validating output shapes
    /// - Converting outputs to specific dimensions
    /// - Interpreting the semantic meaning of outputs
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of (name, tensor) pairs specifying the model inputs
    ///
    /// # Returns
    ///
    /// A vector of (name, TensorOutput) pairs representing all model outputs.
    /// The caller (model layer) is responsible for interpreting and validating
    /// these outputs according to the model's requirements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Single input
    /// let inputs = vec![("image", TensorInput::Array4(&batch))];
    /// let outputs = inference.infer(&inputs)?;
    /// let result = outputs[0].1.try_into_array2_f32()?;
    ///
    /// // Multiple inputs
    /// let inputs = vec![
    ///     ("image", TensorInput::Array4(&batch)),
    ///     ("scale_factor", TensorInput::Array2(&scale)),
    ///     ("im_shape", TensorInput::Array2(&shape)),
    /// ];
    /// let outputs = inference.infer(&inputs)?;
    /// ```
    pub fn infer(
        &self,
        inputs: &[(&str, TensorInput)],
    ) -> Result<Vec<(String, TensorOutput)>, OCRError> {
        if inputs.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No inputs provided for inference".to_string(),
            });
        }

        // Get primary input shape for error reporting (use first input)
        let input_shape = inputs[0].1.shape();

        // Build context string for error messages
        let input_names: Vec<&str> = inputs.iter().map(|(name, _)| *name).collect();
        let context = format!(
            "ONNX Runtime inference failed with inputs: {}",
            input_names.join(", ")
        );

        // Acquire session lock (round-robin)
        let idx = self
            .next_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.sessions.len();
        let mut session_guard = self.sessions[idx]
            .lock()
            .map_err(|_| OCRError::InvalidInput {
                message: format!(
                    "Model '{}': Failed to acquire session lock for session {}/{}",
                    self.model_name,
                    idx,
                    self.sessions.len()
                ),
            })?;

        // Collect declared output names before running
        let output_names: Vec<String> = session_guard
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Convert inputs to TensorRef and build the inputs
        let tensor_refs: Result<Vec<_>, _> = inputs
            .iter()
            .map(|(name, tensor_input)| {
                tensor_input
                    .to_tensor_ref()
                    .map(|tensor_ref| (Cow::Borrowed(*name), tensor_ref.into()))
            })
            .collect();

        let tensor_refs = tensor_refs?;

        // Run inference
        let ort_inputs: SessionInputs<'_, '_, 0> = SessionInputs::ValueMap(tensor_refs);
        let outputs = session_guard.run(ort_inputs).map_err(|e| {
            OCRError::model_inference_error_builder(&self.model_name, "forward_pass")
                .input_shape(&input_shape)
                .context(&context)
                .build(e)
        })?;

        // Extract all outputs without making assumptions
        let mut results = Vec::new();
        for name in &output_names {
            let value = &outputs[name.as_str()];

            // Try to extract as different types (prioritize f32 for most ML models)
            let tensor = if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                TensorOutput::F32 {
                    shape: shape.iter().copied().collect(),
                    data: data.to_vec(),
                }
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                TensorOutput::I64 {
                    shape: shape.iter().copied().collect(),
                    data: data.to_vec(),
                }
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
                TensorOutput::I64 {
                    shape: shape.iter().copied().collect(),
                    data: data.iter().map(|&v| v as i64).collect(),
                }
            } else {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}': Unsupported output type for tensor '{}'. Only f32, i64, and i32 are supported.",
                        self.model_name, name
                    ),
                });
            };

            results.push((name.clone(), tensor));
        }

        Ok(results)
    }
}
