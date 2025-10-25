use super::*;
use ndarray::{ArrayView2, ArrayView3, ArrayView4};
use ort::value::TensorRef;

impl OrtInfer {
    /// Returns the configured or discovered output tensor name.
    fn get_output_name(&self) -> Result<String, OCRError> {
        if let Some(ref name) = self.output_name {
            Ok(name.clone())
        } else {
            let session = self.sessions[0]
                .lock()
                .map_err(|_| OCRError::InvalidInput {
                    message: "Failed to acquire session lock".to_string(),
                })?;
            if let Some(output) = session.outputs.first() {
                Ok(output.name.clone())
            } else {
                Err(OCRError::InvalidInput {
                    message: "No outputs available in session - model may be invalid or corrupted"
                        .to_string(),
                })
            }
        }
    }

    /// Returns the model path associated with this inference engine.
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    /// Returns the model name associated with this inference engine.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    fn run_inference_with_processor<T>(
        &self,
        x: &Tensor4D,
        processor: impl FnOnce(&[i64], &[f32]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let input_shape = x.shape().to_vec();

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

    pub fn infer_4d(&self, x: &Tensor4D) -> Result<Tensor4D, OCRError> {
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 4 {
                return Err(OCRError::tensor_operation_error(
                    "output_validation",
                    &[4],
                    &[output_shape.len()],
                    &format!(
                        "Model '{}' 4D inference: expected 4D output tensor, got {}D with shape {:?}",
                        self.model_name,
                        output_shape.len(),
                        output_shape
                    ),
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

            let array_view = ArrayView4::from_shape(
                (batch_size_out, channels_out, height_out, width_out),
                output_data,
            )
            .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

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

            let array_view = ArrayView2::from_shape((batch_size, num_classes), output_data)
                .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    pub fn infer_3d(&self, x: &Tensor4D) -> Result<Tensor3D, OCRError> {
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 3 {
                return Err(OCRError::tensor_operation_error(
                    "output_validation",
                    &[3],
                    &[output_shape.len()],
                    &format!(
                        "Model '{}' 3D inference: expected 3D output tensor, got {}D with shape {:?}",
                        self.model_name,
                        output_shape.len(),
                        output_shape
                    ),
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

            let array_view = ArrayView3::from_shape(
                (batch_size_out, seq_len, num_classes),
                output_data,
            )
            .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    /// Runs inference with int64 outputs (for models that output token IDs).
    ///
    /// This method is similar to `run_inference_with_processor` but extracts i64 tensors
    /// instead of f32 tensors.
    fn run_inference_with_processor_i64<T>(
        &self,
        x: &Tensor4D,
        processor: impl FnOnce(&[i64], &[i64]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let input_shape = x.shape().to_vec();

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

        // Collect declared output names before running (avoid borrow conflicts later)
        let output_names: Vec<String> = session_guard
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect();

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

        // Try the discovered output name first; if it isn't i64, scan other outputs for an i64 tensor.
        let mut extracted: Option<(Vec<i64>, &[i64])> = None;

        // Helper to try extract by name
        let try_extract_by = |name: &str| -> Option<(Vec<i64>, &[i64])> {
            match outputs[name].try_extract_tensor::<i64>() {
                Ok((shape, data)) => Some((shape.to_vec(), data)),
                Err(_) => None,
            }
        };

        // First attempt: the default output name
        if let Some((shape, data)) = try_extract_by(output_name.as_str()) {
            extracted = Some((shape, data));
        } else {
            // Fallback: iterate declared outputs to find any i64 tensor
            for name in &output_names {
                if name.as_str() == output_name.as_str() {
                    continue;
                }
                if let Some((shape, data)) = try_extract_by(name.as_str()) {
                    extracted = Some((shape, data));
                    break;
                }
            }
        }

        let (output_shape, output_data) = match extracted {
            Some((shape, data)) => (shape, data),
            None => {
                // Build a helpful error listing available outputs
                let available: Vec<String> = output_names.clone();
                return Err(OCRError::model_inference_error(
                    &self.model_name,
                    "output_extraction",
                    0,
                    &input_shape,
                    &format!(
                        "Failed to extract any output as i64. Tried '{}' first. Available outputs: {:?}",
                        output_name, available
                    ),
                    crate::core::errors::SimpleError::new("No i64 output tensor found"),
                ));
            }
        };

        processor(&output_shape, output_data)
    }

    /// Runs inference and returns a 2D int64 tensor.
    ///
    /// This is useful for models that output token IDs (e.g., formula recognition).
    /// The output shape is typically [batch_size, sequence_length].
    pub fn infer_2d_i64(&self, x: &Tensor4D) -> Result<ndarray::Array2<i64>, OCRError> {
        self.run_inference_with_processor_i64(x, |output_shape, output_data| {
            if output_shape.len() != 2 {
                return Err(OCRError::tensor_operation_error(
                    "output_validation",
                    &[2],
                    &[output_shape.len()],
                    &format!(
                        "Model '{}' 2D i64 inference: expected 2D output tensor, got {}D with shape {:?}",
                        self.model_name,
                        output_shape.len(),
                        output_shape
                    ),
                    crate::core::errors::SimpleError::new("Invalid output tensor dimensions"),
                ));
            }

            let batch_size_out = output_shape[0] as usize;
            let seq_len = output_shape[1] as usize;
            let expected_len = batch_size_out * seq_len;

            if output_data.len() != expected_len {
                return Err(OCRError::tensor_operation_error(
                    "output_data_validation",
                    &[expected_len],
                    &[output_data.len()],
                    &format!(
                        "Model '{}' 2D i64 inference: output data size mismatch",
                        self.model_name
                    ),
                    crate::core::errors::SimpleError::new("Output tensor data size mismatch"),
                ));
            }

            let array_view = ArrayView2::from_shape((batch_size_out, seq_len), output_data)
                .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    /// Runs inference with multiple inputs for layout detection models.
    ///
    /// Layout detection models typically require:
    /// - `image`: The preprocessed image tensor [N, 3, H, W]
    /// - `scale_factor`: Scale factors used during preprocessing [N, 2] (for PicoDet)
    /// - `im_shape`: Original image shape [N, 2] (for PP-DocLayout)
    pub fn infer_4d_layout(
        &self,
        x: &Tensor4D,
        scale_factor: Option<ndarray::Array2<f32>>,
        im_shape: Option<ndarray::Array2<f32>>,
    ) -> Result<Tensor4D, OCRError> {
        let idx = self
            .next_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.sessions.len();
        let mut session_guard = self.sessions[idx]
            .lock()
            .map_err(|e| OCRError::InvalidInput {
                message: format!(
                    "Failed to acquire session lock for model '{}': {}",
                    self.model_name, e
                ),
            })?;

        let input_shape = x.shape();
        let _batch_size = input_shape[0];

        // Use the tensor as-is (assumed to be NCHW contiguous)
        let input_tensor_view = x.view();

        // Check which inputs the model expects
        let has_im_shape = session_guard
            .inputs
            .iter()
            .any(|input| input.name == "im_shape");

        // Build inputs based on what's provided and what the model expects
        let outputs = match (im_shape.as_ref(), scale_factor.as_ref(), has_im_shape) {
            (Some(shape), Some(scale), true) => {
                // PP-DocLayout models (L, plus-L) use both im_shape and scale_factor
                let image_tensor = TensorRef::from_array_view(input_tensor_view).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!("Failed to create image tensor: {}", e),
                    }
                })?;
                let shape_tensor = TensorRef::from_array_view(shape.view()).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!("Failed to create im_shape tensor: {}", e),
                    }
                })?;
                let scale_tensor = TensorRef::from_array_view(scale.view()).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!("Failed to create scale_factor tensor: {}", e),
                    }
                })?;
                let inputs = ort::inputs![
                    "image" => image_tensor,
                    "im_shape" => shape_tensor,
                    "scale_factor" => scale_tensor
                ];
                session_guard
                    .run(inputs)
                    .map_err(|e| {
                        OCRError::model_inference_error(
                            &self.model_name,
                            "forward_pass",
                            0,
                            input_shape,
                            "ONNX Runtime inference failed with inputs 'image', 'im_shape', and 'scale_factor'",
                            e,
                        )
                    })?
            }
            (Some(_), Some(scale), false) | (None, Some(scale), _) => {
                // PP-DocLayout models (S, M) or PicoDet models use scale_factor only (no im_shape)
                let image_tensor = TensorRef::from_array_view(input_tensor_view).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!("Failed to create image tensor: {}", e),
                    }
                })?;
                let scale_tensor = TensorRef::from_array_view(scale.view()).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!("Failed to create scale_factor tensor: {}", e),
                    }
                })?;
                let inputs = ort::inputs![
                    "image" => image_tensor,
                    "scale_factor" => scale_tensor
                ];
                session_guard.run(inputs).map_err(|e| {
                    OCRError::model_inference_error(
                        &self.model_name,
                        "forward_pass",
                        0,
                        input_shape,
                        "ONNX Runtime inference failed with inputs 'image' and 'scale_factor'",
                        e,
                    )
                })?
            }
            _ => {
                // Fall back to single input
                let image_tensor = TensorRef::from_array_view(input_tensor_view).map_err(|e| {
                    OCRError::InvalidInput {
                        message: format!("Failed to create image tensor: {}", e),
                    }
                })?;
                let inputs = ort::inputs!["image" => image_tensor];
                session_guard.run(inputs).map_err(|e| {
                    OCRError::model_inference_error(
                        &self.model_name,
                        "forward_pass",
                        0,
                        input_shape,
                        "ONNX Runtime inference failed with single input 'image'",
                        e,
                    )
                })?
            }
        };

        // Extract output
        let default_output_name = "fetch_name_0".to_string();
        let output_name = self.output_name.as_ref().unwrap_or(&default_output_name);
        let output = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                OCRError::model_inference_error(
                    &self.model_name,
                    "output_extraction",
                    0,
                    input_shape,
                    &format!("Failed to extract output tensor '{}' as f32", output_name),
                    e,
                )
            })?;

        let (output_shape, output_data) = output;

        // Validate and convert output
        // Some models output 2D [num_boxes, 6] format instead of 4D
        if output_shape.len() == 2 {
            // 2D output format: [num_boxes, 6] where each box is [class_id, score, x1, y1, x2, y2]
            let num_boxes = output_shape[0] as usize;
            let box_dim = output_shape[1] as usize;

            if box_dim != 6 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Expected box dimension 6, got {} with shape {:?}",
                        box_dim, output_shape
                    ),
                });
            }

            // Convert to 4D format [batch=1, num_boxes, 1, 6]
            let array =
                ndarray::Array::from_shape_vec((1, num_boxes, 1, box_dim), output_data.to_owned())
                    .map_err(|e| {
                        OCRError::tensor_operation_error(
                            "output_reshape",
                            &[1, num_boxes, 1, box_dim],
                            &[output_data.len()],
                            &format!(
                                "Failed to reshape 2D output to 4D for model '{}'",
                                self.model_name
                            ),
                            e,
                        )
                    })?;

            Ok(array)
        } else if output_shape.len() == 4 {
            // Standard 4D output format
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

            let array = ndarray::Array::from_shape_vec(
                (batch_size_out, channels_out, height_out, width_out),
                output_data.to_owned(),
            )
            .map_err(|e| {
                OCRError::tensor_operation_error(
                    "output_reshape",
                    &[batch_size_out, channels_out, height_out, width_out],
                    &[output_data.len()],
                    &format!(
                        "Failed to reshape 4D output for model '{}'",
                        self.model_name
                    ),
                    e,
                )
            })?;

            Ok(array)
        } else {
            Err(OCRError::tensor_operation_error(
                "output_validation",
                &[2, 4],
                &[output_shape.len()],
                &format!(
                    "Model '{}' layout inference: expected 2D or 4D output tensor, got {}D with shape {:?}",
                    self.model_name,
                    output_shape.len(),
                    output_shape
                ),
                crate::core::errors::SimpleError::new("Invalid output tensor dimensions"),
            ))
        }
    }
}
