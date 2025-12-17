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

    /// Generic inference helper that handles the common inference workflow.
    ///
    /// This method:
    /// 1. Gets output name
    /// 2. Converts input tensor
    /// 3. Acquires session lock
    /// 4. Runs inference
    /// 5. Calls the provided processor with outputs and metadata
    ///
    /// The processor receives the raw outputs and can extract tensors as needed.
    /// This design avoids lifetime issues while still reducing code duplication.
    ///
    /// # Type Parameters
    /// - `T`: The return type of the processor
    fn run_inference_core<T>(
        &self,
        x: &Tensor4D,
        processor: impl for<'a> FnOnce(
            &'a ort::session::SessionOutputs<'a>,
            &str,
            &[String],
            &[usize],
        ) -> Result<T, OCRError>,
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
            OCRError::model_inference_error_builder(&self.model_name, "tensor_conversion")
                .input_shape(&input_shape)
                .context(format!(
                    "Failed to convert input tensor with shape {:?}",
                    input_shape
                ))
                .build(e)
        })?;

        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];

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

        // Collect declared output names before running (avoid borrow conflicts later)
        let output_names: Vec<String> = session_guard
            .outputs
            .iter()
            .map(|o| o.name.clone())
            .collect();

        let outputs = session_guard.run(inputs).map_err(|e| {
            OCRError::model_inference_error_builder(&self.model_name, "forward_pass")
                .input_shape(&input_shape)
                .context(format!(
                    "ONNX Runtime inference failed with input '{}' -> output '{}'",
                    self.input_name, output_name
                ))
                .build(e)
        })?;

        processor(&outputs, &output_name, &output_names, &input_shape)
    }

    /// Runs inference with f32 output extraction.
    fn run_inference_with_processor<T>(
        &self,
        x: &Tensor4D,
        processor: impl FnOnce(&[i64], &[f32]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let model_name = self.model_name.clone();

        self.run_inference_core(
            x,
            move |outputs, output_name, _output_names, input_shape| {
                let output = outputs[output_name]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| {
                        OCRError::model_inference_error_builder(&model_name, "output_extraction")
                            .input_shape(input_shape)
                            .context(format!(
                                "Failed to extract output tensor '{}' as f32",
                                output_name
                            ))
                            .build(e)
                    })?;
                let (output_shape, output_data) = output;
                processor(output_shape, output_data)
            },
        )
    }

    pub fn infer_4d(&self, x: &Tensor4D) -> Result<Tensor4D, OCRError> {
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 4 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' 4D inference: expected 4D output tensor, got {}D with shape {:?}",
                        self.model_name,
                        output_shape.len(),
                        output_shape
                    ),
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
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' 2D inference: output data size mismatch for input shape {:?} -> output shape {:?}: expected {}, got {}",
                        self.model_name, input_shape, output_shape, expected_len, output_data.len()
                    ),
                });
            }

            let array_view = ArrayView2::from_shape((batch_size, num_classes), output_data)
                .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    pub fn infer_3d(&self, x: &Tensor4D) -> Result<Tensor3D, OCRError> {
        self.run_inference_with_processor(x, |output_shape, output_data| {
            if output_shape.len() != 3 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' 3D inference: expected 3D output tensor, got {}D with shape {:?}",
                        self.model_name,
                        output_shape.len(),
                        output_shape
                    ),
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
    /// instead of f32 tensors. It includes fallback logic to scan all outputs for an i64 tensor
    /// if the primary output is not i64.
    fn run_inference_with_processor_i64<T>(
        &self,
        x: &Tensor4D,
        processor: impl FnOnce(&[i64], &[i64]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let model_name = self.model_name.clone();

        self.run_inference_core(x, move |outputs, output_name, output_names, _input_shape| {
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
            if let Some((shape, data)) = try_extract_by(output_name) {
                extracted = Some((shape, data));
            } else {
                // Fallback: iterate declared outputs to find any i64 tensor
                for name in output_names {
                    if name.as_str() == output_name {
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
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Model '{}': Failed to extract any output as i64. Tried '{}' first. Available outputs: {:?}",
                            model_name, output_name, output_names
                        ),
                    });
                }
            };

            processor(&output_shape, output_data)
        })
    }

    /// Runs inference and returns a 2D int64 tensor.
    ///
    /// This is useful for models that output token IDs (e.g., formula recognition).
    /// The output shape is typically [batch_size, sequence_length].
    pub fn infer_2d_i64(&self, x: &Tensor4D) -> Result<ndarray::Array2<i64>, OCRError> {
        self.run_inference_with_processor_i64(x, |output_shape, output_data| {
            if output_shape.len() != 2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' 2D i64 inference: expected 2D output tensor, got {}D with shape {:?}",
                        self.model_name,
                        output_shape.len(),
                        output_shape
                    ),
                });
            }

            let batch_size_out = output_shape[0] as usize;
            let seq_len = output_shape[1] as usize;
            let expected_len = batch_size_out * seq_len;

            if output_data.len() != expected_len {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' 2D i64 inference: output data size mismatch - expected {}, got {}",
                        self.model_name, expected_len, output_data.len()
                    ),
                });
            }

            let array_view = ArrayView2::from_shape((batch_size_out, seq_len), output_data)
                .map_err(OCRError::Tensor)?;
            Ok(array_view.to_owned())
        })
    }

    /// Runs inference for models with dual 3D outputs.
    ///
    /// This is used for models like SLANet that output two 3D tensors.
    /// The first output is typically structure/token predictions, and the second
    /// is bounding box predictions or similar auxiliary outputs.
    ///
    /// # Returns
    ///
    /// A tuple of two 3D tensors: (first_output, second_output)
    pub fn infer_dual_3d(&self, x: &Tensor4D) -> Result<(Tensor3D, Tensor3D), OCRError> {
        let model_name = self.model_name.clone();

        self.run_inference_core(x, move |outputs, _output_name, output_names, input_shape| {
            // Expect at least 2 outputs
            if output_names.len() < 2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: expected at least 2 outputs, got {}",
                        model_name,
                        output_names.len()
                    ),
                });
            }

            // Extract first output
            let first_output = outputs[output_names[0].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    OCRError::model_inference_error_builder(&model_name, "output_extraction")
                        .input_shape(input_shape)
                        .batch_index(0)
                        .context(format!(
                            "Failed to extract first output tensor '{}' as f32",
                            output_names[0]
                        ))
                        .build(e)
                })?;

            let (first_shape, first_data) = first_output;

            // Validate first output is 3D
            if first_shape.len() != 3 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: first output expected 3D, got {}D with shape {:?}",
                        model_name,
                        first_shape.len(),
                        first_shape
                    ),
                });
            }

            // Extract second output
            let second_output = outputs[output_names[1].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    OCRError::model_inference_error_builder(&model_name, "output_extraction")
                        .input_shape(input_shape)
                        .batch_index(1)
                        .context(format!(
                            "Failed to extract second output tensor '{}' as f32",
                            output_names[1]
                        ))
                        .build(e)
                })?;

            let (second_shape, second_data) = second_output;

            // Validate second output is 3D
            if second_shape.len() != 3 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: second output expected 3D, got {}D with shape {:?}",
                        model_name,
                        second_shape.len(),
                        second_shape
                    ),
                });
            }

            // Reshape first tensor
            let dim0_1 = first_shape[0] as usize;
            let dim1_1 = first_shape[1] as usize;
            let dim2_1 = first_shape[2] as usize;
            let expected_len_1 = dim0_1 * dim1_1 * dim2_1;

            if first_data.len() != expected_len_1 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: first output data size mismatch - expected {}, got {}",
                        model_name, expected_len_1, first_data.len()
                    ),
                });
            }

            let first_tensor = ArrayView3::from_shape((dim0_1, dim1_1, dim2_1), first_data)
                .map_err(OCRError::Tensor)?
                .to_owned();

            // Reshape second tensor
            let dim0_2 = second_shape[0] as usize;
            let dim1_2 = second_shape[1] as usize;
            let dim2_2 = second_shape[2] as usize;
            let expected_len_2 = dim0_2 * dim1_2 * dim2_2;

            if second_data.len() != expected_len_2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: second output data size mismatch - expected {}, got {}",
                        model_name, expected_len_2, second_data.len()
                    ),
                });
            }

            let second_tensor = ArrayView3::from_shape((dim0_2, dim1_2, dim2_2), second_data)
                .map_err(OCRError::Tensor)?
                .to_owned();

            Ok((first_tensor, second_tensor))
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
                session_guard.run(inputs).map_err(|e| {
                    OCRError::model_inference_error_builder(&self.model_name, "forward_pass")
                        .input_shape(input_shape)
                        .context(
                            "ONNX Runtime inference failed with inputs 'image', 'im_shape', and 'scale_factor'",
                        )
                        .build(e)
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
                    OCRError::model_inference_error_builder(&self.model_name, "forward_pass")
                        .input_shape(input_shape)
                        .context(
                            "ONNX Runtime inference failed with inputs 'image' and 'scale_factor'",
                        )
                        .build(e)
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
                    OCRError::model_inference_error_builder(&self.model_name, "forward_pass")
                        .input_shape(input_shape)
                        .context("ONNX Runtime inference failed with single input 'image'")
                        .build(e)
                })?
            }
        };

        // Extract output
        let default_output_name = "fetch_name_0".to_string();
        let output_name = self.output_name.as_ref().unwrap_or(&default_output_name);
        let output = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                OCRError::model_inference_error_builder(&self.model_name, "output_extraction")
                    .input_shape(input_shape)
                    .context(format!(
                        "Failed to extract output tensor '{}' as f32",
                        output_name
                    ))
                    .build(e)
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
            Err(OCRError::InvalidInput {
                message: format!(
                    "Model '{}' layout inference: expected 2D or 4D output tensor, got {}D with shape {:?}",
                    self.model_name,
                    output_shape.len(),
                    output_shape
                ),
            })
        }
    }
}
