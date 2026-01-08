use super::*;
use ndarray::{ArrayView2, ArrayView3, ArrayView4};
use ort::session::{SessionInputs, SessionOutputs};
use ort::value::TensorRef;

/// Return type for run_inference_core: (output_shape, output_data, output_names, input_shape)
type InferenceCoreResult = (Vec<i64>, Vec<f32>, Vec<String>, Vec<usize>);

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
            if let Some(output) = session.outputs().first() {
                Ok(output.name().to_string())
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

    /// Core generic inference helper that runs pre-built inputs through the session.
    ///
    /// This method centralizes session management and inference execution:
    /// 1. Acquires session lock (round-robin across session pool)
    /// 2. Runs inference with the provided inputs
    /// 3. Passes outputs to the extractor closure for type-specific extraction
    ///
    /// Uses Higher-Ranked Trait Bounds (HRTB) to properly handle the lifetime
    /// of `SessionOutputs` within the closure.
    ///
    /// # Arguments
    /// * `inputs` - Pre-built ONNX session inputs
    /// * `input_shape` - Shape of the primary input tensor (for error messages)
    /// * `context` - Description of the inference context (for error messages)
    /// * `extractor` - Closure that extracts and processes the outputs
    fn run_with_inputs<R, F>(
        &self,
        inputs: SessionInputs<'_, '_>,
        input_shape: &[usize],
        context: &str,
        extractor: F,
    ) -> Result<R, OCRError>
    where
        F: for<'a> FnOnce(SessionOutputs<'a>, &[String]) -> Result<R, OCRError>,
    {
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
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let outputs = session_guard.run(inputs).map_err(|e| {
            OCRError::model_inference_error_builder(&self.model_name, "forward_pass")
                .input_shape(input_shape)
                .context(context)
                .build(e)
        })?;

        extractor(outputs, &output_names)
    }

    /// Convenience helper for single-input inference with standard tensor input.
    ///
    /// This method:
    /// 1. Converts the input tensor to ONNX format
    /// 2. Delegates to `run_with_inputs` for session management
    /// 3. Passes outputs to the processor closure for type-specific extraction
    fn run_inference_with_outputs<T, F>(&self, x: &Tensor4D, processor: F) -> Result<T, OCRError>
    where
        F: for<'a> FnOnce(SessionOutputs<'a>, &[String], &[usize]) -> Result<T, OCRError>,
    {
        let input_shape = x.shape().to_vec();

        let input_dims: Vec<i64> = x.shape().iter().map(|&d| d as i64).collect();
        let input_data = x.as_slice().ok_or_else(|| OCRError::InvalidInput {
            message: "Input tensor is not contiguous in memory".to_string(),
        })?;

        let input_tensor =
            TensorRef::from_array_view((input_dims.clone(), input_data)).map_err(|e| {
                OCRError::model_inference_error_builder(&self.model_name, "tensor_conversion")
                    .input_shape(&input_shape)
                    .context(format!(
                        "Failed to convert input tensor with shape {:?}",
                        input_shape
                    ))
                    .build(e)
            })?;

        let inputs = ort::inputs![self.input_name.as_str() => input_tensor];
        let context = format!(
            "ONNX Runtime inference failed with input '{}'",
            self.input_name
        );

        self.run_with_inputs(
            SessionInputs::ValueMap(inputs),
            &input_shape,
            &context,
            |outputs, output_names| processor(outputs, output_names, &input_shape),
        )
    }

    /// Generic inference helper that handles the common inference workflow.
    ///
    /// This method:
    /// 1. Gets output name
    /// 2. Converts input tensor
    /// 3. Acquires session lock
    /// 4. Runs inference
    /// 5. Extracts and returns the output tensor data
    ///
    /// Returns (output_shape, output_data, output_names, input_shape)
    fn run_inference_core(&self, x: &Tensor4D) -> Result<InferenceCoreResult, OCRError> {
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

        self.run_inference_with_outputs(x, |outputs, output_names, input_shape| {
            let (output_shape, output_data_slice) = outputs[output_name.as_str()]
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

            let output_shape_vec: Vec<i64> = output_shape.iter().copied().collect();
            let output_data_vec: Vec<f32> = output_data_slice.to_vec();

            Ok((
                output_shape_vec,
                output_data_vec,
                output_names.to_vec(),
                input_shape.to_vec(),
            ))
        })
    }

    /// Runs inference with f32 output extraction.
    fn run_inference_with_processor<T>(
        &self,
        x: &Tensor4D,
        processor: impl FnOnce(&[i64], &[f32]) -> Result<T, OCRError>,
    ) -> Result<T, OCRError> {
        let (output_shape, output_data, _output_names, _input_shape) =
            self.run_inference_core(x)?;
        processor(&output_shape, &output_data)
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

        self.run_inference_with_outputs(x, |outputs, output_names, _input_shape| {
            // Try the discovered output name first; if it isn't i64, scan other outputs for an i64 tensor.
            let mut extracted: Option<(Vec<i64>, Vec<i64>)> = None;

            // First attempt: the default output name
            if let Ok((shape, data)) = outputs[output_name.as_str()].try_extract_tensor::<i64>() {
                extracted = Some((shape.iter().copied().collect(), data.to_vec()));
            } else {
                // Fallback: iterate declared outputs to find any i64 tensor
                for name in output_names {
                    if name.as_str() == output_name {
                        continue;
                    }
                    if let Ok((shape, data)) = outputs[name.as_str()].try_extract_tensor::<i64>() {
                        extracted = Some((shape.iter().copied().collect(), data.to_vec()));
                        break;
                    }
                }
            }

            let (output_shape, output_data) = match extracted {
                Some((shape, data)) => (shape, data),
                None => {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Model '{}': Failed to extract any output as i64. Tried '{}' first. Available outputs: {:?}",
                            self.model_name, output_name, output_names
                        ),
                    });
                }
            };

            processor(&output_shape, &output_data)
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
        self.run_inference_with_outputs(x, |outputs, output_names, input_shape| {
            // Expect at least 2 outputs
            if output_names.len() < 2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: expected at least 2 outputs, got {}",
                        self.model_name,
                        output_names.len()
                    ),
                });
            }

            // Extract first output
            let (first_shape, first_data_slice) = outputs[output_names[0].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    OCRError::model_inference_error_builder(&self.model_name, "output_extraction")
                        .input_shape(input_shape)
                        .batch_index(0)
                        .context(format!(
                            "Failed to extract first output tensor '{}' as f32",
                            output_names[0]
                        ))
                        .build(e)
                })?;

            let first_shape_vec: Vec<i64> = first_shape.iter().copied().collect();
            let first_data: Vec<f32> = first_data_slice.to_vec();

            // Validate first output is 3D
            if first_shape_vec.len() != 3 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: first output expected 3D, got {}D with shape {:?}",
                        self.model_name,
                        first_shape_vec.len(),
                        first_shape_vec
                    ),
                });
            }

            // Extract second output
            let (second_shape, second_data_slice) = outputs[output_names[1].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    OCRError::model_inference_error_builder(&self.model_name, "output_extraction")
                        .input_shape(input_shape)
                        .batch_index(1)
                        .context(format!(
                            "Failed to extract second output tensor '{}' as f32",
                            output_names[1]
                        ))
                        .build(e)
                })?;

            let second_shape_vec: Vec<i64> = second_shape.iter().copied().collect();
            let second_data: Vec<f32> = second_data_slice.to_vec();

            // Validate second output is 3D
            if second_shape_vec.len() != 3 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: second output expected 3D, got {}D with shape {:?}",
                        self.model_name,
                        second_shape_vec.len(),
                        second_shape_vec
                    ),
                });
            }

            // Reshape first tensor
            let dim0_1 = first_shape_vec[0] as usize;
            let dim1_1 = first_shape_vec[1] as usize;
            let dim2_1 = first_shape_vec[2] as usize;
            let expected_len_1 = dim0_1 * dim1_1 * dim2_1;

            if first_data.len() != expected_len_1 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: first output data size mismatch - expected {}, got {}",
                        self.model_name,
                        expected_len_1,
                        first_data.len()
                    ),
                });
            }

            let first_tensor = ArrayView3::from_shape((dim0_1, dim1_1, dim2_1), &first_data)
                .map_err(OCRError::Tensor)?
                .to_owned();

            // Reshape second tensor
            let dim0_2 = second_shape_vec[0] as usize;
            let dim1_2 = second_shape_vec[1] as usize;
            let dim2_2 = second_shape_vec[2] as usize;
            let expected_len_2 = dim0_2 * dim1_2 * dim2_2;

            if second_data.len() != expected_len_2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Model '{}' dual 3D inference: second output data size mismatch - expected {}, got {}",
                        self.model_name,
                        expected_len_2,
                        second_data.len()
                    ),
                });
            }

            let second_tensor = ArrayView3::from_shape((dim0_2, dim1_2, dim2_2), &second_data)
                .map_err(OCRError::Tensor)?
                .to_owned();

            Ok((first_tensor, second_tensor))
        })
    }

    /// Checks if the model expects an 'im_shape' input.
    ///
    /// This is needed for layout models to determine which input combination to use.
    fn has_im_shape_input(&self) -> Result<bool, OCRError> {
        let session = self.sessions[0]
            .lock()
            .map_err(|_| OCRError::InvalidInput {
                message: "Failed to acquire session lock to check inputs".to_string(),
            })?;
        Ok(session
            .inputs()
            .iter()
            .any(|input| input.name() == "im_shape"))
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
        let input_shape = x.shape().to_vec();

        // Convert image tensor to tuple form (dims, data)
        let image_dims: Vec<i64> = x.shape().iter().map(|&d| d as i64).collect();
        let image_data = x.as_slice().ok_or_else(|| OCRError::InvalidInput {
            message: "Image tensor is not contiguous in memory".to_string(),
        })?;

        // Check which inputs the model expects
        let has_im_shape = self.has_im_shape_input()?;

        // Build inputs based on what's provided and what the model expects
        let (inputs, context) = match (im_shape.as_ref(), scale_factor.as_ref(), has_im_shape) {
            (Some(shape), Some(scale), true) => {
                // PP-DocLayout models (L, plus-L) use both im_shape and scale_factor
                let image_tensor = TensorRef::from_array_view((image_dims.clone(), image_data))
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("Failed to create image tensor: {}", e),
                    })?;
                let shape_dims: Vec<i64> = shape.shape().iter().map(|&d| d as i64).collect();
                let shape_data = shape.as_slice().ok_or_else(|| OCRError::InvalidInput {
                    message: "im_shape tensor is not contiguous in memory".to_string(),
                })?;
                let shape_tensor =
                    TensorRef::from_array_view((shape_dims, shape_data)).map_err(|e| {
                        OCRError::InvalidInput {
                            message: format!("Failed to create im_shape tensor: {}", e),
                        }
                    })?;
                let scale_dims: Vec<i64> = scale.shape().iter().map(|&d| d as i64).collect();
                let scale_data = scale.as_slice().ok_or_else(|| OCRError::InvalidInput {
                    message: "scale_factor tensor is not contiguous in memory".to_string(),
                })?;
                let scale_tensor =
                    TensorRef::from_array_view((scale_dims, scale_data)).map_err(|e| {
                        OCRError::InvalidInput {
                            message: format!("Failed to create scale_factor tensor: {}", e),
                        }
                    })?;
                (
                    ort::inputs![
                        "image" => image_tensor,
                        "im_shape" => shape_tensor,
                        "scale_factor" => scale_tensor
                    ],
                    "ONNX Runtime inference failed with inputs 'image', 'im_shape', and 'scale_factor'",
                )
            }
            (Some(_), Some(scale), false) | (None, Some(scale), _) => {
                // PP-DocLayout models (S, M) or PicoDet models use scale_factor only (no im_shape)
                let image_tensor = TensorRef::from_array_view((image_dims.clone(), image_data))
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("Failed to create image tensor: {}", e),
                    })?;
                let scale_dims: Vec<i64> = scale.shape().iter().map(|&d| d as i64).collect();
                let scale_data = scale.as_slice().ok_or_else(|| OCRError::InvalidInput {
                    message: "scale_factor tensor is not contiguous in memory".to_string(),
                })?;
                let scale_tensor =
                    TensorRef::from_array_view((scale_dims, scale_data)).map_err(|e| {
                        OCRError::InvalidInput {
                            message: format!("Failed to create scale_factor tensor: {}", e),
                        }
                    })?;
                (
                    ort::inputs![
                        "image" => image_tensor,
                        "scale_factor" => scale_tensor
                    ],
                    "ONNX Runtime inference failed with inputs 'image' and 'scale_factor'",
                )
            }
            _ => {
                // Fall back to single input
                let image_tensor =
                    TensorRef::from_array_view((image_dims, image_data)).map_err(|e| {
                        OCRError::InvalidInput {
                            message: format!("Failed to create image tensor: {}", e),
                        }
                    })?;
                (
                    ort::inputs!["image" => image_tensor],
                    "ONNX Runtime inference failed with single input 'image'",
                )
            }
        };

        // Use the centralized helper for session management and inference
        self.run_with_inputs(
            SessionInputs::ValueMap(inputs),
            &input_shape,
            context,
            |outputs, _output_names| {
                // Extract output
                let default_output_name = "fetch_name_0".to_string();
                let output_name = self.output_name.as_ref().unwrap_or(&default_output_name);
                let (output_shape, output_data) = outputs[output_name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| {
                        OCRError::model_inference_error_builder(&self.model_name, "output_extraction")
                            .input_shape(&input_shape)
                            .context(format!(
                                "Failed to extract output tensor '{}' as f32",
                                output_name
                            ))
                            .build(e)
                    })?;

                // Validate and convert output
                // Some models output 2D [num_boxes, N] format instead of 4D
                // N can be 6 (PP-DocLayout) or 8 (PP-DocLayoutV2 with reading order: col_index, row_index)
                // We pass through raw data and let the postprocessor handle format-specific logic
                match output_shape.len() {
                    2 => {
                        let num_boxes = output_shape[0] as usize;
                        let box_dim = output_shape[1] as usize;

                        match box_dim {
                            // 8-dim format: [class_id, score, x1, y1, x2, y2, col_index, row_index]
                            // Pass through raw data for postprocessor to handle reading order sorting
                            8 => {
                                ndarray::Array::from_shape_vec((1, num_boxes, 1, 8), output_data.to_vec())
                                    .map_err(|e| {
                                        OCRError::tensor_operation_error(
                                            "output_reshape",
                                            &[1, num_boxes, 1, 8],
                                            &[output_data.len()],
                                            &format!(
                                                "Failed to reshape 8-dim output to 4D for model '{}'",
                                                self.model_name
                                            ),
                                            e,
                                        )
                                    })
                            }
                            // 6-dim format: [class_id, score, x1, y1, x2, y2]
                            // Convert directly to 4D format [batch=1, num_boxes, 1, 6]
                            6 => {
                                ndarray::Array::from_shape_vec((1, num_boxes, 1, 6), output_data.to_vec())
                                    .map_err(|e| {
                                        OCRError::tensor_operation_error(
                                            "output_reshape",
                                            &[1, num_boxes, 1, 6],
                                            &[output_data.len()],
                                            &format!(
                                                "Failed to reshape 2D output to 4D for model '{}'",
                                                self.model_name
                                            ),
                                            e,
                                        )
                                    })
                            }
                            _ => Err(OCRError::InvalidInput {
                                message: format!(
                                    "Expected box dimension 6 or 8, got {} with shape {:?}",
                                    box_dim, output_shape
                                ),
                            }),
                        }
                    }
                    // Standard 4D output format
                    4 => {
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

                        ndarray::Array::from_shape_vec(
                            (batch_size_out, channels_out, height_out, width_out),
                            output_data.to_vec(),
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
                        })
                    }
                    _ => Err(OCRError::InvalidInput {
                        message: format!(
                            "Model '{}' layout inference: expected 2D or 4D output tensor, got {}D with shape {:?}",
                            self.model_name,
                            output_shape.len(),
                            output_shape
                        ),
                    }),
                }
            },
        )
    }
}
