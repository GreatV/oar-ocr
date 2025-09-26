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
}
