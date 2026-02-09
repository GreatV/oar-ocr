//! Generic tensor output representation for inference results.
//!
//! This module defines a backend-agnostic way to represent tensor outputs
//! from inference engines, supporting multiple data types and dimensions.

use crate::core::errors::OCRError;
use ndarray::{Array2, Array3, Array4, ArrayD};

/// Generic tensor output supporting multiple data types.
///
/// This enum represents the raw output from an inference engine without
/// making assumptions about the semantic meaning of the data. It's the
/// responsibility of the caller (model layer) to interpret and validate
/// the output shape and type.
#[derive(Debug, Clone)]
pub enum TensorOutput {
    /// 32-bit floating point tensor
    F32 { shape: Vec<i64>, data: Vec<f32> },
    /// 64-bit integer tensor (commonly used for token IDs)
    I64 { shape: Vec<i64>, data: Vec<i64> },
}

impl TensorOutput {
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[i64] {
        match self {
            TensorOutput::F32 { shape, .. } => shape,
            TensorOutput::I64 { shape, .. } => shape,
        }
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Returns the total number of elements.
    pub fn len(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }

    /// Returns true if the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to extract as a 2D f32 array.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not f32 type
    /// - The tensor is not 2-dimensional
    /// - The data length doesn't match the shape
    pub fn try_into_array2_f32(self) -> Result<Array2<f32>, OCRError> {
        match self {
            TensorOutput::F32 { shape, data } => {
                if shape.len() != 2 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Expected 2D tensor, got {}D with shape {:?}",
                            shape.len(),
                            shape
                        ),
                    });
                }

                let dim0 = shape[0] as usize;
                let dim1 = shape[1] as usize;
                let expected_len = dim0 * dim1;

                if data.len() != expected_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Data length mismatch: expected {}, got {}",
                            expected_len,
                            data.len()
                        ),
                    });
                }

                Array2::from_shape_vec((dim0, dim1), data).map_err(OCRError::Tensor)
            }
            TensorOutput::I64 { .. } => Err(OCRError::InvalidInput {
                message: "Expected f32 tensor, got i64".to_string(),
            }),
        }
    }

    /// Attempts to extract as a 3D f32 array.
    pub fn try_into_array3_f32(self) -> Result<Array3<f32>, OCRError> {
        match self {
            TensorOutput::F32 { shape, data } => {
                if shape.len() != 3 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Expected 3D tensor, got {}D with shape {:?}",
                            shape.len(),
                            shape
                        ),
                    });
                }

                let dim0 = shape[0] as usize;
                let dim1 = shape[1] as usize;
                let dim2 = shape[2] as usize;
                let expected_len = dim0 * dim1 * dim2;

                if data.len() != expected_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Data length mismatch: expected {}, got {}",
                            expected_len,
                            data.len()
                        ),
                    });
                }

                Array3::from_shape_vec((dim0, dim1, dim2), data).map_err(OCRError::Tensor)
            }
            TensorOutput::I64 { .. } => Err(OCRError::InvalidInput {
                message: "Expected f32 tensor, got i64".to_string(),
            }),
        }
    }

    /// Attempts to extract as a 4D f32 array.
    pub fn try_into_array4_f32(self) -> Result<Array4<f32>, OCRError> {
        match self {
            TensorOutput::F32 { shape, data } => {
                if shape.len() != 4 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Expected 4D tensor, got {}D with shape {:?}",
                            shape.len(),
                            shape
                        ),
                    });
                }

                let dim0 = shape[0] as usize;
                let dim1 = shape[1] as usize;
                let dim2 = shape[2] as usize;
                let dim3 = shape[3] as usize;
                let expected_len = dim0 * dim1 * dim2 * dim3;

                if data.len() != expected_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Data length mismatch: expected {}, got {}",
                            expected_len,
                            data.len()
                        ),
                    });
                }

                Array4::from_shape_vec((dim0, dim1, dim2, dim3), data).map_err(OCRError::Tensor)
            }
            TensorOutput::I64 { .. } => Err(OCRError::InvalidInput {
                message: "Expected f32 tensor, got i64".to_string(),
            }),
        }
    }

    /// Attempts to extract as a dynamic-dimensional f32 array.
    pub fn try_into_array_f32(self) -> Result<ArrayD<f32>, OCRError> {
        match self {
            TensorOutput::F32 { shape, data } => {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                let expected_len: usize = dims.iter().product();

                if data.len() != expected_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Data length mismatch: expected {}, got {}",
                            expected_len,
                            data.len()
                        ),
                    });
                }

                ArrayD::from_shape_vec(dims, data).map_err(OCRError::Tensor)
            }
            TensorOutput::I64 { .. } => Err(OCRError::InvalidInput {
                message: "Expected f32 tensor, got i64".to_string(),
            }),
        }
    }

    /// Attempts to extract as a 2D i64 array.
    pub fn try_into_array2_i64(self) -> Result<Array2<i64>, OCRError> {
        match self {
            TensorOutput::I64 { shape, data } => {
                if shape.len() != 2 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Expected 2D tensor, got {}D with shape {:?}",
                            shape.len(),
                            shape
                        ),
                    });
                }

                let dim0 = shape[0] as usize;
                let dim1 = shape[1] as usize;
                let expected_len = dim0 * dim1;

                if data.len() != expected_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Data length mismatch: expected {}, got {}",
                            expected_len,
                            data.len()
                        ),
                    });
                }

                Array2::from_shape_vec((dim0, dim1), data).map_err(OCRError::Tensor)
            }
            TensorOutput::F32 { .. } => Err(OCRError::InvalidInput {
                message: "Expected i64 tensor, got f32".to_string(),
            }),
        }
    }

    /// Attempts to extract as a 3D i64 array.
    pub fn try_into_array3_i64(self) -> Result<Array3<i64>, OCRError> {
        match self {
            TensorOutput::I64 { shape, data } => {
                if shape.len() != 3 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Expected 3D tensor, got {}D with shape {:?}",
                            shape.len(),
                            shape
                        ),
                    });
                }

                let dim0 = shape[0] as usize;
                let dim1 = shape[1] as usize;
                let dim2 = shape[2] as usize;
                let expected_len = dim0 * dim1 * dim2;

                if data.len() != expected_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Data length mismatch: expected {}, got {}",
                            expected_len,
                            data.len()
                        ),
                    });
                }

                Array3::from_shape_vec((dim0, dim1, dim2), data).map_err(OCRError::Tensor)
            }
            TensorOutput::F32 { .. } => Err(OCRError::InvalidInput {
                message: "Expected i64 tensor, got f32".to_string(),
            }),
        }
    }
}
