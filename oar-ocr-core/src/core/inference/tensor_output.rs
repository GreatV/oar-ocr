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

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(shape: &[i64], data: Vec<f32>) -> TensorOutput {
        TensorOutput::F32 {
            shape: shape.to_vec(),
            data,
        }
    }

    fn i64_tensor(shape: &[i64], data: Vec<i64>) -> TensorOutput {
        TensorOutput::I64 {
            shape: shape.to_vec(),
            data,
        }
    }

    // -- accessor tests --

    #[test]
    fn test_shape_and_ndim() {
        let t = f32_tensor(&[2, 3, 4], vec![0.0; 24]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
    }

    #[test]
    fn test_len_and_is_empty() {
        let t = f32_tensor(&[2, 3], vec![0.0; 6]);
        assert_eq!(t.len(), 6);
        assert!(!t.is_empty());

        let empty = f32_tensor(&[0, 3], vec![]);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    // -- try_into_array2_f32 --

    #[test]
    fn test_array2_f32_ok() {
        let t = f32_tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr = t.try_into_array2_f32().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_array2_f32_wrong_ndim() {
        let t = f32_tensor(&[2, 3, 4], vec![0.0; 24]);
        let err = t.try_into_array2_f32().unwrap_err().to_string();
        assert!(err.contains("Expected 2D"));
    }

    #[test]
    fn test_array2_f32_data_length_mismatch() {
        let t = f32_tensor(&[2, 3], vec![0.0; 7]);
        let err = t.try_into_array2_f32().unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"));
    }

    #[test]
    fn test_array2_f32_wrong_type() {
        let t = i64_tensor(&[2, 3], vec![0; 6]);
        let err = t.try_into_array2_f32().unwrap_err().to_string();
        assert!(err.contains("Expected f32"));
    }

    // -- try_into_array3_f32 --

    #[test]
    fn test_array3_f32_ok() {
        let t = f32_tensor(&[1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr = t.try_into_array3_f32().unwrap();
        assert_eq!(arr.shape(), &[1, 2, 3]);
        assert_eq!(arr[[0, 1, 2]], 6.0);
    }

    #[test]
    fn test_array3_f32_wrong_ndim() {
        let t = f32_tensor(&[2, 3], vec![0.0; 6]);
        let err = t.try_into_array3_f32().unwrap_err().to_string();
        assert!(err.contains("Expected 3D"));
    }

    #[test]
    fn test_array3_f32_data_length_mismatch() {
        let t = f32_tensor(&[1, 2, 3], vec![0.0; 5]);
        let err = t.try_into_array3_f32().unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"));
    }

    // -- try_into_array4_f32 --

    #[test]
    fn test_array4_f32_ok() {
        let t = f32_tensor(&[1, 2, 3, 4], vec![0.0; 24]);
        let arr = t.try_into_array4_f32().unwrap();
        assert_eq!(arr.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_array4_f32_wrong_ndim() {
        let t = f32_tensor(&[2, 3], vec![0.0; 6]);
        let err = t.try_into_array4_f32().unwrap_err().to_string();
        assert!(err.contains("Expected 4D"));
    }

    #[test]
    fn test_array4_f32_data_length_mismatch() {
        let t = f32_tensor(&[1, 2, 3, 4], vec![0.0; 25]);
        let err = t.try_into_array4_f32().unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"));
    }

    // -- try_into_array_f32 (dynamic) --

    #[test]
    fn test_array_f32_ok_various_dims() {
        let t1 = f32_tensor(&[6], vec![0.0; 6]);
        let arr1 = t1.try_into_array_f32().unwrap();
        assert_eq!(arr1.shape(), &[6]);

        let t5 = f32_tensor(&[1, 2, 1, 3, 1], vec![0.0; 6]);
        let arr5 = t5.try_into_array_f32().unwrap();
        assert_eq!(arr5.shape(), &[1, 2, 1, 3, 1]);
    }

    #[test]
    fn test_array_f32_data_length_mismatch() {
        let t = f32_tensor(&[2, 3], vec![0.0; 5]);
        let err = t.try_into_array_f32().unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"));
    }

    #[test]
    fn test_array_f32_wrong_type() {
        let t = i64_tensor(&[2, 3], vec![0; 6]);
        let err = t.try_into_array_f32().unwrap_err().to_string();
        assert!(err.contains("Expected f32"));
    }

    // -- try_into_array2_i64 --

    #[test]
    fn test_array2_i64_ok() {
        let t = i64_tensor(&[2, 3], vec![10, 20, 30, 40, 50, 60]);
        let arr = t.try_into_array2_i64().unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 1]], 20);
    }

    #[test]
    fn test_array2_i64_wrong_ndim() {
        let t = i64_tensor(&[2, 3, 4], vec![0; 24]);
        let err = t.try_into_array2_i64().unwrap_err().to_string();
        assert!(err.contains("Expected 2D"));
    }

    #[test]
    fn test_array2_i64_data_length_mismatch() {
        let t = i64_tensor(&[2, 3], vec![0; 7]);
        let err = t.try_into_array2_i64().unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"));
    }

    #[test]
    fn test_array2_i64_wrong_type() {
        let t = f32_tensor(&[2, 3], vec![0.0; 6]);
        let err = t.try_into_array2_i64().unwrap_err().to_string();
        assert!(err.contains("Expected i64"));
    }

    // -- try_into_array3_i64 --

    #[test]
    fn test_array3_i64_ok() {
        let t = i64_tensor(&[1, 2, 3], vec![1, 2, 3, 4, 5, 6]);
        let arr = t.try_into_array3_i64().unwrap();
        assert_eq!(arr.shape(), &[1, 2, 3]);
        assert_eq!(arr[[0, 0, 2]], 3);
    }

    #[test]
    fn test_array3_i64_wrong_ndim() {
        let t = i64_tensor(&[6], vec![0; 6]);
        let err = t.try_into_array3_i64().unwrap_err().to_string();
        assert!(err.contains("Expected 3D"));
    }

    #[test]
    fn test_array3_i64_data_length_mismatch() {
        let t = i64_tensor(&[1, 2, 3], vec![0; 7]);
        let err = t.try_into_array3_i64().unwrap_err().to_string();
        assert!(err.contains("Data length mismatch"));
    }

    #[test]
    fn test_array3_i64_wrong_type() {
        let t = f32_tensor(&[1, 2, 3], vec![0.0; 6]);
        let err = t.try_into_array3_i64().unwrap_err().to_string();
        assert!(err.contains("Expected i64"));
    }

    // -- scalar / edge cases --

    #[test]
    fn test_scalar_f32_as_array2() {
        let t = f32_tensor(&[1, 1], vec![42.0]);
        let arr = t.try_into_array2_f32().unwrap();
        assert_eq!(arr[[0, 0]], 42.0);
    }

    #[test]
    fn test_empty_tensor_array2() {
        let t = f32_tensor(&[0, 5], vec![]);
        let arr = t.try_into_array2_f32().unwrap();
        assert_eq!(arr.shape(), &[0, 5]);
        assert!(arr.is_empty());
    }
}
