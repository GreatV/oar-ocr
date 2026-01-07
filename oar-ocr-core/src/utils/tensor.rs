//! Tensor utility functions for converting between vectors and tensors.
//!
//! This module provides functions to convert between Rust vectors and
//! multi-dimensional tensors (1D, 2D, 3D, and 4D). It also includes
//! utility functions for tensor operations like slicing and stacking.

use crate::core::OCRError;
use crate::core::Tensor1D;
use crate::core::Tensor2D;
use crate::core::Tensor3D;
use crate::core::Tensor4D;

use ndarray::{Array2, Array3, Array4, ArrayD, Axis};

/// Converts a 2D vector of f32 values into a 2D tensor.
///
/// # Arguments
///
/// * `data` - A slice of vectors containing f32 values.
///
/// # Returns
///
/// * `Ok(Tensor2D)` - A 2D tensor created from the input data.
/// * `Err(OCRError)` - An error if the input data is invalid (e.g., empty, inconsistent row lengths).
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::vec_to_tensor2d;
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let tensor = vec_to_tensor2d(&data);
/// ```
pub fn vec_to_tensor2d(data: &[Vec<f32>]) -> Result<Tensor2D, OCRError> {
    if data.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "Empty data".to_string(),
        });
    }

    let rows = data.len();
    let cols = data[0].len();

    if cols == 0 {
        return Err(OCRError::InvalidInput {
            message: "Cannot create tensor with zero-width columns".to_string(),
        });
    }

    let total_size = rows
        .checked_mul(cols)
        .ok_or_else(|| OCRError::InvalidInput {
            message: format!("Tensor dimensions ({rows}, {cols}) would cause integer overflow"),
        })?;

    const MAX_TENSOR_ELEMENTS: usize = 1_000_000_000;
    if total_size > MAX_TENSOR_ELEMENTS {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Tensor size {total_size} exceeds maximum allowed size {MAX_TENSOR_ELEMENTS}"
            ),
        });
    }

    for (i, row) in data.iter().enumerate() {
        if row.len() != cols {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Inconsistent row lengths at row {}: expected {}, got {}",
                    i,
                    cols,
                    row.len()
                ),
            });
        }
    }

    let flat_data: Vec<f32> = data.iter().flat_map(|row| row.iter().cloned()).collect();
    let flat_data_len = flat_data.len();
    Array2::from_shape_vec((rows, cols), flat_data).map_err(|e| {
        OCRError::tensor_operation_error(
            "vec_to_tensor2d",
            &[rows, cols],
            &[flat_data_len],
            &format!(
                "Failed to create 2D tensor from {} rows x {} cols data",
                rows, cols
            ),
            e,
        )
    })
}

/// Converts a 2D tensor into a 2D vector of f32 values.
///
/// # Arguments
///
/// * `tensor` - A reference to a 2D tensor.
///
/// # Returns
///
/// * `Vec<Vec<f32>>` - A 2D vector created from the tensor.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{tensor2d_to_vec, vec_to_tensor2d};
/// // Create a tensor first
/// let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let tensor = vec_to_tensor2d(&data).unwrap();
/// let vec_data = tensor2d_to_vec(&tensor);
/// ```
pub fn tensor2d_to_vec(tensor: &Tensor2D) -> Vec<Vec<f32>> {
    tensor.outer_iter().map(|row| row.to_vec()).collect()
}

/// Converts a 3D vector of f32 values into a 3D tensor.
///
/// # Arguments
///
/// * `data` - A slice of 3D vectors containing f32 values.
///
/// # Returns
///
/// * `Ok(Tensor3D)` - A 3D tensor created from the input data.
/// * `Err(OCRError)` - An error if the input data is invalid (e.g., empty, inconsistent dimensions).
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::vec_to_tensor3d;
/// let data = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
/// let tensor = vec_to_tensor3d(&data);
/// ```
pub fn vec_to_tensor3d(data: &[Vec<Vec<f32>>]) -> Result<Tensor3D, OCRError> {
    if data.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "Empty data".to_string(),
        });
    }

    let dim0 = data.len();
    let dim1 = data[0].len();

    let dim2 = if dim1 > 0 && !data[0].is_empty() {
        data[0][0].len()
    } else {
        0
    };

    for (i, outer) in data.iter().enumerate() {
        if outer.len() != dim1 {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Inconsistent dimension 1 at index {}: expected {}, got {}",
                    i,
                    dim1,
                    outer.len()
                ),
            });
        }
        for (j, inner) in outer.iter().enumerate() {
            if inner.len() != dim2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Inconsistent dimension 2 at index [{}, {}]: expected {}, got {}",
                        i,
                        j,
                        dim2,
                        inner.len()
                    ),
                });
            }
        }
    }

    let total_size = dim0
        .checked_mul(dim1)
        .and_then(|size| size.checked_mul(dim2))
        .ok_or_else(|| OCRError::InvalidInput {
            message: format!(
                "Tensor dimensions ({dim0}, {dim1}, {dim2}) would cause integer overflow"
            ),
        })?;

    const MAX_TENSOR_ELEMENTS: usize = 1_000_000_000;
    if total_size > MAX_TENSOR_ELEMENTS {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Tensor size {total_size} exceeds maximum allowed size {MAX_TENSOR_ELEMENTS}"
            ),
        });
    }

    let flat_data: Vec<f32> = data
        .iter()
        .flat_map(|slice| slice.iter().flat_map(|row| row.iter().cloned()))
        .collect();
    let flat_data_len = flat_data.len();

    Array3::from_shape_vec((dim0, dim1, dim2), flat_data).map_err(|e| {
        OCRError::tensor_operation_error(
            "vec_to_tensor3d",
            &[dim0, dim1, dim2],
            &[flat_data_len],
            &format!(
                "Failed to create 3D tensor from {}x{}x{} data",
                dim0, dim1, dim2
            ),
            e,
        )
    })
}

/// Converts a 3D tensor into a 3D vector of f32 values.
///
/// # Arguments
///
/// * `tensor` - A reference to a 3D tensor.
///
/// # Returns
///
/// * `Vec<Vec<Vec<f32>>>` - A 3D vector created from the tensor.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{tensor3d_to_vec, vec_to_tensor3d};
/// // Create a tensor first
/// let data = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
/// let tensor = vec_to_tensor3d(&data).unwrap();
/// let vec_data = tensor3d_to_vec(&tensor);
/// ```
pub fn tensor3d_to_vec(tensor: &Tensor3D) -> Vec<Vec<Vec<f32>>> {
    tensor
        .outer_iter()
        .map(|slice| slice.outer_iter().map(|row| row.to_vec()).collect())
        .collect()
}

/// Converts a 4D vector of f32 values into a 4D tensor.
///
/// # Arguments
///
/// * `data` - A slice of 4D vectors containing f32 values.
///
/// # Returns
///
/// * `Ok(Tensor4D)` - A 4D tensor created from the input data.
/// * `Err(OCRError)` - An error if the input data is invalid (e.g., empty, inconsistent dimensions).
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::vec_to_tensor4d;
/// let data = vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]];
/// let tensor = vec_to_tensor4d(&data);
/// ```
pub fn vec_to_tensor4d(data: &[Vec<Vec<Vec<f32>>>]) -> Result<Tensor4D, OCRError> {
    if data.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "Empty data".to_string(),
        });
    }

    let dim0 = data.len();
    let dim1 = data[0].len();

    let dim2 = if dim1 > 0 && !data[0].is_empty() {
        data[0][0].len()
    } else {
        0
    };

    let dim3 = if dim2 > 0 && !data[0].is_empty() && !data[0][0].is_empty() {
        data[0][0][0].len()
    } else {
        0
    };

    for (i, outer) in data.iter().enumerate() {
        if outer.len() != dim1 {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Inconsistent dimension 1 at index {}: expected {}, got {}",
                    i,
                    dim1,
                    outer.len()
                ),
            });
        }
        for (j, middle) in outer.iter().enumerate() {
            if middle.len() != dim2 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Inconsistent dimension 2 at index [{}, {}]: expected {}, got {}",
                        i,
                        j,
                        dim2,
                        middle.len()
                    ),
                });
            }
            for (k, inner) in middle.iter().enumerate() {
                if inner.len() != dim3 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Inconsistent dimension 3 at index [{}, {}, {}]: expected {}, got {}",
                            i,
                            j,
                            k,
                            dim3,
                            inner.len()
                        ),
                    });
                }
            }
        }
    }

    let total_size = dim0
        .checked_mul(dim1)
        .and_then(|size| size.checked_mul(dim2))
        .and_then(|size| size.checked_mul(dim3))
        .ok_or_else(|| OCRError::InvalidInput {
            message: format!(
                "Tensor dimensions ({dim0}, {dim1}, {dim2}, {dim3}) would cause integer overflow"
            ),
        })?;

    const MAX_TENSOR_ELEMENTS: usize = 1_000_000_000;
    if total_size > MAX_TENSOR_ELEMENTS {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Tensor size {total_size} exceeds maximum allowed size {MAX_TENSOR_ELEMENTS}"
            ),
        });
    }

    Ok(Array4::from_shape_fn(
        (dim0, dim1, dim2, dim3),
        |(i, j, k, l)| data[i][j][k][l],
    ))
}

/// Converts a 4D tensor into a 4D vector of f32 values.
///
/// # Arguments
///
/// * `tensor` - A reference to a 4D tensor.
///
/// # Returns
///
/// * `Vec<Vec<Vec<Vec<f32>>>>` - A 4D vector created from the tensor.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{tensor4d_to_vec, vec_to_tensor4d};
/// // Create a tensor first
/// let data = vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]];
/// let tensor = vec_to_tensor4d(&data).unwrap();
/// let vec_data = tensor4d_to_vec(&tensor);
/// ```
pub fn tensor4d_to_vec(tensor: &Tensor4D) -> Vec<Vec<Vec<Vec<f32>>>> {
    tensor
        .outer_iter()
        .map(|batch| {
            batch
                .outer_iter()
                .map(|slice| slice.outer_iter().map(|row| row.to_vec()).collect())
                .collect()
        })
        .collect()
}

/// Converts a 1D vector of f32 values into a 1D tensor with the specified shape.
///
/// # Arguments
///
/// * `data` - A vector of f32 values.
/// * `shape` - A slice of usize values representing the shape of the tensor.
///
/// # Returns
///
/// * `Ok(Tensor1D)` - A 1D tensor created from the input data.
/// * `Err(OCRError)` - An error if the input data is invalid.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::vec_to_tensor1d;
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let shape = &[2, 2];
/// let tensor = vec_to_tensor1d(data, shape);
/// ```
pub fn vec_to_tensor1d(data: Vec<f32>, shape: &[usize]) -> Result<Tensor1D, OCRError> {
    let data_len = data.len();
    ArrayD::from_shape_vec(shape, data).map_err(|e| {
        OCRError::tensor_operation_error(
            "vec_to_tensor1d",
            shape,
            &[data_len],
            &format!(
                "Failed to create 1D tensor with shape {:?} from {} elements",
                shape, data_len
            ),
            e,
        )
    })
}

/// Converts a 1D tensor into a 1D vector of f32 values.
///
/// # Arguments
///
/// * `tensor` - A reference to a 1D tensor.
///
/// # Returns
///
/// * `Vec<f32>` - A 1D vector created from the tensor.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{tensor1d_to_vec, vec_to_tensor1d};
/// // Create a tensor first
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let shape = &[4];
/// let tensor = vec_to_tensor1d(data, shape).unwrap();
/// let vec_data = tensor1d_to_vec(&tensor);
/// ```
pub fn tensor1d_to_vec(tensor: &Tensor1D) -> Vec<f32> {
    tensor.as_slice().unwrap_or(&[]).to_vec()
}

/// Extracts a 3D slice from a 4D tensor at the specified index.
///
/// # Arguments
///
/// * `tensor` - A reference to a 4D tensor.
/// * `index` - The index of the slice to extract.
///
/// # Returns
///
/// * `Ok(Tensor3D)` - A 3D tensor slice extracted from the input tensor.
/// * `Err(OCRError)` - An error if the index is out of bounds.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{tensor4d_slice, vec_to_tensor4d};
/// // Create a tensor first
/// let data = vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]];
/// let tensor = vec_to_tensor4d(&data).unwrap();
/// let slice = tensor4d_slice(&tensor, 0);
/// ```
pub fn tensor4d_slice(tensor: &Tensor4D, index: usize) -> Result<Tensor3D, OCRError> {
    if index >= tensor.shape()[0] {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Index {} out of bounds for tensor with shape {:?}",
                index,
                tensor.shape()
            ),
        });
    }
    Ok(tensor.index_axis(Axis(0), index).to_owned())
}

/// Extracts a 2D slice from a 3D tensor at the specified index.
///
/// # Arguments
///
/// * `tensor` - A reference to a 3D tensor.
/// * `index` - The index of the slice to extract.
///
/// # Returns
///
/// * `Ok(Tensor2D)` - A 2D tensor slice extracted from the input tensor.
/// * `Err(OCRError)` - An error if the index is out of bounds.
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{tensor3d_slice, vec_to_tensor3d};
/// // Create a tensor first
/// let data = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
/// let tensor = vec_to_tensor3d(&data).unwrap();
/// let slice = tensor3d_slice(&tensor, 0);
/// ```
pub fn tensor3d_slice(tensor: &Tensor3D, index: usize) -> Result<Tensor2D, OCRError> {
    if index >= tensor.shape()[0] {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Index {} out of bounds for tensor with shape {:?}",
                index,
                tensor.shape()
            ),
        });
    }
    Ok(tensor.index_axis(Axis(0), index).to_owned())
}

/// Stacks a slice of 3D tensors into a single 4D tensor.
///
/// # Arguments
///
/// * `tensors` - A slice of 3D tensors to stack.
///
/// # Returns
///
/// * `Ok(Tensor4D)` - A 4D tensor created by stacking the input tensors.
/// * `Err(OCRError)` - An error if the input tensors are invalid (e.g., empty, inconsistent shapes).
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{stack_tensor3d, vec_to_tensor3d};
/// // Create tensors first
/// let data1 = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
/// let data2 = vec![vec![vec![5.0, 6.0], vec![7.0, 8.0]]];
/// let tensor1 = vec_to_tensor3d(&data1).unwrap();
/// let tensor2 = vec_to_tensor3d(&data2).unwrap();
/// let tensors = vec![tensor1, tensor2];
/// let stacked_tensor = stack_tensor3d(&tensors);
/// ```
pub fn stack_tensor3d(tensors: &[Tensor3D]) -> Result<Tensor4D, OCRError> {
    if tensors.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "No tensors to stack".to_string(),
        });
    }

    let first_shape = tensors[0].shape();

    if first_shape.contains(&0) {
        return Err(OCRError::InvalidInput {
            message: format!("Cannot stack tensors with zero dimensions: shape {first_shape:?}"),
        });
    }

    for (i, tensor) in tensors.iter().enumerate() {
        if tensor.is_empty() {
            return Err(OCRError::InvalidInput {
                message: format!("Tensor {i} is empty and cannot be stacked"),
            });
        }

        if i > 0 && tensor.shape() != first_shape {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Tensor shape mismatch during 3D tensor stacking at index {}: expected {:?}, got {:?}",
                    i,
                    first_shape,
                    tensor.shape()
                ),
            });
        }
    }

    let result_size = tensors
        .len()
        .checked_mul(first_shape.iter().product::<usize>())
        .ok_or_else(|| OCRError::InvalidInput {
            message: format!(
                "Stacking {} tensors of shape {:?} would cause integer overflow",
                tensors.len(),
                first_shape
            ),
        })?;

    const MAX_TENSOR_ELEMENTS: usize = 1_000_000_000;
    if result_size > MAX_TENSOR_ELEMENTS {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Stacked tensor size {result_size} exceeds maximum allowed size {MAX_TENSOR_ELEMENTS}"
            ),
        });
    }

    let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();

    ndarray::stack(Axis(0), &views).map_err(|e| {
        OCRError::tensor_operation_error(
            "stack_tensor3d",
            &[
                tensors.len(),
                first_shape[0],
                first_shape[1],
                first_shape[2],
            ],
            &[result_size],
            &format!(
                "Failed to stack {} 3D tensors of shape {:?}",
                tensors.len(),
                first_shape
            ),
            e,
        )
    })
}

/// Stacks a slice of 2D tensors into a single 3D tensor.
///
/// # Arguments
///
/// * `tensors` - A slice of 2D tensors to stack.
///
/// # Returns
///
/// * `Ok(Tensor3D)` - A 3D tensor created by stacking the input tensors.
/// * `Err(OCRError)` - An error if the input tensors are invalid (e.g., empty, inconsistent shapes).
///
/// # Examples
///
/// ```
/// use oar_ocr_core::utils::tensor::{stack_tensor2d, vec_to_tensor2d};
/// // Create tensors first
/// let data1 = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let data2 = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
/// let tensor1 = vec_to_tensor2d(&data1).unwrap();
/// let tensor2 = vec_to_tensor2d(&data2).unwrap();
/// let tensors = vec![tensor1, tensor2];
/// let stacked_tensor = stack_tensor2d(&tensors);
/// ```
pub fn stack_tensor2d(tensors: &[Tensor2D]) -> Result<Tensor3D, OCRError> {
    if tensors.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "No tensors to stack".to_string(),
        });
    }

    let first_shape = tensors[0].shape();

    if first_shape.contains(&0) {
        return Err(OCRError::InvalidInput {
            message: format!("Cannot stack tensors with zero dimensions: shape {first_shape:?}"),
        });
    }

    for (i, tensor) in tensors.iter().enumerate() {
        if tensor.is_empty() {
            return Err(OCRError::InvalidInput {
                message: format!("Tensor {i} is empty and cannot be stacked"),
            });
        }

        if i > 0 && tensor.shape() != first_shape {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "All tensors must have the same shape for stacking. Tensor 0 has shape {:?}, tensor {} has shape {:?}",
                    first_shape,
                    i,
                    tensor.shape()
                ),
            });
        }
    }

    let result_size = tensors
        .len()
        .checked_mul(first_shape.iter().product::<usize>())
        .ok_or_else(|| OCRError::InvalidInput {
            message: format!(
                "Stacking {} tensors of shape {:?} would cause integer overflow",
                tensors.len(),
                first_shape
            ),
        })?;

    const MAX_TENSOR_ELEMENTS: usize = 1_000_000_000;
    if result_size > MAX_TENSOR_ELEMENTS {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Stacked tensor size {result_size} exceeds maximum allowed size {MAX_TENSOR_ELEMENTS}"
            ),
        });
    }

    let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();

    ndarray::stack(Axis(0), &views).map_err(OCRError::Tensor)
}
