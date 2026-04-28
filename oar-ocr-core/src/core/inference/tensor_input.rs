/// Represents a tensor input of various dimensions for generic inference.
///
/// This enum allows passing tensors of different dimensions to an inference
/// backend without hardcoding model-specific input semantics.
///
/// `TensorInput` and `TensorOutput` live at the `core::inference` layer because
/// they are the backend-neutral boundary types shared by ONNX Runtime and RKNN.
#[derive(Debug)]
pub enum TensorInput<'a> {
    /// A 2D tensor reference (e.g., scale_factor, im_shape).
    Array2(&'a ndarray::Array2<f32>),
    /// A 3D tensor reference.
    Array3(&'a ndarray::Array3<f32>),
    /// A 4D tensor reference (e.g., image batch).
    Array4(&'a ndarray::Array4<f32>),
}

impl TensorInput<'_> {
    #[cfg(feature = "ort")]
    pub(crate) fn shape(&self) -> Vec<usize> {
        match self {
            TensorInput::Array2(arr) => arr.shape().to_vec(),
            TensorInput::Array3(arr) => arr.shape().to_vec(),
            TensorInput::Array4(arr) => arr.shape().to_vec(),
        }
    }
}
