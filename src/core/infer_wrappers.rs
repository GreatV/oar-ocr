//! Small helpers that wrap OrtInfer into concrete dimensional InferenceEngine implementations.

use crate::core::{
    InferenceEngine as GInferenceEngine, OCRError, OrtInfer, Tensor2D, Tensor3D, Tensor4D,
};

#[derive(Debug)]
pub struct OrtInfer2D(pub OrtInfer);
impl GInferenceEngine for OrtInfer2D {
    type Input = Tensor4D;
    type Output = Tensor2D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        self.0.infer_2d(input.clone())
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-2D".to_string()
    }
}

#[derive(Debug)]
pub struct OrtInfer3D(pub OrtInfer);
impl GInferenceEngine for OrtInfer3D {
    type Input = Tensor4D;
    type Output = Tensor3D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        self.0.infer_3d(input.clone())
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-3D".to_string()
    }
}

#[derive(Debug)]
pub struct OrtInfer4D(pub OrtInfer);
impl GInferenceEngine for OrtInfer4D {
    type Input = Tensor4D;
    type Output = Tensor4D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        self.0.infer_4d(input.clone())
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-4D".to_string()
    }
}
