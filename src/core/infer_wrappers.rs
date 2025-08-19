//! Small helpers that wrap OrtInfer into concrete dimensional InferenceEngine implementations.

use crate::core::{
    InferenceEngine as GInferenceEngine, OCRError, OrtInfer, Tensor2D, Tensor3D, Tensor4D,
};

#[derive(Debug)]
pub struct OrtInfer2D(OrtInfer);

impl OrtInfer2D {
    /// Creates a new OrtInfer2D wrapper around an OrtInfer instance.
    pub fn new(inner: OrtInfer) -> Self {
        Self(inner)
    }

    /// Returns a reference to the inner OrtInfer instance.
    pub fn inner(&self) -> &OrtInfer {
        &self.0
    }

    /// Returns a mutable reference to the inner OrtInfer instance.
    pub fn inner_mut(&mut self) -> &mut OrtInfer {
        &mut self.0
    }

    /// Consumes the wrapper and returns the inner OrtInfer instance.
    pub fn into_inner(self) -> OrtInfer {
        self.0
    }
}

impl From<OrtInfer> for OrtInfer2D {
    fn from(inner: OrtInfer) -> Self {
        Self::new(inner)
    }
}

impl GInferenceEngine for OrtInfer2D {
    type Input = Tensor4D;
    type Output = Tensor2D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        // Performance improvement: Pass reference instead of cloning the tensor
        self.0.infer_2d(input)
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-2D".to_string()
    }
}

#[derive(Debug)]
pub struct OrtInfer3D(OrtInfer);

impl OrtInfer3D {
    /// Creates a new OrtInfer3D wrapper around an OrtInfer instance.
    pub fn new(inner: OrtInfer) -> Self {
        Self(inner)
    }

    /// Returns a reference to the inner OrtInfer instance.
    pub fn inner(&self) -> &OrtInfer {
        &self.0
    }

    /// Returns a mutable reference to the inner OrtInfer instance.
    pub fn inner_mut(&mut self) -> &mut OrtInfer {
        &mut self.0
    }

    /// Consumes the wrapper and returns the inner OrtInfer instance.
    pub fn into_inner(self) -> OrtInfer {
        self.0
    }
}

impl From<OrtInfer> for OrtInfer3D {
    fn from(inner: OrtInfer) -> Self {
        Self::new(inner)
    }
}

impl GInferenceEngine for OrtInfer3D {
    type Input = Tensor4D;
    type Output = Tensor3D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        // Performance improvement: Pass reference instead of cloning the tensor
        self.0.infer_3d(input)
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-3D".to_string()
    }
}

#[derive(Debug)]
pub struct OrtInfer4D(OrtInfer);

impl OrtInfer4D {
    /// Creates a new OrtInfer4D wrapper around an OrtInfer instance.
    pub fn new(inner: OrtInfer) -> Self {
        Self(inner)
    }

    /// Returns a reference to the inner OrtInfer instance.
    pub fn inner(&self) -> &OrtInfer {
        &self.0
    }

    /// Returns a mutable reference to the inner OrtInfer instance.
    pub fn inner_mut(&mut self) -> &mut OrtInfer {
        &mut self.0
    }

    /// Consumes the wrapper and returns the inner OrtInfer instance.
    pub fn into_inner(self) -> OrtInfer {
        self.0
    }
}

impl From<OrtInfer> for OrtInfer4D {
    fn from(inner: OrtInfer) -> Self {
        Self::new(inner)
    }
}

impl GInferenceEngine for OrtInfer4D {
    type Input = Tensor4D;
    type Output = Tensor4D;
    fn infer(&self, input: &Self::Input) -> Result<Self::Output, OCRError> {
        // Performance improvement: Pass reference instead of cloning the tensor
        self.0.infer_4d(input)
    }
    fn engine_info(&self) -> String {
        "ONNXRuntime-4D".to_string()
    }
}
