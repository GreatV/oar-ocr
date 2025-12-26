use crate::core::OCRError;
use candle_core::{IndexOp, Tensor};

pub fn candle_to_ocr_inference(
    model_name: &str,
    context: impl Into<String>,
    err: candle_core::Error,
) -> OCRError {
    OCRError::Inference {
        model_name: model_name.to_string(),
        context: context.into(),
        source: Box::new(err),
    }
}

pub fn candle_to_ocr_processing(
    kind: crate::core::errors::ProcessingStage,
    context: impl Into<String>,
    err: candle_core::Error,
) -> OCRError {
    OCRError::Processing {
        kind,
        context: context.into(),
        source: Box::new(err),
    }
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor, OCRError> {
    let d = x.dim(candle_core::D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "rotate_half dim failed",
            e,
        )
    })?;
    let half = d / 2;
    let x1 = x.i((.., .., .., 0..half)).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "rotate_half slice x1 failed",
            e,
        )
    })?;
    let x2 = x.i((.., .., .., half..d)).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "rotate_half slice x2 failed",
            e,
        )
    })?;
    let nx2 = x2.neg().map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "rotate_half neg failed",
            e,
        )
    })?;
    Tensor::cat(&[&nx2, &x1], candle_core::D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "rotate_half cat failed",
            e,
        )
    })
}
