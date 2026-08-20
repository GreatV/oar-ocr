//! Shared tensor and rotary-embedding helpers.

use crate::api::error::{Error, ProcessingStage};
use crate::runtime::errors::candle_to_ocr_processing;
use candle_core::{D, Device, Tensor};

pub fn vision_inv_freq(
    dim: usize,
    theta: f64,
    model_name: &str,
    device: &Device,
) -> Result<Tensor, Error> {
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|index| (1f64 / theta.powf(index as f64 / dim as f64)) as f32)
        .collect();
    Tensor::from_vec(inv_freq, (dim / 2,), device).map_err(|error| {
        candle_to_ocr_processing(
            ProcessingStage::TensorOperation,
            format!("{model_name}: vision inv_freq tensor failed"),
            error,
        )
    })
}

pub fn rotate_half(tensor: &Tensor) -> Result<Tensor, Error> {
    let dimension = tensor.dim(D::Minus1).map_err(|error| {
        candle_to_ocr_processing(ProcessingStage::TensorOperation, "rotate_half dim", error)
    })?;
    let half = dimension / 2;
    let first = tensor.narrow(D::Minus1, 0, half).map_err(|error| {
        candle_to_ocr_processing(ProcessingStage::TensorOperation, "rotate_half first", error)
    })?;
    let second = tensor
        .narrow(D::Minus1, half, dimension - half)
        .map_err(|error| {
            candle_to_ocr_processing(
                ProcessingStage::TensorOperation,
                "rotate_half second",
                error,
            )
        })?;
    let negative_second = second.neg().map_err(|error| {
        candle_to_ocr_processing(
            ProcessingStage::TensorOperation,
            "rotate_half negate",
            error,
        )
    })?;
    Tensor::cat(&[&negative_second, &first], D::Minus1).map_err(|error| {
        candle_to_ocr_processing(
            ProcessingStage::TensorOperation,
            "rotate_half concatenate",
            error,
        )
    })
}
