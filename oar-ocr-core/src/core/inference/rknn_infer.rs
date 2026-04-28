//! `InferenceBackend` implementation backed by the RKNN runtime (RK3588 NPU).
//!
//! Active only on aarch64 + `feature = "rknpu"`. On any other configuration
//! this file expands to nothing so the parent `mod` declaration stays cheap.

#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
use std::borrow::Cow;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use std::path::{Path, PathBuf};
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use std::sync::Mutex;

#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
use crate::core::config::RknnInputMode;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use crate::core::config::{ConfigValidator, RknnCoreMaskConfig, RknnSessionConfig};
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use crate::core::errors::OCRError;
#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
use crate::core::inference::TensorInput;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use crate::core::inference::TensorOutput;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use crate::core::inference::backend::InferenceBackend;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
use crate::core::inference::rknn::{
    RknnContext, RknnCoreMask, RknnInput, RknnTensorAttr, RknnTensorFormat, RknnTensorType,
};

#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
#[derive(Debug)]
struct PreparedInput<'a> {
    bytes: Cow<'a, [u8]>,
    fmt: PreparedTensorFormat,
    pass_through: bool,
}

#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedTensorFormat {
    Nchw,
    Nhwc,
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
impl From<PreparedTensorFormat> for RknnTensorFormat {
    fn from(value: PreparedTensorFormat) -> Self {
        match value {
            PreparedTensorFormat::Nchw => Self::Nchw,
            PreparedTensorFormat::Nhwc => Self::Nhwc,
        }
    }
}

/// RKNN-backed inference engine.
///
/// Holds a pool of duplicated `RknnContext`s plus cached input/output tensor
/// attributes. Each context is protected by a `Mutex`, while calls are
/// distributed round-robin across the pool.
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
pub struct RknnInfer {
    contexts: Vec<Mutex<RknnContext>>,
    next_idx: std::sync::atomic::AtomicUsize,
    input_attrs: Vec<RknnTensorAttr>,
    output_attrs: Vec<RknnTensorAttr>,
    input_name: String,
    model_path: PathBuf,
    model_name: String,
    input_mode: RknnInputMode,
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
impl std::fmt::Debug for RknnInfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RknnInfer")
            .field("input_name", &self.input_name)
            .field("contexts", &self.contexts.len())
            .field("inputs", &self.input_attrs.len())
            .field("outputs", &self.output_attrs.len())
            .field("model_path", &self.model_path)
            .field("model_name", &self.model_name)
            .field("input_mode", &self.input_mode)
            .finish()
    }
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
impl RknnInfer {
    /// Build an `RknnInfer` from a `.rknn` model file.
    pub fn from_file(
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
        model_name: Option<String>,
        rknn_config: Option<&RknnSessionConfig>,
    ) -> Result<Self, OCRError> {
        if let Some(cfg) = rknn_config {
            cfg.validate()?;
        }

        let model_path = model_path.as_ref().to_path_buf();
        let mut ctx = RknnContext::from_file(&model_path)?;

        let (n_in, n_out) = ctx.input_output_num()?;
        let input_attrs: Vec<RknnTensorAttr> = (0..n_in)
            .map(|i| ctx.input_attr(i))
            .collect::<Result<_, _>>()?;
        let output_attrs: Vec<RknnTensorAttr> = (0..n_out)
            .map(|i| ctx.output_attr(i))
            .collect::<Result<_, _>>()?;

        // Pick a default input name: caller override > first input's reported
        // name > "x" (matches OrtInfer's fallback).
        let resolved_input_name = input_name
            .map(str::to_owned)
            .or_else(|| input_attrs.first().map(|a| a.name.clone()))
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "x".to_string());

        let model_name = model_name.unwrap_or_else(|| {
            model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("rknn_model")
                .to_string()
        });

        let num_contexts = rknn_config
            .map(RknnSessionConfig::effective_num_contexts)
            .unwrap_or(1);
        let input_mode = rknn_config.map(|cfg| cfg.input_mode).unwrap_or_default();
        if input_mode == RknnInputMode::NormalizedNchwPassThrough
            && input_attrs
                .first()
                .map(|attr| attr.fmt != RknnTensorFormat::Nchw)
                .unwrap_or(false)
        {
            let attr = input_attrs.first().expect("checked above");
            return Err(OCRError::ConfigError {
                message: format!(
                    "RKNN model '{}' primary input '{}' reports format {:?}; \
                     NormalizedNchwPassThrough requires a model converted for NCHW input",
                    model_name, attr.name, attr.fmt
                ),
            });
        }

        if let Some(mask) = rknn_config.and_then(|cfg| cfg.core_mask_for_context(0)) {
            ctx.set_core_mask(mask.into())?;
        }

        let mut contexts = Vec::with_capacity(num_contexts);
        contexts.push(Mutex::new(ctx));

        for idx in 1..num_contexts {
            let mut duplicated = {
                let first = contexts.first_mut().expect("primary context exists");
                let primary = first
                    .get_mut()
                    .expect("primary RKNN context mutex is not shared during pool construction");
                primary.duplicate()?
            };
            if let Some(mask) = rknn_config.and_then(|cfg| cfg.core_mask_for_context(idx)) {
                duplicated.set_core_mask(mask.into())?;
            }
            contexts.push(Mutex::new(duplicated));
        }

        Ok(Self {
            contexts,
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_attrs,
            output_attrs,
            input_name: resolved_input_name,
            model_path,
            model_name,
            input_mode,
        })
    }

    fn input_index_for(&self, name: &str) -> Option<u32> {
        // Fast path: a single-input model never needs a name lookup.
        if self.input_attrs.len() == 1 {
            return Some(0);
        }
        self.input_attrs
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.index)
    }
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
impl From<RknnCoreMaskConfig> for RknnCoreMask {
    fn from(value: RknnCoreMaskConfig) -> Self {
        match value {
            RknnCoreMaskConfig::Auto => Self::Auto,
            RknnCoreMaskConfig::Core0 => Self::Core0,
            RknnCoreMaskConfig::Core1 => Self::Core1,
            RknnCoreMaskConfig::Core2 => Self::Core2,
            RknnCoreMaskConfig::Core01 => Self::Core01,
            RknnCoreMaskConfig::Core012 => Self::Core012,
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
impl InferenceBackend for RknnInfer {
    fn model_path(&self) -> &Path {
        &self.model_path
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn input_name(&self) -> &str {
        &self.input_name
    }

    fn input_names_from_model(&self) -> Vec<String> {
        self.input_attrs.iter().map(|a| a.name.clone()).collect()
    }

    fn primary_input_shape(&self) -> Option<Vec<i64>> {
        self.input_attrs
            .first()
            .map(|a| a.dims.iter().map(|&d| d as i64).collect())
    }

    fn infer(
        &self,
        inputs: &[(&str, TensorInput<'_>)],
    ) -> Result<Vec<(String, TensorOutput)>, OCRError> {
        if inputs.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No inputs provided for RKNN inference".to_string(),
            });
        }

        // ToolkitLayout transposes 4-D OCR inputs to NHWC for librknnrt's
        // input-processing path. NormalizedNchwPassThrough skips that copy and
        // expects a model converted to consume normalized NCHW input directly.
        let mut input_buffers: Vec<Cow<'_, [u8]>> = Vec::with_capacity(inputs.len());
        let mut indices: Vec<u32> = Vec::with_capacity(inputs.len());
        let mut input_meta: Vec<(RknnTensorFormat, bool)> = Vec::with_capacity(inputs.len());
        for (name, ti) in inputs {
            let idx = self
                .input_index_for(name)
                .ok_or_else(|| OCRError::InvalidInput {
                    message: format!(
                        "RKNN model '{}' has no input named '{}' (declared inputs: {:?})",
                        self.model_name,
                        name,
                        self.input_attrs.iter().map(|a| &a.name).collect::<Vec<_>>(),
                    ),
                })?;
            let prepared = tensor_input_to_rknn_bytes(ti, self.input_mode).ok_or_else(|| {
                OCRError::InvalidInput {
                    message: format!(
                        "Input '{}' for RKNN model '{}' is not a contiguous f32 tensor",
                        name, self.model_name
                    ),
                }
            })?;
            let PreparedInput {
                bytes,
                fmt,
                pass_through,
            } = prepared;

            indices.push(idx);
            input_buffers.push(bytes);
            input_meta.push((fmt.into(), pass_through));
        }

        let rknn_inputs: Vec<RknnInput<'_>> = indices
            .iter()
            .zip(input_buffers.iter())
            .zip(input_meta.iter())
            .map(|((&index, data), &(fmt, pass_through))| RknnInput {
                index,
                data: data.as_ref(),
                pass_through,
                fmt,
                dtype: RknnTensorType::Fp32,
            })
            .collect();

        let n_outputs = self.output_attrs.len() as u32;
        let outputs_raw = {
            let idx = self
                .next_idx
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % self.contexts.len();
            let mut guard = self.contexts[idx]
                .lock()
                .map_err(|_| OCRError::InvalidInput {
                    message: format!(
                        "Model '{}': Failed to acquire RKNN context lock for context {}/{}",
                        self.model_name,
                        idx,
                        self.contexts.len()
                    ),
                })?;
            guard.set_inputs(&rknn_inputs)?;
            guard.run()?;
            guard.outputs_get(true, n_outputs)?
        };

        // Reinterpret each output's bytes as f32 and pair with the shape from
        // its `output_attr`. We assume `want_float = true`, so any quantized
        // output is dequantized by librknnrt before we see it.
        let mut results = Vec::with_capacity(outputs_raw.len());
        for out in outputs_raw {
            // OCR models have only a small number of outputs, so a linear
            // lookup keeps the cached metadata simple.
            let attr = self
                .output_attrs
                .iter()
                .find(|a| a.index == out.index)
                .ok_or_else(|| OCRError::InvalidInput {
                    message: format!(
                        "RKNN output index {} not declared by model '{}'",
                        out.index, self.model_name
                    ),
                })?;

            if out.data.len() % std::mem::size_of::<f32>() != 0 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "RKNN output '{}' has {} bytes which is not a multiple of f32 size",
                        attr.name,
                        out.data.len()
                    ),
                });
            }
            let data = f32_vec_from_ne_bytes(&out.data);

            let shape: Vec<i64> = attr.dims.iter().map(|&d| d as i64).collect();
            results.push((attr.name.clone(), TensorOutput::F32 { shape, data }));
        }

        Ok(results)
    }
}

#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
fn tensor_input_to_rknn_bytes<'a>(
    ti: &TensorInput<'a>,
    input_mode: RknnInputMode,
) -> Option<PreparedInput<'a>> {
    match ti {
        TensorInput::Array2(a) => Some(PreparedInput {
            bytes: Cow::Borrowed(floats_as_bytes(a.as_slice()?)),
            fmt: PreparedTensorFormat::Nchw,
            // Auxiliary tensors are already raw buffers; when pass-through is
            // enabled, mark them consistently with the image input batch.
            pass_through: matches!(input_mode, RknnInputMode::NormalizedNchwPassThrough),
        }),
        TensorInput::Array3(a) => Some(PreparedInput {
            bytes: Cow::Borrowed(floats_as_bytes(a.as_slice()?)),
            fmt: PreparedTensorFormat::Nchw,
            // Auxiliary tensors are already raw buffers; when pass-through is
            // enabled, mark them consistently with the image input batch.
            pass_through: matches!(input_mode, RknnInputMode::NormalizedNchwPassThrough),
        }),
        TensorInput::Array4(a) => match input_mode {
            RknnInputMode::ToolkitLayout => {
                // (N, C, H, W) -> (N, H, W, C). `permuted_axes` is a
                // stride-only view; force a standard-layout copy.
                let nhwc = a
                    .view()
                    .permuted_axes([0, 2, 3, 1])
                    .as_standard_layout()
                    .to_owned();
                Some(PreparedInput {
                    bytes: Cow::Owned(floats_as_bytes(nhwc.as_slice()?).to_vec()),
                    fmt: PreparedTensorFormat::Nhwc,
                    pass_through: false,
                })
            }
            RknnInputMode::NormalizedNchwPassThrough => Some(PreparedInput {
                bytes: Cow::Borrowed(floats_as_bytes(a.as_slice()?)),
                fmt: PreparedTensorFormat::Nchw,
                pass_through: true,
            }),
        },
    }
}

#[cfg(any(test, all(target_arch = "aarch64", feature = "rknpu")))]
fn floats_as_bytes(slice: &[f32]) -> &[u8] {
    // SAFETY: `u8` has alignment 1 and the returned byte slice is tied to the
    // lifetime of the source `f32` slice.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
fn f32_vec_from_ne_bytes(bytes: &[u8]) -> Vec<f32> {
    let len = bytes.len() / std::mem::size_of::<f32>();
    let mut out = Vec::<f32>::with_capacity(len);
    // SAFETY: the caller has already checked that `bytes.len()` is a multiple
    // of `size_of::<f32>()`. We copy raw bytes into a properly aligned f32 Vec.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        out.set_len(len);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("chunk has four bytes")))
            .collect()
    }

    #[test]
    fn tensor_input_to_rknn_bytes_transposes_nchw_array4_in_toolkit_layout() {
        let tensor = Array4::from_shape_vec(
            (1, 2, 2, 3),
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, //
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
            ],
        )
        .expect("valid shape");

        let prepared =
            tensor_input_to_rknn_bytes(&TensorInput::Array4(&tensor), RknnInputMode::ToolkitLayout)
                .expect("contiguous");

        assert_eq!(
            bytes_to_f32(&prepared.bytes),
            vec![0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0,]
        );
        assert_eq!(prepared.fmt, PreparedTensorFormat::Nhwc);
        assert!(!prepared.pass_through);
    }

    #[test]
    fn tensor_input_to_rknn_bytes_passes_nchw_through_when_configured() {
        let tensor = Array4::from_shape_vec(
            (1, 3, 2, 2),
            vec![
                0.0, 1.0, 2.0, 3.0, //
                10.0, 11.0, 12.0, 13.0, //
                20.0, 21.0, 22.0, 23.0,
            ],
        )
        .expect("valid shape");

        let prepared = tensor_input_to_rknn_bytes(
            &TensorInput::Array4(&tensor),
            RknnInputMode::NormalizedNchwPassThrough,
        )
        .expect("contiguous");

        assert_eq!(
            bytes_to_f32(&prepared.bytes),
            vec![
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0
            ]
        );
        assert_eq!(prepared.fmt, PreparedTensorFormat::Nchw);
        assert!(prepared.pass_through);
    }
}
