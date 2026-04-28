//! Safe wrapper over the raw `sys` FFI bindings to librknnrt.
//!
//! This module hides the unsafe pointer dance and provides idiomatic Rust
//! types for working with an RKNN context. It is feature- and target-gated;
//! the file is empty on builds where the rknpu backend is not active.

#![cfg(all(target_arch = "aarch64", feature = "rknpu"))]

use std::path::Path;
use std::ptr;

use super::error::{RknnError, code_to_msg};
use super::sys;

// -------- public enums --------

/// Tensor data layout (mirrors `rknn_tensor_format`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RknnTensorFormat {
    Nchw,
    Nhwc,
    Nc1hwc2,
    Undefined,
}

impl RknnTensorFormat {
    fn from_raw(raw: sys::rknn_tensor_format) -> Self {
        match raw {
            sys::_rknn_tensor_format_RKNN_TENSOR_NCHW => Self::Nchw,
            sys::_rknn_tensor_format_RKNN_TENSOR_NHWC => Self::Nhwc,
            sys::_rknn_tensor_format_RKNN_TENSOR_NC1HWC2 => Self::Nc1hwc2,
            _ => Self::Undefined,
        }
    }

    fn to_raw(self) -> sys::rknn_tensor_format {
        match self {
            Self::Nchw => sys::_rknn_tensor_format_RKNN_TENSOR_NCHW,
            Self::Nhwc => sys::_rknn_tensor_format_RKNN_TENSOR_NHWC,
            Self::Nc1hwc2 => sys::_rknn_tensor_format_RKNN_TENSOR_NC1HWC2,
            Self::Undefined => sys::_rknn_tensor_format_RKNN_TENSOR_UNDEFINED,
        }
    }
}

/// Tensor element type (mirrors `rknn_tensor_type`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RknnTensorType {
    Fp32,
    Fp16,
    Int8,
    Uint8,
    Int16,
    Int32,
    Int64,
    Bool,
    /// Any other value reported by the runtime that we don't model explicitly
    /// (e.g. INT4, UINT16, UINT32). Stored as the raw enum tag.
    Other(u32),
}

impl RknnTensorType {
    fn from_raw(raw: sys::rknn_tensor_type) -> Self {
        match raw {
            sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32 => Self::Fp32,
            sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT16 => Self::Fp16,
            sys::_rknn_tensor_type_RKNN_TENSOR_INT8 => Self::Int8,
            sys::_rknn_tensor_type_RKNN_TENSOR_UINT8 => Self::Uint8,
            sys::_rknn_tensor_type_RKNN_TENSOR_INT16 => Self::Int16,
            sys::_rknn_tensor_type_RKNN_TENSOR_INT32 => Self::Int32,
            sys::_rknn_tensor_type_RKNN_TENSOR_INT64 => Self::Int64,
            sys::_rknn_tensor_type_RKNN_TENSOR_BOOL => Self::Bool,
            other => Self::Other(other as u32),
        }
    }

    fn to_raw(self) -> sys::rknn_tensor_type {
        match self {
            Self::Fp32 => sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32,
            Self::Fp16 => sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT16,
            Self::Int8 => sys::_rknn_tensor_type_RKNN_TENSOR_INT8,
            Self::Uint8 => sys::_rknn_tensor_type_RKNN_TENSOR_UINT8,
            Self::Int16 => sys::_rknn_tensor_type_RKNN_TENSOR_INT16,
            Self::Int32 => sys::_rknn_tensor_type_RKNN_TENSOR_INT32,
            Self::Int64 => sys::_rknn_tensor_type_RKNN_TENSOR_INT64,
            Self::Bool => sys::_rknn_tensor_type_RKNN_TENSOR_BOOL,
            Self::Other(v) => v as sys::rknn_tensor_type,
        }
    }
}

/// Quantisation kind (mirrors `rknn_tensor_qnt_type`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RknnQuantType {
    None,
    DynamicFixedPoint,
    AffineAsymmetric,
    Other(u32),
}

impl RknnQuantType {
    fn from_raw(raw: sys::rknn_tensor_qnt_type) -> Self {
        match raw {
            sys::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE => Self::None,
            sys::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP => Self::DynamicFixedPoint,
            sys::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => Self::AffineAsymmetric,
            other => Self::Other(other as u32),
        }
    }
}

/// Selects which NPU cores to run on (RK3588 only).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RknnCoreMask {
    Auto,
    Core0,
    Core1,
    Core2,
    Core01,
    Core012,
}

impl RknnCoreMask {
    fn to_raw(self) -> sys::rknn_core_mask {
        match self {
            Self::Auto => sys::_rknn_core_mask_RKNN_NPU_CORE_AUTO,
            Self::Core0 => sys::_rknn_core_mask_RKNN_NPU_CORE_0,
            Self::Core1 => sys::_rknn_core_mask_RKNN_NPU_CORE_1,
            Self::Core2 => sys::_rknn_core_mask_RKNN_NPU_CORE_2,
            Self::Core01 => sys::_rknn_core_mask_RKNN_NPU_CORE_0_1,
            Self::Core012 => sys::_rknn_core_mask_RKNN_NPU_CORE_0_1_2,
        }
    }
}

// -------- public structs --------

/// SDK / driver version reported by `rknn_query(RKNN_QUERY_SDK_VERSION)`.
#[derive(Debug, Clone)]
pub struct RknnSdkVersion {
    pub api_version: String,
    pub drv_version: String,
}

/// Decoded `rknn_tensor_attr`. Only the dimensions actually used (`n_dims`)
/// are copied out of the fixed-size C array.
#[derive(Debug, Clone)]
pub struct RknnTensorAttr {
    pub index: u32,
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u32>,
    pub n_elems: u32,
    pub size: u32,
    pub fmt: RknnTensorFormat,
    pub dtype: RknnTensorType,
    pub qnt_type: RknnQuantType,
    pub scale: f32,
    pub zp: i32,
}

/// Borrowed input buffer description for `rknn_inputs_set`.
pub struct RknnInput<'a> {
    pub index: u32,
    pub data: &'a [u8],
    pub pass_through: bool,
    pub fmt: RknnTensorFormat,
    pub dtype: RknnTensorType,
}

/// Output tensor returned by `rknn_outputs_get`. The bytes are copied out of
/// the runtime-owned buffer before `rknn_outputs_release` is called, so the
/// `Vec` is independently owned.
#[derive(Debug)]
pub struct RknnOutput {
    pub index: u32,
    pub data: Vec<u8>,
    pub want_float: bool,
}

// -------- the context --------

/// A loaded RKNN model. Wraps an `rknn_context` and calls `rknn_destroy` on
/// drop.
#[derive(Debug)]
pub struct RknnContext {
    ctx: sys::rknn_context,
}

// SAFETY: ownership of the opaque handle may move between threads. Stateful
// operations require `&mut self`, so safe Rust callers cannot run the
// set-inputs/run/get-outputs sequence concurrently on the same context.
unsafe impl Send for RknnContext {}

impl RknnContext {
    /// Initialise from an in-memory model buffer. The bytes are passed to
    /// `rknn_init`; librknnrt copies the model internally so the caller does
    /// not need to keep the buffer alive afterwards.
    pub fn from_bytes(model: &[u8]) -> Result<Self, RknnError> {
        let mut ctx: sys::rknn_context = 0;
        // SAFETY: the RKNN API takes `void*` for historical C API reasons, but
        // `vendor/rknn/include/rknn_api.h` documents this argument as the input
        // model buffer for `rknn_init` and librknnrt copies it. We do not let
        // the pointer escape this call.
        let ret = unsafe {
            sys::rknn_init(
                &mut ctx as *mut sys::rknn_context,
                model.as_ptr() as *mut core::ffi::c_void,
                model.len() as u32,
                0,
                ptr::null_mut(),
            )
        };
        check(ret, "rknn_init")?;
        let wrapped = Self { ctx };
        super::custom_ops::register_uvdoc_grid_sample_align_corners(wrapped.ctx)?;
        Ok(wrapped)
    }

    /// Duplicate this context for concurrent multi-context inference.
    pub fn duplicate(&mut self) -> Result<Self, RknnError> {
        // The header does not state whether `context_in` is mutated while
        // duplicating, so the wrapper requires exclusive access to the source.
        let mut ctx: sys::rknn_context = 0;
        let ret = unsafe {
            sys::rknn_dup_context(
                &mut self.ctx as *mut sys::rknn_context,
                &mut ctx as *mut sys::rknn_context,
            )
        };
        check(ret, "rknn_dup_context")?;
        let wrapped = Self { ctx };
        super::custom_ops::register_uvdoc_grid_sample_align_corners(wrapped.ctx)?;
        Ok(wrapped)
    }

    /// Read a model from disk and initialise via [`Self::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self, RknnError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Query the SDK / driver version strings.
    pub fn sdk_version(&self) -> Result<RknnSdkVersion, RknnError> {
        let mut info: sys::rknn_sdk_version = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            sys::rknn_query(
                self.ctx,
                sys::_rknn_query_cmd_RKNN_QUERY_SDK_VERSION,
                &mut info as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<sys::rknn_sdk_version>() as u32,
            )
        };
        check(ret, "rknn_query(SDK_VERSION)")?;
        Ok(RknnSdkVersion {
            api_version: cstr_to_string(&info.api_version),
            drv_version: cstr_to_string(&info.drv_version),
        })
    }

    /// Query the number of input and output tensors.
    pub fn input_output_num(&self) -> Result<(u32, u32), RknnError> {
        let mut info: sys::rknn_input_output_num = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            sys::rknn_query(
                self.ctx,
                sys::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
                &mut info as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<sys::rknn_input_output_num>() as u32,
            )
        };
        check(ret, "rknn_query(IN_OUT_NUM)")?;
        Ok((info.n_input, info.n_output))
    }

    /// Query the attributes of input tensor `idx`.
    pub fn input_attr(&self, idx: u32) -> Result<RknnTensorAttr, RknnError> {
        self.tensor_attr(
            idx,
            sys::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
            "rknn_query(INPUT_ATTR)",
        )
    }

    /// Query the attributes of output tensor `idx`.
    pub fn output_attr(&self, idx: u32) -> Result<RknnTensorAttr, RknnError> {
        self.tensor_attr(
            idx,
            sys::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
            "rknn_query(OUTPUT_ATTR)",
        )
    }

    fn tensor_attr(
        &self,
        idx: u32,
        cmd: sys::rknn_query_cmd,
        api: &'static str,
    ) -> Result<RknnTensorAttr, RknnError> {
        let mut attr: sys::rknn_tensor_attr = unsafe { std::mem::zeroed() };
        attr.index = idx;
        let ret = unsafe {
            sys::rknn_query(
                self.ctx,
                cmd,
                &mut attr as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<sys::rknn_tensor_attr>() as u32,
            )
        };
        check(ret, api)?;

        let n_dims = attr.n_dims;
        let take = (n_dims as usize).min(attr.dims.len());
        let dims = attr.dims[..take].to_vec();

        Ok(RknnTensorAttr {
            index: attr.index,
            name: cstr_to_string(&attr.name),
            n_dims,
            dims,
            n_elems: attr.n_elems,
            size: attr.size,
            fmt: RknnTensorFormat::from_raw(attr.fmt),
            dtype: RknnTensorType::from_raw(attr.type_),
            qnt_type: RknnQuantType::from_raw(attr.qnt_type),
            scale: attr.scale,
            zp: attr.zp,
        })
    }

    /// Configure which NPU cores this context should run on (RK3588).
    pub fn set_core_mask(&mut self, mask: RknnCoreMask) -> Result<(), RknnError> {
        let ret = unsafe { sys::rknn_set_core_mask(self.ctx, mask.to_raw()) };
        check(ret, "rknn_set_core_mask")
    }

    /// Set inputs by binding the given buffers via `rknn_inputs_set`.
    pub fn set_inputs(&mut self, inputs: &[RknnInput<'_>]) -> Result<(), RknnError> {
        if inputs.is_empty() {
            return Err(RknnError::InvalidArg(
                "set_inputs called with empty slice".into(),
            ));
        }

        // Build an array of `rknn_input` referencing the user's buffers.
        // The lifetimes are bounded by this function call: `rknn_inputs_set`
        // synchronously consumes the data.
        let mut raw: Vec<sys::rknn_input> = Vec::with_capacity(inputs.len());
        for inp in inputs {
            let mut r: sys::rknn_input = unsafe { std::mem::zeroed() };
            r.index = inp.index;
            r.buf = inp.data.as_ptr() as *mut core::ffi::c_void;
            r.size = inp.data.len() as u32;
            r.pass_through = u8::from(inp.pass_through);
            r.type_ = inp.dtype.to_raw();
            r.fmt = inp.fmt.to_raw();
            raw.push(r);
        }

        let ret = unsafe { sys::rknn_inputs_set(self.ctx, raw.len() as u32, raw.as_mut_ptr()) };
        check(ret, "rknn_inputs_set")
    }

    /// Trigger a synchronous inference.
    pub fn run(&mut self) -> Result<(), RknnError> {
        let ret = unsafe { sys::rknn_run(self.ctx, ptr::null_mut()) };
        check(ret, "rknn_run")
    }

    /// Get the outputs after [`run`](Self::run). Bytes are copied out before
    /// `rknn_outputs_release` so the returned `Vec`s are independently owned.
    pub fn outputs_get(
        &mut self,
        want_float: bool,
        n_outputs: u32,
    ) -> Result<Vec<RknnOutput>, RknnError> {
        if n_outputs == 0 {
            return Err(RknnError::InvalidArg(
                "outputs_get called with n_outputs = 0".into(),
            ));
        }

        let mut raw: Vec<sys::rknn_output> = (0..n_outputs)
            .map(|_| {
                let mut r: sys::rknn_output = unsafe { std::mem::zeroed() };
                r.want_float = u8::from(want_float);
                r.is_prealloc = 0;
                r
            })
            .collect();

        let ret = unsafe {
            sys::rknn_outputs_get(self.ctx, n_outputs, raw.as_mut_ptr(), ptr::null_mut())
        };
        check(ret, "rknn_outputs_get")?;

        // Copy each output buffer out before releasing.
        let mut out: Vec<RknnOutput> = Vec::with_capacity(n_outputs as usize);
        for r in &raw {
            let bytes: Vec<u8> = if r.buf.is_null() || r.size == 0 {
                Vec::new()
            } else {
                let slice =
                    unsafe { std::slice::from_raw_parts(r.buf as *const u8, r.size as usize) };
                slice.to_vec()
            };
            out.push(RknnOutput {
                index: r.index,
                data: bytes,
                want_float,
            });
        }

        // Release the runtime-owned buffers. We deliberately ignore release
        // errors after a successful copy — the data is already safely ours.
        let _ = unsafe { sys::rknn_outputs_release(self.ctx, n_outputs, raw.as_mut_ptr()) };

        Ok(out)
    }
}

impl Drop for RknnContext {
    fn drop(&mut self) {
        // Best-effort: ignore the return code — there's nothing useful we can
        // do about a destroy failure during drop.
        unsafe {
            let _ = sys::rknn_destroy(self.ctx);
        }
    }
}

// -------- helpers --------

fn check(code: i32, api: &'static str) -> Result<(), RknnError> {
    if code == 0 {
        Ok(())
    } else {
        Err(RknnError::ApiError {
            api,
            code,
            msg: code_to_msg(code),
        })
    }
}

fn cstr_to_string(buf: &[core::ffi::c_char]) -> String {
    // Find the NUL terminator; if missing, take the whole buffer.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };
    let nul = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..nul]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rknn_init` on garbage input is documented as "behavior is
    /// undefined" — librknnrt 2.0 SIGSEGVs rather than returning an error
    /// for an empty buffer, so we don't exercise that path. Instead, smoke
    /// test enum mappings, which is the only piece of the safe wrapper
    /// we can verify without a real model file on disk.
    #[test]
    fn enum_round_trips_smoke() {
        for &fmt in &[
            RknnTensorFormat::Nchw,
            RknnTensorFormat::Nhwc,
            RknnTensorFormat::Nc1hwc2,
            RknnTensorFormat::Undefined,
        ] {
            let raw = fmt.to_raw();
            let back = RknnTensorFormat::from_raw(raw);
            assert!(matches!(
                (fmt, back),
                (RknnTensorFormat::Nchw, RknnTensorFormat::Nchw)
                    | (RknnTensorFormat::Nhwc, RknnTensorFormat::Nhwc)
                    | (RknnTensorFormat::Nc1hwc2, RknnTensorFormat::Nc1hwc2)
                    | (RknnTensorFormat::Undefined, RknnTensorFormat::Undefined)
            ));
        }
    }
}
