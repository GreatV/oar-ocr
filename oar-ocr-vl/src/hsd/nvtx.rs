//! NVTX range / marker instrumentation for HSD profiling.
//!
//! `SpecDecodeStats` already tracks per-phase wall time on the host side. NVTX
//! complements that by exposing the same phase boundaries to NVIDIA Nsight
//! Systems / Compute, so `nsys --trace=cuda,nvtx` traces show CUDA kernels
//! grouped under labelled DSV phases (`dsv.candidate_build`, `dsv.verify_tree`,
//! `dsv.traverse`, `dsv.commit`, `dsv.step_one`, `dsv.fallback_argmax`).
//!
//! The module is feature-gated under `nvtx`. When the feature is off, every
//! function compiles to a no-op and `Range` is a zero-sized type, so leaving
//! the call sites in the hot path costs nothing in release builds.
//!
//! Linkage: when the `nvtx` feature is enabled the build links against
//! `libnvToolsExt`, which ships with the CUDA toolkit. Builds without CUDA
//! installed will fail at link time — `nvtx` should only be turned on for
//! profiling builds on a CUDA host.

#[cfg(feature = "nvtx")]
mod ffi {
    use std::os::raw::{c_char, c_int};

    #[link(name = "nvToolsExt")]
    unsafe extern "C" {
        pub fn nvtxRangePushA(message: *const c_char) -> c_int;
        pub fn nvtxRangePop() -> c_int;
        pub fn nvtxMarkA(message: *const c_char);
    }
}

/// RAII-style NVTX range. The range is opened on construction and closed
/// when the value is dropped.
///
/// When the `nvtx` feature is disabled this is a zero-sized type and all
/// methods compile to no-ops.
#[must_use = "the NVTX range closes immediately on drop; bind it to a local"]
pub struct Range {
    #[cfg(feature = "nvtx")]
    _name: std::ffi::CString,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Range {
    /// Opens an NVTX range with the given label.
    ///
    /// Labels containing interior NUL bytes are silently replaced with a
    /// fixed sentinel (`"invalid_nvtx_name"`) — instrumentation should
    /// never abort the host program.
    #[inline(always)]
    pub fn new(_name: &str) -> Self {
        #[cfg(feature = "nvtx")]
        {
            let cname = std::ffi::CString::new(_name).unwrap_or_else(|_| {
                // SAFETY: the byte string is a fixed NUL-terminated C string
                // with no interior NUL bytes.
                unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"invalid_nvtx_name\0") }
                    .to_owned()
            });
            // SAFETY: `cname` outlives the call (held in `Self`); the FFI
            // signature reads a NUL-terminated string and does not retain it.
            unsafe {
                ffi::nvtxRangePushA(cname.as_ptr());
            }
            Self {
                _name: cname,
                _not_send: std::marker::PhantomData,
            }
        }
        #[cfg(not(feature = "nvtx"))]
        {
            Self {
                _not_send: std::marker::PhantomData,
            }
        }
    }
}

#[cfg(feature = "nvtx")]
impl Drop for Range {
    #[inline(always)]
    fn drop(&mut self) {
        // SAFETY: `nvtxRangePop` decrements the active range stack; it is
        // always safe to call as long as we previously pushed.
        unsafe {
            ffi::nvtxRangePop();
        }
    }
}

/// Emits a single instantaneous NVTX marker. Useful for tagging iteration
/// boundaries that don't correspond to a stack-scoped range.
#[inline(always)]
pub fn mark(_name: &str) {
    #[cfg(feature = "nvtx")]
    {
        if let Ok(c) = std::ffi::CString::new(_name) {
            // SAFETY: see `Range::new`.
            unsafe {
                ffi::nvtxMarkA(c.as_ptr());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The no-op path must compile and run — Range::new should never panic
    /// or do anything observable when the feature is off.
    #[test]
    fn range_no_op_when_feature_disabled() {
        let r = Range::new("test.range");
        drop(r);
        mark("test.mark");
    }

    /// Names containing NUL bytes must not abort.
    #[test]
    fn nul_in_name_does_not_panic() {
        let r = Range::new("bad\0name");
        drop(r);
        mark("bad\0marker");
    }
}
