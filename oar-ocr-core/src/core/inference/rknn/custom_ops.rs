//! RKNN custom CPU operators used by models with unsupported fallback ops.

use std::os::raw::c_int;

use half::f16;

use super::{error::RknnError, sys};

const GRID_SAMPLE_OP_TYPE: &[u8] = b"GridSample";

pub(crate) fn register_uvdoc_grid_sample_align_corners(
    ctx: sys::rknn_context,
) -> Result<(), RknnError> {
    let mut op = sys::rknn_custom_op {
        version: 1,
        target: sys::_rknn_target_type_RKNN_TARGET_TYPE_CPU,
        compute: Some(grid_sample_compute),
        ..Default::default()
    };
    copy_cstr(&mut op.op_type, GRID_SAMPLE_OP_TYPE);

    let ret = unsafe { sys::rknn_register_custom_ops(ctx, &mut op, 1) };
    if ret < 0 {
        return Err(RknnError::ApiError {
            api: "rknn_register_custom_ops(GridSample)",
            code: ret,
            msg: super::error::code_to_msg(ret),
        });
    }
    Ok(())
}

fn copy_cstr<const N: usize>(dst: &mut [std::os::raw::c_char; N], src: &[u8]) {
    let n = src.len().min(N.saturating_sub(1));
    for (d, s) in dst.iter_mut().take(n).zip(src.iter().take(n)) {
        *d = *s as std::os::raw::c_char;
    }
    if N > 0 {
        dst[n] = 0;
    }
}

unsafe extern "C" fn grid_sample_compute(
    op_ctx: *mut sys::rknn_custom_op_context,
    inputs: *mut sys::rknn_custom_op_tensor,
    n_inputs: u32,
    outputs: *mut sys::rknn_custom_op_tensor,
    n_outputs: u32,
) -> c_int {
    let result = unsafe { grid_sample_compute_inner(op_ctx, inputs, n_inputs, outputs, n_outputs) };
    match result {
        Ok(()) => 0,
        Err(code) => code,
    }
}

unsafe fn grid_sample_compute_inner(
    _op_ctx: *mut sys::rknn_custom_op_context,
    inputs: *mut sys::rknn_custom_op_tensor,
    n_inputs: u32,
    outputs: *mut sys::rknn_custom_op_tensor,
    n_outputs: u32,
) -> Result<(), c_int> {
    if n_inputs != 2 || n_outputs != 1 || inputs.is_null() || outputs.is_null() {
        return Err(-1);
    }

    let input = unsafe { &*inputs.add(0) };
    let grid = unsafe { &*inputs.add(1) };
    let output = unsafe { &mut *outputs.add(0) };

    let in_dims = dims(&input.attr);
    let grid_dims = dims(&grid.attr);
    let out_dims = dims(&output.attr);

    if in_dims.len() != 4 || grid_dims.len() != 4 || out_dims.len() != 4 {
        return Err(-1);
    }

    let (n, c, in_h, in_w) = nchw_dims(&input.attr, &in_dims).ok_or(-1)?;
    let (out_n, out_c, out_h, out_w) = nchw_dims(&output.attr, &out_dims).ok_or(-1)?;
    if n != out_n || c != out_c {
        return Err(-1);
    }
    if grid_dims[0] != n || grid_dims[1] != out_h || grid_dims[2] != out_w || grid_dims[3] != 2 {
        return Err(-1);
    }

    let input_ptr = tensor_ptr(&input.mem).ok_or(-1)?;
    let grid_ptr = tensor_ptr(&grid.mem).ok_or(-1)?;
    let output_ptr = tensor_ptr(&output.mem).ok_or(-1)?;

    for b in 0..n {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let gx = read_tensor_f32(
                    grid_ptr,
                    grid.attr.type_,
                    grid_offset(&grid.attr, &grid_dims, b, oh, ow, 0).ok_or(-1)?,
                )?;
                let gy = read_tensor_f32(
                    grid_ptr,
                    grid.attr.type_,
                    grid_offset(&grid.attr, &grid_dims, b, oh, ow, 1).ok_or(-1)?,
                )?;

                // UVDoc exports ONNX GridSample with align_corners=1 and
                // padding_mode=zeros.
                let ix = ((gx + 1.0) * (in_w as f32 - 1.0)) * 0.5;
                let iy = ((gy + 1.0) * (in_h as f32 - 1.0)) * 0.5;

                let x0 = ix.floor() as isize;
                let y0 = iy.floor() as isize;
                let x1 = x0 + 1;
                let y1 = y0 + 1;
                let wx = ix - x0 as f32;
                let wy = iy - y0 as f32;

                for ch in 0..c {
                    let v00 = sample_nchw(input_ptr, &input.attr, &in_dims, b, ch, y0, x0)?;
                    let v01 = sample_nchw(input_ptr, &input.attr, &in_dims, b, ch, y0, x1)?;
                    let v10 = sample_nchw(input_ptr, &input.attr, &in_dims, b, ch, y1, x0)?;
                    let v11 = sample_nchw(input_ptr, &input.attr, &in_dims, b, ch, y1, x1)?;

                    let top = v00 * (1.0 - wx) + v01 * wx;
                    let bottom = v10 * (1.0 - wx) + v11 * wx;
                    let value = top * (1.0 - wy) + bottom * wy;
                    let out_offset =
                        nchw_offset(&output.attr, &out_dims, b, ch, oh, ow).ok_or(-1)?;
                    write_tensor_f32(output_ptr, output.attr.type_, out_offset, value)?;
                }
            }
        }
    }

    Ok(())
}

fn dims(attr: &sys::rknn_tensor_attr) -> Vec<usize> {
    attr.dims
        .iter()
        .take(attr.n_dims as usize)
        .map(|&d| d as usize)
        .collect()
}

fn nchw_dims(attr: &sys::rknn_tensor_attr, dims: &[usize]) -> Option<(usize, usize, usize, usize)> {
    match attr.fmt {
        sys::_rknn_tensor_format_RKNN_TENSOR_NCHW => Some((dims[0], dims[1], dims[2], dims[3])),
        sys::_rknn_tensor_format_RKNN_TENSOR_NHWC => Some((dims[0], dims[3], dims[1], dims[2])),
        _ => None,
    }
}

fn tensor_ptr(mem: &sys::rknn_tensor_mem) -> Option<*mut u8> {
    if mem.virt_addr.is_null() || mem.offset < 0 {
        return None;
    }
    Some(unsafe { (mem.virt_addr as *mut u8).add(mem.offset as usize) })
}

fn nchw_offset(
    attr: &sys::rknn_tensor_attr,
    dims: &[usize],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> Option<usize> {
    match attr.fmt {
        sys::_rknn_tensor_format_RKNN_TENSOR_NCHW => {
            // RKNN's CPU custom-op fallback currently hands us h-contiguous
            // FLOAT32 temporaries, so w_stride is the only padded dimension.
            let stride_w = attr.w_stride.max(dims[3] as u32) as usize;
            Some(((n * dims[1] + c) * dims[2] + h) * stride_w + w)
        }
        sys::_rknn_tensor_format_RKNN_TENSOR_NHWC => {
            let stride_w = attr.w_stride.max(dims[2] as u32) as usize;
            Some(((n * dims[1] + h) * stride_w + w) * dims[3] + c)
        }
        _ => None,
    }
}

fn grid_offset(
    attr: &sys::rknn_tensor_attr,
    dims: &[usize],
    n: usize,
    h: usize,
    w: usize,
    c: usize,
) -> Option<usize> {
    // RKNN's custom-op CPU fallback converts this GridSample's inputs to
    // FLOAT32 temporaries. The grid tensor is reported as fmt=NCHW, but its
    // dimensions keep ONNX GridSample order [N, H_out, W_out, 2]. Treat it as
    // shape-order contiguous data rather than semantic NCHW.
    let stride_w = attr.w_stride.max(dims[2] as u32) as usize;
    Some(((n * dims[1] + h) * stride_w + w) * dims[3] + c)
}

fn sample_nchw(
    ptr: *mut u8,
    attr: &sys::rknn_tensor_attr,
    dims: &[usize],
    n: usize,
    c: usize,
    y: isize,
    x: isize,
) -> Result<f32, c_int> {
    let (_, _, h, w) = nchw_dims(attr, dims).ok_or(-1)?;
    if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
        return Ok(0.0);
    }
    let offset = nchw_offset(attr, dims, n, c, y as usize, x as usize).ok_or(-1)?;
    read_tensor_f32(ptr, attr.type_, offset)
}

fn read_tensor_f32(ptr: *mut u8, dtype: sys::rknn_tensor_type, elem: usize) -> Result<f32, c_int> {
    unsafe {
        match dtype {
            sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32 => Ok(*(ptr as *const f32).add(elem)),
            sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT16 => {
                let bits = *(ptr as *const u16).add(elem);
                Ok(f16::from_bits(bits).to_f32())
            }
            _ => Err(-1),
        }
    }
}

fn write_tensor_f32(
    ptr: *mut u8,
    dtype: sys::rknn_tensor_type,
    elem: usize,
    value: f32,
) -> Result<(), c_int> {
    unsafe {
        match dtype {
            sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT32 => {
                *(ptr as *mut f32).add(elem) = value;
                Ok(())
            }
            sys::_rknn_tensor_type_RKNN_TENSOR_FLOAT16 => {
                *(ptr as *mut u16).add(elem) = f16::from_f32(value).to_bits();
                Ok(())
            }
            _ => Err(-1),
        }
    }
}
