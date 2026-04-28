//! Rockchip RKNN runtime backend.
//!
//! Active only when `feature = "rknpu"` is enabled AND
//! `target_arch = "aarch64"`. On any other configuration this module is an
//! empty stub so the parent module can declare it unconditionally.

#![cfg_attr(not(all(target_arch = "aarch64", feature = "rknpu")), allow(dead_code))]

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
mod custom_ops;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
mod error;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
mod safe;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
mod sys;

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
pub(crate) use error::RknnError;
#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
pub(crate) use safe::{
    RknnContext, RknnCoreMask, RknnInput, RknnOutput, RknnQuantType, RknnSdkVersion,
    RknnTensorAttr, RknnTensorFormat, RknnTensorType,
};

#[cfg(all(test, target_arch = "aarch64", feature = "rknpu"))]
mod probe {
    use super::{RknnContext, RknnInput, RknnTensorFormat, RknnTensorType};

    #[test]
    #[ignore = "set RKNN_PROBE_MODEL=/path/to/model.rknn and run on aarch64 with --features rknpu"]
    fn probe_model_from_env() {
        let path = std::env::var("RKNN_PROBE_MODEL").expect("RKNN_PROBE_MODEL must point to .rknn");
        println!("loading: {}", path);
        let mut ctx = RknnContext::from_file(std::path::Path::new(&path)).expect("from_file");
        println!("sdk: {:?}", ctx.sdk_version().unwrap());

        let (n_in, n_out) = ctx.input_output_num().unwrap();
        println!("n_inputs={n_in} n_outputs={n_out}");

        for i in 0..n_in {
            let a = ctx.input_attr(i).unwrap();
            println!(
                "INPUT[{i}] name={} dims={:?} fmt={:?} dtype={:?} qnt={:?} size={} scale={} zp={}",
                a.name, a.dims, a.fmt, a.dtype, a.qnt_type, a.size, a.scale, a.zp
            );
        }
        for i in 0..n_out {
            let a = ctx.output_attr(i).unwrap();
            println!(
                "OUTPUT[{i}] name={} dims={:?} fmt={:?} dtype={:?} qnt={:?} size={} scale={} zp={}",
                a.name, a.dims, a.fmt, a.dtype, a.qnt_type, a.size, a.scale, a.zp
            );
        }

        let in_attr = ctx.input_attr(0).unwrap();
        let n_elems = in_attr.dims.iter().map(|&d| d as usize).product::<usize>();
        let buf = vec![0u8; n_elems * std::mem::size_of::<f32>()];

        let fmt_arg = std::env::var("RKNN_FMT").unwrap_or_else(|_| "nhwc".into());
        let fmt = match fmt_arg.as_str() {
            "nchw" => RknnTensorFormat::Nchw,
            "nhwc" => RknnTensorFormat::Nhwc,
            _ => panic!("RKNN_FMT must be nchw|nhwc"),
        };

        let iters: usize = std::env::var("RKNN_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            ctx.set_inputs(&[RknnInput {
                index: 0,
                data: &buf,
                pass_through: false,
                fmt,
                dtype: RknnTensorType::Fp32,
            }])
            .expect("set_inputs");
            ctx.run().expect("run");
            let _ = ctx.outputs_get(true, n_out).expect("outputs_get");
        }
        let elapsed = t0.elapsed();
        println!(
            "{} iters in {:.2} ms, mean = {:.3} ms/iter",
            iters,
            elapsed.as_secs_f64() * 1000.0,
            (elapsed.as_secs_f64() * 1000.0) / iters as f64,
        );
    }
}
