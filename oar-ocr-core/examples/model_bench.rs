//! Backend-agnostic model benchmark.
//!
//! Loads any model the [`InferenceBackend`] factory understands (`.onnx` →
//! ORT, `.rknn` → RKNN), feeds zero-filled input tensors of the model's
//! declared primary shape, and times the inference loop.
//!
//! Usage: `model_bench <path> [iters] [--shape NCHW]`. With `--features rknpu`
//! on aarch64, the RKNN backend is enabled.

use ndarray::{Array2, Array3, Array4};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

use oar_ocr_core::core::inference::{self, TensorInput};

fn parse_shape_arg(args: &[String]) -> Option<Vec<usize>> {
    args.windows(2).find_map(|w| {
        if w[0] == "--shape" {
            Some(w[1].split(',').map(|s| s.parse().unwrap()).collect())
        } else {
            None
        }
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: model_bench <model> [iters] [--shape N,C,H,W]");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let iters: usize = args
        .iter()
        .skip(2)
        .find(|s| !s.starts_with("--") && s.parse::<usize>().is_ok())
        .map(|s| s.parse().unwrap())
        .unwrap_or(20);

    let backend = inference::build(None, &path, None).expect("failed to build backend");

    // Decide on input shape: CLI override > model-declared shape (with -1
    // dynamic dims rounded to 1).
    let model_shape = backend.primary_input_shape();
    let shape: Vec<usize> = match parse_shape_arg(&args) {
        Some(s) => s,
        None => model_shape
            .as_ref()
            .map(|v| {
                v.iter()
                    .map(|&d| if d > 0 { d as usize } else { 1 })
                    .collect()
            })
            .expect("model has no declared input shape; pass --shape N,C,H,W"),
    };

    let name = backend.input_name().to_string();
    println!(
        "model={} input_name={} shape={:?} iters={}",
        path.display(),
        name,
        shape,
        iters
    );

    // Materialise the zero tensors once and re-borrow on every iter.
    let (arr2, arr3, arr4) = match shape.len() {
        2 => (Some(Array2::<f32>::zeros((shape[0], shape[1]))), None, None),
        3 => (
            None,
            Some(Array3::<f32>::zeros((shape[0], shape[1], shape[2]))),
            None,
        ),
        4 => (
            None,
            None,
            Some(Array4::<f32>::zeros((
                shape[0], shape[1], shape[2], shape[3],
            ))),
        ),
        n => panic!("unsupported input rank: {}", n),
    };

    let make_input = || match (&arr2, &arr3, &arr4) {
        (Some(a), _, _) => TensorInput::Array2(a),
        (_, Some(a), _) => TensorInput::Array3(a),
        (_, _, Some(a)) => TensorInput::Array4(a),
        _ => unreachable!(),
    };

    // Warm-up.
    if let Err(e) = backend.infer(&[(name.as_str(), make_input())]) {
        eprintln!("warmup failed: {}", e);
        std::process::exit(1);
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        backend
            .infer(&[(name.as_str(), make_input())])
            .expect("infer");
    }
    let elapsed = t0.elapsed();
    let mean_ms = elapsed.as_secs_f64() * 1000.0 / iters as f64;
    println!(
        "{} iters in {:.2} ms, mean = {:.3} ms/iter",
        iters,
        elapsed.as_secs_f64() * 1000.0,
        mean_ms
    );
}
