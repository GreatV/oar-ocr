//! Small, model-independent Metal benchmark for the VLM hot path.
//!
//! Run with:
//! `cargo run --release -p oar-ocr-vl --features metal --example metal_bench`

#[cfg(all(feature = "metal", target_os = "macos"))]
use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "metal", target_os = "macos"))]
use oar_ocr_vl::attention::repeat_kv;
#[cfg(all(feature = "metal", target_os = "macos"))]
use oar_ocr_vl::attention::scaled_dot_product_attention_gqa;
#[cfg(all(feature = "metal", target_os = "macos"))]
use oar_ocr_vl::utils::select_dtype;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "metal", target_os = "macos"))]
fn synchronize(tensor: Tensor) -> Result<()> {
    tensor.device().synchronize()?;
    std::hint::black_box(tensor);
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn measure<F>(name: &str, warmups: usize, iterations: usize, mut operation: F) -> Result<()>
where
    F: FnMut() -> Result<Tensor>,
{
    for _ in 0..warmups {
        synchronize(operation()?)?;
    }

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        synchronize(operation()?)?;
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
    println!(
        "{name:<34} median={:>8.3} ms  mean={:>8.3} ms",
        median.as_secs_f64() * 1_000.0,
        mean.as_secs_f64() * 1_000.0
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn measure_token<F>(name: &str, warmups: usize, iterations: usize, mut operation: F) -> Result<()>
where
    F: FnMut() -> Result<u32>,
{
    for _ in 0..warmups {
        std::hint::black_box(operation()?);
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        std::hint::black_box(operation()?);
        samples.push(start.elapsed());
    }
    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
    println!(
        "{name:<34} median={:>8.3} ms  mean={:>8.3} ms",
        median.as_secs_f64() * 1_000.0,
        mean.as_secs_f64() * 1_000.0
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bench_projection(device: &Device, dtype: DType) -> Result<()> {
    let input = Tensor::randn(0f32, 1f32, (512, 1024), device)?.to_dtype(dtype)?;
    let weight = Tensor::randn(0f32, 0.02f32, (1024, 1024), device)?.to_dtype(dtype)?;
    measure(&format!("projection 512x1024 {dtype:?}"), 3, 9, || {
        input.matmul(&weight)
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn native_softmax_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let weights = (q.matmul(&k.transpose(2, 3)?)? * 0.125)?;
    candle_nn::ops::softmax_last_dim(&weights)?.matmul(v)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn f32_softmax_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    let dtype = weights.dtype();
    candle_nn::ops::softmax_last_dim(&weights.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?
        .matmul(v)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bench_attention(device: &Device, dtype: DType, q_len: usize, kv_len: usize) -> Result<()> {
    let shape_q = (1, 8, q_len, 64);
    let shape_kv = (1, 8, kv_len, 64);
    let q = Tensor::randn(0f32, 1f32, shape_q, device)?.to_dtype(dtype)?;
    let k = Tensor::randn(0f32, 1f32, shape_kv, device)?.to_dtype(dtype)?;
    let v = Tensor::randn(0f32, 1f32, shape_kv, device)?.to_dtype(dtype)?;
    measure(
        &format!("attention/fp32-sm q={q_len} kv={kv_len} {dtype:?}"),
        2,
        7,
        || f32_softmax_attention(&q, &k, &v, 0.125),
    )?;
    measure(
        &format!("attention/native-sm q={q_len} kv={kv_len} {dtype:?}"),
        2,
        7,
        || native_softmax_attention(&q, &k, &v),
    )?;
    if device.is_metal() {
        measure(
            &format!("attention/fused-sdpa q={q_len} kv={kv_len} {dtype:?}"),
            2,
            7,
            || candle_nn::ops::sdpa(&q, &k, &v, None, false, 0.125, 1.0),
        )?;
    }

    if q_len == 512 {
        let reference = f32_softmax_attention(&q, &k, &v, 0.125)?.to_dtype(DType::F32)?;
        let native = native_softmax_attention(&q, &k, &v)?.to_dtype(DType::F32)?;
        let error = (reference - native.clone())?.abs()?;
        let max_abs = error.max_all()?.to_scalar::<f32>()?;
        let mean_abs = error.mean_all()?.to_scalar::<f32>()?;
        println!("  native softmax delta: max_abs={max_abs:.6}, mean_abs={mean_abs:.8}");
        if device.is_metal() {
            let fused =
                candle_nn::ops::sdpa(&q, &k, &v, None, false, 0.125, 1.0)?.to_dtype(DType::F32)?;
            let fused_error = (fused - native)?.abs()?;
            let fused_max_abs = fused_error.max_all()?.to_scalar::<f32>()?;
            let fused_mean_abs = fused_error.mean_all()?.to_scalar::<f32>()?;
            println!(
                "  fused SDPA delta: max_abs={fused_max_abs:.6}, mean_abs={fused_mean_abs:.8}"
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bench_sampling(device: &Device, dtype: DType) -> Result<()> {
    let logits = Tensor::randn(0f32, 1f32, 151_936, device)?.to_dtype(dtype)?;
    measure_token(&format!("sampling/full-download {dtype:?}"), 2, 15, || {
        let values = logits
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        Ok(values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index as u32)
            .unwrap_or(0))
    })?;
    measure_token(&format!("sampling/device-argmax {dtype:?}"), 2, 15, || {
        logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bench_gqa(device: &Device, dtype: DType, q_len: usize, kv_len: usize) -> Result<()> {
    let q = Tensor::randn(0f32, 1f32, (1, 16, q_len, 128), device)?.to_dtype(dtype)?;
    let k = Tensor::randn(0f32, 1f32, (1, 2, kv_len, 128), device)?.to_dtype(dtype)?;
    let v = Tensor::randn(0f32, 1f32, (1, 2, kv_len, 128), device)?.to_dtype(dtype)?;
    measure(
        &format!("gqa/repeat-eager q={q_len} kv={kv_len} {dtype:?}"),
        2,
        7,
        || {
            let repeated_k = repeat_kv(&k, 8)?.contiguous()?;
            let repeated_v = repeat_kv(&v, 8)?.contiguous()?;
            f32_softmax_attention(&q, &repeated_k, &repeated_v, 0.08838835)
        },
    )?;
    measure(
        &format!("gqa/shared-dispatch q={q_len} kv={kv_len} {dtype:?}"),
        2,
        7,
        || scaled_dot_product_attention_gqa(&q, &k, &v, None, 0.08838835, false, 8),
    )?;
    measure(
        &format!("gqa/fused-sdpa q={q_len} kv={kv_len} {dtype:?}"),
        2,
        7,
        || candle_nn::ops::sdpa(&q, &k, &v, None, false, 0.08838835, 1.0),
    )?;

    let repeated_k = repeat_kv(&k, 8)?.contiguous()?;
    let repeated_v = repeat_kv(&v, 8)?.contiguous()?;
    let eager =
        f32_softmax_attention(&q, &repeated_k, &repeated_v, 0.08838835)?.to_dtype(DType::F32)?;
    let fused =
        candle_nn::ops::sdpa(&q, &k, &v, None, false, 0.08838835, 1.0)?.to_dtype(DType::F32)?;
    let error = (eager - fused)?.abs()?;
    println!(
        "  GQA fused delta: max_abs={:.6}, mean_abs={:.8}",
        error.max_all()?.to_scalar::<f32>()?,
        error.mean_all()?.to_scalar::<f32>()?
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn report_softmax_accuracy(device: &Device, dtype: DType, score_stddev: f32) -> Result<()> {
    let scores = (Tensor::randn(0f32, 1f32, (8, 512, 512), device)? * score_stddev as f64)?
        .to_dtype(dtype)?;
    let reference = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?;
    let native = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(DType::F32)?;
    let error = (reference.clone() - &native)?.abs()?;
    let max_abs = error.max_all()?.to_scalar::<f32>()?;
    let mean_abs = error.mean_all()?.to_scalar::<f32>()?;
    let row_sum_error = (native.sum(candle_core::D::Minus1)? - 1f64)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let reference_argmax = reference
        .argmax(candle_core::D::Minus1)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    let native_argmax = native
        .argmax(candle_core::D::Minus1)?
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<u32>()?;
    let mismatches = reference_argmax
        .iter()
        .zip(native_argmax.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "softmax accuracy {dtype:?} std={score_stddev:<3}: max_abs={max_abs:.6}, \
         mean_abs={mean_abs:.8}, max_row_sum_err={row_sum_error:.6}, argmax_mismatch={mismatches}/{}",
        reference_argmax.len()
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!("VLM Metal microbenchmark (synchronized end-to-end kernel latency)");
    println!("shape: batch=1, heads=8, head_dim=64\n");
    println!("project automatic dtype: {:?}\n", select_dtype(&device));

    println!("-- CPU F32 reference --");
    let cpu = Device::Cpu;
    bench_projection(&cpu, DType::F32)?;
    bench_attention(&cpu, DType::F32, 512, 512)?;
    bench_attention(&cpu, DType::F32, 1, 1024)?;
    println!();

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        println!("-- {dtype:?} --");
        let result: Result<()> = (|| {
            bench_projection(&device, dtype)?;
            bench_attention(&device, dtype, 256, 256)?;
            bench_attention(&device, dtype, 512, 512)?;
            bench_attention(&device, dtype, 1, 256)?;
            bench_attention(&device, dtype, 1, 1024)?;
            bench_gqa(&device, dtype, 512, 512)?;
            bench_gqa(&device, dtype, 1, 1024)?;
            bench_sampling(&device, dtype)?;
            Ok(())
        })();
        if let Err(error) = result {
            println!("unsupported or failed: {error}");
        }
        println!();
    }

    println!("-- native Metal softmax numerical stress --");
    for dtype in [DType::F16, DType::BF16] {
        for score_stddev in [1f32, 4f32, 8f32] {
            report_softmax_accuracy(&device, dtype, score_stddev)?;
        }
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("This benchmark requires macOS and --features metal");
}
