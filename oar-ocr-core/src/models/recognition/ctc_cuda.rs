//! Compact CUDA readback without modifying the recognizer's graph/fusions.
use crate::core::OCRError;
use crate::processors::{CTCArgmaxOutput, CTCLabelDecode};
use ort::{ep, memory::AllocationDevice, session::Session, value::Tensor};
use std::sync::Mutex;

// ONNX IR 8 / opset 13, float x[B,T,V]:
// reversed = Slice(x, starts=[-1], ends=[i64::MIN], axes=[2], steps=[-1]);
// indices = ArgMax(reversed, axis=2, keepdims=0); // first on reverse = last on x
// maxima = ReduceMax(x, axes=[2], keepdims=0);
// sums = ReduceSum(x, axes=[2], keepdims=0).
// select_last_index=1 forces ORT ArgMax onto CPU; reversing keeps all nodes on
// CUDA. Sum is only an exception sentinel, never a returned confidence.
const GRAPH: &[u8] = include_bytes!("ctc_reduce.onnx");

#[derive(Debug, Default)]
pub(super) struct CtcReducer(Mutex<Option<Session>>);

impl CtcReducer {
    pub(super) fn reduce(
        &self,
        input: &Tensor<f32>,
        device: i32,
        decoder: &CTCLabelDecode,
    ) -> Result<CTCArgmaxOutput, OCRError> {
        self.reduce_inner(input, device, decoder)
            .map_err(|e| OCRError::Inference {
                model_name: "CRNN".into(),
                context: "CUDA CTC reduction".into(),
                source: Box::new(e),
            })
    }

    fn reduce_inner(
        &self,
        input: &Tensor<f32>,
        device: i32,
        decoder: &CTCLabelDecode,
    ) -> ort::Result<CTCArgmaxOutput> {
        let shape = input.shape();
        let [batch, time, vocab] = shape.as_ref() else {
            return Err(ort::Error::new("expected CTC output [batch, time, vocab]"));
        };
        let (batch, time, vocab) = (*batch as usize, *time as usize, *vocab as usize);
        if batch == 0 || time == 0 || vocab == 0 {
            return host_reduce(input, decoder);
        }
        let mut guard = self
            .0
            .lock()
            .map_err(|_| ort::Error::new("CTC reducer lock poisoned"))?;
        if guard.is_none() {
            *guard = Some(
                Session::builder()?
                    .with_intra_threads(1)?
                    .with_inter_threads(1)?
                    .with_execution_providers([ep::CUDA::default()
                        .with_device_id(device)
                        .build()
                        .error_on_failure()])?
                    .with_disable_cpu_fallback()?
                    .commit_from_memory(GRAPH)?,
            );
        }
        let session = guard.as_mut().expect("initialized reducer");
        // The reverse buffer can be hundreds of MB. Return its idle arena
        // allocations after reduction rather than retaining them beside the
        // structure pipeline's other model sessions.
        let mut options = ort::session::RunOptions::new()?;
        options.set(
            "memory.enable_memory_arena_shrinkage",
            format!("gpu:{device}"),
        )?;
        let outputs = session.run_with_options(ort::inputs![input], &options)?;
        let (_, indices) = outputs["indices"].try_extract_tensor::<i64>()?;
        let (_, probabilities) = outputs["maxima"].try_extract_tensor::<f32>()?;
        let (_, sums) = outputs["sums"].try_extract_tensor::<f32>()?;
        // Host SIMD NaN and signed-zero confidence semantics are target dependent.
        // CUDA ReduceMax also seeds some reductions with -FLT_MAX, not -infinity.
        // Conservatively retain the unchanged host path for the whole batch if
        // any row has a nonfinite sum or zero maximum. Overflow may over-trigger
        // this fallback, but healthy softmax probabilities do not overflow.
        if sums.iter().any(|v| !v.is_finite()) || probabilities.contains(&0.0) {
            drop(outputs);
            drop(guard);
            return host_reduce(input, decoder);
        }
        Ok(CTCArgmaxOutput {
            batch_size: batch,
            sequence_length: time,
            indices: indices
                .iter()
                .map(|&index| vocab - 1 - index as usize)
                .collect(),
            probabilities: probabilities.to_vec(),
        })
    }
}

fn host_reduce(input: &Tensor<f32>, decoder: &CTCLabelDecode) -> ort::Result<CTCArgmaxOutput> {
    let host = input.to(AllocationDevice::CPU, 0)?;
    let (shape, data) = host.try_extract_tensor::<f32>()?;
    let view = ndarray::ArrayView3::from_shape(
        (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        data,
    )
    .map_err(|e| ort::Error::new(e.to_string()))?;
    Ok(decoder.argmax_predictions(&view))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_ctc_reduction_matches_host_bits() -> ort::Result<()> {
        use ort::ep::ExecutionProvider;
        let mut builder = Session::builder()?;
        if ep::CUDA::default().register(&mut builder).is_err() || ep::get_gpu_device().is_err() {
            eprintln!("skipping CUDA CTC parity: no CUDA device");
            return Ok(());
        }
        let decoder = CTCLabelDecode::new(None, false);
        let reducer = CtcReducer::default();
        for width in [1, 7, 8, 9, 16, 17, 32, 18708] {
            let mut rows: Vec<Vec<f32>> = (0..32)
                .map(|seed| {
                    (0..width)
                        .map(|i| ((i * 37 + seed * 11) % 41) as f32 - 20.0)
                        .collect()
                })
                .collect();
            rows.extend([
                vec![f32::NEG_INFINITY; width],
                vec![f32::INFINITY; width],
                vec![f32::from_bits(1); width],
                vec![-f32::from_bits(1); width],
                (0..width)
                    .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
                    .collect(),
                vec![f32::from_bits(0x7fc01234); width],
            ]);
            for pos in [0, width / 2, width - 1] {
                let mut row = vec![5.0; width];
                row[0] = 9.0;
                row[pos] = f32::NAN;
                rows.push(row);
            }
            let subnormals: Vec<Vec<f32>> = [1u32, 7, 0x7fffff, 0x800000, 0x80000001, 0x807fffff]
                .into_iter()
                .map(|bits| {
                    let mut row = vec![-f32::MIN_POSITIVE; width];
                    row[width / 2] = f32::from_bits(bits);
                    row
                })
                .collect();
            // Separate finite and exceptional batches so the test exercises both
            // the device reduction and the conservative full-batch fallback.
            for rows in [&rows[..32], subnormals.as_slice(), &rows[32..]] {
                let data: Vec<_> = rows.iter().flatten().copied().collect();
                let view = ndarray::ArrayView3::from_shape((1, rows.len(), width), &data).unwrap();
                let expected = decoder.argmax_predictions(&view);
                let input = Tensor::from_array(([1, rows.len(), width], data))?
                    .to(AllocationDevice::CUDA, 0)?;
                let actual = reducer.reduce_inner(&input, 0, &decoder)?;
                assert_eq!(actual.indices, expected.indices, "width={width}");
                assert_eq!(
                    actual
                        .probabilities
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .probabilities
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "width={width}"
                );
            }
        }
        Ok(())
    }
}
