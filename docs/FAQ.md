# Frequently Asked Questions

Common build and runtime questions, distilled from GitHub issues.

## Windows: linking fails with `LNK2001: unresolved external symbol __std_find_trivial_8`

Unresolved `__std_*` symbols (`__std_find_trivial_*`, `__std_max_element_*`, `__std_search_*`, ...) mean the linker's MSVC STL is too old. The prebuilt ONNX Runtime binaries that `ort-sys` downloads are compiled with Visual Studio 2022 (MSVC v143) and reference vectorized STL helpers that do not exist in the VS 2019 (v142) link libraries.

Fix:

1. Install **Visual Studio 2022** or **Build Tools for Visual Studio 2022** with the "Desktop development with C++" workload (MSVC v143 + Windows SDK).
2. Run `cargo clean` and rebuild. Rustc picks the newest installed MSVC toolset automatically.

VS 2019 and VS 2022 build tools can coexist. Only the newer one needs to be present for linking. See issue [#105](https://github.com/GreatV/oar-ocr/issues/105).

## GPU inference is slower than CPU for PP-OCRv6 tiny/small

Expected. The tiny and small models are so compact that per-call overhead, including host-to-device tensor copies, kernel launches, CPU/GPU synchronization, and CPU pre/post-processing, can outweigh the computation saved by the GPU. The following results were measured on an RTX 4090 with an i9-13900KF using a single image and excluding warmup:

| Model  | CPU        | GPU (CUDA EP)             |
| ------ | ---------- | ------------------------- |
| tiny   | 34 ms/img  | 44 ms/img                 |
| small  | 59 ms/img  | 77 ms/img                 |
| medium | 404 ms/img | **173 ms/img** (2.3× faster) |

Guidelines:

- For tiny/small, use the default CPU mode (the `simd` feature is on by default).
- Use the medium model, or batch several images per `predict()` call, when GPU acceleration matters.
- Exclude the first call when benchmarking because it includes cuDNN initialization and algorithm selection (~5× slower than steady state).

Also note that requesting `OrtExecutionProvider::CUDA` without building with `--features cuda` makes the pipeline builder return an error. Check the `Result` of `.build()`. Without the `cuda` feature, the downloaded ONNX Runtime is CPU-only and the GPU is never used. See issue [#151](https://github.com/GreatV/oar-ocr/issues/151).
