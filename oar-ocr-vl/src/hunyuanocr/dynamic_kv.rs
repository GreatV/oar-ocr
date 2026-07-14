use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, CustomOp3, InplaceOp3, Layout, Result, Shape};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/hunyuan_dynamic_kv.ptx"));

pub(super) struct DynamicKvAppend {
    pub query_len: usize,
    pub cache_len: usize,
}

pub(super) struct DynamicPagedKvAppend {
    pub query_len: usize,
    pub cache_len: usize,
}

pub(super) struct FusedXdRope {
    pub projection_width: usize,
    pub projection_offset: usize,
    pub num_heads: usize,
    pub query_len: usize,
    pub head_dim: usize,
}

pub(super) struct FusedXdRopeRmsNormF16 {
    pub projection_width: usize,
    pub projection_offset: usize,
    pub num_heads: usize,
    pub query_len: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub include_v: bool,
}

pub(super) struct FusedRmsNormRopeBf16 {
    pub projection_width: usize,
    pub projection_offset: usize,
    pub num_heads: usize,
    pub query_len: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub include_v: bool,
}

pub(super) struct FusedRopeBf16;

pub(super) struct FusedAddRmsNormBf16 {
    pub eps: f32,
}

pub(super) struct FusedSiluMulBf16;

impl CustomOp2 for FusedSiluMulBf16 {
    fn name(&self) -> &'static str {
        "hunyuan-fused-silu-mul-bf16"
    }

    fn cpu_fwd(
        &self,
        _gate: &CpuStorage,
        _gate_layout: &Layout,
        _up: &CpuStorage,
        _up_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused SiLU*up is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        gate: &candle_core::CudaStorage,
        gate_layout: &Layout,
        up: &candle_core::CudaStorage,
        up_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !gate_layout.is_contiguous()
            || !up_layout.is_contiguous()
            || gate_layout.shape() != up_layout.shape()
            || gate.dtype() != candle_core::DType::BF16
            || up.dtype() != candle_core::DType::BF16
        {
            candle_core::bail!("fused SiLU*up requires matching contiguous BF16 tensors")
        }
        let device = gate.device().clone();
        let gate = gate.as_cuda_slice::<half::bf16>()?;
        let up = up.as_cuda_slice::<half::bf16>()?;
        let gate = gate.slice(gate_layout.start_offset()..);
        let up = up.slice(up_layout.start_offset()..);
        let count = gate_layout.shape().elem_count();
        let output = unsafe { device.alloc::<half::bf16>(count) }?;
        let function =
            device.get_or_load_custom_func("silu_mul_bf16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&gate);
        builder.arg(&up);
        builder.arg(&output);
        candle_core::builder_arg!(builder, count as u32);
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            gate_layout.shape().clone(),
        ))
    }
}

impl CustomOp3 for FusedXdRopeRmsNormF16 {
    fn name(&self) -> &'static str {
        "hunyuan-fused-xdrope-rmsnorm-f16"
    }

    fn cpu_fwd(
        &self,
        _qkv: &CpuStorage,
        _qkv_layout: &Layout,
        _cos_sin: &CpuStorage,
        _cos_sin_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused XDRoPE+RMSNorm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        qkv: &candle_core::CudaStorage,
        qkv_layout: &Layout,
        cos_sin: &candle_core::CudaStorage,
        cos_sin_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !qkv_layout.is_contiguous()
            || !cos_sin_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("fused XDRoPE+RMSNorm requires contiguous tensors")
        }
        let (batch, qkv_len, projection_width) = qkv_layout.shape().dims3()?;
        if batch != 1
            || qkv_len != self.query_len
            || projection_width != self.projection_width
            || self.head_dim != 128
            || self.projection_offset + self.num_heads * self.head_dim > projection_width
            || cos_sin_layout.shape().elem_count() != 2 * self.query_len * self.head_dim
            || weight_layout.shape().dims1()? != self.head_dim
            || (self.include_v
                && self.projection_offset + 2 * self.num_heads * self.head_dim > projection_width)
        {
            candle_core::bail!(
                "fused XDRoPE+RMSNorm shape mismatch qkv={:?} cos_sin={:?} weight={:?}",
                qkv_layout.shape(),
                cos_sin_layout.shape(),
                weight_layout.shape()
            )
        }
        let device = qkv.device().clone();
        let qkv = qkv.as_cuda_slice::<half::bf16>()?;
        let cos_sin = cos_sin.as_cuda_slice::<f32>()?;
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let qkv = qkv.slice(qkv_layout.start_offset()..);
        let cos_sin = cos_sin.slice(cos_sin_layout.start_offset()..);
        let weight = weight.slice(weight_layout.start_offset()..);
        let count = self.num_heads * self.query_len * self.head_dim;
        let output_count = if self.include_v { 2 * count } else { count };
        let output = unsafe { device.alloc::<half::f16>(output_count) }?;
        let function =
            device.get_or_load_custom_func("xdrope_rmsnorm_f16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&qkv);
        builder.arg(&cos_sin);
        builder.arg(&weight);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.projection_width as u32,
            self.projection_offset as u32,
            self.num_heads as u32,
            self.query_len as u32,
            self.head_dim as u32,
            self.eps,
            u32::from(self.include_v)
        );
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((self.num_heads * self.query_len) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let shape = if self.include_v {
            Shape::from_dims(&[2, 1, self.num_heads, self.query_len, self.head_dim])
        } else {
            Shape::from_dims(&[1, self.num_heads, self.query_len, self.head_dim])
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            shape,
        ))
    }
}

impl CustomOp3 for FusedRmsNormRopeBf16 {
    fn name(&self) -> &'static str {
        "hunyuan-fused-rmsnorm-rope-bf16"
    }

    fn cpu_fwd(
        &self,
        _qkv: &CpuStorage,
        _qkv_layout: &Layout,
        _cos_sin: &CpuStorage,
        _cos_sin_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused RMSNorm+RoPE is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        qkv: &candle_core::CudaStorage,
        qkv_layout: &Layout,
        cos_sin: &candle_core::CudaStorage,
        cos_sin_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !qkv_layout.is_contiguous()
            || !cos_sin_layout.is_contiguous()
            || !weight_layout.is_contiguous()
        {
            candle_core::bail!("fused RMSNorm+RoPE requires contiguous tensors")
        }
        let (batch, qkv_len, projection_width) = qkv_layout.shape().dims3()?;
        if batch != 1
            || qkv_len != self.query_len
            || projection_width != self.projection_width
            || self.head_dim != 128
            || self.projection_offset + self.num_heads * self.head_dim > projection_width
            || cos_sin_layout.shape().elem_count() != 2 * self.query_len * self.head_dim
            || weight_layout.shape().dims1()? != self.head_dim
            || (self.include_v
                && self.projection_offset + 2 * self.num_heads * self.head_dim > projection_width)
        {
            candle_core::bail!(
                "fused RMSNorm+RoPE shape mismatch qkv={:?} cos_sin={:?} weight={:?}",
                qkv_layout.shape(),
                cos_sin_layout.shape(),
                weight_layout.shape()
            )
        }
        let device = qkv.device().clone();
        let qkv = qkv.as_cuda_slice::<half::bf16>()?;
        let cos_sin = cos_sin.as_cuda_slice::<half::bf16>()?;
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let qkv = qkv.slice(qkv_layout.start_offset()..);
        let cos_sin = cos_sin.slice(cos_sin_layout.start_offset()..);
        let weight = weight.slice(weight_layout.start_offset()..);
        let count = self.num_heads * self.query_len * self.head_dim;
        let output_count = if self.include_v { 2 * count } else { count };
        let output = unsafe { device.alloc::<half::bf16>(output_count) }?;
        let function =
            device.get_or_load_custom_func("rmsnorm_rope_bf16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&qkv);
        builder.arg(&cos_sin);
        builder.arg(&weight);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.projection_width as u32,
            self.projection_offset as u32,
            self.num_heads as u32,
            self.query_len as u32,
            self.head_dim as u32,
            self.eps,
            u32::from(self.include_v)
        );
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: ((self.num_heads * self.query_len) as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let shape = if self.include_v {
            Shape::from_dims(&[2, 1, self.num_heads, self.query_len, self.head_dim])
        } else {
            Shape::from_dims(&[1, self.num_heads, self.query_len, self.head_dim])
        };
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            shape,
        ))
    }
}

impl CustomOp3 for FusedAddRmsNormBf16 {
    fn name(&self) -> &'static str {
        "hunyuan-fused-add-rmsnorm-bf16"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _delta: &CpuStorage,
        _delta_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused add+RMSNorm is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        delta: &candle_core::CudaStorage,
        delta_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !input_layout.is_contiguous()
            || !delta_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || input_layout.shape() != delta_layout.shape()
        {
            candle_core::bail!("fused add+RMSNorm requires matching contiguous tensors")
        }
        let dims = input_layout.shape().dims();
        let ncols = *dims
            .last()
            .ok_or_else(|| candle_core::Error::Msg("RMSNorm input is scalar".into()))?;
        if ncols != 1024 || weight_layout.shape().dims1()? != ncols {
            candle_core::bail!(
                "fused add+RMSNorm requires hidden width 1024, got input={:?} weight={:?}",
                input_layout.shape(),
                weight_layout.shape()
            )
        }
        let element_count = input_layout.shape().elem_count();
        let rows = element_count / ncols;
        let device = input.device().clone();
        let input = input.as_cuda_slice::<half::bf16>()?;
        let delta = delta.as_cuda_slice::<half::bf16>()?;
        let weight = weight.as_cuda_slice::<half::bf16>()?;
        let input = input.slice(input_layout.start_offset()..);
        let delta = delta.slice(delta_layout.start_offset()..);
        let weight = weight.slice(weight_layout.start_offset()..);
        let output = unsafe { device.alloc::<half::bf16>(2 * element_count) }?;
        let function =
            device.get_or_load_custom_func("add_rmsnorm_bf16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&input);
        builder.arg(&delta);
        builder.arg(&weight);
        builder.arg(&output);
        candle_core::builder_arg!(builder, ncols as u32, self.eps);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (ncols as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
        let mut output_shape = Vec::with_capacity(dims.len() + 1);
        output_shape.push(2);
        output_shape.extend_from_slice(dims);
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, device),
            Shape::from_dims(&output_shape),
        ))
    }
}

impl InplaceOp3 for FusedRopeBf16 {
    fn name(&self) -> &'static str {
        "hunyuan-fused-rope-bf16"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _cos_sin: &CpuStorage,
        _cos_sin_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("fused BF16 RoPE is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        output: &mut candle_core::CudaStorage,
        output_layout: &Layout,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        cos_sin: &candle_core::CudaStorage,
        cos_sin_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !output_layout.is_contiguous()
            || !input_layout.is_contiguous()
            || !cos_sin_layout.is_contiguous()
            || output_layout.shape() != input_layout.shape()
        {
            candle_core::bail!("fused BF16 RoPE requires matching contiguous tensors")
        }
        let (_, heads, query_len, head_dim) = output_layout.shape().dims4()?;
        if !head_dim.is_multiple_of(2)
            || cos_sin_layout.shape().elem_count() != 2 * query_len * head_dim
        {
            candle_core::bail!(
                "fused BF16 RoPE shape mismatch input={:?} cos_sin={:?}",
                input_layout.shape(),
                cos_sin_layout.shape()
            )
        }
        let device = output.device().clone();
        let output = output.as_cuda_slice_mut::<half::bf16>()?;
        let input = input.as_cuda_slice::<half::bf16>()?;
        let cos_sin = cos_sin.as_cuda_slice::<half::bf16>()?;
        let mut output = output.slice_mut(output_layout.start_offset()..);
        let input = input.slice(input_layout.start_offset()..);
        let cos_sin = cos_sin.slice(cos_sin_layout.start_offset()..);
        let count = heads * query_len * head_dim;
        let function = device.get_or_load_custom_func("rope_bf16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&mut output);
        builder.arg(&input);
        builder.arg(&cos_sin);
        candle_core::builder_arg!(builder, heads as u32, query_len as u32, head_dim as u32);
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok(())
    }
}

impl InplaceOp3 for FusedXdRope {
    fn name(&self) -> &'static str {
        "hunyuan-fused-xdrope"
    }

    fn cpu_fwd(
        &self,
        _output: &mut CpuStorage,
        _output_layout: &Layout,
        _qkv: &CpuStorage,
        _qkv_layout: &Layout,
        _cos_sin: &CpuStorage,
        _cos_sin_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("fused XDRoPE is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        output: &mut candle_core::CudaStorage,
        output_layout: &Layout,
        qkv: &candle_core::CudaStorage,
        qkv_layout: &Layout,
        cos_sin: &candle_core::CudaStorage,
        cos_sin_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !output_layout.is_contiguous()
            || !qkv_layout.is_contiguous()
            || !cos_sin_layout.is_contiguous()
        {
            candle_core::bail!("fused XDRoPE requires contiguous tensors")
        }
        let (_, heads, query_len, head_dim) = output_layout.shape().dims4()?;
        let (_, qkv_len, projection_width) = qkv_layout.shape().dims3()?;
        if heads != self.num_heads
            || query_len != self.query_len
            || qkv_len != query_len
            || head_dim != self.head_dim
            || projection_width != self.projection_width
            || self.projection_offset + heads * head_dim > projection_width
            || cos_sin_layout.shape().elem_count() != 2 * query_len * head_dim
        {
            candle_core::bail!(
                "fused XDRoPE shape mismatch output={:?} qkv={:?} cos_sin={:?}",
                output_layout.shape(),
                qkv_layout.shape(),
                cos_sin_layout.shape()
            )
        }
        let device = output.device().clone();
        let output = output.as_cuda_slice_mut::<half::bf16>()?;
        let qkv = qkv.as_cuda_slice::<half::bf16>()?;
        let cos_sin = cos_sin.as_cuda_slice::<f32>()?;
        let mut output = output.slice_mut(output_layout.start_offset()..);
        let qkv = qkv.slice(qkv_layout.start_offset()..);
        let cos_sin = cos_sin.slice(cos_sin_layout.start_offset()..);
        let count = heads * query_len * head_dim;
        let function = device.get_or_load_custom_func("xdrope_bf16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&mut output);
        builder.arg(&qkv);
        builder.arg(&cos_sin);
        candle_core::builder_arg!(
            builder,
            projection_width as u32,
            self.projection_offset as u32,
            heads as u32,
            query_len as u32,
            head_dim as u32
        );
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok(())
    }
}

impl InplaceOp3 for DynamicKvAppend {
    fn name(&self) -> &'static str {
        "hunyuan-dynamic-kv-append"
    }

    fn cpu_fwd(
        &self,
        _cache: &mut CpuStorage,
        _cache_layout: &Layout,
        _source: &CpuStorage,
        _source_layout: &Layout,
        _lengths: &CpuStorage,
        _lengths_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("dynamic KV append is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        cache: &mut candle_core::CudaStorage,
        cache_layout: &Layout,
        source: &candle_core::CudaStorage,
        source_layout: &Layout,
        lengths: &candle_core::CudaStorage,
        lengths_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        let (_, num_heads, cache_len, head_dim) = cache_layout.shape().dims4()?;
        let (_, source_heads, query_len, source_head_dim) = source_layout.shape().dims4()?;
        if num_heads != source_heads
            || head_dim != source_head_dim
            || query_len != self.query_len
            || cache_len != self.cache_len
        {
            candle_core::bail!(
                "dynamic KV shape mismatch cache={:?} source={:?}",
                cache_layout.shape(),
                source_layout.shape()
            )
        }
        if lengths_layout.shape().dims1()? != 2 {
            candle_core::bail!("dynamic KV cumulative lengths must have shape [2]")
        }
        let device = cache.device().clone();
        let lengths = lengths.as_cuda_slice::<u32>()?;
        let lengths = lengths.slice(lengths_layout.start_offset()..);
        let count = num_heads * query_len * head_dim;
        macro_rules! launch {
            ($ty:ty, $function:literal) => {{
                let cache = cache.as_cuda_slice_mut::<$ty>()?;
                let source = source.as_cuda_slice::<$ty>()?;
                let mut cache = cache.slice_mut(cache_layout.start_offset()..);
                let source = source.slice(source_layout.start_offset()..);
                let function =
                    device.get_or_load_custom_func($function, "hunyuan_dynamic_kv", PTX)?;
                let mut builder = function.builder();
                builder.arg(&mut cache);
                builder.arg(&source);
                builder.arg(&lengths);
                candle_core::builder_arg!(
                    builder,
                    query_len as u32,
                    num_heads as u32,
                    head_dim as u32,
                    cache_len as u32
                );
                unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
            }};
        }
        match cache.dtype() {
            candle_core::DType::F16 => launch!(half::f16, "append_kv_f16"),
            candle_core::DType::BF16 => launch!(half::bf16, "append_kv_bf16"),
            dtype => candle_core::bail!("dynamic KV append does not support {dtype:?}"),
        }
        Ok(())
    }
}

impl InplaceOp3 for DynamicPagedKvAppend {
    fn name(&self) -> &'static str {
        "hunyuan-dynamic-paged-kv-append"
    }

    fn cpu_fwd(
        &self,
        _cache: &mut CpuStorage,
        _cache_layout: &Layout,
        _source: &CpuStorage,
        _source_layout: &Layout,
        _lengths: &CpuStorage,
        _lengths_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("dynamic paged KV append is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        cache: &mut candle_core::CudaStorage,
        cache_layout: &Layout,
        source: &candle_core::CudaStorage,
        source_layout: &Layout,
        lengths: &candle_core::CudaStorage,
        lengths_layout: &Layout,
    ) -> Result<()> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};

        let (blocks, page_size, num_heads, head_dim) = cache_layout.shape().dims4()?;
        let (_, source_heads, query_len, source_head_dim) = source_layout.shape().dims4()?;
        if num_heads != source_heads
            || head_dim != source_head_dim
            || query_len != self.query_len
            || blocks * page_size != self.cache_len
        {
            candle_core::bail!(
                "dynamic paged KV shape mismatch cache={:?} source={:?}",
                cache_layout.shape(),
                source_layout.shape()
            )
        }
        if lengths_layout.shape().dims1()? != 2 {
            candle_core::bail!("dynamic paged KV cumulative lengths must have shape [2]")
        }
        if cache.dtype() != candle_core::DType::BF16 || source.dtype() != candle_core::DType::BF16 {
            candle_core::bail!("dynamic paged KV append requires BF16")
        }
        let device = cache.device().clone();
        let cache = cache.as_cuda_slice_mut::<half::bf16>()?;
        let source = source.as_cuda_slice::<half::bf16>()?;
        let lengths = lengths.as_cuda_slice::<u32>()?;
        let mut cache = cache.slice_mut(cache_layout.start_offset()..);
        let source = source.slice(source_layout.start_offset()..);
        let lengths = lengths.slice(lengths_layout.start_offset()..);
        let count = num_heads * query_len * head_dim;
        let function =
            device.get_or_load_custom_func("append_paged_kv_bf16", "hunyuan_dynamic_kv", PTX)?;
        let mut builder = function.builder();
        builder.arg(&mut cache);
        builder.arg(&source);
        builder.arg(&lengths);
        candle_core::builder_arg!(
            builder,
            query_len as u32,
            num_heads as u32,
            head_dim as u32,
            self.cache_len as u32
        );
        unsafe { builder.launch(LaunchConfig::for_num_elems(count as u32)) }.w()?;
        Ok(())
    }
}
