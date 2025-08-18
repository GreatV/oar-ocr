//! Shared predictor components to reduce duplication across predictor implementations.

use crate::core::{BatchSampler, DefaultImageReader, OrtInfer, ToBatch};
use crate::processors::NormalizeImage;

/// Generic container of common predictor components.
///
/// This groups the shared pipeline elements used by most predictors so that
/// predictor structs don't have to repeat these fields.
#[derive(Debug)]
pub struct PredictorComponents<PostOp> {
    /// Batch sampler for processing images in batches
    pub batch_sampler: BatchSampler,
    /// Image reader for loading images from file paths
    pub read_image: DefaultImageReader,
    /// Image normalizer for preprocessing images before inference
    pub normalize: NormalizeImage,
    /// Batch converter for converting images to tensors
    pub to_batch: ToBatch,
    /// ONNX Runtime inference engine
    pub infer: OrtInfer,
    /// Post-processing operation
    pub post_op: PostOp,
}
