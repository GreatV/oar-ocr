//! Constants used throughout the OCR pipeline.
//!
//! This module defines various constants that are used across different
//! components of the OCR pipeline, such as default values for image processing
//! parameters, batch sizes, and tensor size limits.

/// The default maximum width for images in the OCR pipeline.
pub const DEFAULT_MAX_IMG_WIDTH: usize = 3200;

/// The default maximum size for any side of an image.
pub const DEFAULT_MAX_SIDE_LIMIT: u32 = 4000;

/// The default size to which image sides are limited during processing.
pub const DEFAULT_LIMIT_SIDE_LEN: u32 = 736;

/// The minimum number of items required before parallel processing is used.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 4;

/// The default shape (channels, height, width) for recognition images.
pub const DEFAULT_REC_IMAGE_SHAPE: [usize; 3] = [3, 48, 320];

/// The default number of items processed together in a batch.
pub const DEFAULT_BATCH_SIZE: usize = 6;

/// The default number of top results to select in classification tasks.
pub const DEFAULT_TOPK: usize = 4;

/// The default shape (height, width) for classification images.
pub const DEFAULT_CLASSIFICATION_INPUT_SHAPE: (u32, u32) = (224, 224);

/// The maximum number of elements allowed in a tensor, to prevent memory issues.
pub const MAX_TENSOR_SIZE: usize = 100_000_000;
