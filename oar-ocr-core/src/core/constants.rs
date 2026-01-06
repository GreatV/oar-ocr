//! Constants used throughout the OCR pipeline.
//!
//! This module defines various constants that are used across different
//! components of the OCR pipeline, such as default values for image processing
//! parameters, batch sizes, and tensor size limits.

/// The default maximum width for images.
///
/// This constant defines the maximum width that an image can have
/// when processing in the OCR pipeline.
pub const DEFAULT_MAX_IMG_WIDTH: usize = 3200;

/// The default maximum side limit for images.
///
/// This constant defines the maximum size for any side of an image
/// when processing in the OCR pipeline.
pub const DEFAULT_MAX_SIDE_LIMIT: u32 = 4000;

/// The default limit for the side length of images.
///
/// This constant defines the default size to which image sides
/// are limited during processing in the OCR pipeline.
pub const DEFAULT_LIMIT_SIDE_LEN: u32 = 736;

/// The default threshold for parallel processing.
///
/// This constant defines the minimum number of items that need
/// to be processed before parallel processing is used.
pub const DEFAULT_PARALLEL_THRESHOLD: usize = 4;

/// The default shape for recognition images.
///
/// This constant defines the default shape (channels, height, width)
/// for images used in the recognition phase of the OCR pipeline.
pub const DEFAULT_REC_IMAGE_SHAPE: [usize; 3] = [3, 48, 320];

/// The default batch size for processing.
///
/// This constant defines the default number of items processed
/// together in a batch in the OCR pipeline.
pub const DEFAULT_BATCH_SIZE: usize = 6;

/// The default value for top-k selection.
///
/// This constant defines the default number of top results
/// to select in classification tasks.
pub const DEFAULT_TOPK: usize = 4;

/// The default input shape for classification.
///
/// This constant defines the default shape (height, width)
/// for images used in classification tasks.
pub const DEFAULT_CLASSIFICATION_INPUT_SHAPE: (u32, u32) = (224, 224);

/// The maximum allowed tensor size.
///
/// This constant defines the maximum number of elements
/// allowed in a tensor to prevent memory issues.
pub const MAX_TENSOR_SIZE: usize = 100_000_000;
