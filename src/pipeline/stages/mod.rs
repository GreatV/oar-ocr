//! Shared stage processors for the OCR pipeline.
//!
//! This module provides reusable stage processors that eliminate code duplication
//! across different pipeline flows (single-image, in-memory, dynamic batching,
//! detection-from-memory). Each processor encapsulates the logic for a specific
//! pipeline stage with consistent error handling, logging, and performance characteristics.
//!
//! # Extensible Pipeline Architecture
//!
//! The pipeline supports extensible stages through the `PipelineStage` trait and
//! `StageRegistry` system, allowing new stages to be added without modifying core
//! pipeline code.
//!
//! # Stage Processor Pattern Abstraction
//!
//! This module includes helper utilities that reduce code duplication across
//! stage processors by abstracting common lifecycle patterns:
//!
//! - **Timing management** - Automatic start/stop timing for operations
//! - **Empty input handling** - Consistent behavior for empty input collections
//! - **Parallel processing** - Automatic decisions between sequential and parallel processing
//! - **Metrics collection** - Standardized success/failure counting and metadata
//! - **Result wrapping** - Consistent `StageResult` creation with metrics
//!
//! These abstractions reduce repetitive code by approximately 25-35% across
//! the orientation, cropping, and recognition processors while maintaining
//! the same functionality and API compatibility.
//!
//! See [`processor_helper`] for the concrete helper implementations.

pub mod config;
mod cropping;
mod extensible;
mod orientation;
mod processor_helper;
mod recognition;
mod registry;
mod text_detection;
mod types;

// Re-export public types and processors
pub use cropping::{
    CroppingConfig, CroppingResult, CroppingStageProcessor, ExtensibleCroppingStage,
};
pub use orientation::{
    ExtensibleOrientationStage, OrientationConfig, OrientationResult, OrientationStageProcessor,
};
pub use processor_helper::{
    BatchConfig, BatchProcessor, SingleItemProcessor, StageAlgorithm, process_items,
    run_with_metrics, run_with_metrics_and_fallback,
};
pub use recognition::{
    ExtensibleRecognitionStage, GroupingStrategy, GroupingStrategyConfig, GroupingStrategyFactory,
    OrientationCorrectionConfig, OrientationCorrector, RecognitionConfig, RecognitionResult,
    RecognitionStageProcessor,
};
pub use types::{StageMetrics, StageProcessor, StageProcessorHelper, StageResult};

// Re-export extensible pipeline components
pub use config::{ExtensiblePipelineConfig, GlobalPipelineSettings};
pub use extensible::{PipelineStage, StageContext, StageData, StageDependency, StageId};
pub use registry::{ExtensiblePipeline, PipelineExecutor};

// Re-export example stages
pub use text_detection::{
    ExtensibleTextDetectionStage, ExtensibleTextLineOrientationStage, TextDetectionConfig,
    TextDetectionResult, TextLineOrientationConfig,
};
