//! The OCR pipeline module.
//!
//! This module provides the task graph-based OCR pipeline implementation that combines
//! multiple model adapters to perform document orientation classification, text
//! detection, text recognition, and text line classification.
//!
//! # Task Graph Architecture
//!
//! The pipeline uses a flexible task graph architecture that allows:
//! - Declarative pipeline configuration via JSON or code
//! - Dynamic model swapping without code changes
//! - Edge processors for data transformation between tasks
//! - Comprehensive validation of pipeline structure
//!
//! See [`TaskGraphBuilder`] and [`TaskGraphConfig`] for details.

mod processors;
mod result;
mod task_graph_builder;
mod task_graph_config;
mod validation;

pub use processors::{EdgeProcessor, EdgeProcessorConfig, EdgeProcessorFactory};
pub use result::{ErrorMetrics, OAROCRResult, TextRegion};
pub use task_graph_builder::TaskGraphBuilder;
pub use task_graph_config::{ModelBinding, TaskGraphConfig, TaskNode};
pub use validation::{
    create_expected_schema, validate_adapter_schema, validate_registry_schemas,
    validate_task_connection, validate_task_dependencies,
};
