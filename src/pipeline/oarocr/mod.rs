//! Task graph-based OCR pipeline implementation.

mod image_processing;
mod processors;
mod result;
mod task_graph_builder;
mod task_graph_config;
mod validation;

pub use image_processing::ImageProcessor;
pub use processors::{EdgeProcessor, EdgeProcessorConfig, EdgeProcessorFactory};
pub use result::{ErrorMetrics, OAROCRResult, TextRegion};
pub use task_graph_builder::TaskGraphBuilder;
pub use task_graph_config::{ModelBinding, TaskGraphConfig, TaskNode};
pub use validation::{
    create_expected_schema, validate_adapter_schema, validate_registry_schemas,
    validate_task_connection, validate_task_dependencies,
};
