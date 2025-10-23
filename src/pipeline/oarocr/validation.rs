//! Validation utilities for task output schemas and model compatibility.
//!
//! This module provides validation functions to ensure that models produce outputs
//! compatible with what downstream tasks expect, and to surface mismatches early.

use crate::core::traits::{ModelAdapter, TaskSchema, TaskType};
use crate::core::{ModelRegistry, OCRError};
use std::collections::HashMap;

/// Validates that a model adapter is compatible with a task schema.
///
/// # Arguments
///
/// * `adapter` - The model adapter to validate
/// * `expected_schema` - The expected task schema
///
/// # Returns
///
/// Result indicating success or a detailed error describing the incompatibility
pub fn validate_adapter_schema<A: ModelAdapter>(
    adapter: &A,
    expected_schema: &TaskSchema,
) -> Result<(), OCRError> {
    let adapter_schema = adapter.schema();

    // Check task type compatibility
    if adapter_schema.task_type != expected_schema.task_type {
        return Err(OCRError::ConfigError {
            message: format!(
                "Task type mismatch: adapter produces {:?} but expected {:?}",
                adapter_schema.task_type, expected_schema.task_type
            ),
        });
    }

    // Check input types compatibility
    for expected_input in &expected_schema.input_types {
        if !adapter_schema.input_types.contains(expected_input) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Input type mismatch: adapter does not support input type '{}'",
                    expected_input
                ),
            });
        }
    }

    // Check output types compatibility
    for expected_output in &expected_schema.output_types {
        if !adapter_schema.output_types.contains(expected_output) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Output type mismatch: adapter does not produce output type '{}'",
                    expected_output
                ),
            });
        }
    }

    Ok(())
}

/// Validates that all adapters in a registry are compatible with their task types.
///
/// # Arguments
///
/// * `registry` - The model registry to validate
///
/// # Returns
///
/// Result indicating success or a detailed error describing incompatibilities
pub fn validate_registry_schemas(registry: &ModelRegistry) -> Result<(), OCRError> {
    let adapters = registry.list_all_with_ids()?;

    for (registry_id, adapter_info) in adapters {
        // Create expected schema for this task type
        let expected_schema = create_expected_schema(adapter_info.task_type);

        // Look up the adapter
        let adapter = registry
            .lookup(adapter_info.task_type, &registry_id)?
            .ok_or_else(|| OCRError::ConfigError {
                message: format!(
                    "Adapter '{}' (id '{}') not found in registry",
                    adapter_info.model_name, registry_id
                ),
            })?;

        // Validate schema compatibility
        if adapter.task_type() != expected_schema.task_type {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Adapter '{}' (id '{}') has task type {:?} but expected {:?}",
                    adapter_info.model_name,
                    registry_id,
                    adapter.task_type(),
                    expected_schema.task_type
                ),
            });
        }
    }

    Ok(())
}

/// Creates an expected schema for a given task type.
///
/// # Arguments
///
/// * `task_type` - The task type to create a schema for
///
/// # Returns
///
/// The expected task schema for the given task type
pub fn create_expected_schema(task_type: TaskType) -> TaskSchema {
    match task_type {
        TaskType::TextDetection => TaskSchema::new(
            TaskType::TextDetection,
            vec!["image".to_string()],
            vec!["text_boxes".to_string(), "scores".to_string()],
        ),
        TaskType::TextRecognition => TaskSchema::new(
            TaskType::TextRecognition,
            vec!["text_boxes".to_string()],
            vec!["text_strings".to_string(), "scores".to_string()],
        ),
        TaskType::DocumentOrientation => TaskSchema::new(
            TaskType::DocumentOrientation,
            vec!["image".to_string()],
            vec!["orientation_labels".to_string(), "scores".to_string()],
        ),
        TaskType::TextLineOrientation => TaskSchema::new(
            TaskType::TextLineOrientation,
            vec!["image".to_string()],
            vec!["orientation_labels".to_string(), "scores".to_string()],
        ),
        TaskType::DocumentRectification => TaskSchema::new(
            TaskType::DocumentRectification,
            vec!["image".to_string()],
            vec!["rectified_image".to_string()],
        ),
        TaskType::LayoutDetection => TaskSchema::new(
            TaskType::LayoutDetection,
            vec!["image".to_string()],
            vec!["layout_elements".to_string()],
        ),
        TaskType::TableCellDetection => TaskSchema::new(
            TaskType::TableCellDetection,
            vec!["image".to_string()],
            vec!["table_cells".to_string()],
        ),
        TaskType::FormulaRecognition => TaskSchema::new(
            TaskType::FormulaRecognition,
            vec!["image".to_string()],
            vec!["latex_formula".to_string(), "confidence".to_string()],
        ),
        TaskType::SealTextDetection => TaskSchema::new(
            TaskType::SealTextDetection,
            vec!["image".to_string()],
            vec!["seal_text_boxes".to_string(), "scores".to_string()],
        ),
    }
}

/// Validates a pipeline configuration to ensure all task dependencies are satisfied.
///
/// # Arguments
///
/// * `task_graph` - Map of task IDs to their dependencies
///
/// # Returns
///
/// Result indicating success or a detailed error describing missing dependencies
pub fn validate_task_dependencies(
    task_graph: &HashMap<String, Vec<String>>,
) -> Result<(), OCRError> {
    // Check that all dependencies exist
    for (task_id, dependencies) in task_graph {
        for dep in dependencies {
            if !task_graph.contains_key(dep) {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "Task '{}' depends on '{}' which does not exist",
                        task_id, dep
                    ),
                });
            }
        }
    }

    // Check for circular dependencies
    if has_circular_dependencies(task_graph) {
        return Err(OCRError::ConfigError {
            message: "Task graph contains circular dependencies".to_string(),
        });
    }

    Ok(())
}

/// Checks if a task graph has circular dependencies.
fn has_circular_dependencies(task_graph: &HashMap<String, Vec<String>>) -> bool {
    use std::collections::HashSet;

    fn has_cycle(
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if has_cycle(neighbor, graph, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(neighbor) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    for node in task_graph.keys() {
        if !visited.contains(node) && has_cycle(node, task_graph, &mut visited, &mut rec_stack) {
            return true;
        }
    }

    false
}

/// Validates that task outputs match expected inputs for downstream tasks.
///
/// # Arguments
///
/// * `upstream_task` - The upstream task type
/// * `downstream_task` - The downstream task type
///
/// # Returns
///
/// Result indicating success or a detailed error describing the mismatch
pub fn validate_task_connection(
    upstream_task: TaskType,
    downstream_task: TaskType,
) -> Result<(), OCRError> {
    let upstream_schema = create_expected_schema(upstream_task);
    let downstream_schema = create_expected_schema(downstream_task);

    // Check if any upstream output matches any downstream input
    let has_compatible_connection = upstream_schema
        .output_types
        .iter()
        .any(|output| downstream_schema.input_types.contains(output));

    if !has_compatible_connection {
        return Err(OCRError::ConfigError {
            message: format!(
                "Task connection mismatch: {:?} outputs {:?} but {:?} requires {:?}",
                upstream_task,
                upstream_schema.output_types,
                downstream_task,
                downstream_schema.input_types
            ),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_expected_schema() {
        let schema = create_expected_schema(TaskType::TextDetection);
        assert_eq!(schema.task_type, TaskType::TextDetection);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"text_boxes".to_string()));
    }

    #[test]
    fn test_validate_task_dependencies_no_cycles() {
        let mut graph = HashMap::new();
        graph.insert("task1".to_string(), vec![]);
        graph.insert("task2".to_string(), vec!["task1".to_string()]);
        graph.insert("task3".to_string(), vec!["task2".to_string()]);

        assert!(validate_task_dependencies(&graph).is_ok());
    }

    #[test]
    fn test_validate_task_dependencies_with_cycle() {
        let mut graph = HashMap::new();
        graph.insert("task1".to_string(), vec!["task2".to_string()]);
        graph.insert("task2".to_string(), vec!["task1".to_string()]);

        assert!(validate_task_dependencies(&graph).is_err());
    }

    #[test]
    fn test_validate_task_connection() {
        // Detection -> Recognition should be valid
        assert!(
            validate_task_connection(TaskType::TextDetection, TaskType::TextRecognition).is_ok()
        );

        // Recognition -> Detection should be invalid
        assert!(
            validate_task_connection(TaskType::TextRecognition, TaskType::TextDetection).is_err()
        );
    }
}
