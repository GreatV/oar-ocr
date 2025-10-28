//! Task graph configuration for the OAROCR pipeline.
//!
//! This module provides configuration types for declaring task graphs and model bindings,
//! enabling flexible pipeline composition and model swapping.

use crate::core::traits::TaskType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for a single model binding in the task graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBinding {
    /// Name of the model (e.g., "DB", "CRNN", "RT-DETR")
    pub model_name: String,

    /// Path to the model file (ONNX, etc.)
    pub model_path: PathBuf,

    /// Task type this model handles
    pub task_type: TaskType,

    /// Optional model-specific configuration as JSON
    #[serde(default)]
    pub config: Option<serde_json::Value>,

    /// Batch size for this model
    #[serde(default)]
    pub batch_size: Option<usize>,

    /// Session pool size for ONNX runtime
    #[serde(default)]
    pub session_pool_size: Option<usize>,

    /// Enable logging for this model
    #[serde(default)]
    pub enable_logging: Option<bool>,
}

impl ModelBinding {
    /// Creates a new model binding.
    pub fn new(
        model_name: impl Into<String>,
        model_path: impl Into<PathBuf>,
        task_type: TaskType,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            model_path: model_path.into(),
            task_type,
            config: None,
            batch_size: None,
            session_pool_size: None,
            enable_logging: None,
        }
    }

    /// Sets the model-specific configuration.
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = Some(config);
        self
    }

    /// Sets the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets the session pool size.
    pub fn with_session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = Some(size);
        self
    }

    /// Sets whether to enable logging.
    pub fn with_logging(mut self, enable: bool) -> Self {
        self.enable_logging = Some(enable);
        self
    }
}

/// Represents a node in the task graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    /// Unique identifier for this task node
    pub id: String,

    /// Task type for this node
    pub task_type: TaskType,

    /// Model binding for this task (references a model by name)
    pub model_binding: String,

    /// Dependencies - IDs of tasks that must complete before this one
    #[serde(default)]
    pub dependencies: Vec<String>,

    /// Edge processors - processors to apply to data from dependencies
    /// Map from dependency ID to processor configuration
    #[serde(default)]
    pub edge_processors:
        std::collections::HashMap<String, crate::oarocr::processors::EdgeProcessorConfig>,

    /// Whether this task is optional (can be skipped if model not available)
    #[serde(default)]
    pub optional: bool,

    /// Whether this task is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

impl TaskNode {
    /// Creates a new task node.
    pub fn new(
        id: impl Into<String>,
        task_type: TaskType,
        model_binding: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            task_type,
            model_binding: model_binding.into(),
            dependencies: Vec::new(),
            edge_processors: std::collections::HashMap::new(),
            optional: false,
            enabled: true,
        }
    }

    /// Adds a dependency to this task node.
    pub fn with_dependency(mut self, dependency: impl Into<String>) -> Self {
        self.dependencies.push(dependency.into());
        self
    }

    /// Sets whether this task is optional.
    pub fn with_optional(mut self, optional: bool) -> Self {
        self.optional = optional;
        self
    }

    /// Sets whether this task is enabled.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Adds an edge processor for a specific dependency.
    pub fn with_edge_processor(
        mut self,
        dependency_id: impl Into<String>,
        processor: crate::oarocr::processors::EdgeProcessorConfig,
    ) -> Self {
        self.edge_processors.insert(dependency_id.into(), processor);
        self
    }
}

/// Task graph configuration defining the pipeline structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraphConfig {
    /// Model bindings - maps model names to their configurations
    pub model_bindings: HashMap<String, ModelBinding>,

    /// Task nodes defining the pipeline structure
    pub task_nodes: Vec<TaskNode>,

    /// Path to character dictionary for text recognition
    #[serde(default)]
    pub character_dict_path: Option<PathBuf>,
}

impl TaskGraphConfig {
    /// Creates a new empty task graph configuration.
    pub fn new() -> Self {
        Self {
            model_bindings: HashMap::new(),
            task_nodes: Vec::new(),
            character_dict_path: None,
        }
    }

    /// Adds a model binding to the configuration.
    pub fn add_model_binding(mut self, name: impl Into<String>, binding: ModelBinding) -> Self {
        self.model_bindings.insert(name.into(), binding);
        self
    }

    /// Adds a task node to the configuration.
    pub fn add_task_node(mut self, node: TaskNode) -> Self {
        self.task_nodes.push(node);
        self
    }

    /// Sets the character dictionary path.
    pub fn with_character_dict(mut self, path: impl Into<PathBuf>) -> Self {
        self.character_dict_path = Some(path.into());
        self
    }

    /// Validates the task graph configuration.
    pub fn validate(&self) -> Result<(), String> {
        // Check that all task nodes reference valid model bindings
        for node in &self.task_nodes {
            if !self.model_bindings.contains_key(&node.model_binding) {
                return Err(format!(
                    "Task node '{}' references unknown model binding '{}'",
                    node.id, node.model_binding
                ));
            }

            // Check that model binding task type matches node task type
            let binding = &self.model_bindings[&node.model_binding];
            if binding.task_type != node.task_type {
                return Err(format!(
                    "Task node '{}' has task type {:?} but model binding '{}' has task type {:?}",
                    node.id, node.task_type, node.model_binding, binding.task_type
                ));
            }
        }

        // Check that all dependencies reference valid task nodes
        let node_ids: std::collections::HashSet<_> =
            self.task_nodes.iter().map(|n| n.id.as_str()).collect();

        for node in &self.task_nodes {
            for dep in &node.dependencies {
                if !node_ids.contains(dep.as_str()) {
                    return Err(format!(
                        "Task node '{}' has dependency '{}' which doesn't exist",
                        node.id, dep
                    ));
                }
            }
        }

        // Check for circular dependencies
        if self.has_circular_dependencies() {
            return Err("Task graph contains circular dependencies".to_string());
        }

        Ok(())
    }

    /// Checks if the task graph has circular dependencies.
    fn has_circular_dependencies(&self) -> bool {
        use std::collections::{HashMap, HashSet};

        // Build adjacency list
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.task_nodes {
            graph.insert(
                &node.id,
                node.dependencies.iter().map(|s| s.as_str()).collect(),
            );
        }

        // DFS to detect cycles
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        fn has_cycle<'a>(
            node: &'a str,
            graph: &HashMap<&'a str, Vec<&'a str>>,
            visited: &mut HashSet<&'a str>,
            rec_stack: &mut HashSet<&'a str>,
        ) -> bool {
            visited.insert(node);
            rec_stack.insert(node);

            if let Some(neighbors) = graph.get(node) {
                for &neighbor in neighbors {
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

        for node in &self.task_nodes {
            if !visited.contains(node.id.as_str())
                && has_cycle(&node.id, &graph, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    /// Returns the task nodes in topological order (dependencies first).
    pub fn topological_order(&self) -> Result<Vec<&TaskNode>, String> {
        use std::collections::HashMap;

        // Build adjacency list and in-degree map
        let mut graph: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        for node in &self.task_nodes {
            graph.insert(&node.id, vec![]);
            in_degree.insert(&node.id, 0);
        }

        for node in &self.task_nodes {
            for dep in &node.dependencies {
                graph.get_mut(dep.as_str()).unwrap().push(&node.id);
                *in_degree.get_mut(node.id.as_str()).unwrap() += 1;
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();

        while let Some(node_id) = queue.pop() {
            let node = self.task_nodes.iter().find(|n| n.id == node_id).unwrap();
            result.push(node);

            if let Some(neighbors) = graph.get(node_id) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }

        if result.len() != self.task_nodes.len() {
            return Err("Task graph contains circular dependencies".to_string());
        }

        Ok(result)
    }
}

impl Default for TaskGraphConfig {
    fn default() -> Self {
        Self::new()
    }
}
