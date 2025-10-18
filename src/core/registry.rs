//! Model Registry for dynamic adapter lookup and management.
//!
//! This module provides a registry system for managing model adapters,
//! allowing dynamic lookup and configuration of models for different tasks.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::TaskType,
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, RwLock};

/// A type-erased model adapter that can be stored in the registry.
///
/// This trait extends ModelAdapter to support dynamic dispatch and cloning.
pub trait DynModelAdapter: Send + Sync + Debug {
    /// Returns information about this adapter.
    fn info(&self) -> AdapterInfo;

    /// Returns the task type this adapter handles.
    fn task_type(&self) -> TaskType;

    /// Returns whether this adapter supports batching.
    fn supports_batching(&self) -> bool;

    /// Returns the recommended batch size.
    fn recommended_batch_size(&self) -> usize;
}

/// Wrapper to make ModelAdapter trait object-safe.
#[derive(Debug)]
struct AdapterWrapper<A: ModelAdapter> {
    adapter: A,
}

impl<A: ModelAdapter> AdapterWrapper<A> {
    fn new(adapter: A) -> Self {
        Self { adapter }
    }
}

impl<A: ModelAdapter + 'static> DynModelAdapter for AdapterWrapper<A> {
    fn info(&self) -> AdapterInfo {
        self.adapter.info()
    }

    fn task_type(&self) -> TaskType {
        self.adapter.info().task_type
    }

    fn supports_batching(&self) -> bool {
        self.adapter.supports_batching()
    }

    fn recommended_batch_size(&self) -> usize {
        self.adapter.recommended_batch_size()
    }
}

/// Type alias for the adapter storage map
type AdapterMap = Arc<RwLock<HashMap<(TaskType, String), Arc<dyn DynModelAdapter>>>>;

/// Registry for managing model adapters.
///
/// The registry allows registering, looking up, and managing model adapters
/// for different tasks. It supports multiple adapters per task type.
pub struct ModelRegistry {
    /// Map from (task_type, model_name) to adapter
    adapters: AdapterMap,
}

impl ModelRegistry {
    /// Creates a new empty model registry.
    pub fn new() -> Self {
        Self {
            adapters: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Registers a model adapter in the registry.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The adapter to register
    ///
    /// # Returns
    ///
    /// Result indicating success or error if registration fails
    pub fn register<A: ModelAdapter + 'static>(&self, adapter: A) -> Result<(), OCRError> {
        let info = adapter.info();
        let key = (info.task_type, info.model_name.clone());

        let mut adapters = self.adapters.write().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire write lock on registry: {}", e),
        })?;

        if adapters.contains_key(&key) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Adapter for task {:?} with model '{}' is already registered",
                    key.0, key.1
                ),
            });
        }

        adapters.insert(key, Arc::new(AdapterWrapper::new(adapter)));
        Ok(())
    }

    /// Looks up an adapter by task type and model name.
    ///
    /// # Arguments
    ///
    /// * `task_type` - The type of task
    /// * `model_name` - The name of the model
    ///
    /// # Returns
    ///
    /// Option containing the adapter if found
    pub fn lookup(
        &self,
        task_type: TaskType,
        model_name: &str,
    ) -> Result<Option<Arc<dyn DynModelAdapter>>, OCRError> {
        let adapters = self.adapters.read().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire read lock on registry: {}", e),
        })?;

        Ok(adapters.get(&(task_type, model_name.to_string())).cloned())
    }

    /// Lists all registered adapters for a given task type.
    ///
    /// # Arguments
    ///
    /// * `task_type` - The type of task to filter by
    ///
    /// # Returns
    ///
    /// Vector of adapter info for the specified task type
    pub fn list_by_task(&self, task_type: TaskType) -> Result<Vec<AdapterInfo>, OCRError> {
        let adapters = self.adapters.read().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire read lock on registry: {}", e),
        })?;

        Ok(adapters
            .iter()
            .filter(|((t, _), _)| *t == task_type)
            .map(|(_, adapter)| adapter.info())
            .collect())
    }

    /// Lists all registered adapters.
    ///
    /// # Returns
    ///
    /// Vector of all adapter info in the registry
    pub fn list_all(&self) -> Result<Vec<AdapterInfo>, OCRError> {
        let adapters = self.adapters.read().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire read lock on registry: {}", e),
        })?;

        Ok(adapters.values().map(|adapter| adapter.info()).collect())
    }

    /// Removes an adapter from the registry.
    ///
    /// # Arguments
    ///
    /// * `task_type` - The type of task
    /// * `model_name` - The name of the model
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    pub fn unregister(&self, task_type: TaskType, model_name: &str) -> Result<(), OCRError> {
        let mut adapters = self.adapters.write().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire write lock on registry: {}", e),
        })?;

        let key = (task_type, model_name.to_string());
        if adapters.remove(&key).is_none() {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Adapter for task {:?} with model '{}' not found",
                    task_type, model_name
                ),
            });
        }

        Ok(())
    }

    /// Clears all adapters from the registry.
    pub fn clear(&self) -> Result<(), OCRError> {
        let mut adapters = self.adapters.write().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire write lock on registry: {}", e),
        })?;

        adapters.clear();
        Ok(())
    }

    /// Returns the number of registered adapters.
    pub fn len(&self) -> Result<usize, OCRError> {
        let adapters = self.adapters.read().map_err(|e| OCRError::ConfigError {
            message: format!("Failed to acquire read lock on registry: {}", e),
        })?;

        Ok(adapters.len())
    }

    /// Returns whether the registry is empty.
    pub fn is_empty(&self) -> Result<bool, OCRError> {
        Ok(self.len()? == 0)
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for ModelRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let adapters = self.adapters.read().unwrap();
        f.debug_struct("ModelRegistry")
            .field("adapter_count", &adapters.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert!(registry.is_empty().unwrap());
        assert_eq!(registry.len().unwrap(), 0);
    }

    #[test]
    fn test_registry_operations() {
        let registry = ModelRegistry::new();

        // Test listing empty registry
        let all = registry.list_all().unwrap();
        assert_eq!(all.len(), 0);

        // Test listing by task
        let detection_adapters = registry.list_by_task(TaskType::TextDetection).unwrap();
        assert_eq!(detection_adapters.len(), 0);
    }
}
