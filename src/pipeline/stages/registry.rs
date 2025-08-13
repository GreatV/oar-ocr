//! Stage registry and extensible pipeline executor.
//!
//! This module provides the registry system for managing pipeline stages
//! and the executor for running extensible pipelines.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{debug, info};

use super::extensible::{PipelineStage, StageContext, StageData, StageDependency, StageId};
use super::types::StageResult;
use crate::core::OCRError;
use crate::core::config::{ConfigError, ConfigValidator};

/// Trait for type-erased configuration that can be validated.
pub trait ErasedConfig: Any + Send + Sync + Debug {
    /// Validate the configuration
    fn validate_erased(&self) -> Result<(), ConfigError>;
    /// Get the default configuration as a boxed Any
    fn default_erased() -> Box<dyn ErasedConfig>
    where
        Self: Sized;
    /// Clone the configuration
    fn clone_erased(&self) -> Box<dyn ErasedConfig>;
    /// Downcast to concrete type
    fn as_any(&self) -> &dyn Any;
}

impl<T> ErasedConfig for T
where
    T: ConfigValidator + Default + Clone + Any + Send + Sync + Debug,
{
    fn validate_erased(&self) -> Result<(), ConfigError> {
        self.validate()
    }

    fn default_erased() -> Box<dyn ErasedConfig> {
        Box::new(T::get_defaults())
    }

    fn clone_erased(&self) -> Box<dyn ErasedConfig> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A dummy configuration type that represents an invalid BoxedConfig.
/// This is used when BoxedConfig::default() or BoxedConfig::get_defaults()
/// are called directly, which should never happen in normal usage.
#[derive(Debug, Clone)]
#[allow(dead_code)] // This is intentionally only used in error cases
struct InvalidBoxedConfig;

impl ErasedConfig for InvalidBoxedConfig {
    fn validate_erased(&self) -> Result<(), ConfigError> {
        Err(ConfigError::InvalidConfig {
            message: "Invalid BoxedConfig: This configuration was created through BoxedConfig::default() or BoxedConfig::get_defaults(), which should never be called directly. Use ErasedConfig::default_erased() or box a concrete config type instead.".to_string(),
        })
    }

    fn default_erased() -> Box<dyn ErasedConfig> {
        Box::new(InvalidBoxedConfig)
    }

    fn clone_erased(&self) -> Box<dyn ErasedConfig> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Type alias for a boxed stage configuration.
type BoxedConfig = Box<dyn ErasedConfig>;

impl ConfigValidator for BoxedConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        self.validate_erased()
    }

    fn get_defaults() -> Self {
        // This method should never be called directly on BoxedConfig.
        // BoxedConfig instances should be created through ErasedConfig::default_erased()
        // or by boxing concrete config types.
        // Return an invalid config that will fail validation with a clear error message.
        //
        // DESIGN NOTE: This intentionally returns an InvalidBoxedConfig that will always
        // fail validation. The reason for this design choice is that BoxedConfig is a
        // type-erased trait object, and it doesn't make sense to create a "default"
        // BoxedConfig without knowing the concrete type underneath. Instead, users should:
        // 1. Create a default of a concrete config type and then box it
        // 2. Use ErasedConfig::default_erased() on a concrete type
        // This prevents runtime errors from using an incorrectly typed configuration.
        Box::new(InvalidBoxedConfig)
    }
}

impl Default for BoxedConfig {
    fn default() -> Self {
        // This method should never be called directly on BoxedConfig.
        // BoxedConfig instances should be created through ErasedConfig::default_erased()
        // or by boxing concrete config types.
        // Return an invalid config that will fail validation with a clear error message.
        //
        // DESIGN NOTE: This intentionally returns an InvalidBoxedConfig that will always
        // fail validation. The reason for this design choice is that BoxedConfig is a
        // type-erased trait object, and it doesn't make sense to create a "default"
        // BoxedConfig without knowing the concrete type underneath. Instead, users should:
        // 1. Create a default of a concrete config type and then box it
        // 2. Use ErasedConfig::default_erased() on a concrete type
        // This prevents runtime errors from using an incorrectly typed configuration.
        Box::new(InvalidBoxedConfig)
    }
}

// Extension trait for downcasting BoxedConfig
pub trait BoxedConfigExt {
    /// Downcast to a concrete configuration type
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
}

impl BoxedConfigExt for BoxedConfig {
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

/// Type alias for a boxed stage result.
type BoxedResult = Box<dyn Any + Send + Sync>;

/// Type alias for a registered pipeline stage.
type RegisteredStage = Arc<dyn PipelineStage<Config = BoxedConfig, Result = BoxedResult>>;

/// Registry for managing pipeline stages.
///
/// The registry allows dynamic registration of stages and provides
/// dependency resolution and execution ordering.
#[derive(Debug)]
pub struct StageRegistry {
    /// Registered stages
    stages: HashMap<StageId, RegisteredStage>,
    /// Stage configurations
    configs: HashMap<StageId, BoxedConfig>,
    /// Execution order cache
    execution_order: Option<Vec<StageId>>,
}

impl StageRegistry {
    /// Create a new empty stage registry.
    pub fn new() -> Self {
        Self {
            stages: HashMap::new(),
            configs: HashMap::new(),
            execution_order: None,
        }
    }

    /// Register a stage with the registry.
    ///
    /// # Arguments
    ///
    /// * `stage` - The stage to register
    /// * `config` - Optional configuration for the stage
    pub fn register_stage<S, C>(&mut self, stage: S, config: Option<C>) -> Result<(), OCRError>
    where
        S: PipelineStage<Config = C> + 'static,
        C: Send + Sync + Debug + Clone + ConfigValidator + Default + 'static,
    {
        let stage_id = stage.stage_id();

        // Validate configuration if provided
        if let Some(ref cfg) = config {
            stage.validate_config(cfg)?;
        }

        // Create type-erased wrappers
        let erased_stage = Arc::new(TypeErasedStage::new(stage));

        // Store the stage
        self.stages.insert(stage_id.clone(), erased_stage);

        // Store configuration if provided
        if let Some(cfg) = config {
            self.configs.insert(stage_id, Box::new(cfg));
        }

        // Invalidate execution order cache
        self.execution_order = None;

        Ok(())
    }

    /// Get a registered stage by ID.
    pub fn get_stage(&self, stage_id: &StageId) -> Option<&RegisteredStage> {
        self.stages.get(stage_id)
    }

    /// Get configuration for a stage.
    pub fn get_config(&self, stage_id: &StageId) -> Option<&BoxedConfig> {
        self.configs.get(stage_id)
    }

    /// Get all registered stage IDs.
    #[allow(dead_code)]
    pub fn stage_ids(&self) -> Vec<StageId> {
        self.stages.keys().cloned().collect()
    }

    /// Resolve execution order based on dependencies.
    pub fn resolve_execution_order(&mut self) -> Result<Vec<StageId>, OCRError> {
        if let Some(ref order) = self.execution_order {
            return Ok(order.clone());
        }

        let order = self.topological_sort()?;
        self.execution_order = Some(order.clone());
        Ok(order)
    }

    /// Perform topological sort to determine execution order.
    fn topological_sort(&self) -> Result<Vec<StageId>, OCRError> {
        let mut in_degree: HashMap<StageId, usize> = HashMap::new();
        let mut graph: HashMap<StageId, Vec<StageId>> = HashMap::new();

        // Initialize in-degree and graph
        for stage_id in self.stages.keys() {
            in_degree.insert(stage_id.clone(), 0);
            graph.insert(stage_id.clone(), Vec::new());
        }

        // Build dependency graph
        for (stage_id, stage) in &self.stages {
            for dependency in stage.dependencies() {
                match dependency {
                    StageDependency::After(dep_id) | StageDependency::Requires(dep_id) => {
                        if self.stages.contains_key(&dep_id) {
                            graph.get_mut(&dep_id).unwrap().push(stage_id.clone());
                            *in_degree.get_mut(stage_id).unwrap() += 1;
                        }
                    }
                    StageDependency::Before(dep_id) | StageDependency::Provides(dep_id) => {
                        if self.stages.contains_key(&dep_id) {
                            graph.get_mut(stage_id).unwrap().push(dep_id.clone());
                            *in_degree.get_mut(&dep_id).unwrap() += 1;
                        }
                    }
                }
            }
        }

        // Kahn's algorithm for topological sorting
        let mut queue: VecDeque<StageId> = VecDeque::new();
        let mut result: Vec<StageId> = Vec::new();

        // Find all nodes with no incoming edges
        for (stage_id, &degree) in &in_degree {
            if degree == 0 {
                queue.push_back(stage_id.clone());
            }
        }

        while let Some(stage_id) = queue.pop_front() {
            result.push(stage_id.clone());

            // For each neighbor of the current stage
            if let Some(neighbors) = graph.get(&stage_id) {
                for neighbor in neighbors {
                    let degree = in_degree.get_mut(neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        // Check for cycles
        if result.len() != self.stages.len() {
            return Err(OCRError::ConfigError {
                message: "Circular dependency detected in pipeline stages".to_string(),
            });
        }

        Ok(result)
    }
}

impl Default for StageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Type-erased wrapper for pipeline stages.
#[derive(Debug)]
struct TypeErasedStage<S> {
    inner: S,
}

impl<S> TypeErasedStage<S>
where
    S: PipelineStage,
{
    fn new(stage: S) -> Self {
        Self { inner: stage }
    }
}

impl<S> PipelineStage for TypeErasedStage<S>
where
    S: PipelineStage + 'static,
    S::Config: Clone + 'static,
    S::Result: 'static,
{
    type Config = BoxedConfig;
    type Result = Box<dyn Any + Send + Sync>;

    fn stage_id(&self) -> StageId {
        self.inner.stage_id()
    }

    fn stage_name(&self) -> &str {
        self.inner.stage_name()
    }

    fn dependencies(&self) -> Vec<StageDependency> {
        self.inner.dependencies()
    }

    fn is_enabled(&self, context: &StageContext, config: Option<&Self::Config>) -> bool {
        let typed_config = config.and_then(|c| BoxedConfigExt::downcast_ref::<S::Config>(c));
        self.inner.is_enabled(context, typed_config)
    }

    fn process(
        &self,
        context: &mut StageContext,
        data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError> {
        let typed_config = config.and_then(|c| BoxedConfigExt::downcast_ref::<S::Config>(c));

        let result = self.inner.process(context, data, typed_config)?;

        Ok(StageResult::new(
            Box::new(result.data) as Box<dyn Any + Send + Sync>,
            result.metrics,
        ))
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        if let Some(typed_config) = BoxedConfigExt::downcast_ref::<S::Config>(config) {
            self.inner.validate_config(typed_config)
        } else {
            Err(OCRError::pipeline_stage_error(
                self.stage_name(),
                &self.stage_id().to_string(),
                1, // config validation is for single config
                "validate_config",
                crate::core::errors::SimpleError::new("Invalid configuration type for stage"),
            ))
        }
    }

    fn default_config(&self) -> Self::Config {
        Box::new(self.inner.default_config())
    }
}

/// Extensible pipeline that uses the stage registry.
pub struct ExtensiblePipeline {
    registry: StageRegistry,
}

impl ExtensiblePipeline {
    /// Create a new extensible pipeline.
    pub fn new() -> Self {
        Self {
            registry: StageRegistry::new(),
        }
    }

    /// Register a stage with the pipeline.
    pub fn register_stage<S, C>(&mut self, stage: S, config: Option<C>) -> Result<(), OCRError>
    where
        S: PipelineStage<Config = C> + 'static,
        C: Send + Sync + Debug + Clone + ConfigValidator + Default + 'static,
    {
        self.registry.register_stage(stage, config)
    }

    /// Execute the pipeline with the given context.
    #[allow(dead_code)]
    pub fn execute(&mut self, context: &mut StageContext) -> Result<(), OCRError> {
        let execution_order = self.registry.resolve_execution_order()?;

        for stage_id in execution_order {
            let stage =
                self.registry
                    .get_stage(&stage_id)
                    .ok_or_else(|| OCRError::ConfigError {
                        message: format!("Stage not found: {}", stage_id.as_str()),
                    })?;

            let config = self.registry.get_config(&stage_id);
            let stage_data = StageData::new((*context.original_image).clone());
            let result = stage.process(context, stage_data, config)?;
            context.set_stage_result(stage_id, result);
        }

        Ok(())
    }

    /// Get the stage registry.
    #[allow(dead_code)]
    pub fn registry(&self) -> &StageRegistry {
        &self.registry
    }

    /// Get the stage registry mutably.
    #[allow(dead_code)]
    pub fn registry_mut(&mut self) -> &mut StageRegistry {
        &mut self.registry
    }
}

impl Default for ExtensiblePipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline executor for running extensible pipelines.
pub struct PipelineExecutor;

impl PipelineExecutor {
    /// Execute a pipeline with the given context and data.
    pub fn execute(
        pipeline: &mut ExtensiblePipeline,
        mut context: StageContext,
        initial_data: StageData,
    ) -> Result<StageData, OCRError> {
        let execution_order = pipeline.registry.resolve_execution_order()?;
        let mut current_data = initial_data;

        info!("Executing pipeline with {} stages", execution_order.len());

        for stage_id in execution_order {
            let stage =
                pipeline
                    .registry
                    .get_stage(&stage_id)
                    .ok_or_else(|| OCRError::ConfigError {
                        message: format!("Stage not found: {}", stage_id.as_str()),
                    })?;

            let config = pipeline.registry.get_config(&stage_id);

            // Check if stage is enabled
            if !stage.is_enabled(&context, config) {
                debug!("Skipping disabled stage: {}", stage.stage_name());
                continue;
            }

            debug!("Executing stage: {}", stage.stage_name());

            // Execute the stage
            let stage_result = stage.process(&mut context, current_data, config)?;

            // Store the result in context for other stages
            context.set_stage_result(stage_id.clone(), stage_result.data);

            // Update current data - stages that modify the image should update the context
            // For now, we'll keep the current image from the context
            current_data = StageData::new(context.current_image.as_ref().clone());

            debug!(
                "Stage {} completed in {:?}",
                stage.stage_name(),
                stage_result.metrics.processing_time
            );
        }

        Ok(current_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxed_config_default_returns_invalid_config() {
        let config = BoxedConfig::default();
        let result = config.validate();
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("Invalid BoxedConfig"));
        assert!(error_message.contains("should never be called directly"));
    }

    #[test]
    fn test_boxed_config_get_defaults_returns_invalid_config() {
        let config = BoxedConfig::get_defaults();
        let result = config.validate();
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("Invalid BoxedConfig"));
        assert!(error_message.contains("should never be called directly"));
    }
}
