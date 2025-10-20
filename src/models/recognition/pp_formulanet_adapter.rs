//! PP-FormulaNet formula recognition adapter.

use super::paddle_formula_base::{
    PaddleFormulaAdapter, PaddleFormulaAdapterBase, PaddleFormulaAdapterBuilderBase,
    PaddleFormulaSpec,
};
use crate::core::OCRError;
use crate::core::traits::adapter::{AdapterBuilder, AdapterInfo, ModelAdapter};
use crate::core::traits::task::Task;
use crate::domain::tasks::{FormulaRecognitionConfig, FormulaRecognitionTask};
use std::path::Path;

#[derive(Debug)]
pub(crate) struct PPFormulaNetSpec;

impl PaddleFormulaSpec for PPFormulaNetSpec {
    fn model_name() -> &'static str {
        "PP-FormulaNet"
    }

    fn description() -> &'static str {
        "PaddleOCR PP-FormulaNet formula recognition model"
    }
}

/// Adapter for the PP-FormulaNet formula recognition model.
#[derive(Debug)]
pub struct PPFormulaNetAdapter {
    inner: PaddleFormulaAdapter<PPFormulaNetSpec>,
}

impl PPFormulaNetAdapter {
    pub(crate) fn new(base: PaddleFormulaAdapterBase<PPFormulaNetSpec>) -> Self {
        Self {
            inner: PaddleFormulaAdapter::new(base),
        }
    }
}

impl ModelAdapter for PPFormulaNetAdapter {
    type Task = FormulaRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.inner.info()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        self.inner.execute(input, config)
    }

    fn supports_batching(&self) -> bool {
        self.inner.supports_batching()
    }

    fn recommended_batch_size(&self) -> usize {
        self.inner.recommended_batch_size()
    }
}

/// Builder for [`PPFormulaNetAdapter`].
#[derive(Debug)]
pub struct PPFormulaNetAdapterBuilder {
    base: PaddleFormulaAdapterBuilderBase<PPFormulaNetSpec>,
}

impl PPFormulaNetAdapterBuilder {
    pub fn new() -> Self {
        Self {
            base: PaddleFormulaAdapterBuilderBase::new(),
        }
    }

    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.base = self.base.target_size(width, height);
        self
    }

    pub fn tokenizer_path<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.base = self.base.tokenizer_path(path);
        self
    }

    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.base = self.base.session_pool_size(size);
        self
    }

    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.base = self.base.model_name(name);
        self
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.base = self.base.score_threshold(threshold);
        self
    }

    pub fn max_length(mut self, length: usize) -> Self {
        self.base = self.base.max_length(length);
        self
    }
}

impl Default for PPFormulaNetAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for PPFormulaNetAdapterBuilder {
    type Config = FormulaRecognitionConfig;
    type Adapter = PPFormulaNetAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let base = self.base.build(model_path)?;
        Ok(PPFormulaNetAdapter::new(base))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.base = self.base.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "FormulaRecognition-PP-FormulaNet"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::adapter::AdapterBuilder;

    #[test]
    fn builder_returns_correct_type() {
        let builder = PPFormulaNetAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "FormulaRecognition-PP-FormulaNet");
    }
}
