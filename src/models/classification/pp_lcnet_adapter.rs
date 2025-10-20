//! Generic PP-LCNet Classifier Adapter
//!
//! This module provides a generic adapter for classification tasks like document orientation
//! and text line orientation. It uses a trait-based approach to handle task-specific
//! configurations while sharing the core preprocessing, inference, and postprocessing logic.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{ImageTaskInput, Task},
};
use crate::core::{OCRError, Tensor2D, Tensor4D};
use crate::domain::tasks::{
    DocumentOrientationConfig, DocumentOrientationTask, TextLineOrientationConfig,
    TextLineOrientationTask,
};
use crate::processors::{ChannelOrder, NormalizeImage};
use crate::utils::topk::Topk;
use image::{DynamicImage, RgbImage, imageops::FilterType};
use std::marker::PhantomData;
use std::path::Path;

/// A trait to encapsulate the specific details of a classification task.
pub trait ClassificationTask: Task<Input = ImageTaskInput> {
    /// Provides the default input shape (height, width) for the model.
    fn default_input_shape() -> (u32, u32);
    /// Specifies the resizing filter to be used during preprocessing.
    fn resize_filter() -> FilterType;
    /// Returns the list of class labels for the task.
    fn labels() -> Vec<String>;
    /// Provides a recommended batch size for the task.
    fn recommended_batch_size() -> usize;
    /// Constructs the adapter's information block.
    fn adapter_info() -> AdapterInfo;
    /// Extracts the top-k value from the task-specific configuration.
    fn get_topk(config: &Self::Config) -> usize;
    /// Generates a default label name from a class ID if not provided by Topk.
    fn default_label_from_id(id: usize) -> String;
    /// Creates a new instance of the task's output type.
    fn new_output(
        class_ids: Vec<Vec<usize>>,
        scores: Vec<Vec<f32>>,
        label_names: Vec<Vec<String>>,
    ) -> Self::Output;
}

/// Generic adapter for PP-LCNet-based classification models.
#[derive(Debug)]
pub struct PPLCNetAdapter<T: ClassificationTask> {
    inference: OrtInfer,
    normalizer: NormalizeImage,
    topk_processor: Topk,
    input_shape: (u32, u32),
    info: AdapterInfo,
    config: T::Config,
    resize_filter: FilterType,
    _task: PhantomData<T>,
}

impl<T: ClassificationTask> PPLCNetAdapter<T> {
    /// Creates a new generic classification adapter.
    pub fn new(
        inference: OrtInfer,
        normalizer: NormalizeImage,
        topk_processor: Topk,
        input_shape: (u32, u32),
        info: AdapterInfo,
        config: T::Config,
        resize_filter: FilterType,
    ) -> Self {
        Self {
            inference,
            normalizer,
            topk_processor,
            input_shape,
            info,
            config,
            resize_filter,
            _task: PhantomData,
        }
    }

    /// Preprocesses images for classification.
    fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        let resized_images: Vec<DynamicImage> = images
            .into_iter()
            .map(|img| {
                DynamicImage::ImageRgb8(image::imageops::resize(
                    &img,
                    self.input_shape.1,
                    self.input_shape.0,
                    self.resize_filter,
                ))
            })
            .collect();

        let batch_tensor = self.normalizer.normalize_batch_to(resized_images)?;
        Ok(batch_tensor)
    }

    /// Postprocesses model predictions to labels.
    fn postprocess(&self, predictions: &Tensor2D, config: &T::Config) -> T::Output {
        let predictions_vec: Vec<Vec<f32>> =
            predictions.outer_iter().map(|row| row.to_vec()).collect();

        let topk_result = self
            .topk_processor
            .process(&predictions_vec, T::get_topk(config))
            .unwrap_or_else(|_| crate::utils::topk::TopkResult {
                indexes: vec![],
                scores: vec![],
                class_names: None,
            });

        let class_ids = topk_result.indexes;
        let scores = topk_result.scores;
        let label_names = topk_result.class_names.unwrap_or_else(|| {
            class_ids
                .iter()
                .map(|ids| ids.iter().map(|&id| T::default_label_from_id(id)).collect())
                .collect()
        });

        T::new_output(class_ids, scores, label_names)
    }
}

impl<T: ClassificationTask> ModelAdapter for PPLCNetAdapter<T> {
    type Task = T;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let images = input.images.clone();

        let effective_config = config.unwrap_or(&self.config);
        let batch_tensor = self.preprocess(images)?;
        let predictions = self.inference.infer_2d(&batch_tensor)?;
        let output = self.postprocess(&predictions, effective_config);

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        T::recommended_batch_size()
    }
}

/// Builder for the generic classification adapter.
pub struct PPLCNetAdapterBuilder<T>
where
    T: ClassificationTask,
    T::Config: Default,
{
    task_config: T::Config,
    input_shape: (u32, u32),
    session_pool_size: usize,
    model_name_override: Option<String>,
    _task: PhantomData<T>,
}

impl<T> PPLCNetAdapterBuilder<T>
where
    T: ClassificationTask,
    T::Config: Default,
{
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            task_config: T::Config::default(),
            input_shape: T::default_input_shape(),
            session_pool_size: 1,
            model_name_override: None,
            _task: PhantomData,
        }
    }

    /// Sets the input shape.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = input_shape;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets a custom model name for registry identification.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name_override = Some(model_name.into());
        self
    }
}

impl<T> Default for PPLCNetAdapterBuilder<T>
where
    T: ClassificationTask,
    T::Config: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AdapterBuilder for PPLCNetAdapterBuilder<T>
where
    T: ClassificationTask,
    T::Config: Default,
{
    type Config = T::Config;
    type Adapter = PPLCNetAdapter<T>;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let inference = if self.session_pool_size > 1 {
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ..Default::default()
            };
            OrtInfer::from_common(&common_config, model_path, Some("x"))?
        } else {
            OrtInfer::new(model_path, Some("x"))?
        };

        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.485, 0.456, 0.406]),
            Some(vec![0.229, 0.224, 0.225]),
            Some(ChannelOrder::CHW),
        )?;

        let topk_processor = Topk::from_class_names(T::labels());

        let mut info = T::adapter_info();
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(PPLCNetAdapter::new(
            inference,
            normalizer,
            topk_processor,
            self.input_shape,
            info,
            self.task_config,
            T::resize_filter(),
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "Generic-PPLCNet-Classification"
    }
}

/// Document orientation classifier adapter alias.
pub type DocOrientationAdapter = PPLCNetAdapter<DocumentOrientationTask>;

/// Builder wrapper for `DocOrientationAdapter`.
pub struct DocOrientationAdapterBuilder {
    inner: PPLCNetAdapterBuilder<DocumentOrientationTask>,
}

impl DocOrientationAdapterBuilder {
    /// Creates a new document orientation adapter builder.
    pub fn new() -> Self {
        Self {
            inner: PPLCNetAdapterBuilder::new(),
        }
    }

    /// Applies a task-specific configuration.
    pub fn with_config(mut self, config: DocumentOrientationConfig) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    /// Overrides the input shape (height, width).
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.inner = self.inner.input_shape(input_shape);
        self
    }

    /// Sets the session pool size for ONNX Runtime.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.inner = self.inner.session_pool_size(size);
        self
    }

    /// Overrides the model name used for registry identification.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.inner = self.inner.model_name(model_name);
        self
    }
}

impl Default for DocOrientationAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for DocOrientationAdapterBuilder {
    type Config = DocumentOrientationConfig;
    type Adapter = DocOrientationAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "DocumentOrientation-PPLCNet"
    }
}

/// Text line orientation classifier adapter alias.
pub type TextLineOrientationAdapter = PPLCNetAdapter<TextLineOrientationTask>;

/// Builder wrapper for `TextLineOrientationAdapter`.
pub struct TextLineOrientationAdapterBuilder {
    inner: PPLCNetAdapterBuilder<TextLineOrientationTask>,
}

impl TextLineOrientationAdapterBuilder {
    /// Creates a new text line orientation adapter builder.
    pub fn new() -> Self {
        Self {
            inner: PPLCNetAdapterBuilder::new(),
        }
    }

    /// Applies a task-specific configuration.
    pub fn with_config(mut self, config: TextLineOrientationConfig) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    /// Overrides the input shape (height, width).
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.inner = self.inner.input_shape(input_shape);
        self
    }

    /// Sets the session pool size for ONNX Runtime.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.inner = self.inner.session_pool_size(size);
        self
    }

    /// Overrides the model name used for registry identification.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.inner = self.inner.model_name(model_name);
        self
    }
}

impl Default for TextLineOrientationAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for TextLineOrientationAdapterBuilder {
    type Config = TextLineOrientationConfig;
    type Adapter = TextLineOrientationAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "TextLineOrientation-PPLCNet"
    }
}
