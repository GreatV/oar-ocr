//! Text Recognition Task Adapter
//!
//! This adapter uses the CRNN model and adapts its output to the TextRecognition task format.

use crate::apply_ort_config;
use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::domain::tasks::{TextRecognitionConfig, TextRecognitionOutput, TextRecognitionTask};
use crate::impl_adapter_builder;
use crate::models::recognition::crnn::{CRNNModel, CRNNModelBuilder, CRNNPreprocessConfig};

/// Text recognition adapter that uses the CRNN model.
#[derive(Debug)]
pub struct TextRecognitionAdapter {
    /// The underlying CRNN model
    model: CRNNModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TextRecognitionConfig,
    /// Whether to return character positions for word box generation
    return_word_box: bool,
}

impl ModelAdapter for TextRecognitionAdapter {
    type Task = TextRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);
        let batch_len = input.images.len();

        // Use the CRNN model to recognize text
        let model_output = self
            .model
            .forward(input.images, self.return_word_box)
            .map_err(|e| {
                OCRError::adapter_execution_error(
                    "TextRecognitionAdapter",
                    format!(
                        "forward (batch_size={}, return_word_box={})",
                        batch_len, self.return_word_box
                    ),
                    e,
                )
            })?;

        // Apply score threshold filtering while preserving index correspondence.
        // Items below threshold are kept but with empty text to maintain 1:1 mapping
        // between inputs and outputs, which is critical for batched processing.
        let mut result_texts = Vec::with_capacity(model_output.texts.len());
        let mut result_scores = Vec::with_capacity(model_output.scores.len());
        let mut result_positions = Vec::with_capacity(model_output.texts.len());
        let mut result_col_indices = Vec::with_capacity(model_output.texts.len());
        let mut result_seq_lengths = Vec::with_capacity(model_output.texts.len());

        for (((text, score), positions), (col_indices, seq_len)) in model_output
            .texts
            .into_iter()
            .zip(model_output.scores)
            .zip(
                model_output
                    .char_positions
                    .into_iter()
                    .chain(std::iter::repeat(Vec::new())),
            )
            .zip(
                model_output
                    .char_col_indices
                    .into_iter()
                    .zip(model_output.sequence_lengths.into_iter())
                    .chain(std::iter::repeat((Vec::new(), 0))),
            )
        {
            if score >= effective_config.score_threshold {
                result_texts.push(text);
                result_scores.push(score);
                result_positions.push(positions);
                result_col_indices.push(col_indices);
                result_seq_lengths.push(seq_len);
            } else {
                // Keep entry to preserve index correspondence, but mark as filtered
                result_texts.push(String::new());
                result_scores.push(score);
                result_positions.push(Vec::new());
                result_col_indices.push(Vec::new());
                result_seq_lengths.push(seq_len);
            }
        }

        Ok(TextRecognitionOutput {
            texts: result_texts,
            scores: result_scores,
            char_positions: result_positions,
            char_col_indices: result_col_indices,
            sequence_lengths: result_seq_lengths,
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        32
    }
}

impl_adapter_builder! {
    builder_name: TextRecognitionAdapterBuilder,
    adapter_name: TextRecognitionAdapter,
    config_type: TextRecognitionConfig,
    adapter_type: "text_recognition",
    adapter_desc: "Recognizes text content from image regions",
    task_type: TextRecognition,

    fields: {
        preprocess_config: CRNNPreprocessConfig = CRNNPreprocessConfig::default(),
        character_dict: Option<Vec<String>> = None,
        return_word_box: bool = false,
    },

    methods: {
        /// Sets the model input shape.
        pub fn model_input_shape(mut self, shape: [usize; 3]) -> Self {
            self.preprocess_config.model_input_shape = shape;
            self
        }

        /// Sets the character dictionary.
        pub fn character_dict(mut self, character_dict: Vec<String>) -> Self {
            self.character_dict = Some(character_dict);
            self
        }

        /// Sets the score threshold.
        pub fn score_thresh(mut self, score_thresh: f32) -> Self {
            self.config.task_config.score_threshold = score_thresh;
            self
        }

        /// Sets the maximum image width.
        pub fn max_img_w(mut self, max_img_w: usize) -> Self {
            self.preprocess_config.max_img_w = Some(max_img_w);
            self
        }

        /// Sets whether to return character positions for word box generation.
        pub fn return_word_box(mut self, enable: bool) -> Self {
            self.return_word_box = enable;
            self
        }
    }

    build: |builder: TextRecognitionAdapterBuilder, model_path: &std::path::Path| -> Result<TextRecognitionAdapter, OCRError> {
        let (task_config, ort_config) = builder.config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the CRNN model
        let mut model_builder = CRNNModelBuilder::new().preprocess_config(builder.preprocess_config);

        if let Some(character_dict) = builder.character_dict {
            model_builder = model_builder.character_dict(character_dict);
        }

        let model = apply_ort_config!(model_builder, ort_config).build(model_path)?;

        // Create adapter info using the helper
        let info = TextRecognitionAdapterBuilder::base_adapter_info();

        Ok(TextRecognitionAdapter {
            model,
            info,
            config: task_config,
            return_word_box: builder.return_word_box,
        })
    },
}
