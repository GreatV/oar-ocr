//! Predictor implementations for various OCR tasks.
//!
//! This module contains implementations of different predictors used in the OCR pipeline:
//! - Text detection (finding text regions in images)
//! - Text recognition (recognizing characters in text regions)
//! - Document orientation classification (determining document orientation)
//! - Document rectification (correcting document perspective)
//! - Text line classification (classifying text line properties)
//!
//! Each predictor module contains both the predictor implementation and its builder.

/// Text recognition predictor using CRNN (Convolutional Recurrent Neural Network)
pub mod crnn_recognizer;

/// Text detection predictor using DB (Differentiable Binarization) algorithm
pub mod db_detector;

/// Document orientation classifier for determining document orientation
pub mod doc_orientation_classifier;

/// Document rectifier using DocTR (Document Text Recognition) models
pub mod doctr_rectifier;

/// Text line classifier for classifying properties of text lines
pub mod text_line_classifier;

// Re-exports for easier access to predictor types
pub use crnn_recognizer::{TextRecPredictor, TextRecPredictorBuilder};
pub use db_detector::{TextDetPredictor, TextDetPredictorBuilder};
pub use doc_orientation_classifier::{DocOrientationClassifier, DocOrientationClassifierBuilder};
pub use doctr_rectifier::{DoctrRectifierPredictor, DoctrRectifierPredictorBuilder};
pub use text_line_classifier::{TextLineClasPredictor, TextLineClasPredictorBuilder};

/// Implements the `BasePredictor` trait for a predictor type.
///
/// This macro provides a standard implementation of the `BasePredictor` trait
/// for different predictor types. It handles common functionality like
/// processing batches, error handling, and tracing.
///
/// # Variants
///
/// 1. Basic variant: `impl_standard_predictor!(PredictorType, ResultType, ErrorType, "predictor_name")`
///    - Calls `process_internal` method on the predictor
///
/// 2. Method variant: `impl_standard_predictor!(PredictorType, ResultType, ErrorType, "predictor_name", method_name)`
///    - Calls the specified method on the predictor
///
/// 3. Method with args variant: `impl_standard_predictor!(PredictorType, ResultType, ErrorType, "predictor_name", method_name(arg1, arg2))`
///    - Calls the specified method with additional arguments
///
/// # Parameters
///
/// - `$predictor_type`: The type of the predictor implementing the trait
/// - `$result_type`: The result type returned by the predictor
/// - `$error_type`: The error type that can be returned by the predictor
/// - `$predictor_name`: A string literal representing the predictor name (for tracing)
/// - `$method`: Optional method name to call instead of `process_internal`
/// - `$args`: Optional additional arguments to pass to the method
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_standard_predictor;
/// # struct MyPredictor;
/// # struct MyResult;
/// # struct MyError;
/// impl_standard_predictor!(MyPredictor, MyResult, MyError, "my_predictor");
/// ```
#[macro_export]
macro_rules! impl_standard_predictor {


    ($predictor_type:ty, $result_type:ty, $error_type:ty, $predictor_name:expr) => {
        impl $crate::core::traits::BasePredictor for $predictor_type {
            type Result = $result_type;
            type Error = $error_type;

            #[track_caller]
            fn process(&mut self, batch_data: $crate::core::batch::BatchData) -> Result<Self::Result, Self::Error> {
                use tracing::error;

                let span = tracing::span!(
                    tracing::Level::DEBUG,
                    "predictor_process",
                    predictor_type = $predictor_name,
                    batch_size = batch_data.len()
                );
                let _enter = span.enter();


                match self.process_internal(batch_data) {
                    Ok(result) => {
                        tracing::debug!("Processing completed successfully");
                        Ok(result)
                    }
                    Err(e) => {
                        error!("Processing failed: {}", e);
                        Err(e)
                    }
                }
            }

            fn convert_to_prediction_result(&self, result: Self::Result) -> $crate::core::predictions::PredictionResult<'static> {
                result.into_prediction()
            }

            fn batch_sampler(&self) -> &$crate::core::batch::BatchSampler {
                &self.batch_sampler
            }

            fn model_name(&self) -> &str {
                &self.model_name
            }

            fn predictor_type_name(&self) -> &str {
                $predictor_name
            }
        }
    };


    ($predictor_type:ty, $result_type:ty, $error_type:ty, $predictor_name:expr, $method:ident) => {
        impl $crate::core::traits::BasePredictor for $predictor_type {
            type Result = $result_type;
            type Error = $error_type;

            #[track_caller]
            fn process(&mut self, batch_data: $crate::core::batch::BatchData) -> Result<Self::Result, Self::Error> {
                use tracing::error;

                let span = tracing::span!(
                    tracing::Level::DEBUG,
                    "predictor_process",
                    predictor_type = $predictor_name,
                    batch_size = batch_data.len(),
                    method = stringify!($method)
                );
                let _enter = span.enter();

                match self.$method(batch_data) {
                    Ok(result) => {
                        tracing::debug!("Processing completed successfully");
                        Ok(result)
                    }
                    Err(e) => {
                        error!("Processing failed: {}", e);
                        Err(e)
                    }
                }
            }

            fn convert_to_prediction_result(&self, result: Self::Result) -> $crate::core::predictions::PredictionResult<'static> {
                result.into_prediction()
            }

            fn batch_sampler(&self) -> &$crate::core::batch::BatchSampler {
                &self.batch_sampler
            }

            fn model_name(&self) -> &str {
                &self.model_name
            }

            fn predictor_type_name(&self) -> &str {
                $predictor_name
            }
        }
    };



    ($predictor_type:ty, $result_type:ty, $error_type:ty, $predictor_name:expr, $method:ident($($args:expr),*)) => {
        impl $crate::core::traits::BasePredictor for $predictor_type {
            type Result = $result_type;
            type Error = $error_type;

            #[track_caller]
            fn process(&mut self, batch_data: $crate::core::batch::BatchData) -> Result<Self::Result, Self::Error> {
                use tracing::error;

                let span = tracing::span!(
                    tracing::Level::DEBUG,
                    "predictor_process",
                    predictor_type = $predictor_name,
                    batch_size = batch_data.len(),
                    method = stringify!($method)
                );
                let _enter = span.enter();

                match self.$method(batch_data, $($args),*) {
                    Ok(result) => {
                        tracing::debug!("Processing completed successfully");
                        Ok(result)
                    }
                    Err(e) => {
                        error!("Processing failed: {}", e);
                        Err(e)
                    }
                }
            }

            fn convert_to_prediction_result(&self, result: Self::Result) -> $crate::core::predictions::PredictionResult<'static> {
                result.into_prediction()
            }

            fn batch_sampler(&self) -> &$crate::core::batch::BatchSampler {
                &self.batch_sampler
            }

            fn model_name(&self) -> &str {
                &self.model_name
            }

            fn predictor_type_name(&self) -> &str {
                $predictor_name
            }
        }
    };
}

/// Implements the `PredictorBuilder` trait for a predictor builder type.
///
/// This macro provides a standard implementation of the `PredictorBuilder` trait
/// for predictor builder types. It handles common functionality like building
/// predictors from model paths.
///
/// # Parameters
///
/// - `$builder_type`: The type of the builder implementing the trait
/// - `$predictor_type`: The predictor type that will be built
/// - `$predictor_name`: A string literal representing the predictor name
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_standard_predictor_builder;
/// # struct MyPredictorBuilder;
/// # struct MyPredictor;
/// impl_standard_predictor_builder!(MyPredictorBuilder, MyPredictor, "my_predictor");
/// ```
#[macro_export]
macro_rules! impl_standard_predictor_builder {
    ($builder_type:ty, $predictor_type:ty, $predictor_name:expr) => {
        impl $crate::core::traits::PredictorBuilder for $builder_type {
            type Config = ();
            type Predictor = $predictor_type;

            fn build_typed(
                self,
                model_path: &std::path::Path,
            ) -> Result<Self::Predictor, $crate::core::errors::OCRError> {
                self.build_internal(model_path)
            }

            fn build_predictor(
                self,
                model_path: &std::path::Path,
            ) -> Result<Box<dyn $crate::core::traits::Predictor>, $crate::core::errors::OCRError>
            {
                Ok(Box::new(self.build_internal(model_path)?))
            }

            fn predictor_type(&self) -> &str {
                $predictor_name
            }

            fn with_config(self, _config: Self::Config) -> Self {
                self
            }
        }
    };
}
