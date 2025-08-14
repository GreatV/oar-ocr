//! Prediction result types for the OCR pipeline.
//!
//! This module defines various types and traits for representing and working with
//! prediction results in the OCR pipeline. It includes enums for different types
//! of predictions (detection, recognition, classification, rectification) and
//! traits for converting between different representations.

use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;

/// Enum representing different types of prediction results.
///
/// This enum is used to represent the results of different types of predictions
/// in the OCR pipeline, such as text detection, text recognition, image classification,
/// and image rectification.
///
/// # Type Parameters
///
/// * `'a` - The lifetime of the borrowed data.
/// * `I` - The type of the input images.
#[derive(Debug, Clone)]
pub enum PredictionResult<'a, I = Arc<RgbImage>> {
    /// Results from text detection.
    Detection {
        /// The input paths of the images.
        input_path: Vec<Cow<'a, str>>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        input_img: Vec<I>,
        /// The detected polygons.
        dt_polys: Vec<Vec<crate::processors::BoundingBox>>,
        /// The scores for the detected polygons.
        dt_scores: Vec<Vec<f32>>,
    },
    /// Results from text recognition.
    Recognition {
        /// The input paths of the images.
        input_path: Vec<Cow<'a, str>>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        input_img: Vec<I>,
        /// The recognized text.
        rec_text: Vec<Cow<'a, str>>,
        /// The scores for the recognized text.
        rec_score: Vec<f32>,
    },
    /// Results from image classification.
    Classification {
        /// The input paths of the images.
        input_path: Vec<Cow<'a, str>>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        input_img: Vec<I>,
        /// The class IDs for the classifications.
        class_ids: Vec<Vec<usize>>,
        /// The scores for the classifications.
        scores: Vec<Vec<f32>>,
        /// The label names for the classifications.
        label_names: Vec<Vec<Cow<'a, str>>>,
    },
    /// Results from image rectification.
    Rectification {
        /// The input paths of the images.
        input_path: Vec<Cow<'a, str>>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        input_img: Vec<I>,
        /// The rectified images.
        rectified_img: Vec<I>,
    },
}

/// Enum representing owned prediction results.
///
/// This enum is similar to PredictionResult, but uses owned String values instead
/// of borrowed Cow values. It also implements Serialize and Deserialize traits
/// for easy serialization and deserialization.
///
/// # Type Parameters
///
/// * `I` - The type of the input images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OwnedPredictionResult<I = Arc<RgbImage>> {
    /// Results from text detection.
    Detection {
        /// The input paths of the images.
        input_path: Vec<String>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        #[serde(skip)]
        input_img: Vec<I>,
        /// The detected polygons.
        dt_polys: Vec<Vec<crate::processors::BoundingBox>>,
        /// The scores for the detected polygons.
        dt_scores: Vec<Vec<f32>>,
    },
    /// Results from text recognition.
    Recognition {
        /// The input paths of the images.
        input_path: Vec<String>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        #[serde(skip)]
        input_img: Vec<I>,
        /// The recognized text.
        rec_text: Vec<String>,
        /// The scores for the recognized text.
        rec_score: Vec<f32>,
    },
    /// Results from image classification.
    Classification {
        /// The input paths of the images.
        input_path: Vec<String>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        #[serde(skip)]
        input_img: Vec<I>,
        /// The class IDs for the classifications.
        class_ids: Vec<Vec<usize>>,
        /// The scores for the classifications.
        scores: Vec<Vec<f32>>,
        /// The label names for the classifications.
        label_names: Vec<Vec<String>>,
    },
    /// Results from image rectification.
    Rectification {
        /// The input paths of the images.
        input_path: Vec<String>,
        /// The indices of the images in the batch.
        index: Vec<usize>,
        /// The input images.
        #[serde(skip)]
        input_img: Vec<I>,
        /// The rectified images.
        #[serde(skip)]
        rectified_img: Vec<I>,
    },
}

/// Implementation of methods for PredictionResult.
impl<'a, I> PredictionResult<'a, I> {
    /// Gets the input paths of the images.
    ///
    /// # Returns
    ///
    /// A slice of the input paths.
    pub fn input_paths(&self) -> &[Cow<'a, str>] {
        match self {
            PredictionResult::Detection { input_path, .. } => input_path,
            PredictionResult::Recognition { input_path, .. } => input_path,
            PredictionResult::Classification { input_path, .. } => input_path,
            PredictionResult::Rectification { input_path, .. } => input_path,
        }
    }

    /// Gets the indices of the images in the batch.
    ///
    /// # Returns
    ///
    /// A slice of the indices.
    pub fn indices(&self) -> &[usize] {
        match self {
            PredictionResult::Detection { index, .. } => index,
            PredictionResult::Recognition { index, .. } => index,
            PredictionResult::Classification { index, .. } => index,
            PredictionResult::Rectification { index, .. } => index,
        }
    }

    /// Gets the input images.
    ///
    /// # Returns
    ///
    /// A slice of the input images.
    pub fn input_images(&self) -> &[I] {
        match self {
            PredictionResult::Detection { input_img, .. } => input_img,
            PredictionResult::Recognition { input_img, .. } => input_img,
            PredictionResult::Classification { input_img, .. } => input_img,
            PredictionResult::Rectification { input_img, .. } => input_img,
        }
    }

    /// Checks if the prediction result is a detection result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a detection result, false otherwise.
    pub fn is_detection(&self) -> bool {
        matches!(self, PredictionResult::Detection { .. })
    }

    /// Checks if the prediction result is a recognition result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a recognition result, false otherwise.
    pub fn is_recognition(&self) -> bool {
        matches!(self, PredictionResult::Recognition { .. })
    }

    /// Checks if the prediction result is a classification result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a classification result, false otherwise.
    pub fn is_classification(&self) -> bool {
        matches!(self, PredictionResult::Classification { .. })
    }

    /// Checks if the prediction result is a rectification result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a rectification result, false otherwise.
    pub fn is_rectification(&self) -> bool {
        matches!(self, PredictionResult::Rectification { .. })
    }

    /// Converts the prediction result to an owned prediction result.
    ///
    /// # Returns
    ///
    /// An OwnedPredictionResult with the same data.
    pub fn into_owned(self) -> OwnedPredictionResult<I> {
        match self {
            PredictionResult::Detection {
                input_path,
                index,
                input_img,
                dt_polys,
                dt_scores,
            } => OwnedPredictionResult::Detection {
                input_path: input_path.into_iter().map(|cow| cow.into_owned()).collect(),
                index,
                input_img,
                dt_polys,
                dt_scores,
            },
            PredictionResult::Recognition {
                input_path,
                index,
                input_img,
                rec_text,
                rec_score,
            } => OwnedPredictionResult::Recognition {
                input_path: input_path.into_iter().map(|cow| cow.into_owned()).collect(),
                index,
                input_img,
                rec_text: rec_text.into_iter().map(|cow| cow.into_owned()).collect(),
                rec_score,
            },
            PredictionResult::Classification {
                input_path,
                index,
                input_img,
                class_ids,
                scores,
                label_names,
            } => OwnedPredictionResult::Classification {
                input_path: input_path.into_iter().map(|cow| cow.into_owned()).collect(),
                index,
                input_img,
                class_ids,
                scores,
                label_names: label_names
                    .into_iter()
                    .map(|vec| vec.into_iter().map(|cow| cow.into_owned()).collect())
                    .collect(),
            },
            PredictionResult::Rectification {
                input_path,
                index,
                input_img,
                rectified_img,
            } => OwnedPredictionResult::Rectification {
                input_path: input_path.into_iter().map(|cow| cow.into_owned()).collect(),
                index,
                input_img,
                rectified_img,
            },
        }
    }
}

/// Implementation of methods for OwnedPredictionResult.
impl<I> OwnedPredictionResult<I> {
    /// Gets the input paths of the images.
    ///
    /// # Returns
    ///
    /// A slice of the input paths.
    pub fn input_paths(&self) -> &[String] {
        match self {
            OwnedPredictionResult::Detection { input_path, .. } => input_path,
            OwnedPredictionResult::Recognition { input_path, .. } => input_path,
            OwnedPredictionResult::Classification { input_path, .. } => input_path,
            OwnedPredictionResult::Rectification { input_path, .. } => input_path,
        }
    }

    /// Gets the indices of the images in the batch.
    ///
    /// # Returns
    ///
    /// A slice of the indices.
    pub fn indices(&self) -> &[usize] {
        match self {
            OwnedPredictionResult::Detection { index, .. } => index,
            OwnedPredictionResult::Recognition { index, .. } => index,
            OwnedPredictionResult::Classification { index, .. } => index,
            OwnedPredictionResult::Rectification { index, .. } => index,
        }
    }

    /// Gets the input images.
    ///
    /// # Returns
    ///
    /// A slice of the input images.
    pub fn input_images(&self) -> &[I] {
        match self {
            OwnedPredictionResult::Detection { input_img, .. } => input_img,
            OwnedPredictionResult::Recognition { input_img, .. } => input_img,
            OwnedPredictionResult::Classification { input_img, .. } => input_img,
            OwnedPredictionResult::Rectification { input_img, .. } => input_img,
        }
    }

    /// Checks if the prediction result is a detection result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a detection result, false otherwise.
    pub fn is_detection(&self) -> bool {
        matches!(self, OwnedPredictionResult::Detection { .. })
    }

    /// Checks if the prediction result is a recognition result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a recognition result, false otherwise.
    pub fn is_recognition(&self) -> bool {
        matches!(self, OwnedPredictionResult::Recognition { .. })
    }

    /// Checks if the prediction result is a classification result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a classification result, false otherwise.
    pub fn is_classification(&self) -> bool {
        matches!(self, OwnedPredictionResult::Classification { .. })
    }

    /// Checks if the prediction result is a rectification result.
    ///
    /// # Returns
    ///
    /// True if the prediction result is a rectification result, false otherwise.
    pub fn is_rectification(&self) -> bool {
        matches!(self, OwnedPredictionResult::Rectification { .. })
    }

    /// Converts the owned prediction result to a borrowed prediction result.
    ///
    /// # Returns
    ///
    /// A PredictionResult with borrowed data.
    pub fn as_prediction_result(&self) -> PredictionResult<'_, &I> {
        match self {
            OwnedPredictionResult::Detection {
                input_path,
                index,
                input_img,
                dt_polys,
                dt_scores,
            } => PredictionResult::Detection {
                input_path: input_path
                    .iter()
                    .map(|s| Cow::Borrowed(s.as_str()))
                    .collect(),
                index: index.clone(),
                input_img: input_img.iter().collect(),
                dt_polys: dt_polys.clone(),
                dt_scores: dt_scores.clone(),
            },
            OwnedPredictionResult::Recognition {
                input_path,
                index,
                input_img,
                rec_text,
                rec_score,
            } => PredictionResult::Recognition {
                input_path: input_path
                    .iter()
                    .map(|s| Cow::Borrowed(s.as_str()))
                    .collect(),
                index: index.clone(),
                input_img: input_img.iter().collect(),
                rec_text: rec_text.iter().map(|s| Cow::Borrowed(s.as_str())).collect(),
                rec_score: rec_score.clone(),
            },
            OwnedPredictionResult::Classification {
                input_path,
                index,
                input_img,
                class_ids,
                scores,
                label_names,
            } => PredictionResult::Classification {
                input_path: input_path
                    .iter()
                    .map(|s| Cow::Borrowed(s.as_str()))
                    .collect(),
                index: index.clone(),
                input_img: input_img.iter().collect(),
                class_ids: class_ids.clone(),
                scores: scores.clone(),
                label_names: label_names
                    .iter()
                    .map(|vec| vec.iter().map(|s| Cow::Borrowed(s.as_str())).collect())
                    .collect(),
            },
            OwnedPredictionResult::Rectification {
                input_path,
                index,
                input_img,
                rectified_img,
            } => PredictionResult::Rectification {
                input_path: input_path
                    .iter()
                    .map(|s| Cow::Borrowed(s.as_str()))
                    .collect(),
                index: index.clone(),
                input_img: input_img.iter().collect(),
                rectified_img: rectified_img.iter().collect(),
            },
        }
    }
}

/// Trait for converting a type into a prediction result.
///
/// This trait is used to convert a type into a prediction result.
pub trait IntoPrediction {
    /// The output type.
    type Out;
    /// Converts the type into a prediction result.
    ///
    /// # Returns
    ///
    /// The prediction result.
    fn into_prediction(self) -> Self::Out;
}

/// Trait for converting a type into an owned prediction result.
///
/// This trait is used to convert a type into an owned prediction result.
pub trait IntoOwnedPrediction {
    /// The output type.
    type Out;
    /// Converts the type into an owned prediction result.
    ///
    /// # Returns
    ///
    /// The owned prediction result.
    fn into_owned_prediction(self) -> Self::Out;
}

/// Implementation of IntoOwnedPrediction for types that implement IntoPrediction.
///
/// This implementation allows types that implement IntoPrediction to be converted
/// into owned prediction results.
impl<T> IntoOwnedPrediction for T
where
    T: IntoPrediction,
    T::Out: Into<OwnedPredictionResult>,
{
    type Out = OwnedPredictionResult;

    fn into_owned_prediction(self) -> Self::Out {
        self.into_prediction().into()
    }
}

/// Implementation of From for converting PredictionResult to OwnedPredictionResult.
///
/// This implementation allows PredictionResult to be converted to OwnedPredictionResult.
impl<I> From<PredictionResult<'_, I>> for OwnedPredictionResult<I> {
    fn from(result: PredictionResult<'_, I>) -> Self {
        result.into_owned()
    }
}

/// Statistics for the OCR pipeline.
///
/// This struct is used to track statistics for the OCR pipeline, such as the number
/// of images processed, the number of successful and failed predictions, and the
/// average inference time.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// The total number of images processed.
    pub total_processed: usize,
    /// The number of successful predictions.
    pub successful_predictions: usize,
    /// The number of failed predictions.
    pub failed_predictions: usize,
    /// The average inference time in milliseconds.
    pub average_inference_time_ms: f64,
}

/// Implementation of methods for PipelineStats.
impl PipelineStats {
    /// Creates a new PipelineStats instance with default values.
    ///
    /// # Returns
    ///
    /// A new PipelineStats instance.
    pub fn new() -> Self {
        Self {
            total_processed: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            average_inference_time_ms: 0.0,
        }
    }

    /// Gets the success rate as a percentage.
    ///
    /// # Returns
    ///
    /// Success rate as a percentage (0.0 to 100.0)
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            0.0
        } else {
            (self.successful_predictions as f64 / self.total_processed as f64) * 100.0
        }
    }

    /// Gets the failure rate as a percentage.
    ///
    /// # Returns
    ///
    /// Failure rate as a percentage (0.0 to 100.0)
    pub fn failure_rate(&self) -> f64 {
        if self.total_processed == 0 {
            0.0
        } else {
            (self.failed_predictions as f64 / self.total_processed as f64) * 100.0
        }
    }

    /// Gets the average processing speed in images per second.
    ///
    /// # Returns
    ///
    /// Processing speed in images per second
    pub fn images_per_second(&self) -> f64 {
        if self.average_inference_time_ms == 0.0 {
            0.0
        } else {
            1000.0 / self.average_inference_time_ms
        }
    }
}

/// Implementation of Default for PipelineStats.
///
/// This implementation allows PipelineStats to be created with default values.
impl Default for PipelineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of Display for PipelineStats.
///
/// This implementation allows PipelineStats to be easily printed.
impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pipeline Statistics:")?;
        writeln!(f, "  Total processed: {}", self.total_processed)?;
        writeln!(
            f,
            "  Successful: {} ({:.1}%)",
            self.successful_predictions,
            self.success_rate()
        )?;
        writeln!(
            f,
            "  Failed: {} ({:.1}%)",
            self.failed_predictions,
            self.failure_rate()
        )?;
        writeln!(
            f,
            "  Average inference time: {:.2} ms",
            self.average_inference_time_ms
        )?;
        writeln!(
            f,
            "  Processing speed: {:.2} images/sec",
            self.images_per_second()
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_stats_success_rate_zero_processed() {
        let stats = PipelineStats {
            total_processed: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            average_inference_time_ms: 0.0,
        };
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_pipeline_stats_success_rate_all_successful() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 10,
            failed_predictions: 0,
            average_inference_time_ms: 50.0,
        };
        assert_eq!(stats.success_rate(), 100.0);
    }

    #[test]
    fn test_pipeline_stats_success_rate_partial() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 7,
            failed_predictions: 3,
            average_inference_time_ms: 50.0,
        };
        assert_eq!(stats.success_rate(), 70.0);
    }

    #[test]
    fn test_pipeline_stats_failure_rate_zero_processed() {
        let stats = PipelineStats {
            total_processed: 0,
            successful_predictions: 0,
            failed_predictions: 0,
            average_inference_time_ms: 0.0,
        };
        assert_eq!(stats.failure_rate(), 0.0);
    }

    #[test]
    fn test_pipeline_stats_failure_rate_all_failed() {
        let stats = PipelineStats {
            total_processed: 5,
            successful_predictions: 0,
            failed_predictions: 5,
            average_inference_time_ms: 100.0,
        };
        assert_eq!(stats.failure_rate(), 100.0);
    }

    #[test]
    fn test_pipeline_stats_failure_rate_partial() {
        let stats = PipelineStats {
            total_processed: 8,
            successful_predictions: 6,
            failed_predictions: 2,
            average_inference_time_ms: 75.0,
        };
        assert_eq!(stats.failure_rate(), 25.0);
    }

    #[test]
    fn test_pipeline_stats_images_per_second_zero_time() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 10,
            failed_predictions: 0,
            average_inference_time_ms: 0.0,
        };
        assert_eq!(stats.images_per_second(), 0.0);
    }

    #[test]
    fn test_pipeline_stats_images_per_second_normal() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 10,
            failed_predictions: 0,
            average_inference_time_ms: 100.0, // 100ms per image
        };
        assert_eq!(stats.images_per_second(), 10.0); // 1000ms / 100ms = 10 images/sec
    }

    #[test]
    fn test_pipeline_stats_images_per_second_fast() {
        let stats = PipelineStats {
            total_processed: 100,
            successful_predictions: 100,
            failed_predictions: 0,
            average_inference_time_ms: 50.0, // 50ms per image
        };
        assert_eq!(stats.images_per_second(), 20.0); // 1000ms / 50ms = 20 images/sec
    }

    #[test]
    fn test_pipeline_stats_display() {
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 8,
            failed_predictions: 2,
            average_inference_time_ms: 125.0,
        };

        let display_str = format!("{}", stats);
        assert!(display_str.contains("Pipeline Statistics:"));
        assert!(display_str.contains("Total processed: 10"));
        assert!(display_str.contains("Successful: 8 (80.0%)"));
        assert!(display_str.contains("Failed: 2 (20.0%)"));
        assert!(display_str.contains("Average inference time: 125.00 ms"));
        assert!(display_str.contains("Processing speed: 8.00 images/sec"));
    }

    #[test]
    fn test_pipeline_stats_edge_case_inconsistent_counts() {
        // Test edge case where successful + failed != total (shouldn't happen in practice)
        let stats = PipelineStats {
            total_processed: 10,
            successful_predictions: 7,
            failed_predictions: 2, // 7 + 2 = 9, not 10
            average_inference_time_ms: 100.0,
        };

        // Should still calculate rates based on total_processed
        assert_eq!(stats.success_rate(), 70.0);
        assert_eq!(stats.failure_rate(), 20.0);
    }
}
