//! Concrete task implementations for layout detection.
//!
//! This module provides the layout detection task that identifies document layout elements.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::processors::BoundingBox;
use crate::utils::{ScoreValidator, validate_max_value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bounding box merge mode for layout elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MergeBboxMode {
    /// Keep the larger bounding box
    #[default]
    Large,
    /// Merge to union of bounding boxes
    Union,
    /// Keep the smaller bounding box
    Small,
}

/// Unclip ratio configuration for layout detection.
/// Controls how bounding boxes are scaled while keeping the center fixed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnclipRatio {
    /// Single ratio applied to both width and height
    Uniform(f32),
    /// Separate ratios for (width, height)
    Separate(f32, f32),
    /// Per-class ratios: class_id -> (width_ratio, height_ratio)
    PerClass(HashMap<usize, (f32, f32)>),
}

impl Default for UnclipRatio {
    fn default() -> Self {
        UnclipRatio::Separate(1.0, 1.0)
    }
}

/// Configuration for layout detection task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDetectionConfig {
    /// Default score threshold for detection (default: 0.5)
    pub score_threshold: f32,
    /// Maximum number of layout elements (default: 100)
    pub max_elements: usize,
    /// Per-class score thresholds (overrides score_threshold for specific classes)
    /// PP-StructureV3 defaults:
    /// - paragraph_title: 0.3
    /// - formula: 0.3
    /// - text: 0.4
    /// - seal: 0.45
    /// - others: 0.5
    #[serde(default)]
    pub class_thresholds: Option<HashMap<String, f32>>,
    /// Per-class bounding box merge modes
    #[serde(default)]
    pub class_merge_modes: Option<HashMap<String, MergeBboxMode>>,
    /// Enable NMS for layout detection (default: true)
    #[serde(default = "default_layout_nms")]
    pub layout_nms: bool,
    /// NMS threshold (default: 0.5)
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f32,
    /// Unclip ratio for expanding/shrinking bounding boxes (PP-StructureV3)
    /// Default: [1.0, 1.0] (no change)
    #[serde(default)]
    pub layout_unclip_ratio: Option<UnclipRatio>,
}

fn default_layout_nms() -> bool {
    true
}

fn default_nms_threshold() -> f32 {
    0.5
}

impl Default for LayoutDetectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            max_elements: 100,
            class_thresholds: None,
            class_merge_modes: None,
            layout_nms: true,
            nms_threshold: 0.5,
            layout_unclip_ratio: None,
        }
    }
}

impl LayoutDetectionConfig {
    /// Creates a config with PP-StructureV3 default class thresholds.
    ///
    /// PP-StructureV3 uses different thresholds for different element types:
    /// - paragraph_title: 0.3
    /// - formula: 0.3
    /// - text: 0.4
    /// - seal: 0.45
    /// - others: 0.5 (default)
    pub fn with_pp_structurev3_thresholds() -> Self {
        let mut class_thresholds = HashMap::new();
        class_thresholds.insert("paragraph_title".to_string(), 0.3);
        class_thresholds.insert("formula".to_string(), 0.3);
        class_thresholds.insert("text".to_string(), 0.4);
        class_thresholds.insert("seal".to_string(), 0.45);

        Self {
            score_threshold: 0.5,
            max_elements: 100,
            class_thresholds: Some(class_thresholds),
            class_merge_modes: None,
            layout_nms: true,
            nms_threshold: 0.5,
            layout_unclip_ratio: Some(UnclipRatio::Separate(1.0, 1.0)),
        }
    }

    /// Creates a config with PP-StructureV3 default thresholds, merge modes, and unclip ratio.
    ///
    /// Merge modes follow standard configuration:
    /// - "large": paragraph_title, image, formula, chart
    /// - "union": all other PP-DocLayout_plus-L classes
    pub fn with_pp_structurev3_defaults() -> Self {
        let mut cfg = Self::with_pp_structurev3_thresholds();

        let mut merge_modes = HashMap::new();
        merge_modes.insert("paragraph_title".to_string(), MergeBboxMode::Large);
        merge_modes.insert("image".to_string(), MergeBboxMode::Large);
        merge_modes.insert("text".to_string(), MergeBboxMode::Union);
        merge_modes.insert("number".to_string(), MergeBboxMode::Union);
        merge_modes.insert("abstract".to_string(), MergeBboxMode::Union);
        merge_modes.insert("content".to_string(), MergeBboxMode::Union);
        merge_modes.insert("figure_table_chart_title".to_string(), MergeBboxMode::Union);
        merge_modes.insert("formula".to_string(), MergeBboxMode::Large);
        merge_modes.insert("table".to_string(), MergeBboxMode::Union);
        merge_modes.insert("reference".to_string(), MergeBboxMode::Union);
        merge_modes.insert("doc_title".to_string(), MergeBboxMode::Union);
        merge_modes.insert("footnote".to_string(), MergeBboxMode::Union);
        merge_modes.insert("header".to_string(), MergeBboxMode::Union);
        merge_modes.insert("algorithm".to_string(), MergeBboxMode::Union);
        merge_modes.insert("footer".to_string(), MergeBboxMode::Union);
        merge_modes.insert("seal".to_string(), MergeBboxMode::Union);
        merge_modes.insert("chart".to_string(), MergeBboxMode::Large);
        merge_modes.insert("formula_number".to_string(), MergeBboxMode::Union);
        merge_modes.insert("aside_text".to_string(), MergeBboxMode::Union);
        merge_modes.insert("reference_content".to_string(), MergeBboxMode::Union);

        cfg.class_merge_modes = Some(merge_modes);
        cfg.layout_unclip_ratio = Some(UnclipRatio::Separate(1.0, 1.0));
        cfg
    }

    /// Gets the threshold for a specific class.
    ///
    /// Returns the class-specific threshold if configured, otherwise the default threshold.
    pub fn get_class_threshold(&self, class_name: &str) -> f32 {
        self.class_thresholds
            .as_ref()
            .and_then(|thresholds| thresholds.get(class_name).copied())
            .unwrap_or(self.score_threshold)
    }

    /// Gets the merge mode for a specific class.
    ///
    /// Returns the class-specific merge mode if configured, otherwise Large (default).
    pub fn get_class_merge_mode(&self, class_name: &str) -> MergeBboxMode {
        self.class_merge_modes
            .as_ref()
            .and_then(|modes| modes.get(class_name).copied())
            .unwrap_or(MergeBboxMode::Large)
    }
}

impl_config_validator!(LayoutDetectionConfig {
    score_threshold: range(0.0, 1.0),
    max_elements: min(1),
});

/// A detected layout element.
#[derive(Debug, Clone)]
pub struct LayoutElement {
    /// Bounding box of the element
    pub bbox: BoundingBox,
    /// Type of layout element (string label)
    pub element_type: String,
    /// Confidence score
    pub score: f32,
}

/// Output from layout detection task.
#[derive(Debug, Clone)]
pub struct LayoutDetectionOutput {
    /// Detected layout elements per image
    pub elements: Vec<Vec<LayoutElement>>,
    /// Whether elements are already sorted by reading order (e.g., from PP-DocLayoutV2)
    ///
    /// When `true`, downstream consumers can skip reading order sorting algorithms
    /// as the elements are already in the correct reading order based on model output.
    pub is_reading_order_sorted: bool,
}

impl LayoutDetectionOutput {
    /// Creates an empty layout detection output.
    pub fn empty() -> Self {
        Self {
            elements: Vec::new(),
            is_reading_order_sorted: false,
        }
    }

    /// Creates a layout detection output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: Vec::with_capacity(capacity),
            is_reading_order_sorted: false,
        }
    }

    /// Sets the reading order sorted flag.
    pub fn with_reading_order_sorted(mut self, sorted: bool) -> Self {
        self.is_reading_order_sorted = sorted;
        self
    }
}

/// Layout detection task implementation.
#[derive(Debug, Default)]
pub struct LayoutDetectionTask {
    config: LayoutDetectionConfig,
}

impl LayoutDetectionTask {
    /// Creates a new layout detection task.
    pub fn new(config: LayoutDetectionConfig) -> Self {
        Self { config }
    }
}

impl Task for LayoutDetectionTask {
    type Config = LayoutDetectionConfig;
    type Input = ImageTaskInput;
    type Output = LayoutDetectionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::LayoutDetection
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::LayoutDetection,
            vec!["image".to_string()],
            vec!["layout_elements".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for layout detection")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        for (idx, elements) in output.elements.iter().enumerate() {
            // Validate element count
            validate_max_value(
                elements.len(),
                self.config.max_elements,
                "element count",
                &format!("Image {}", idx),
            )?;

            // Validate scores
            let scores: Vec<f32> = elements.iter().map(|e| e.score).collect();
            validator.validate_scores_with(&scores, |elem_idx| {
                format!("Image {}, element {}", idx, elem_idx)
            })?;
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        LayoutDetectionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::Point;
    use image::RgbImage;

    #[test]
    fn test_layout_detection_task_creation() {
        let task = LayoutDetectionTask::default();
        assert_eq!(task.task_type(), TaskType::LayoutDetection);
    }

    #[test]
    fn test_input_validation() {
        let task = LayoutDetectionTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = LayoutDetectionTask::default();

        // Valid output should pass
        let box1 = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let element = LayoutElement {
            bbox: box1,
            element_type: "text".to_string(),
            score: 0.95,
        };
        let output = LayoutDetectionOutput {
            elements: vec![vec![element]],
            is_reading_order_sorted: false,
        };
        assert!(task.validate_output(&output).is_ok());
    }

    #[test]
    fn test_schema() {
        let task = LayoutDetectionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::LayoutDetection);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"layout_elements".to_string()));
    }
}
