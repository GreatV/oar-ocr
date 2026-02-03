//! Layout Detection Adapter
//!
//! This module provides adapters for layout detection models.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::core::{OCRError, TaskType, Tensor4D};
use crate::domain::tasks::{
    LayoutDetectionConfig, LayoutDetectionElement, LayoutDetectionOutput, LayoutDetectionTask,
    MergeBboxMode, UnclipRatio,
};
use crate::models::detection::{
    PPDocLayoutModel, PPDocLayoutModelBuilder, PPDocLayoutPostprocessConfig, PicoDetModel,
    PicoDetModelBuilder, PicoDetPostprocessConfig, RTDetrModel, RTDetrModelBuilder,
    RTDetrPostprocessConfig,
};
use crate::processors::{ImageScaleInfo, LayoutPostProcess, apply_nms_with_merge, unclip_boxes};
use ndarray::Axis;
use std::collections::HashMap;
use std::path::Path;

/// Result type for layout detection box filtering and merging operations.
type LayoutBoxResult = (
    Vec<crate::processors::BoundingBox>,
    Vec<usize>,
    Vec<f32>,
    Vec<(f32, f32)>,
);

/// Configuration for layout detection models.
#[derive(Debug, Clone)]
pub struct LayoutModelConfig {
    /// Model name
    pub model_name: String,
    /// Number of classes
    pub num_classes: usize,
    /// Class label mapping (class_id -> label string)
    pub class_labels: HashMap<usize, String>,
    /// Model type (e.g., "picodet", "rtdetr", "pp-doclayout")
    pub model_type: String,
    /// Optional fixed input image size (height, width)
    pub input_size: Option<(u32, u32)>,
}

impl LayoutModelConfig {
    /// Create configuration for PicoDet layout 1x model (5 classes).
    pub fn picodet_layout_1x() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "text".to_string());
        class_labels.insert(1, "title".to_string());
        class_labels.insert(2, "list".to_string());
        class_labels.insert(3, "table".to_string());
        class_labels.insert(4, "figure".to_string());

        Self {
            model_name: "picodet_layout_1x".to_string(),
            num_classes: 5,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((800, 608)),
        }
    }

    /// Create configuration for PicoDet layout 1x table-only model.
    pub fn picodet_layout_1x_table() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "table".to_string());

        Self {
            model_name: "picodet_layout_1x_table".to_string(),
            num_classes: 1,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((800, 608)),
        }
    }

    /// Create configuration for PicoDet-S layout 3 class model.
    pub fn picodet_s_layout_3cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "image".to_string());
        class_labels.insert(1, "table".to_string());
        class_labels.insert(2, "seal".to_string()); // seal treated as separate class

        Self {
            model_name: "picodet-s_layout_3cls".to_string(),
            num_classes: 3,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((480, 480)),
        }
    }

    /// Create configuration for PicoDet-L layout 3 class model.
    pub fn picodet_l_layout_3cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "image".to_string());
        class_labels.insert(1, "table".to_string());
        class_labels.insert(2, "seal".to_string());

        Self {
            model_name: "picodet-l_layout_3cls".to_string(),
            num_classes: 3,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PicoDet-S layout 17 class model.
    pub fn picodet_s_layout_17cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());

        Self {
            model_name: "picodet-s_layout_17cls".to_string(),
            num_classes: 17,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((480, 480)),
        }
    }

    /// Create configuration for PicoDet-L layout 17 class model.
    pub fn picodet_l_layout_17cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());

        Self {
            model_name: "picodet-l_layout_17cls".to_string(),
            num_classes: 17,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for RT-DETR-H layout 3 class model.
    pub fn rtdetr_h_layout_3cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "figure".to_string()); // image
        class_labels.insert(1, "table".to_string());
        class_labels.insert(2, "seal".to_string()); // seal

        Self {
            model_name: "rt-detr-h_layout_3cls".to_string(),
            num_classes: 3,
            class_labels,
            model_type: "rtdetr".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for RT-DETR-H layout 17 class model.
    pub fn rtdetr_h_layout_17cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());

        Self {
            model_name: "rt-detr-h_layout_17cls".to_string(),
            num_classes: 17,
            class_labels,
            model_type: "rtdetr".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocBlockLayout model (1 class: Region).
    /// This model uses 640x640 input size and only detects generic regions.
    pub fn pp_docblocklayout() -> Self {
        let mut class_labels = HashMap::new();
        // PP-DocBlockLayout has only 1 class: Region (generic layout block)
        class_labels.insert(0, "region".to_string());

        Self {
            model_name: "pp-docblocklayout".to_string(),
            num_classes: 1,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocLayout-S model (23 classes).
    pub fn pp_doclayout_s() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());
        class_labels.insert(17, "chart_title".to_string());
        class_labels.insert(18, "chart".to_string());
        class_labels.insert(19, "formula_number".to_string());
        class_labels.insert(20, "header_image".to_string());
        class_labels.insert(21, "footer_image".to_string());
        class_labels.insert(22, "aside_text".to_string());

        Self {
            model_name: "pp-doclayout-s".to_string(),
            num_classes: 23,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((480, 480)),
        }
    }

    /// Create configuration for PP-DocLayout-M model (23 classes).
    pub fn pp_doclayout_m() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());
        class_labels.insert(17, "chart_title".to_string());
        class_labels.insert(18, "chart".to_string());
        class_labels.insert(19, "formula_number".to_string());
        class_labels.insert(20, "header_image".to_string());
        class_labels.insert(21, "footer_image".to_string());
        class_labels.insert(22, "aside_text".to_string());

        Self {
            model_name: "pp-doclayout-m".to_string(),
            num_classes: 23,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocLayout-L model (23 classes).
    pub fn pp_doclayout_l() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());
        class_labels.insert(17, "chart_title".to_string());
        class_labels.insert(18, "chart".to_string());
        class_labels.insert(19, "formula_number".to_string());
        class_labels.insert(20, "header_image".to_string());
        class_labels.insert(21, "footer_image".to_string());
        class_labels.insert(22, "aside_text".to_string());

        Self {
            model_name: "pp-doclayout-l".to_string(),
            num_classes: 23,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocLayout-plus-L model (20 classes).
    pub fn pp_doclayout_plus_l() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "reference".to_string());
        class_labels.insert(10, "doc_title".to_string());
        class_labels.insert(11, "footnote".to_string());
        class_labels.insert(12, "header".to_string());
        class_labels.insert(13, "algorithm".to_string());
        class_labels.insert(14, "footer".to_string());
        class_labels.insert(15, "seal".to_string());
        class_labels.insert(16, "chart".to_string());
        class_labels.insert(17, "formula_number".to_string());
        class_labels.insert(18, "aside_text".to_string());
        class_labels.insert(19, "reference_content".to_string());

        Self {
            model_name: "pp-doclayout_plus-l".to_string(),
            num_classes: 20,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((800, 800)),
        }
    }

    /// Create configuration for PP-DocLayoutV2 model (25 classes).
    pub fn pp_doclayoutv2() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "abstract".to_string());
        class_labels.insert(1, "algorithm".to_string());
        class_labels.insert(2, "aside_text".to_string());
        class_labels.insert(3, "chart".to_string());
        class_labels.insert(4, "content".to_string());
        class_labels.insert(5, "display_formula".to_string());
        class_labels.insert(6, "doc_title".to_string());
        class_labels.insert(7, "figure_title".to_string());
        class_labels.insert(8, "footer".to_string());
        class_labels.insert(9, "footer_image".to_string());
        class_labels.insert(10, "footnote".to_string());
        class_labels.insert(11, "formula_number".to_string());
        class_labels.insert(12, "header".to_string());
        class_labels.insert(13, "header_image".to_string());
        class_labels.insert(14, "image".to_string());
        class_labels.insert(15, "inline_formula".to_string());
        class_labels.insert(16, "number".to_string());
        class_labels.insert(17, "paragraph_title".to_string());
        class_labels.insert(18, "reference".to_string());
        class_labels.insert(19, "reference_content".to_string());
        class_labels.insert(20, "seal".to_string());
        class_labels.insert(21, "table".to_string());
        class_labels.insert(22, "text".to_string());
        class_labels.insert(23, "vertical_text".to_string());
        class_labels.insert(24, "vision_footnote".to_string());

        Self {
            model_name: "pp-doclayoutv2".to_string(),
            num_classes: 25,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((800, 800)),
        }
    }

    /// Create configuration for PP-DocLayoutV3 model (25 classes).
    pub fn pp_doclayoutv3() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "abstract".to_string());
        class_labels.insert(1, "algorithm".to_string());
        class_labels.insert(2, "aside_text".to_string());
        class_labels.insert(3, "chart".to_string());
        class_labels.insert(4, "content".to_string());
        class_labels.insert(5, "display_formula".to_string());
        class_labels.insert(6, "doc_title".to_string());
        class_labels.insert(7, "figure_title".to_string());
        class_labels.insert(8, "footer".to_string());
        class_labels.insert(9, "footer_image".to_string());
        class_labels.insert(10, "footnote".to_string());
        class_labels.insert(11, "formula_number".to_string());
        class_labels.insert(12, "header".to_string());
        class_labels.insert(13, "header_image".to_string());
        class_labels.insert(14, "image".to_string());
        class_labels.insert(15, "inline_formula".to_string());
        class_labels.insert(16, "number".to_string());
        class_labels.insert(17, "paragraph_title".to_string());
        class_labels.insert(18, "reference".to_string());
        class_labels.insert(19, "reference_content".to_string());
        class_labels.insert(20, "seal".to_string());
        class_labels.insert(21, "table".to_string());
        class_labels.insert(22, "text".to_string());
        class_labels.insert(23, "vertical_text".to_string());
        class_labels.insert(24, "vision_footnote".to_string());

        Self {
            model_name: "pp-doclayoutv3".to_string(),
            num_classes: 25,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((800, 800)),
        }
    }
}

/// Enum for different layout detection model types.
#[derive(Debug)]
enum LayoutModel {
    PicoDet(PicoDetModel),
    RTDetr(RTDetrModel),
    PPDocLayout(PPDocLayoutModel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PpDocLayoutOrderMode {
    None,
    V2,
    V3,
}

/// Generic layout detection adapter.
///
/// This adapter uses one of the layout detection models (PicoDet, RT-DETR, or PP-DocLayout)
/// and adapts the model output to the LayoutDetectionTask output format.
#[derive(Debug)]
pub struct LayoutDetectionAdapter {
    model: LayoutModel,
    postprocessor: LayoutPostProcess,
    model_config: LayoutModelConfig,
    info: AdapterInfo,
    config: LayoutDetectionConfig,
}

impl LayoutDetectionAdapter {
    /// Creates a new layout detection adapter with PicoDet model.
    pub fn new_picodet(
        model: PicoDetModel,
        postprocessor: LayoutPostProcess,
        model_config: LayoutModelConfig,
        info: AdapterInfo,
        config: LayoutDetectionConfig,
    ) -> Self {
        Self {
            model: LayoutModel::PicoDet(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    /// Creates a new layout detection adapter with RT-DETR model.
    pub fn new_rtdetr(
        model: RTDetrModel,
        postprocessor: LayoutPostProcess,
        model_config: LayoutModelConfig,
        info: AdapterInfo,
        config: LayoutDetectionConfig,
    ) -> Self {
        Self {
            model: LayoutModel::RTDetr(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    /// Creates a new layout detection adapter with PP-DocLayout model.
    pub fn new_pp_doclayout(
        model: PPDocLayoutModel,
        postprocessor: LayoutPostProcess,
        model_config: LayoutModelConfig,
        info: AdapterInfo,
        config: LayoutDetectionConfig,
    ) -> Self {
        Self {
            model: LayoutModel::PPDocLayout(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    /// Postprocesses model predictions to layout elements.
    fn postprocess(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<ImageScaleInfo>,
        config: &LayoutDetectionConfig,
    ) -> LayoutDetectionOutput {
        if self.model_config.model_type == "pp-doclayout" {
            return self.postprocess_pp_doclayout(predictions, img_shapes, config);
        }

        let (boxes, class_ids, scores) = self.postprocessor.apply(predictions, img_shapes);

        let mut elements = Vec::with_capacity(boxes.len());

        // Convert to layout elements
        for img_idx in 0..boxes.len() {
            let mut img_boxes = boxes[img_idx].clone();
            let mut img_classes = class_ids[img_idx].clone();
            let mut img_scores = scores[img_idx].clone();

            // Apply unclip ratio if configured (PP-StructureV3 layout_unclip_ratio)
            if let Some(ref unclip_ratio) = config.layout_unclip_ratio {
                let (width_ratio, height_ratio, per_class_ratios) = match unclip_ratio {
                    UnclipRatio::Uniform(r) => (*r, *r, None),
                    UnclipRatio::Separate(w, h) => (*w, *h, None),
                    UnclipRatio::PerClass(ratios) => (1.0, 1.0, Some(ratios)),
                };
                img_boxes = unclip_boxes(
                    &img_boxes,
                    &img_classes,
                    width_ratio,
                    height_ratio,
                    per_class_ratios,
                );
            }

            // Apply NMS with merge modes if configured (PP-StructureV3 merge_bboxes_mode)
            if let Some(ref merge_modes) = config.class_merge_modes {
                (img_boxes, img_classes, img_scores) = apply_nms_with_merge(
                    img_boxes,
                    img_classes,
                    img_scores,
                    &self.model_config.class_labels,
                    merge_modes,
                    config.nms_threshold,
                    config.max_elements,
                );
            }

            let mut img_elements = Vec::new();

            for ((bbox, &class_id), &score) in img_boxes
                .iter()
                .zip(img_classes.iter())
                .zip(img_scores.iter())
            {
                // Map class ID to element type
                let element_type = self
                    .model_config
                    .class_labels
                    .get(&class_id)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());

                // Use per-class threshold if configured, otherwise fall back to default
                let threshold = config.get_class_threshold(&element_type);

                if score >= threshold {
                    let element = LayoutDetectionElement {
                        bbox: bbox.clone(),
                        element_type,
                        score,
                    };

                    img_elements.push(element);

                    if img_elements.len() >= config.max_elements {
                        break;
                    }
                }
            }

            elements.push(img_elements);
        }

        LayoutDetectionOutput {
            elements,
            is_reading_order_sorted: false, // Will be set by execute() based on model output
        }
    }

    fn postprocess_pp_doclayout(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<ImageScaleInfo>,
        config: &LayoutDetectionConfig,
    ) -> LayoutDetectionOutput {
        let feature_dim = predictions.shape().get(3).copied().unwrap_or(0);
        let order_mode = match feature_dim {
            8 => PpDocLayoutOrderMode::V2,
            7 => PpDocLayoutOrderMode::V3,
            _ => PpDocLayoutOrderMode::None,
        };

        let class_thresholds = config.class_thresholds.as_ref().map(|thresholds| {
            let mut by_class = HashMap::new();
            for (class_id, label) in &self.model_config.class_labels {
                if let Some(threshold) = thresholds.get(label) {
                    by_class.insert(*class_id, *threshold);
                }
            }
            by_class
        });

        let class_merge_modes = config.class_merge_modes.as_ref().map(|merge_modes| {
            let mut by_class = HashMap::new();
            for (class_id, label) in &self.model_config.class_labels {
                if let Some(mode) = merge_modes.get(label) {
                    by_class.insert(*class_id, *mode);
                }
            }
            by_class
        });

        let image_class_id = self
            .model_config
            .class_labels
            .iter()
            .find_map(|(id, label)| (label == "image").then_some(*id));
        let formula_class_id = self
            .model_config
            .class_labels
            .iter()
            .find_map(|(id, label)| (label == "formula").then_some(*id));

        let mut elements = Vec::with_capacity(predictions.shape()[0]);

        for (img_idx, img_shape) in img_shapes.iter().enumerate() {
            let pred = predictions.index_axis(Axis(0), img_idx);
            let num_boxes = pred.shape()[0];

            let orig_width = img_shape.src_w;
            let orig_height = img_shape.src_h;

            let mut boxes = Vec::new();
            let mut classes = Vec::new();
            let mut scores = Vec::new();
            let mut order_pairs: Vec<(f32, f32)> = Vec::new();

            for box_idx in 0..num_boxes {
                let class_id = pred[[box_idx, 0, 0]] as i32;
                let score = pred[[box_idx, 0, 1]];

                if class_id < 0 || (class_id as usize) >= self.model_config.num_classes {
                    continue;
                }

                let threshold = match class_thresholds.as_ref() {
                    Some(map) => map.get(&(class_id as usize)).copied().unwrap_or(0.5),
                    None => config.score_threshold.max(0.0),
                };

                if score < threshold {
                    continue;
                }

                let x1 = pred[[box_idx, 0, 2]];
                let y1 = pred[[box_idx, 0, 3]];
                let x2 = pred[[box_idx, 0, 4]];
                let y2 = pred[[box_idx, 0, 5]];

                let (sx1, sy1, sx2, sy2) =
                    Self::convert_bbox_coords(x1, y1, x2, y2, orig_width, orig_height);

                if !Self::is_valid_box(sx1, sy1, sx2, sy2) {
                    continue;
                }

                let bbox = crate::processors::BoundingBox::from_coords(sx1, sy1, sx2, sy2);
                boxes.push(bbox);
                classes.push(class_id as usize);
                scores.push(score);

                let order_pair = match order_mode {
                    PpDocLayoutOrderMode::V2 => (pred[[box_idx, 0, 6]], pred[[box_idx, 0, 7]]),
                    PpDocLayoutOrderMode::V3 => (pred[[box_idx, 0, 6]], 0.0),
                    PpDocLayoutOrderMode::None => (0.0, 0.0),
                };
                order_pairs.push(order_pair);
            }

            if !boxes.is_empty() {
                Self::round_boxes(&mut boxes);
            }

            if config.layout_nms && !boxes.is_empty() {
                let keep = Self::paddlex_layout_nms(&boxes, &classes, &scores);
                boxes = Self::select_by_indices(&boxes, &keep);
                classes = Self::select_by_indices(&classes, &keep);
                scores = Self::select_by_indices(&scores, &keep);
                order_pairs = Self::select_by_indices(&order_pairs, &keep);
            }

            if let Some(image_id) = image_class_id
                && boxes.len() > 1
            {
                let filtered = Self::filter_large_image_boxes(
                    &boxes,
                    &classes,
                    &scores,
                    &order_pairs,
                    orig_width,
                    orig_height,
                    image_id,
                );
                if let Some((new_boxes, new_classes, new_scores, new_orders)) = filtered {
                    boxes = new_boxes;
                    classes = new_classes;
                    scores = new_scores;
                    order_pairs = new_orders;
                }
            }

            if let Some(ref merge_modes) = class_merge_modes
                && !merge_modes.is_empty()
                && !boxes.is_empty()
            {
                let (new_boxes, new_classes, new_scores, new_orders) =
                    Self::apply_paddlex_merge_modes(
                        &boxes,
                        &classes,
                        &scores,
                        &order_pairs,
                        merge_modes,
                        formula_class_id,
                    );
                boxes = new_boxes;
                classes = new_classes;
                scores = new_scores;
                order_pairs = new_orders;
            }

            if order_mode != PpDocLayoutOrderMode::None && !boxes.is_empty() {
                let mut indices: Vec<usize> = (0..boxes.len()).collect();
                match order_mode {
                    PpDocLayoutOrderMode::V2 => {
                        indices.sort_by(|&i, &j| {
                            let (col_i, row_i) = order_pairs[i];
                            let (col_j, row_j) = order_pairs[j];
                            col_i
                                .total_cmp(&col_j)
                                .then_with(|| row_j.total_cmp(&row_i))
                        });
                    }
                    PpDocLayoutOrderMode::V3 => {
                        indices.sort_by(|&i, &j| order_pairs[i].0.total_cmp(&order_pairs[j].0));
                    }
                    PpDocLayoutOrderMode::None => {}
                }

                boxes = Self::select_by_indices(&boxes, &indices);
                classes = Self::select_by_indices(&classes, &indices);
                scores = Self::select_by_indices(&scores, &indices);
                order_pairs = Self::select_by_indices(&order_pairs, &indices);
            }

            if let Some(ref unclip_ratio) = config.layout_unclip_ratio {
                let (width_ratio, height_ratio, per_class_ratios) = match unclip_ratio {
                    UnclipRatio::Uniform(r) => (*r, *r, None),
                    UnclipRatio::Separate(w, h) => (*w, *h, None),
                    UnclipRatio::PerClass(ratios) => (1.0, 1.0, Some(ratios)),
                };
                boxes = unclip_boxes(
                    &boxes,
                    &classes,
                    width_ratio,
                    height_ratio,
                    per_class_ratios,
                );
            }

            let mut img_elements = Vec::new();
            for ((bbox, &class_id), &score) in boxes.iter().zip(classes.iter()).zip(scores.iter()) {
                let element_type = self
                    .model_config
                    .class_labels
                    .get(&class_id)
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());

                img_elements.push(LayoutDetectionElement {
                    bbox: bbox.clone(),
                    element_type,
                    score,
                });

                if img_elements.len() >= config.max_elements {
                    break;
                }
            }

            elements.push(img_elements);
        }

        LayoutDetectionOutput {
            elements,
            is_reading_order_sorted: false,
        }
    }

    fn convert_bbox_coords(
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        orig_width: f32,
        orig_height: f32,
    ) -> (f32, f32, f32, f32) {
        let normalized = x2 <= 1.05
            && y2 <= 1.05
            && x1 >= -0.05
            && y1 >= -0.05
            && orig_width > 0.0
            && orig_height > 0.0;

        if normalized {
            (
                x1.clamp(0.0, 1.0) * orig_width,
                y1.clamp(0.0, 1.0) * orig_height,
                x2.clamp(0.0, 1.0) * orig_width,
                y2.clamp(0.0, 1.0) * orig_height,
            )
        } else {
            (
                x1.clamp(0.0, orig_width),
                y1.clamp(0.0, orig_height),
                x2.clamp(0.0, orig_width),
                y2.clamp(0.0, orig_height),
            )
        }
    }

    fn is_valid_box(x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
        x2 > x1 && y2 > y1 && x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite()
    }

    fn round_boxes(boxes: &mut [crate::processors::BoundingBox]) {
        fn round_half_even(value: f32) -> f32 {
            let base = value.floor();
            let frac = value - base;
            if frac < 0.5 {
                base
            } else if frac > 0.5 {
                base + 1.0
            } else if (base as i64) % 2 == 0 {
                base
            } else {
                base + 1.0
            }
        }

        for bbox in boxes.iter_mut() {
            *bbox = crate::processors::BoundingBox::from_coords(
                round_half_even(bbox.x_min()),
                round_half_even(bbox.y_min()),
                round_half_even(bbox.x_max()),
                round_half_even(bbox.y_max()),
            );
        }
    }

    fn paddlex_layout_nms(
        boxes: &[crate::processors::BoundingBox],
        classes: &[usize],
        scores: &[f32],
    ) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..boxes.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut selected = Vec::new();
        while !indices.is_empty() {
            let current = indices[0];
            let current_class = classes[current];
            let current_box = &boxes[current];
            selected.push(current);

            let mut filtered = Vec::new();
            for &idx in indices.iter().skip(1) {
                let threshold = if classes[idx] == current_class {
                    0.6
                } else {
                    0.98
                };
                let iou = Self::paddlex_iou(current_box, &boxes[idx]);
                if iou < threshold {
                    filtered.push(idx);
                }
            }
            indices = filtered;
        }
        selected
    }

    fn paddlex_iou(
        box1: &crate::processors::BoundingBox,
        box2: &crate::processors::BoundingBox,
    ) -> f32 {
        let (x1, y1, x2, y2) = (box1.x_min(), box1.y_min(), box1.x_max(), box1.y_max());
        let (x1p, y1p, x2p, y2p) = (box2.x_min(), box2.y_min(), box2.x_max(), box2.y_max());

        let inter_w = (x2.min(x2p) - x1.max(x1p) + 1.0).max(0.0);
        let inter_h = (y2.min(y2p) - y1.max(y1p) + 1.0).max(0.0);
        let inter_area = inter_w * inter_h;

        let area1 = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);
        let area2 = (x2p - x1p + 1.0) * (y2p - y1p + 1.0);
        let union = area1 + area2 - inter_area;

        if union > 0.0 { inter_area / union } else { 0.0 }
    }

    fn filter_large_image_boxes(
        boxes: &[crate::processors::BoundingBox],
        classes: &[usize],
        scores: &[f32],
        order_pairs: &[(f32, f32)],
        orig_width: f32,
        orig_height: f32,
        image_class_id: usize,
    ) -> Option<LayoutBoxResult> {
        let area_thres = if orig_width > orig_height { 0.82 } else { 0.93 };
        let img_area = orig_width * orig_height;

        let mut keep_indices = Vec::new();
        for (idx, bbox) in boxes.iter().enumerate() {
            if classes[idx] != image_class_id {
                keep_indices.push(idx);
                continue;
            }

            let xmin = bbox.x_min().max(0.0);
            let ymin = bbox.y_min().max(0.0);
            let xmax = bbox.x_max().min(orig_width);
            let ymax = bbox.y_max().min(orig_height);
            let area = (xmax - xmin) * (ymax - ymin);
            if area <= area_thres * img_area {
                keep_indices.push(idx);
            }
        }

        if keep_indices.is_empty() {
            return None;
        }

        Some((
            Self::select_by_indices(boxes, &keep_indices),
            Self::select_by_indices(classes, &keep_indices),
            Self::select_by_indices(scores, &keep_indices),
            Self::select_by_indices(order_pairs, &keep_indices),
        ))
    }

    fn apply_paddlex_merge_modes(
        boxes: &[crate::processors::BoundingBox],
        classes: &[usize],
        scores: &[f32],
        order_pairs: &[(f32, f32)],
        merge_modes: &HashMap<usize, MergeBboxMode>,
        formula_class_id: Option<usize>,
    ) -> LayoutBoxResult {
        let mut keep_mask = vec![true; boxes.len()];

        for (class_id, mode) in merge_modes {
            if matches!(mode, MergeBboxMode::Union) {
                continue;
            }

            let (contains_other, contained_by_other) =
                Self::check_containment(boxes, classes, formula_class_id, *class_id, *mode);

            match mode {
                MergeBboxMode::Large => {
                    for (idx, flag) in contained_by_other.iter().enumerate() {
                        if *flag == 1 {
                            keep_mask[idx] = false;
                        }
                    }
                }
                MergeBboxMode::Small => {
                    for idx in 0..keep_mask.len() {
                        if !(contains_other[idx] == 0 || contained_by_other[idx] == 1) {
                            keep_mask[idx] = false;
                        }
                    }
                }
                MergeBboxMode::Union => {}
            }
        }

        (
            Self::select_by_mask(boxes, &keep_mask),
            Self::select_by_mask(classes, &keep_mask),
            Self::select_by_mask(scores, &keep_mask),
            Self::select_by_mask(order_pairs, &keep_mask),
        )
    }

    fn check_containment(
        boxes: &[crate::processors::BoundingBox],
        classes: &[usize],
        formula_class_id: Option<usize>,
        target_class_id: usize,
        mode: MergeBboxMode,
    ) -> (Vec<i32>, Vec<i32>) {
        let n = boxes.len();
        let mut contains_other = vec![0; n];
        let mut contained_by_other = vec![0; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if let Some(formula_id) = formula_class_id
                    && classes[i] == formula_id
                    && classes[j] != formula_id
                {
                    continue;
                }

                match mode {
                    MergeBboxMode::Large if classes[j] == target_class_id => {
                        if Self::is_contained(&boxes[i], &boxes[j]) {
                            contained_by_other[i] = 1;
                            contains_other[j] = 1;
                        }
                    }
                    MergeBboxMode::Small if classes[i] == target_class_id => {
                        if Self::is_contained(&boxes[i], &boxes[j]) {
                            contained_by_other[i] = 1;
                            contains_other[j] = 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        (contains_other, contained_by_other)
    }

    fn is_contained(
        inner: &crate::processors::BoundingBox,
        outer: &crate::processors::BoundingBox,
    ) -> bool {
        let (x1, y1, x2, y2) = (inner.x_min(), inner.y_min(), inner.x_max(), inner.y_max());
        let (x1p, y1p, x2p, y2p) = (outer.x_min(), outer.y_min(), outer.x_max(), outer.y_max());

        let box_area = (x2 - x1) * (y2 - y1);
        if box_area <= 0.0 {
            return false;
        }

        let xi1 = x1.max(x1p);
        let yi1 = y1.max(y1p);
        let xi2 = x2.min(x2p);
        let yi2 = y2.min(y2p);
        let inter_w = (xi2 - xi1).max(0.0);
        let inter_h = (yi2 - yi1).max(0.0);
        let inter_area = inter_w * inter_h;
        let iou = inter_area / box_area;
        iou >= 0.9
    }

    fn select_by_indices<T: Clone>(items: &[T], indices: &[usize]) -> Vec<T> {
        indices.iter().map(|&idx| items[idx].clone()).collect()
    }

    fn select_by_mask<T: Clone>(items: &[T], mask: &[bool]) -> Vec<T> {
        items
            .iter()
            .zip(mask.iter())
            .filter_map(|(item, keep)| keep.then_some(item.clone()))
            .collect()
    }
}

impl ModelAdapter for LayoutDetectionAdapter {
    type Task = LayoutDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        // Use provided config or fall back to stored config
        let effective_config = config.unwrap_or(&self.config);
        let batch_len = input.images.len();

        // Run model-specific forward pass
        let (predictions, img_shapes) = match &self.model {
            LayoutModel::PicoDet(model) => {
                let postprocess_config = PicoDetPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model
                    .forward(input.images.clone(), &postprocess_config)
                    .map_err(|e| {
                        OCRError::adapter_execution_error(
                            "LayoutDetectionAdapter",
                            format!("PicoDet forward (batch_size={})", batch_len),
                            e,
                        )
                    })?;
                (output.predictions, img_shapes)
            }
            LayoutModel::RTDetr(model) => {
                let postprocess_config = RTDetrPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model
                    .forward(input.images.clone(), &postprocess_config)
                    .map_err(|e| {
                        OCRError::adapter_execution_error(
                            "LayoutDetectionAdapter",
                            format!("RTDetr forward (batch_size={})", batch_len),
                            e,
                        )
                    })?;
                (output.predictions, img_shapes)
            }
            LayoutModel::PPDocLayout(model) => {
                let postprocess_config = PPDocLayoutPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model
                    .forward(input.images, &postprocess_config)
                    .map_err(|e| {
                        tracing::error!("PPDocLayout forward error: {:?}", e);
                        OCRError::adapter_execution_error(
                            "LayoutDetectionAdapter",
                            format!("PPDocLayout forward (batch_size={})", batch_len),
                            e,
                        )
                    })?;
                (output.predictions, img_shapes)
            }
        };

        // Check if predictions include reading order info (8-dim format from PP-DocLayoutV2)
        // Shape is [batch, num_boxes, 1, N] where N=8 indicates reading order is included
        let has_reading_order = predictions.shape().get(3).copied().unwrap_or(0) == 8;

        // Postprocess predictions
        let mut output = self.postprocess(&predictions, img_shapes, effective_config);
        output.is_reading_order_sorted = has_reading_order;

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        4 // Default batch size for layout detection
    }
}

/// Builder for layout detection adapters.
#[derive(Debug, Default)]
pub struct LayoutDetectionAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<LayoutDetectionConfig>,
    model_config: Option<LayoutModelConfig>,
}

impl LayoutDetectionAdapterBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model configuration.
    pub fn model_config(mut self, config: LayoutModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.config.task_config.score_threshold = threshold;
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.config.task_config.max_elements = max;
        self
    }

    /// Builds the adapter with the specified model configuration.
    fn build_with_config(
        self,
        model_path: &Path,
        model_config: LayoutModelConfig,
    ) -> Result<LayoutDetectionAdapter, OCRError> {
        let (task_config, ort_config) =
            self.config
                .into_validated_parts()
                .map_err(|err| OCRError::ConfigError {
                    message: err.to_string(),
                })?;

        // Create ONNX inference engine with proper input name based on model type
        let inference = if ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let input_name = match model_config.model_type.as_str() {
                "pp-doclayout" => Some("image"),
                _ => None,
            };
            let common_config = ModelInferenceConfig {
                ort_session: ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common_config, model_path, input_name)?
        } else {
            match model_config.model_type.as_str() {
                "pp-doclayout" => {
                    // PP-DocLayout models use "image" as the input name
                    OrtInfer::new(model_path, Some("image"))?
                }
                _ => {
                    // Other models use default or auto-detect
                    OrtInfer::new(model_path, None)?
                }
            }
        };

        // Create postprocessor
        let postprocessor = LayoutPostProcess::new(
            model_config.num_classes,
            task_config.score_threshold,
            task_config.nms_threshold, // Use config value instead of hardcoded 0.5
            task_config.max_elements,
            model_config.model_type.clone(),
        );

        // Create adapter info
        let info = AdapterInfo::new(
            format!("LayoutDetection_{}", model_config.model_name),
            TaskType::LayoutDetection,
            format!(
                "Layout detection adapter for {} with {} classes",
                model_config.model_name, model_config.num_classes
            ),
        );

        // Build model based on model type
        let adapter = match model_config.model_type.as_str() {
            "picodet" => {
                let mut builder = PicoDetModelBuilder::new();
                if let Some((height, width)) = model_config.input_size {
                    builder = builder.image_shape(height, width);
                }
                let model = builder.build(inference)?;
                LayoutDetectionAdapter::new_picodet(
                    model,
                    postprocessor,
                    model_config,
                    info,
                    task_config,
                )
            }
            "rtdetr" => {
                let model = RTDetrModelBuilder::new().build(inference)?;
                LayoutDetectionAdapter::new_rtdetr(
                    model,
                    postprocessor,
                    model_config,
                    info,
                    task_config,
                )
            }
            "pp-doclayout" => {
                let model = match model_config.input_size {
                    Some((height, width)) => PPDocLayoutModelBuilder::new()
                        .image_shape(height, width)
                        .build(inference)?,
                    None => PPDocLayoutModelBuilder::new().build(inference)?,
                };
                LayoutDetectionAdapter::new_pp_doclayout(
                    model,
                    postprocessor,
                    model_config,
                    info,
                    task_config,
                )
            }
            _ => {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Unknown model type: '{}'. Supported types: picodet, rtdetr, pp-doclayout",
                        model_config.model_type
                    ),
                });
            }
        };

        Ok(adapter)
    }
}

impl AdapterBuilder for LayoutDetectionAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = LayoutDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let model_config = self
            .model_config
            .clone()
            .ok_or_else(|| OCRError::InvalidInput {
                message: "Model configuration is required".to_string(),
            })?;

        self.build_with_config(model_path, model_config)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "LayoutDetection"
    }
}

impl crate::core::traits::OrtConfigurable for LayoutDetectionAdapterBuilder {
    fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.config = self.config.with_ort_config(config);
        self
    }
}

// Type aliases and builders for specific models

/// PicoDet layout detection adapter.
pub type PicoDetLayoutAdapter = LayoutDetectionAdapter;

/// Builder for PicoDet layout detection adapter.
pub struct PicoDetLayoutAdapterBuilder {
    inner: LayoutDetectionAdapterBuilder,
}

impl Default for PicoDetLayoutAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_layout_1x()),
        }
    }
}

impl PicoDetLayoutAdapterBuilder {
    /// Creates a new builder with default PicoDet layout 1x configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with PicoDet-S layout 3 class configuration.
    pub fn new_3cls() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_s_layout_3cls()),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.inner = self.inner.max_elements(max);
        self
    }
}

impl crate::core::traits::OrtConfigurable for PicoDetLayoutAdapterBuilder {
    fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.inner = self.inner.with_ort_config(config);
        self
    }
}

impl AdapterBuilder for PicoDetLayoutAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = PicoDetLayoutAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "PicoDetLayout"
    }
}

/// RT-DETR layout detection adapter.
pub type RTDetrLayoutAdapter = LayoutDetectionAdapter;

/// Builder for RT-DETR layout detection adapter.
pub struct RTDetrLayoutAdapterBuilder {
    inner: LayoutDetectionAdapterBuilder,
}

impl Default for RTDetrLayoutAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::rtdetr_h_layout_3cls()),
        }
    }
}

impl RTDetrLayoutAdapterBuilder {
    /// Creates a new builder with default RT-DETR-H layout 3 class configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with RT-DETR-H layout 17 class configuration.
    pub fn new_17cls() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::rtdetr_h_layout_17cls()),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.inner = self.inner.max_elements(max);
        self
    }
}

impl crate::core::traits::OrtConfigurable for RTDetrLayoutAdapterBuilder {
    fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.inner = self.inner.with_ort_config(config);
        self
    }
}

impl AdapterBuilder for RTDetrLayoutAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = RTDetrLayoutAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "RTDetrLayout"
    }
}

/// PP-DocLayout detection adapter.
pub type PPDocLayoutAdapter = LayoutDetectionAdapter;

/// Builder for PP-DocLayout detection adapter.
pub struct PPDocLayoutAdapterBuilder {
    inner: LayoutDetectionAdapterBuilder,
}

impl Default for PPDocLayoutAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::pp_doclayout_l()),
        }
    }
}

impl PPDocLayoutAdapterBuilder {
    /// Creates a new builder with the specified PP-DocLayout model variant.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Model variant name. Supported values:
    ///   - `"pp-doclayout-s"` or `"pp_doclayout_s"` - Small model (480x480)
    ///   - `"pp-doclayout-m"` or `"pp_doclayout_m"` - Medium model (640x640)
    ///   - `"pp-doclayout-l"` or `"pp_doclayout_l"` - Large model (640x640, default)
    ///   - `"pp-doclayout_plus-l"` or `"pp_doclayout_plus_l"` - Plus-Large model (800x800)
    ///   - `"pp-doclayoutv2"` or `"pp_doclayoutv2"` - PP-DocLayoutV2 model (800x800)
    ///   - `"pp-doclayoutv3"` or `"pp_doclayoutv3"` - PP-DocLayoutV3 model (800x800)
    ///   - `"pp-docblocklayout"` or `"pp_docblocklayout"` - Block layout model (640x640)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use oar_ocr_core::core::traits::adapter::AdapterBuilder;
    /// use oar_ocr_core::domain::adapters::PPDocLayoutAdapterBuilder;
    ///
    /// # fn main() -> Result<(), oar_ocr_core::core::OCRError> {
    /// let _adapter = PPDocLayoutAdapterBuilder::new("pp-doclayout-s")
    ///     .build(Path::new("model.onnx"))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(model_name: impl AsRef<str>) -> Self {
        let name = model_name.as_ref();
        let config = match name {
            "PP-DocLayout-S" => LayoutModelConfig::pp_doclayout_s(),
            "PP-DocLayout-M" => LayoutModelConfig::pp_doclayout_m(),
            "PP-DocLayout-L" => LayoutModelConfig::pp_doclayout_l(),
            "PP-DocLayout_plus-L" => LayoutModelConfig::pp_doclayout_plus_l(),
            "PP-DocLayoutV2" | "PP-DocLayout-V2" => LayoutModelConfig::pp_doclayoutv2(),
            "PP-DocLayoutV3" | "PP-DocLayout-V3" => LayoutModelConfig::pp_doclayoutv3(),
            "PP-DocBlockLayout" => LayoutModelConfig::pp_docblocklayout(),
            _ => {
                // Default to pp-doclayout-l for unknown variants
                LayoutModelConfig::pp_doclayout_l()
            }
        };

        Self {
            inner: LayoutDetectionAdapterBuilder::new().model_config(config),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.inner = self.inner.max_elements(max);
        self
    }
}

impl crate::core::traits::OrtConfigurable for PPDocLayoutAdapterBuilder {
    fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.inner = self.inner.with_ort_config(config);
        self
    }
}

impl AdapterBuilder for PPDocLayoutAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = PPDocLayoutAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "PPDocLayout"
    }
}
