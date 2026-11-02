//! Shared detector and document layout selection.
//!
//! PaddleX layout_analysis/processors.py and object_detection/processors.py
//! (Copyright PaddlePaddle Authors, Apache-2.0) define the detector stages.

use super::{bbox_iou, polygon::polygon_overlap};
use crate::document::geometry::BoundingBox;
use crate::pipeline::layout::{ContainmentMode, LayoutPostprocessOptions, LayoutPrediction};

#[derive(Debug, Clone)]
pub(crate) struct LayoutCandidate {
    pub bbox: [f32; 4],
    pub label: String,
    pub class_id: usize,
    pub score: f32,
    pub query: usize,
    pub rank: usize,
}

fn area(b: [f32; 4]) -> f64 {
    (f64::from(b[2]) - f64::from(b[0])).max(0.) * (f64::from(b[3]) - f64::from(b[1])).max(0.)
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> f64 {
    (f64::from(a[2].min(b[2])) - f64::from(a[0].max(b[0]))).max(0.)
        * (f64::from(a[3].min(b[3])) - f64::from(a[1].max(b[1]))).max(0.)
}

pub(crate) fn filter_candidates(
    mut boxes: Vec<LayoutCandidate>,
    width: u32,
    height: u32,
    options: &LayoutPostprocessOptions,
) -> Vec<LayoutCandidate> {
    if options.round_boxes {
        for candidate in &mut boxes {
            candidate.bbox = candidate.bbox.map(f32::round_ties_even);
        }
    }
    if options.nms_same_class.is_some() || options.nms_cross_class.is_some() {
        let mut indices: Vec<_> = (0..boxes.len()).collect();
        indices.sort_by(|&a, &b| boxes[b].score.total_cmp(&boxes[a].score).then(b.cmp(&a)));
        let mut kept = Vec::new();
        for i in indices {
            let suppressed = kept.iter().any(|&j: &usize| {
                let threshold = if boxes[i].class_id == boxes[j].class_id {
                    options.nms_same_class
                } else {
                    options.nms_cross_class
                };
                threshold.is_some_and(|t| bbox_iou(boxes[i].bbox, boxes[j].bbox, 1.) >= t)
            });
            if !suppressed {
                kept.push(i);
            }
        }
        boxes = kept.into_iter().map(|i| boxes[i].clone()).collect();
    }
    if boxes.len() > 1 {
        let limit = if width > height {
            options.landscape_image_ratio
        } else {
            options.portrait_image_ratio
        };
        if let Some(limit) = limit {
            let page_area = f64::from(width) * f64::from(height);
            let retained: Vec<_> = boxes
                .iter()
                .filter(|candidate| {
                    let b = candidate.bbox;
                    candidate.label != "image"
                        || area([
                            b[0].max(0.),
                            b[1].max(0.),
                            b[2].min(width as f32),
                            b[3].min(height as f32),
                        ]) <= limit * page_area
                })
                .cloned()
                .collect();
            if !retained.is_empty() {
                boxes = retained;
            }
        }
    }
    let mut keep = vec![true; boxes.len()];
    // Each class policy observes the same candidate set.
    for rule in &options.containment_rules {
        if rule.mode == ContainmentMode::Union {
            continue;
        }
        let mut contains = vec![false; boxes.len()];
        let mut contained = vec![false; boxes.len()];
        for (i, inner) in boxes.iter().enumerate() {
            if options.preserved_labels.contains(&inner.label) {
                continue;
            }
            let inner_area = area(inner.bbox);
            if inner_area == 0. {
                continue;
            }
            for (j, outer) in boxes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let eligible = match rule.mode {
                    ContainmentMode::Large => outer.class_id == rule.class_id,
                    ContainmentMode::Small => inner.class_id == rule.class_id,
                    ContainmentMode::Union => false,
                };
                if eligible
                    && intersection(inner.bbox, outer.bbox) / inner_area
                        >= options.containment_ratio
                {
                    contained[i] = true;
                    contains[j] = true;
                }
            }
        }
        for i in 0..boxes.len() {
            keep[i] &= match rule.mode {
                ContainmentMode::Large => !contained[i],
                ContainmentMode::Small => !contains[i] || contained[i],
                ContainmentMode::Union => true,
            };
        }
    }
    boxes = boxes
        .into_iter()
        .zip(keep)
        .filter_map(|(b, keep)| keep.then_some(b))
        .collect();
    boxes.sort_by_key(|b| b.rank);
    boxes
}

pub(crate) fn filter_document_layout(
    prediction: &mut LayoutPrediction,
    options: &LayoutPostprocessOptions,
) {
    let boxes = &prediction.detections.elements;
    let mut dropped: Vec<_> = boxes
        .iter()
        .map(|b| options.drop_labels.contains(&b.element_type))
        .collect();
    for i in 0..boxes.len() {
        let a = &boxes[i];
        dropped[i] |= f64::from(a.bbox.x_max() - a.bbox.x_min()) < options.min_size
            || f64::from(a.bbox.y_max() - a.bbox.y_min()) < options.min_size;
        for j in i + 1..boxes.len() {
            if dropped[i] || dropped[j] {
                continue;
            }
            let b = &boxes[j];
            let bounds = |b: &BoundingBox| [b.x_min(), b.y_min(), b.x_max(), b.y_max()];
            let (a_box, b_box) = (bounds(&a.bbox), bounds(&b.bbox));
            let min_area = area(a_box).min(area(b_box));
            let ratio = if options.single_precision_overlap {
                f64::from(super::calculate_overlap_ratio(&a.bbox, &b.bbox, "small"))
            } else if min_area > 0. {
                intersection(a_box, b_box) / min_area
            } else {
                0.
            };
            if options.inline_formula_overlap.is_some_and(|t| ratio > t) {
                if a.element_type == "inline_formula" {
                    dropped[i] = true;
                    continue;
                }
                if b.element_type == "inline_formula" {
                    dropped[j] = true;
                    continue;
                }
            }
            let threshold = options.overlap_threshold.map(|t| {
                if options.single_precision_overlap {
                    f64::from(t as f32)
                } else {
                    t
                }
            });
            if !threshold.is_some_and(|t| ratio > t) {
                continue;
            }
            let visual_a = options.visual_labels.contains(&a.element_type);
            let visual_b = options.visual_labels.contains(&b.element_type);
            let table_pair = a.element_type == "table" || b.element_type == "table";
            if a.element_type != b.element_type
                && (visual_a || visual_b)
                && (!options.table_visual_exemption || !table_pair || (visual_a && visual_b))
            {
                continue;
            }
            if let Some(threshold) = options.polygon_overlap_threshold
                && let Some(Some(polygon_a)) = prediction.polygons.get(i)
            {
                let polygon_b = prediction
                    .polygons
                    .get(j)
                    .and_then(Option::as_ref)
                    .unwrap_or(&b.bbox);
                let overlap = polygon_overlap(polygon_a, polygon_b, false);
                if if options.polygon_overlap_strict {
                    overlap <= threshold
                } else {
                    overlap < threshold
                } {
                    continue;
                }
            }
            let area_order = if options.single_precision_overlap {
                super::calculate_bbox_area(&a.bbox) >= super::calculate_bbox_area(&b.bbox)
            } else {
                area(a_box) >= area(b_box)
            };
            if area_order {
                dropped[j] = true;
            } else {
                dropped[i] = true;
            }
        }
    }
    let elements = std::mem::take(&mut prediction.detections.elements);
    prediction.detections.elements = elements
        .into_iter()
        .enumerate()
        .filter_map(|(i, b)| (!dropped[i]).then_some(b))
        .collect();
    let polygons = std::mem::take(&mut prediction.polygons);
    prediction.polygons = polygons
        .into_iter()
        .enumerate()
        .filter_map(|(i, p)| (!dropped.get(i).copied().unwrap_or(true)).then_some(p))
        .collect();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::layout::{ContainmentRule, LayoutDetectionElement, LayoutDetections};

    fn candidate(query: usize, class_id: usize, bbox: [f32; 4], score: f32) -> LayoutCandidate {
        LayoutCandidate {
            query,
            rank: query,
            class_id,
            label: if class_id == 14 { "image" } else { "text" }.into(),
            bbox,
            score,
        }
    }

    #[test]
    fn detector_and_document_filters_keep_reference_stage_order() {
        let boundary = vec![
            super::super::DetectedBox {
                bbox: BoundingBox::from_coords(0., 0., 10., 10.),
                label: "text".into(),
                score: 0.9,
            },
            super::super::DetectedBox {
                bbox: BoundingBox::from_coords(3., 0., 13., 10.),
                label: "text".into(),
                score: 0.8,
            },
        ];
        assert_eq!(super::super::filter_overlap_boxes(boundary, 0.7).len(), 2);
        let options = LayoutPostprocessOptions::paddlex();
        let boxes = vec![
            candidate(0, 22, [0., 0., 100., 100.], 0.8),
            candidate(1, 22, [0., 0., 100., 100.], 0.9),
            candidate(2, 21, [0., 0., 100., 100.], 0.7),
            candidate(3, 22, [150., 0., 180., 30.], 0.6),
        ];
        let retained = filter_candidates(boxes, 200, 200, &options);
        assert_eq!(
            retained.iter().map(|b| b.query).collect::<Vec<_>>(),
            vec![1, 3]
        );
        let options = LayoutPostprocessOptions::default()
            .with_containment_ratio(0.8)
            .with_containment_rules(vec![
                ContainmentRule::default()
                    .with_class_id(21)
                    .with_mode(ContainmentMode::Large),
            ]);
        let boxes = vec![
            candidate(0, 22, [0., 0., 10., 10.], 0.9),
            candidate(1, 21, [2., 0., 12., 10.], 0.8),
        ];
        assert_eq!(filter_candidates(boxes, 100, 100, &options)[0].query, 1);
        // An all-image page keeps its candidates when the size filter removes all.
        let boxes = vec![
            candidate(0, 14, [0., 0., 100., 100.], 0.9),
            candidate(1, 14, [0., 0., 100., 100.], 0.8),
        ];
        let options = LayoutPostprocessOptions::default().with_portrait_image_ratio(Some(0.93));
        assert_eq!(filter_candidates(boxes, 100, 100, &options).len(), 2);

        let mut prediction = LayoutPrediction::from(LayoutDetections::new(vec![
            LayoutDetectionElement {
                bbox: BoundingBox::from_coords(0., 0., 30., 30.),
                element_type: "text".into(),
                score: 0.9,
            },
            LayoutDetectionElement {
                bbox: BoundingBox::from_coords(5., 5., 20., 20.),
                element_type: "inline_formula".into(),
                score: 0.9,
            },
            LayoutDetectionElement {
                bbox: BoundingBox::from_coords(40., 0., 45., 20.),
                element_type: "text".into(),
                score: 0.9,
            },
        ]));
        prediction.polygons = vec![None, Some(BoundingBox::from_coords(5., 5., 20., 20.)), None];
        filter_document_layout(&mut prediction, &LayoutPostprocessOptions::paddlex());
        assert_eq!(prediction.detections.elements.len(), 1);
        assert_eq!(prediction.polygons.len(), 1);
    }
}
