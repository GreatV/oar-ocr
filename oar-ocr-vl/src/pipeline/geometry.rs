//! Geometry and crop preparation used by parsing pipelines.

use crate::document::geometry::BoundingBox;
use image::{GrayImage, RgbImage};
use std::collections::HashSet;

#[inline]
pub fn calculate_bbox_area(bbox: &BoundingBox) -> f32 {
    (bbox.x_max() - bbox.x_min()).abs() * (bbox.y_max() - bbox.y_min()).abs()
}

pub fn calculate_overlap_ratio(bbox1: &BoundingBox, bbox2: &BoundingBox, mode: &str) -> f32 {
    let x_min_inter = bbox1.x_min().max(bbox2.x_min());
    let y_min_inter = bbox1.y_min().max(bbox2.y_min());
    let x_max_inter = bbox1.x_max().min(bbox2.x_max());
    let y_max_inter = bbox1.y_max().min(bbox2.y_max());
    let inter_area = (x_max_inter - x_min_inter).max(0.0) * (y_max_inter - y_min_inter).max(0.0);
    let bbox1_area = calculate_bbox_area(bbox1);
    let bbox2_area = calculate_bbox_area(bbox2);
    let ref_area = match mode {
        "union" => bbox1_area + bbox2_area - inter_area,
        "small" => bbox1_area.min(bbox2_area),
        "large" => bbox1_area.max(bbox2_area),
        _ => bbox1_area + bbox2_area - inter_area,
    };
    if ref_area == 0.0 {
        0.0
    } else {
        inter_area / ref_area
    }
}

pub fn calculate_projection_overlap_ratio(
    bbox1: &BoundingBox,
    bbox2: &BoundingBox,
    direction: &str,
    mode: &str,
) -> f32 {
    let (start1, end1, start2, end2) = if direction == "horizontal" {
        (bbox1.x_min(), bbox1.x_max(), bbox2.x_min(), bbox2.x_max())
    } else {
        (bbox1.y_min(), bbox1.y_max(), bbox2.y_min(), bbox2.y_max())
    };
    let overlap = end1.min(end2) - start1.max(start2);
    if overlap <= 0.0 {
        return 0.0;
    }
    let reference = match mode {
        "union" => end1.max(end2) - start1.min(start2),
        "small" => (end1 - start1).min(end2 - start2),
        "large" => (end1 - start1).max(end2 - start2),
        _ => end1.max(end2) - start1.min(start2),
    };
    if reference > 0.0 {
        overlap / reference
    } else {
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct DetectedBox {
    pub bbox: BoundingBox,
    pub label: String,
    pub score: f32,
}

pub fn filter_overlap_boxes(boxes: Vec<DetectedBox>, overlap_threshold: f32) -> Vec<DetectedBox> {
    let boxes: Vec<_> = boxes
        .into_iter()
        .filter(|candidate| candidate.label != "reference")
        .collect();
    let mut dropped = HashSet::new();
    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            if dropped.contains(&i) || dropped.contains(&j) {
                continue;
            }
            if calculate_overlap_ratio(&boxes[i].bbox, &boxes[j].bbox, "small") <= overlap_threshold
            {
                continue;
            }
            if (boxes[i].label == "image" || boxes[j].label == "image")
                && boxes[i].label != boxes[j].label
            {
                continue;
            }
            if calculate_bbox_area(&boxes[i].bbox) >= calculate_bbox_area(&boxes[j].bbox) {
                dropped.insert(j);
            } else {
                dropped.insert(i);
            }
        }
    }
    boxes
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !dropped.contains(index))
        .map(|(_, candidate)| candidate)
        .collect()
}

pub fn crop_margin(image: &RgbImage) -> RgbImage {
    let gray: GrayImage = image::imageops::grayscale(image);
    let (min_value, max_value) = gray.pixels().fold((255u8, 0u8), |(min, max), pixel| {
        (min.min(pixel.0[0]), max.max(pixel.0[0]))
    });
    if max_value == min_value {
        return image.clone();
    }
    let (mut x_min, mut y_min) = (image.width(), image.height());
    let (mut x_max, mut y_max) = (0, 0);
    let mut found = false;
    for (x, y, pixel) in gray.enumerate_pixels() {
        let normalized = ((pixel.0[0] as f32 - min_value as f32)
            / (max_value as f32 - min_value as f32)
            * 255.0) as u8;
        if normalized < 200 {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            found = true;
        }
    }
    if !found {
        return image.clone();
    }
    let width = (x_max - x_min + 1).min(image.width() - x_min);
    let height = (y_max - y_min + 1).min(image.height() - y_min);
    if width == 0 || height == 0 {
        return image.clone();
    }
    image::imageops::crop_imm(image, x_min, y_min, width, height).to_image()
}
