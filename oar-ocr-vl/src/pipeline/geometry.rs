//! Geometry and crop preparation used by parsing pipelines.

use crate::document::geometry::BoundingBox;
use image::RgbImage;
mod calipers;
pub(crate) mod layout_filter;
pub(crate) mod polygon;
mod raster;

pub(crate) fn cover_ratio(container: [f32; 4], bbox: [f32; 4]) -> f32 {
    let area = (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0);
    if area == 0.0 {
        return 0.0;
    }
    (container[2].min(bbox[2]) - container[0].max(bbox[0])).max(0.0)
        * (container[3].min(bbox[3]) - container[1].max(bbox[1])).max(0.0)
        / area
}

pub(crate) fn bbox_iou(a: [f32; 4], b: [f32; 4], pixel_offset: f64) -> f64 {
    let a = a.map(f64::from);
    let b = b.map(f64::from);
    let intersection = (a[2].min(b[2]) - a[0].max(b[0]) + pixel_offset).max(0.)
        * (a[3].min(b[3]) - a[1].max(b[1]) + pixel_offset).max(0.);
    let area = |bbox: [f64; 4]| {
        (bbox[2] - bbox[0] + pixel_offset).max(0.) * (bbox[3] - bbox[1] + pixel_offset).max(0.)
    };
    let union = area(a) + area(b) - intersection;
    if union > 0. { intersection / union } else { 0. }
}

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
    let mut prediction = crate::pipeline::layout::LayoutPrediction::from(
        crate::pipeline::layout::LayoutDetections::new(
            boxes
                .into_iter()
                .map(|b| crate::pipeline::layout::LayoutDetectionElement {
                    bbox: b.bbox,
                    element_type: b.label,
                    score: b.score,
                })
                .collect(),
        ),
    );
    layout_filter::filter_document_layout(
        &mut prediction,
        &crate::pipeline::layout::LayoutPostprocessOptions::default()
            .with_overlap_threshold(Some(f64::from(overlap_threshold))),
    );
    prediction
        .detections
        .elements
        .into_iter()
        .map(|b| DetectedBox {
            bbox: b.bbox,
            label: b.element_type,
            score: b.score,
        })
        .collect()
}

pub fn crop_margin(image: &RgbImage) -> RgbImage {
    // OpenCV BT.601 RGB-to-gray coefficients, rather than image's Rec.709.
    let gray: Vec<u8> = image
        .pixels()
        .map(|p| {
            ((9798u32 * p[0] as u32 + 19235u32 * p[1] as u32 + 3735u32 * p[2] as u32 + 16384) >> 15)
                as u8
        })
        .collect();
    let (min_value, max_value) = gray.iter().fold((255u8, 0u8), |(min, max), &pixel| {
        (min.min(pixel), max.max(pixel))
    });
    if max_value == min_value {
        return image.clone();
    }
    let (mut x_min, mut y_min) = (image.width(), image.height());
    let (mut x_max, mut y_max) = (0, 0);
    let mut found = false;
    for (index, pixel) in gray.into_iter().enumerate() {
        let (x, y) = (index as u32 % image.width(), index as u32 / image.width());
        let normalized = ((pixel as f64 - min_value as f64) / (max_value as f64 - min_value as f64)
            * 255.0) as u8;
        if normalized <= 200 {
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
    if width <= 2 || height <= 2 {
        return image.clone();
    }
    image::imageops::crop_imm(image, x_min, y_min, width, height).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_margins_use_bt601_inclusive_threshold_and_two_pixel_guard() {
        let mut image = RgbImage::from_pixel(7, 5, image::Rgb([255; 3]));
        for y in 1..4 {
            for x in 3..6 {
                image.put_pixel(x, y, image::Rgb([0; 3]));
            }
        }
        image.put_pixel(2, 1, image::Rgb([200; 3]));
        image.put_pixel(1, 1, image::Rgb([120, 255, 0]));
        image.put_pixel(0, 1, image::Rgb([201; 3]));
        assert_eq!(crop_margin(&image).dimensions(), (5, 3));
        let mut single = RgbImage::from_pixel(5, 5, image::Rgb([255; 3]));
        single.put_pixel(2, 2, image::Rgb([0; 3]));
        assert_eq!(crop_margin(&single).dimensions(), (5, 5));
    }
}
