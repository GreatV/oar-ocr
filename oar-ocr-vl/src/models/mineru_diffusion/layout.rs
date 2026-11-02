//! The Diffusion runner's layout, crop and formatting rules.
//! Adapted from opendatalab/MinerU-Diffusion scripts/run_end2end.py (MIT).

use crate::{DocumentBlock, Error};
use image::RgbImage;

pub(super) fn task_budget(kind: &str, cap: usize) -> usize {
    cap.min(if matches!(kind, "layout" | "table") {
        2048
    } else {
        1024
    })
}

pub(super) fn parse_layout(raw: &str) -> Vec<DocumentBlock> {
    raw.lines()
        .flat_map(|line| {
            let line = line.trim();
            if line.starts_with("<|box_start|>") {
                crate::pipeline::mineru_layout::parse_layout_records(line)
                    .into_iter()
                    .take(1)
                    .map(|(b, _)| b)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            }
        })
        .collect()
}

pub(super) fn resize_crop(
    image: &RgbImage,
    min_edge: u32,
    max_ratio: f32,
) -> Result<RgbImage, Error> {
    crate::runtime::image::resize_padded(
        image,
        min_edge,
        max_ratio,
        crate::runtime::image::ResizeRounding::Nearest,
    )
}

pub(super) fn crop_block(
    image: &RgbImage,
    block: &DocumentBlock,
    min_edge: u32,
    max_ratio: f32,
) -> Result<RgbImage, Error> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(Error::invalid_input("MinerU-Diffusion page is empty"));
    }
    // Layout coordinates were integer thousandths before PageDocument's f32
    // normalization. Restore that grid before floor/ceil at pixel boundaries.
    let bbox = block
        .bbox
        .map(|value| (value as f64 * 1000.).round_ties_even() / 1000.);
    let crop = crate::pipeline::crop::crop_with_options(
        image,
        [
            bbox[0] * width as f64,
            bbox[1] * height as f64,
            bbox[2] * width as f64,
            bbox[3] * height as f64,
        ],
        &crate::pipeline::crop::CropOptions {
            rounding: crate::pipeline::crop::CropRounding::Outward,
            minimum_one_pixel: true,
            angle: block.angle,
            ..Default::default()
        },
    )?;
    resize_crop(&crop, min_edge, max_ratio)
}

pub(super) fn format_content(kind: &str, text: &str) -> String {
    let text = text.trim();
    match kind {
        "table" => crate::render::table::convert_otsl_to_html_with_options(
            text,
            &super::adapter::table_options(),
        ),
        "equation" => crate::render::text::ensure_math_wrappers(text, r"\[", r"\]"),
        _ => text.into(),
    }
}
