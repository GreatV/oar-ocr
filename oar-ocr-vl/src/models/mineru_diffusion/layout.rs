//! The Diffusion runner's layout, crop and formatting rules.
//! Adapted from opendatalab/MinerU-Diffusion scripts/run_end2end.py (MIT).

use crate::{DocumentBlock, Error};
use image::{RgbImage, imageops};

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
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || !max_ratio.is_finite() || max_ratio <= 0. {
        return Err(Error::invalid_input(
            "MinerU-Diffusion crop and aspect bound must be nonempty and positive",
        ));
    }
    let (mut w, mut h) = (width, height);
    if width.max(height) as f64 / width.min(height) as f64 > max_ratio as f64 {
        if w > h {
            h = (w as f64 / max_ratio as f64).ceil() as u32;
        } else {
            w = (h as f64 / max_ratio as f64).ceil() as u32;
        }
    }
    let mut crop = RgbImage::from_pixel(w, h, image::Rgb([255; 3]));
    imageops::overlay(
        &mut crop,
        image,
        ((w - width) / 2) as i64,
        ((h - height) / 2) as i64,
    );
    if w.min(h) < min_edge {
        let scale = min_edge as f64 / w.min(h) as f64;
        crop = crate::runtime::image::pil_bicubic_resize(
            &crop,
            (w as f64 * scale).round_ties_even().max(1.) as u32,
            (h as f64 * scale).round_ties_even().max(1.) as u32,
        )?;
    }
    Ok(crop)
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
    let x1 = (bbox[0] * width as f64)
        .floor()
        .clamp(0., (width - 1) as f64) as u32;
    let y1 = (bbox[1] * height as f64)
        .floor()
        .clamp(0., (height - 1) as f64) as u32;
    let x2 = (bbox[2] * width as f64)
        .ceil()
        .clamp((x1 + 1) as f64, width as f64) as u32;
    let y2 = (bbox[3] * height as f64)
        .ceil()
        .clamp((y1 + 1) as f64, height as f64) as u32;
    let crop = imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
    let crop = match block.angle {
        Some(90) => imageops::rotate270(&crop),
        Some(180) => imageops::rotate180(&crop),
        Some(270) => imageops::rotate90(&crop),
        _ => crop,
    };
    resize_crop(&crop, min_edge, max_ratio)
}

pub(super) fn format_content(kind: &str, text: &str) -> String {
    let text = text.trim();
    match kind {
        "table" => crate::pipeline::mineru_table::diffusion_otsl_to_html(text),
        "equation" if !text.is_empty() => {
            let mut text = text.to_string();
            if !text.starts_with(r"\[") {
                text = format!("\\[\n{text}");
            }
            if !text.ends_with(r"\]") {
                text.push_str("\n\\]");
            }
            text
        }
        _ => text.into(),
    }
}
