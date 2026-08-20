//! Model-native two-stage MinerU-Diffusion page parser.

use super::{DiffusionGenerationConfig, MinerUDiffusion};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{PageDocument, ParseDiagnostic};
use crate::pipeline::mineru_layout::{
    LAYOUT_IMAGE_SIZE, LAYOUT_PROMPT, parse_layout_output, prepare_for_extract,
};
use crate::render::table::convert_otsl_to_html;
use crate::render::text::truncate_repetitive_content;
use image::{RgbImage, imageops};

/// Options for MinerU-Diffusion's native two-stage page parser.
#[derive(Debug, Clone)]
pub struct MinerUDiffusionParseOptions {
    pub generation: DiffusionGenerationConfig,
    pub min_image_edge: u32,
    pub max_image_edge_ratio: f32,
}

impl Default for MinerUDiffusionParseOptions {
    fn default() -> Self {
        Self {
            generation: DiffusionGenerationConfig::default(),
            min_image_edge: 28,
            max_image_edge_ratio: 50.0,
        }
    }
}

impl PageParser for MinerUDiffusion {
    type Options = MinerUDiffusionParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let layout_image = imageops::resize(
            image,
            LAYOUT_IMAGE_SIZE,
            LAYOUT_IMAGE_SIZE,
            imageops::FilterType::CatmullRom,
        );
        let raw_layout = self.generate_raw(&layout_image, LAYOUT_PROMPT, &options.generation)?;
        let mut blocks = parse_layout_output(&raw_layout);
        let mut diagnostics = Vec::new();
        let (images, prompts, indices) = prepare_for_extract(
            image,
            &blocks,
            options.min_image_edge,
            options.max_image_edge_ratio,
        );

        for ((image, prompt), block_index) in images.iter().zip(&prompts).zip(indices) {
            match self.generate_one(image, prompt, &options.generation) {
                Ok(content) => {
                    let content = truncate_repetitive_content(&content, 10, 10, 10);
                    blocks[block_index].content =
                        Some(if blocks[block_index].block_type == "table" {
                            convert_otsl_to_html(&content)
                        } else {
                            content.trim().to_string()
                        });
                }
                Err(error) => diagnostics.push(ParseDiagnostic {
                    block_index: Some(block_index),
                    stage: "recognition".to_string(),
                    message: error.to_string(),
                }),
            }
        }

        Ok(PageDocument {
            blocks,
            raw_output: Some(raw_layout),
            diagnostics,
            ..PageDocument::default()
        })
    }
}
