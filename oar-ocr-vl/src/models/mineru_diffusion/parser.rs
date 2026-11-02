//! Model-native two-stage MinerU-Diffusion page parser.

use super::layout::{crop_block, format_content, parse_layout, task_budget};
use super::{DiffusionGenerationConfig, MinerUDiffusion};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{PageDocument, ParseDiagnostic};
use crate::pipeline::mineru_layout::{LAYOUT_IMAGE_SIZE, LAYOUT_PROMPT};
use crate::render::text::truncate_repetitive_content;
use crate::runtime::image::pil_bicubic_resize;
use image::RgbImage;

/// Options for MinerU-Diffusion's native two-stage page parser.
#[derive(Debug, Clone)]
pub struct MinerUDiffusionParseOptions {
    /// Decode controls; `gen_length` caps each task (default cap: 2048).
    /// Native task budgets are layout/table 2048 and text/formula 1024.
    pub generation: DiffusionGenerationConfig,
    pub min_image_edge: u32,
    pub max_image_edge_ratio: f32,
}

impl Default for MinerUDiffusionParseOptions {
    fn default() -> Self {
        Self {
            generation: DiffusionGenerationConfig {
                gen_length: 2048,
                ..DiffusionGenerationConfig::default()
            },
            min_image_edge: 28,
            max_image_edge_ratio: 50.0,
        }
    }
}

impl PageParser for MinerUDiffusion {
    type Options = MinerUDiffusionParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let layout_image = pil_bicubic_resize(image, LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE)?;
        let raw_layout = generate_task(
            self,
            &layout_image,
            LAYOUT_PROMPT,
            "layout",
            &options.generation,
        )?;
        let mut blocks = parse_layout(&raw_layout);
        let mut diagnostics = Vec::new();
        for (block_index, block) in blocks.iter_mut().enumerate() {
            if matches!(block.block_type.as_str(), "image" | "equation_block") {
                continue;
            }
            let prompt = match block.block_type.as_str() {
                "table" => "\nTable Recognition:",
                "equation" => "\nFormula Recognition:",
                _ => "\nText Recognition:",
            };
            let crop = crop_block(
                image,
                block,
                options.min_image_edge,
                options.max_image_edge_ratio,
            )?;
            match generate_task(self, &crop, prompt, &block.block_type, &options.generation) {
                Ok(content) => {
                    let content = truncate_repetitive_content(&content, 10, 10, 10);
                    block.content = Some(format_content(&block.block_type, &content));
                }
                Err(error) => diagnostics.push(ParseDiagnostic {
                    block_index: Some(block_index),
                    stage: "recognition".to_string(),
                    message: error.to_string(),
                }),
            }
        }

        let mut page = PageDocument {
            blocks,
            raw_output: Some(raw_layout),
            diagnostics,
            ..PageDocument::default()
        };
        page.markdown = Some(page.render_markdown());
        Ok(page)
    }
}

fn generate_task(
    model: &MinerUDiffusion,
    image: &RgbImage,
    prompt: &str,
    kind: &str,
    controls: &DiffusionGenerationConfig,
) -> Result<String, Error> {
    let budget = task_budget(kind, controls.gen_length);
    if budget == 0 {
        return Ok(String::new());
    }
    let mut generation = controls.clone();
    generation.gen_length = budget
        .checked_next_multiple_of(generation.block_length.max(1))
        .ok_or_else(|| Error::invalid_input("MinerU-Diffusion generation length overflow"))?;
    let mut ids = model.generate_token_ids(image, prompt, &generation)?;
    ids.truncate(budget);
    model.decode_tokens(&ids, kind != "layout")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DocumentBlock;

    #[test]
    fn diffusion_keeps_native_labels_and_crop_and_equation_rules() {
        let cap = MinerUDiffusionParseOptions::default().generation.gen_length;
        assert_eq!(
            [
                task_budget("layout", cap),
                task_budget("table", cap),
                task_budget("text", cap)
            ],
            [2048, 2048, 1024]
        );
        assert_eq!(task_budget("table", 256), 256);
        assert_eq!(
            super::super::layout::resize_crop(&RgbImage::new(7, 3), 28, 50.)
                .unwrap()
                .dimensions(),
            (65, 28)
        );
        let raw = "<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>list<|ref_end|>\n<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>unknown<|ref_end|>";
        assert_eq!(
            parse_layout(raw)
                .iter()
                .map(|b| b.block_type.as_str())
                .collect::<Vec<_>>(),
            ["list", "unknown"]
        );
        let block = DocumentBlock {
            block_type: "text".into(),
            bbox: [0.11, 0.11, 0.89, 0.89],
            angle: None,
            content: None,
        };
        assert_eq!(
            crop_block(&RgbImage::new(10, 10), &block, 0, 50.)
                .unwrap()
                .dimensions(),
            (8, 8)
        );
        assert_eq!(
            format_content("equation", r"\big{\rangle}x"),
            "\\[\n\\big{\\rangle}x\n\\]"
        );
        assert_eq!(format_content("text", "[Non-Text]"), "[Non-Text]");
    }
}
