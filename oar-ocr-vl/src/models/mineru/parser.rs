//! Model-native two-stage MinerU page parser.

use super::MinerU;
use super::adapter::table_options;
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{PageDocument, ParseDiagnostic};
use crate::pipeline::mineru_layout::{
    LAYOUT_IMAGE_SIZE, LAYOUT_PROMPT, parse_layout_details, prepare_for_extract,
};
use crate::render::table::convert_otsl_to_html_with_options;
use crate::render::text::truncate_repetitive_content;
use crate::runtime::image::pil_bicubic_resize;
use image::RgbImage;

/// Generation and crop scheduling options for MinerU's native parser.
#[derive(Debug, Clone)]
pub struct MinerUParseOptions {
    pub max_tokens: usize,
    pub region_batch_size: usize,
    pub min_image_edge: u32,
    pub max_image_edge_ratio: f32,
}

impl Default for MinerUParseOptions {
    fn default() -> Self {
        Self {
            max_tokens: 16_384,
            region_batch_size: 2,
            min_image_edge: 28,
            max_image_edge_ratio: 50.0,
        }
    }
}

impl PageParser for MinerU {
    type Options = MinerUParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let layout_image = pil_bicubic_resize(image, LAYOUT_IMAGE_SIZE, LAYOUT_IMAGE_SIZE)?;
        let layout_tokens = self
            .generate_tokens(&[layout_image], &[LAYOUT_PROMPT], options.max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("MinerU layout detection returned no result"))?;
        let raw_layout = self.decode_tokens(&layout_tokens)?;
        let records = parse_layout_details(&raw_layout);
        let continuations: Vec<_> = records
            .iter()
            .filter(|(b, merge)| *merge && b.block_type == "text")
            .map(|(b, _)| b.bbox)
            .collect();
        let mut blocks: Vec<_> = records.into_iter().map(|(block, _)| block).collect();
        let mut diagnostics = Vec::new();
        if blocks.iter().any(|b| b.block_type == "chart") {
            diagnostics.push(ParseDiagnostic { block_index: None, stage: "recognition".into(), message: "Chart recognition is disabled until Image Analysis replies are parsed; chart boxes are retained".into() });
        }

        let (images, prompts, indices) = prepare_for_extract(
            image,
            &blocks,
            options.min_image_edge,
            options.max_image_edge_ratio,
        );
        let batch_size = options.region_batch_size.max(1);
        for start in (0..images.len()).step_by(batch_size) {
            let end = (start + batch_size).min(images.len());
            let expected = end - start;
            let batch = self.generate_tokens(
                &images[start..end],
                &prompts[start..end],
                options.max_tokens,
            );
            let results: Vec<Result<Vec<u32>, Error>> = match batch {
                Ok(tokens) if tokens.len() == expected => tokens.into_iter().map(Ok).collect(),
                Ok(tokens) => {
                    diagnostics.push(ParseDiagnostic {
                        block_index: None,
                        stage: "batch".to_string(),
                        message: format!(
                            "MinerU returned {} results for {expected} blocks; retried individually",
                            tokens.len()
                        ),
                    });
                    retry_individually(self, &images, &prompts, start, end, options.max_tokens)
                }
                Err(error) if expected == 1 => vec![Err(error)],
                Err(error) => {
                    diagnostics.push(ParseDiagnostic {
                        block_index: None,
                        stage: "batch".to_string(),
                        message: format!("{error}; retried individually"),
                    });
                    retry_individually(self, &images, &prompts, start, end, options.max_tokens)
                }
            };

            for (&block_index, result) in indices[start..end].iter().zip(results) {
                match result.and_then(|tokens| self.decode_tokens(&tokens)) {
                    Ok(content) => {
                        let content = truncate_repetitive_content(&content, 10, 10, 10);
                        blocks[block_index].content =
                            Some(if blocks[block_index].block_type == "table" {
                                convert_otsl_to_html_with_options(&content, &table_options())
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
        }

        let blocks = crate::pipeline::mineru_postprocess::postprocess_blocks(blocks);
        let mut page = PageDocument {
            blocks,
            raw_output: Some(raw_layout),
            diagnostics,
            ..PageDocument::default()
        };
        page.markdown = Some(render_with_continuations(&page, &continuations));
        Ok(page)
    }
}

fn render_with_continuations(page: &PageDocument, continuations: &[[f32; 4]]) -> String {
    let mut chunks: Vec<String> = Vec::new();
    let mut last_text = None;
    for block in &page.blocks {
        let chunk = PageDocument {
            blocks: vec![block.clone()],
            ..PageDocument::default()
        }
        .render_markdown();
        if chunk.trim().is_empty() {
            continue;
        }
        if block.block_type == "text"
            && continuations.contains(&block.bbox)
            && let Some(index) = last_text
        {
            let previous: &mut String = &mut chunks[index];
            if !chunk
                .chars()
                .any(|ch| matches!(ch, '\u{3400}'..='\u{4dbf}'|'\u{4e00}'..='\u{9fff}'))
            {
                previous.push(' ');
            }
            previous.push_str(&chunk);
        } else {
            if block.block_type == "text" {
                last_text = Some(chunks.len());
            }
            chunks.push(chunk);
        }
    }
    chunks.join("\n\n")
}

fn retry_individually(
    model: &MinerU,
    images: &[RgbImage],
    prompts: &[String],
    start: usize,
    end: usize,
    max_tokens: usize,
) -> Vec<Result<Vec<u32>, Error>> {
    (start..end)
        .map(|position| {
            let mut outputs = model.generate_tokens(
                std::slice::from_ref(&images[position]),
                std::slice::from_ref(&prompts[position]),
                max_tokens,
            )?;
            if outputs.len() != 1 {
                return Err(Error::invalid_input(format!(
                    "MinerU returned {} results for one block",
                    outputs.len()
                )));
            }
            outputs
                .pop()
                .ok_or_else(|| Error::invalid_input("MinerU returned no block result"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DocumentBlock;

    #[test]
    fn continuation_merges_the_last_text_even_across_a_table() {
        let block = |kind: &str, y, text: &str| DocumentBlock {
            block_type: kind.into(),
            bbox: [0., y, 1., y + 0.1],
            angle: None,
            content: Some(text.into()),
        };
        let page = PageDocument {
            blocks: vec![
                block("text", 0., "First"),
                block("table", 0.2, "<table><tr><td>T</td></tr></table>"),
                block("text", 0.4, "second"),
                block("text", 0.6, "中文"),
            ],
            ..PageDocument::default()
        };
        let markdown =
            render_with_continuations(&page, &[page.blocks[2].bbox, page.blocks[3].bbox]);
        assert!(markdown.starts_with("First second中文\n\n"), "{markdown}");
        assert!(markdown.contains("<table"));
    }
}
