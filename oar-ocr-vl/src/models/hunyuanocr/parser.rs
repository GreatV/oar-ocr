//! Complete-page parser adapter for HunyuanOCR.

use super::{HunyuanOcr, HunyuanOcrVersion};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::PageDocument;
use image::RgbImage;

/// HunyuanOCR prompt-driven complete-page parsing options.
#[derive(Debug, Clone)]
pub struct HunyuanOcrParseOptions {
    /// Instruction override; empty selects the official prompt for the loaded version.
    pub prompt: String,
    pub max_new_tokens: usize,
}

impl Default for HunyuanOcrParseOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_new_tokens: 16_384,
        }
    }
}

impl PageParser for HunyuanOcr {
    type Options = HunyuanOcrParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let raw = self
            .generate(
                std::slice::from_ref(image),
                &[if options.prompt.is_empty() {
                    document_prompt(self.version())
                } else {
                    options.prompt.as_str()
                }],
                options.max_new_tokens,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("HunyuanOCR returned no page result"))??;
        Ok(PageDocument {
            markdown: Some(clean_output(&raw, self.version())),
            raw_output: Some(raw),
            ..PageDocument::default()
        })
    }
}

fn document_prompt(version: HunyuanOcrVersion) -> &'static str {
    match version {
        HunyuanOcrVersion::V1 => {
            "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order."
        }
        HunyuanOcrVersion::V1_5 => {
            "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
        }
    }
}

pub(super) fn clean_output(text: &str, version: HunyuanOcrVersion) -> String {
    crate::render::text::clean_repetition(
        text,
        false,
        &crate::render::text::RepetitionOptions {
            algorithm: crate::render::text::RepeatAlgorithm::ExactSuffix,
            minimum_chars: match version {
                HunyuanOcrVersion::V1 => 8000,
                HunyuanOcrVersion::V1_5 => 2000,
            },
            min_period: 2,
            trim_result: true,
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_cleanup_respects_version_and_unicode_character_thresholds() {
        let text = format!("{}{}", "正文".repeat(995), "甲乙".repeat(10));
        assert_eq!(clean_output(&text, HunyuanOcrVersion::V1), text);
        assert_eq!(
            clean_output(&text, HunyuanOcrVersion::V1_5),
            format!("{}甲乙", "正文".repeat(995))
        );
        assert_eq!(
            clean_output("甲乙甲乙甲乙", HunyuanOcrVersion::V1_5),
            "甲乙甲乙甲乙"
        );
    }
}
