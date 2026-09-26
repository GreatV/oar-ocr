//! Complete-page parser adapter for jina-ocr-v1.

use super::{DEFAULT_MAX_NEW_TOKENS, JinaOcr};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{PageDocument, ParseDiagnostic};
use image::RgbImage;

/// jina-ocr-v1 complete-page parsing options.
#[derive(Debug, Clone)]
pub struct JinaOcrParseOptions {
    pub max_new_tokens: usize,
}

impl Default for JinaOcrParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

/// Diagnostic emitted when the token budget runs out before EOS — the
/// official generation applies an n-gram ban that makes runaway repeats
/// impossible, so a budget exhaustion means the page did not finish.
pub(crate) fn truncation_diagnostic(max_new_tokens: usize) -> ParseDiagnostic {
    ParseDiagnostic {
        block_index: None,
        stage: "generation".to_string(),
        message: format!(
            "JinaOCR generation truncated after {max_new_tokens} tokens without an EOS token; increase max_new_tokens"
        ),
    }
}

impl PageParser for JinaOcr {
    type Options = JinaOcrParseOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let (tokens, finished) = self.generate_one(image, options.max_new_tokens)?;
        let markdown = self.decode_tokens(&tokens)?;
        let diagnostics = if finished {
            Vec::new()
        } else {
            vec![truncation_diagnostic(options.max_new_tokens)]
        };
        Ok(PageDocument {
            markdown: Some(markdown.clone()),
            raw_output: Some(markdown),
            diagnostics,
            ..PageDocument::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_diagnostic_reports_stage_and_budget() {
        let diagnostic = truncation_diagnostic(4_096);
        assert_eq!(diagnostic.block_index, None);
        assert_eq!(diagnostic.stage, "generation");
        assert!(diagnostic.message.contains("4096"));
        assert!(diagnostic.message.contains("EOS"));
    }
}
