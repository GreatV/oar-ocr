//! Complete-page parser adapter for WeVisDoc.

use super::{DEFAULT_MAX_NEW_TOKENS, WeVisDoc};
use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::document::page::{PageDocument, ParseDiagnostic};
use image::RgbImage;

/// WeVisDoc complete-page parsing options.
#[derive(Debug, Clone)]
pub struct WeVisDocParseOptions {
    pub max_new_tokens: usize,
}

impl Default for WeVisDocParseOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

/// Diagnostic emitted when the token budget runs out before an EOS token —
/// the official `wevisdoc/local.py` rejects such pages, the parser reports
/// them per-page instead of failing the whole batch.
pub(crate) fn truncation_diagnostic(max_new_tokens: usize) -> ParseDiagnostic {
    ParseDiagnostic {
        block_index: None,
        stage: "generation".to_string(),
        message: format!(
            "WeVisDoc generation truncated after {max_new_tokens} tokens without an EOS token; increase max_new_tokens"
        ),
    }
}

impl PageParser for WeVisDoc {
    type Options = WeVisDocParseOptions;

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
        let diagnostic = truncation_diagnostic(8_192);
        assert_eq!(diagnostic.block_index, None);
        assert_eq!(diagnostic.stage, "generation");
        assert!(diagnostic.message.contains("8192"));
        assert!(diagnostic.message.contains("EOS"));
    }
}
