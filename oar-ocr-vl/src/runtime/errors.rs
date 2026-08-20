//! Candle error translation with stable crate-level context.

use crate::api::error::{Error, ProcessingStage};

pub fn candle_to_ocr_inference(
    model_name: &str,
    context: impl Into<String>,
    source: candle_core::Error,
) -> Error {
    Error::Inference {
        model_name: model_name.to_string(),
        context: context.into(),
        source: Box::new(source),
    }
}

pub fn candle_to_ocr_processing(
    kind: ProcessingStage,
    context: impl Into<String>,
    source: candle_core::Error,
) -> Error {
    Error::Processing {
        kind,
        context: context.into(),
        source: Box::new(source),
    }
}

pub fn error_chain_message(prefix: &str, error: &(dyn std::error::Error + 'static)) -> String {
    use std::fmt::Write;
    let mut chain = format!("{prefix}: {error}");
    let mut current = error.source();
    while let Some(source) = current {
        let _ = write!(chain, "\n  caused by: {source}");
        current = source.source();
    }
    chain
}
