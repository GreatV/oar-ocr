//! Optional parsing of PaddleOCR-VL's raw spotting protocol.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Recognized lines and quadrilaterals from a spotting response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PaddleOcrVlSpottingOutput {
    /// Line texts joined with blank lines, as in the PaddleX converter.
    pub markdown: String,
    /// Quadrilaterals in caller-supplied image pixel coordinates.
    pub rec_polys: Vec<[[f32; 2]; 4]>,
    /// Recognized text for each quadrilateral, in output order.
    pub rec_texts: Vec<String>,
}

/// Parse raw `Spotting` output without changing standalone generation.
/// `width` and `height` are the dimensions to which 0–1000 LOC coordinates
/// should be mapped, normally those of the supplied source crop.
pub fn postprocess_spotting_output(
    input: &str,
    width: u32,
    height: u32,
) -> PaddleOcrVlSpottingOutput {
    // PaddleX paddleocr_vl/uilts.py post_process_for_spotting (Apache-2.0).
    static TEXT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<\|TEXT_START\|>(.*?)<\|TEXT_END\|>").unwrap());
    static BLOCK: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<\|LOC_BEGIN\|>(.*?)<\|LOC_END\|>").unwrap());
    static LOC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<\|LOC_(\d+)\|>").unwrap());
    let polygon = |text: &str| {
        let values: Vec<f32> = LOC
            .captures_iter(text)
            .take(8)
            .filter_map(|c| c[1].parse().ok())
            .collect();
        (values.len() == 8).then(|| {
            std::array::from_fn(|i| {
                let point = crate::document::geometry::point_from_grid(
                    [values[2 * i], values[2 * i + 1]],
                    1000.,
                    width,
                    height,
                );
                [point.x, point.y]
            })
        })
    };
    let mut output = PaddleOcrVlSpottingOutput::default();
    for (text, block) in TEXT.captures_iter(input).zip(BLOCK.captures_iter(input)) {
        if let Some(poly) = polygon(&block[1]) {
            output.rec_texts.push(text[1].to_string());
            output.rec_polys.push(poly);
        }
    }
    if output.rec_polys.is_empty() {
        let tokens: Vec<_> = LOC.find_iter(input).collect();
        let mut previous = 0;
        for group in tokens.as_chunks::<8>().0 {
            if let Some(poly) = polygon(&input[group[0].start()..group[7].end()]) {
                output
                    .rec_texts
                    .push(input[previous..group[0].start()].trim().to_string());
                output.rec_polys.push(poly);
            }
            previous = group[7].end();
        }
    }
    output.markdown = output.rec_texts.join("\n\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spotting_protocol_and_plain_fallback_map_source_coordinates() {
        let loc =
            "<|LOC_0|><|LOC_0|><|LOC_1000|><|LOC_0|><|LOC_1000|><|LOC_1000|><|LOC_0|><|LOC_1000|>";
        for raw in [
            format!("<|TEXT_START|>A<|TEXT_END|><|LOC_BEGIN|>{loc}<|LOC_END|>"),
            format!("A{loc}"),
        ] {
            let result = postprocess_spotting_output(&raw, 200, 100);
            assert_eq!(result.markdown, "A");
            assert_eq!(
                result.rec_polys,
                vec![[[0., 0.], [200., 0.], [200., 100.], [0., 100.]]]
            );
        }
    }
}
