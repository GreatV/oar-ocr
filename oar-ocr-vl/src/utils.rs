//! Backwards-compatible utility facade.
//!
//! New internal code should import functionality from its owning `runtime`,
//! `pipeline`, or `render` layer rather than adding more helpers here.

pub mod image {
    //! Backwards-compatible image preprocessing exports.
    pub use crate::runtime::image::*;
}

pub mod table {
    //! Backwards-compatible table rendering exports.
    pub use crate::render::table::*;
}

pub mod text {
    //! Backwards-compatible text rendering exports.
    pub use crate::render::text::*;
}

pub use crate::pipeline::geometry::{
    DetectedBox, calculate_bbox_area, calculate_overlap_ratio, calculate_projection_overlap_ratio,
    crop_margin, filter_overlap_boxes,
};
pub use crate::render::markdown::{
    DEFAULT_MARKDOWN_IGNORE_LABELS, default_markdown_ignore_labels, to_markdown,
};
pub use crate::render::table::convert_otsl_to_html;
pub use crate::render::text::truncate_repetitive_content;
pub use crate::runtime::checkpoint::{
    collect_safetensors, default_rescale_factor, default_true, load_json_config,
    load_optional_json_config, validate_image_mean_std, validate_patch_merge_temporal,
};
pub use crate::runtime::device::{free_device_memory, parse_device, select_dtype};
pub use crate::runtime::errors::{
    candle_to_ocr_inference, candle_to_ocr_processing, error_chain_message,
};
pub use crate::runtime::tensor::{rotate_half, vision_inv_freq};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::Error;
    use crate::document::geometry::BoundingBox;
    use crate::document::structure::{LayoutElement, LayoutElementType};
    use crate::runtime::device::{bf16_works, dtype_from_str};
    use candle_core::{DType, Device};

    fn element(label: &str, text: &str) -> LayoutElement {
        LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, 10.0, 10.0),
            LayoutElementType::from_label(label),
            1.0,
        )
        .with_label(label)
        .with_text(text)
    }

    #[test]
    fn parse_device_cpu() {
        assert!(parse_device("cpu").unwrap().is_cpu());
    }

    #[test]
    fn parse_device_invalid() {
        assert!(parse_device("invalid").is_err());
    }

    #[test]
    fn dtype_names_are_case_insensitive() {
        assert_eq!(dtype_from_str("BF16"), Some(DType::BF16));
        assert_eq!(dtype_from_str("fp16"), Some(DType::F16));
        assert_eq!(dtype_from_str("float32"), Some(DType::F32));
        assert_eq!(dtype_from_str("f64"), None);
    }

    #[test]
    fn cpu_dtype_and_probe_are_safe() {
        assert_eq!(select_dtype(&Device::Cpu), DType::F32);
        assert!(!bf16_works(&Device::Cpu));
    }

    #[test]
    fn error_chain_preserves_root_cause() {
        let error = Error::Inference {
            model_name: "test".to_string(),
            context: "decode".to_string(),
            source: Box::new(std::io::Error::other("out of memory")),
        };
        assert!(error_chain_message("failed", &error).contains("out of memory"));
    }

    #[test]
    fn markdown_heading_and_pretty_modes_are_preserved() {
        let elements = vec![
            element("doc_title", "A Paper"),
            element("paragraph_title", "1.2 Background"),
            element("figure_title", "Figure 1"),
        ];
        let ignored = default_markdown_ignore_labels();
        let pretty = to_markdown(&elements, &ignored, true);
        let plain = to_markdown(&elements, &ignored, false);
        assert!(pretty.contains("# A Paper"));
        assert!(pretty.contains("### 1.2 Background"));
        assert!(pretty.contains("text-align: center"));
        assert!(!plain.contains("text-align: center"));
    }
}
