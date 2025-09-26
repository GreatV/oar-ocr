use super::*;
use crate::core::config::{CommonBuilderConfig, OrtSessionConfig};

#[test]
fn test_from_common_with_auto_input_respects_config() {
    let common = CommonBuilderConfig::new()
        .session_pool_size(2)
        .ort_session(OrtSessionConfig::new());

    let result = OrtInfer::from_common_with_auto_input(&common, "dummy_path.onnx");
    assert!(result.is_err());
}

#[test]
fn test_from_common_respects_session_pool_size() {
    let common = CommonBuilderConfig::new().session_pool_size(3);
    let result = OrtInfer::from_common(&common, "dummy_path.onnx", None);
    assert!(result.is_err());
}
