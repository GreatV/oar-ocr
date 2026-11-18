use oar_ocr_core::core::inference::{OrtGlobalThreadPoolOptions, load_session};

// This integration binary starts with a fresh process-wide ORT environment.
#[test]
fn global_pool_is_committed_once_and_shared_by_sessions() -> Result<(), Box<dyn std::error::Error>>
{
    assert!(
        OrtGlobalThreadPoolOptions::new()
            .with_intra_threads(1)
            .commit()?
    );
    let graph = include_bytes!("../src/models/recognition/ctc_reduce.onnx");
    let _session = load_session(graph.as_slice())?;
    assert!(!OrtGlobalThreadPoolOptions::new().commit()?);
    Ok(())
}
