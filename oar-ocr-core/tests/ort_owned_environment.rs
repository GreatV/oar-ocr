use oar_ocr_core::core::config::{ModelInferenceConfig, OrtSessionConfig};
use oar_ocr_core::core::inference::{OrtInfer, initialize_ort_environment};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::{Context, SubscriberExt};

struct InfoCounter(Arc<AtomicUsize>);

impl<S: tracing::Subscriber> Layer<S> for InfoCounter {
    fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
        if event.metadata().target() == "ort::logging"
            && *event.metadata().level() == tracing::Level::INFO
        {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

// Isolate ORT's process-global configuration from other model tests.
#[test]
fn explicit_session_level_lowers_owned_environment_without_raising_it() -> ort::Result<()> {
    let messages = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(InfoCounter(messages.clone())),
    )
    .unwrap();
    // Initialize early, as provider probing does, before applying session options.
    assert!(initialize_ort_environment()?);
    for level in [1, 3] {
        let mut config = ModelInferenceConfig::new();
        config.ort_session = Some(
            OrtSessionConfig::new()
                .with_intra_threads(1)
                .with_log_severity_level(level),
        );
        assert!(OrtInfer::from_config(&config, Vec::<u8>::new(), None).is_err());
    }
    messages.store(0, Ordering::Relaxed);
    // With no session override, constructor logging inherits the global level.
    // Invalid ONNX bytes trigger construction without requiring model weights.
    assert!(
        ort::session::Session::builder()?
            .with_intra_threads(1)
            .map_err(ort::Error::<()>::from)?
            .commit_from_memory(&[])
            .is_err()
    );
    assert!(messages.load(Ordering::Relaxed) > 0);
    Ok(())
}
