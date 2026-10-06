use oar_ocr_core::core::inference::initialize_ort_environment;
use ort::environment::Environment;
use ort::logging::LogLevel;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

// This test runs in its own process because ORT configuration is global.
#[test]
fn preserves_user_environment_logging() -> ort::Result<()> {
    let info_messages = Arc::new(AtomicUsize::new(0));
    let captured = info_messages.clone();
    assert!(
        ort::init()
            .with_logger(Arc::new(move |level, _, _, _, _| {
                if level == LogLevel::Info {
                    captured.fetch_add(1, Ordering::Relaxed);
                }
            }))
            .commit()
    );
    Environment::current()?.set_log_level(LogLevel::Info);
    info_messages.store(0, Ordering::Relaxed);

    assert!(!initialize_ort_environment()?);
    // An invalid model needs no weights but still logs session construction
    // at Info. Changing the global severity to Error would hide those messages.
    let result = ort::session::Session::builder()?
        .with_intra_threads(1)
        .map_err(ort::Error::<()>::from)?
        .commit_from_memory(&[]);
    assert!(result.is_err());
    assert!(info_messages.load(Ordering::Relaxed) > 0);
    Ok(())
}
