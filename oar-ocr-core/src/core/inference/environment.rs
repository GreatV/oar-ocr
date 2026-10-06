//! Initialization of the process-wide ONNX Runtime environment.

use ort::environment::{Environment, EnvironmentBuilder};
use ort::logging::LogLevel;
use std::sync::Mutex;

// Retain ownership if native environment creation fails, so a retry still
// applies our logging default. The lock covers commit, creation, and setup.
static PENDING_OWNED_ENVIRONMENT: Mutex<bool> = Mutex::new(false);

/// Initializes ONNX Runtime with Error logging if no environment is configured.
///
/// Returns `true` when this call commits the default configuration. An existing
/// configuration, including one implicitly created by ONNX Runtime, is retained.
/// Applications that configure ONNX Runtime themselves should do so before this
/// function or any OAR model construction.
pub fn initialize_ort_environment() -> ort::Result<bool> {
    commit_environment(ort::init())
}

pub(super) fn commit_environment(builder: EnvironmentBuilder) -> ort::Result<bool> {
    let mut pending = PENDING_OWNED_ENVIRONMENT
        .lock()
        .map_err(|_| ort::Error::new("ONNX Runtime environment initialization lock poisoned"))?;
    let committed = builder.commit();
    *pending |= committed;
    if *pending {
        Environment::current()?.set_log_level(LogLevel::Error);
        *pending = false;
    }
    Ok(committed)
}
