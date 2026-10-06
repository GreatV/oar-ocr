//! Initialization of the process-wide ONNX Runtime environment.

use ort::environment::{Environment, EnvironmentBuilder};
use ort::logging::LogLevel;
use std::sync::Mutex;

struct EnvironmentState {
    pending: bool,
    owned: bool,
    log_level: LogLevel,
}

// Retain ownership if native environment creation fails, so a retry still
// applies our logging default. The lock covers commit, creation, and setup.
static ENVIRONMENT_STATE: Mutex<EnvironmentState> = Mutex::new(EnvironmentState {
    pending: false,
    owned: false,
    log_level: LogLevel::Error,
});

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
    let mut state = ENVIRONMENT_STATE
        .lock()
        .map_err(|_| ort::Error::new("ONNX Runtime environment initialization lock poisoned"))?;
    let committed = builder.commit();
    state.pending |= committed;
    state.owned |= committed;
    if state.pending {
        Environment::current()?.set_log_level(state.log_level);
        state.pending = false;
    }
    Ok(committed)
}

pub(super) fn lower_owned_environment_log_level(level: LogLevel) -> ort::Result<()> {
    let mut state = ENVIRONMENT_STATE
        .lock()
        .map_err(|_| ort::Error::new("ONNX Runtime environment initialization lock poisoned"))?;
    if state.owned && level < state.log_level {
        Environment::current()?.set_log_level(level);
        state.log_level = level;
    }
    Ok(())
}
