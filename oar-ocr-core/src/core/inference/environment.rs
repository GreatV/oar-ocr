//! Initialization of the process-wide ONNX Runtime environment.

use ort::environment::{Environment, EnvironmentBuilder};
use ort::logging::LogLevel;
use std::sync::{Mutex, MutexGuard};

struct InstalledEnvironment {
    env: Environment,
    // Application-owned environments retain their application's logging policy.
    owned_log_level: Option<LogLevel>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}. Pass the existing handle to set_ort_environment before creating OAR sessions")]
struct UninstalledEnvironment(#[source] ort::Error);

impl InstalledEnvironment {
    fn build(builder: EnvironmentBuilder) -> ort::Result<Self> {
        let env = builder.build().map_err(|error| {
            if error.message() == "only one environment is allowed per process" {
                ort::Error::wrap(UninstalledEnvironment(error))
            } else {
                error
            }
        })?;
        env.set_log_level(LogLevel::Error);
        Ok(Self {
            env,
            owned_log_level: Some(LogLevel::Error),
        })
    }
}

fn state() -> ort::Result<MutexGuard<'static, Option<InstalledEnvironment>>> {
    static STATE: Mutex<Option<InstalledEnvironment>> = Mutex::new(None);
    STATE
        .lock()
        .map_err(|_| ort::Error::new("ONNX Runtime environment initialization lock poisoned"))
}

/// Returns the shared ORT environment, creating it with Error logging on first use.
///
/// Applications that build an environment themselves should first call [`set_ort_environment`].
pub fn initialize_ort_environment() -> ort::Result<Environment> {
    let mut state = state()?;
    let installed = match state.as_ref() {
        Some(installed) => installed,
        None => state.insert(InstalledEnvironment::build(ort::init())?),
    };
    Ok(installed.env.clone())
}

/// Installs an application-built ORT environment before device probing or model construction.
///
/// OAR retains the handle and preserves the application's logging policy. Returns an error
/// if an environment was already installed or initialized by OAR.
pub fn set_ort_environment(env: Environment) -> ort::Result<()> {
    let mut state = state()?;
    if state.is_some() {
        return Err(ort::Error::new(
            "OAR already has an ONNX Runtime environment",
        ));
    }
    *state = Some(InstalledEnvironment {
        env,
        owned_log_level: None,
    });
    Ok(())
}

pub(super) fn commit_environment(builder: EnvironmentBuilder) -> ort::Result<bool> {
    let mut state = state()?;
    if state.is_some() {
        return Ok(false);
    }
    *state = Some(InstalledEnvironment::build(builder)?);
    Ok(true)
}

pub(super) fn lower_owned_environment_log_level(level: LogLevel) -> ort::Result<()> {
    if let Some(InstalledEnvironment {
        env,
        owned_log_level: Some(previous),
    }) = state()?.as_mut()
        && level < *previous
    {
        env.set_log_level(level);
        *previous = level;
    }
    Ok(())
}
