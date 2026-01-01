//! Common utilities for examples.

#[cfg(feature = "vl")]
pub mod candle_device;
pub mod device_config;

/// Initializes the tracing subscriber for logging in examples.
pub fn init_tracing() {
    // Reference utility functions to prevent dead_code warnings when building with --all-features
    let _ = device_config::parse_device_config;
    #[cfg(feature = "vl")]
    {
        let _ = candle_device::parse_candle_device;
    }

    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}
