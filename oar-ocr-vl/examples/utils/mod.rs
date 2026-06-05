//! Common utilities for oar-ocr-vl examples.

#[allow(dead_code)]
pub mod structure_match;

#[allow(dead_code)]
pub mod mineru_layout;

/// Initializes the tracing subscriber for logging in examples.
#[allow(dead_code)]
pub fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}

#[allow(dead_code)]
pub fn print_preview(tag: &str, text: &str) {
    let preview: String = text.chars().take(400).collect();
    let preview_chars = preview.chars().count();
    let total_chars = text.chars().count();
    println!("--- {tag} (first {preview_chars} chars) ---");
    println!("{preview}");
    if total_chars > preview_chars {
        println!("... [{} more chars]", total_chars - preview_chars);
    }
}

#[allow(dead_code)]
pub fn print_diff(baseline: &str, other: &str) {
    let common = baseline
        .chars()
        .zip(other.chars())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "Lengths: baseline={}, other={}, common prefix={}",
        baseline.len(),
        other.len(),
        common
    );
    let snippet =
        |s: &str, start: usize| -> String { s.chars().skip(start).take(80).collect::<String>() };
    eprintln!("baseline[{common}..]:  {:?}", snippet(baseline, common));
    eprintln!("other[{common}..]:     {:?}", snippet(other, common));
}
