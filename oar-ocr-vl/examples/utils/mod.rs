//! Common utilities for oar-ocr-vl examples.

#[allow(dead_code)]
pub mod structure_match;

use std::time::Duration;

use oar_ocr_vl::hsd::types::{DsvConfig, HsdConfig, HsdStats, SpecDecodeStats};

/// Initializes the tracing subscriber for logging in examples.
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
pub fn make_hsd_cfg(
    max_tokens: usize,
    tau: f32,
    window_len: usize,
    max_candidates_per_step: usize,
    max_suffix_len: usize,
    enable_stage1: bool,
) -> HsdConfig {
    HsdConfig {
        dsv: DsvConfig {
            tau,
            window_len,
            max_candidates_per_step,
            max_suffix_len,
        },
        enable_stage1,
        enable_stage2: true,
        max_page_tokens: max_tokens,
        max_region_tokens: max_tokens,
    }
}

#[allow(dead_code)]
pub fn print_hsd_stats(
    baseline_dur: Duration,
    hsd_dur: Duration,
    stats: &HsdStats,
    include_stage1: bool,
) {
    let baseline_decode_estimate = baseline_dur.saturating_sub(stats.stage2.vision_prefill);
    let sr_decode = if stats.stage2.decode.as_secs_f64() > 0.0 {
        baseline_decode_estimate.as_secs_f64() / stats.stage2.decode.as_secs_f64()
    } else {
        0.0
    };
    let sr_e2e = if hsd_dur.as_secs_f64() > 0.0 {
        baseline_dur.as_secs_f64() / hsd_dur.as_secs_f64()
    } else {
        0.0
    };

    println!("\n=== STATS ===");
    println!("baseline e2e:           {:?}", baseline_dur);
    println!("HSD e2e:                {:?}", hsd_dur);
    println!("  drafter prep:         {:?}", stats.drafter);
    if include_stage1 {
        println!("  stage1 prep:          {:?}", stats.stage1.draft_prep);
        println!("  stage1 vision+prefill:{:?}", stats.stage1.vision_prefill);
        println!("  stage1 decode:        {:?}", stats.stage1.decode);
        println!("  stage1 forward passes:{}", stats.stage1.forward_passes);
    }
    println!("  vision+prefill:       {:?}", stats.stage2.vision_prefill);
    println!("  decode:               {:?}", stats.stage2.decode);
    println!("  forward passes:       {}", stats.stage2.forward_passes);
    println!("  emitted tokens:       {}", stats.stage2.emitted_tokens);
    println!("  verify steps:         {}", stats.stage2.accept.num_steps);
    println!(
        "  fallback steps:       {}",
        stats.stage2.accept.num_fallbacks
    );
    println!("  AAL:                  {:.2}", stats.stage2.accept.aal());
    print_dsv_stats(&stats.stage2.dsv);
    println!("\nSR_decode (estimated):  {:.2}x", sr_decode);
    println!("SR_e2e:                 {:.2}x", sr_e2e);
}

#[allow(dead_code)]
pub fn print_dsv_stats(dsv: &SpecDecodeStats) {
    println!("  DSV candidate build:  {:?}", dsv.candidate_build);
    println!(
        "  DSV verify_tree:      {:?} (calls={}, avg_nodes={:.1}, max_nodes={})",
        dsv.verify_tree,
        dsv.verify_tree_calls,
        dsv.avg_tree_nodes(),
        dsv.tree_nodes_max
    );
    println!("  DSV traverse:         {:?}", dsv.traverse);
    println!("  DSV commit:           {:?}", dsv.commit);
    println!(
        "  DSV step_one:         {:?} (calls={})",
        dsv.step_one, dsv.step_one_calls
    );
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
pub fn print_diff(baseline: &str, hsd: &str) {
    let common = baseline
        .chars()
        .zip(hsd.chars())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "Lengths: baseline={}, hsd={}, common prefix={}",
        baseline.len(),
        hsd.len(),
        common
    );
    let snippet =
        |s: &str, start: usize| -> String { s.chars().skip(start).take(80).collect::<String>() };
    eprintln!("baseline[{common}..]:  {:?}", snippet(baseline, common));
    eprintln!("hsd[{common}..]:       {:?}", snippet(hsd, common));
}
