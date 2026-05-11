//! Decoupled Speculative Verification (DSV) — `SpecDecode` operator.
//!
//! This is the model-facing surface of HSD. Concrete VLM backends provide a
//! thin adapter implementing [`SpecBackend`]; the verification driver in this
//! file ([`spec_decode`]) is shared across all backends.
//!
//! ## Algorithm (paper §3.2)
//!
//! At every iteration `spec_decode` does the following:
//!
//! 1. Take the most recent `n = cfg.window_len` accepted tokens as the
//!    reference window and slide it over each draft. Extract the suffixes that
//!    follow each match ([`super::matching::collect_candidates`]).
//! 2. Merge the suffixes into a prefix tree
//!    ([`super::prefix_tree::build_prefix_tree`]).
//! 3. If the tree is empty, fall back to a single-token greedy step. Otherwise
//!    call [`SpecBackend::verify_tree`] which appends the packed tree tokens to
//!    the KV cache under a tree-ancestry mask and returns per-node log-probs.
//! 4. Greedily traverse the tree from the root. At each node, choose the child
//!    with the highest target-model log-probability and accept it iff it is
//!    within `log τ` of the model's unrestricted argmax token (paper eq. 11).
//!    Stop at a leaf or upon rejection.
//! 5. Append the accepted path tokens to the output, then append the greedy
//!    bonus token `û` from the terminal distribution.
//! 6. [`SpecBackend::commit_verify`] gathers the KV cache to keep only the
//!    accepted-path positions, then `step_one(û)` populates the cache for the
//!    bonus token and produces the next-iteration log-probs.
//!
//! [`spec_decode_strict`] is a debugging variant: it replays draft tokens one
//! by one through [`SpecBackend::step_one`] instead of using tree verification.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use std::time::Instant;

use super::matching::collect_candidates;
use super::nvtx;
use super::prefix_tree::{PrefixTree, build_prefix_tree};
use super::types::{AcceptStats, Draft, DsvConfig, SpecDecodeStats};

/// Backend adapter for HSD verification. Each VLM backend (HunyuanOCR,
/// PaddleOCR-VL, MinerU, …) implements this trait once, and reuses the
/// [`spec_decode`] driver.
///
/// All log-probability tensors must be **post log-softmax in F32**. The driver
/// only reads them via `to_vec1::<f32>()`; intermediate compute can run in
/// BF16/F16 inside the backend, but the surface contract is F32.
pub trait SpecBackend {
    /// Decode a single token and advance the KV cache by one. Returns the
    /// next-token log-probability tensor of shape `(vocab,)`.
    fn step_one(&mut self, token: u32) -> CandleResult<Tensor>;

    /// Run a verification forward pass over a packed prefix tree.
    ///
    /// Implementations must:
    /// - Append `tree.num_nodes()` tokens to the KV cache.
    /// - Use a tree-ancestry attention mask
    ///   ([`crate::attention::create_tree_attention_mask`]) so each node only
    ///   sees the accepted prefix and its own ancestor chain.
    /// - Use position ids `accepted_kv_len + tree.depths[i]` for node `i`.
    /// - Return `(num_nodes, vocab)` log-probabilities.
    ///
    /// The KV cache is left in the post-append state; the driver will call
    /// [`SpecBackend::commit_verify`] to gather it down to the accepted path.
    fn verify_tree(&mut self, tree: &PrefixTree) -> CandleResult<Tensor>;

    /// Gather the KV cache so it keeps only the accepted prefix plus the
    /// supplied path-node positions. An empty `accepted_path` means trim back
    /// to the prefix (full rejection).
    ///
    /// `accepted_path` is given as packed-tree indices in walk order
    /// (root → leaf). Each implementation is responsible for translating these
    /// into KV-cache absolute positions (`prefix_kv_len + idx`).
    fn commit_verify(&mut self, accepted_path: &[usize]) -> CandleResult<()>;

    /// Returns true if `tok` is any of the backend's end-of-generation
    /// tokens. Backends typically have several (eod, eos, end-of-turn marker)
    /// — returning a single id and comparing with `==` would let HSD generate
    /// past a real stop token if the model emits a *different* stop than the
    /// one we know about.
    fn is_eos(&self, tok: u32) -> bool;
}

/// Move a 1-D log-prob tensor to CPU/F32 for cheap host-side scanning.
fn lp_to_host(t: &Tensor) -> CandleResult<Vec<f32>> {
    t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.to_vec1()
}

/// Argmax over a host-side log-prob slice. Returns `(token_id, log_prob)`.
fn argmax_host(v: &[f32]) -> (u32, f32) {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best_idx = i;
        }
    }
    (best_idx as u32, best_val)
}

/// GPU-side argmax that copies only the resulting scalar token id back to the
/// host. Replaces `lp_to_host` + `argmax_host` on the fallback / bonus paths,
/// where we don't need the full distribution — only the argmax. On long
/// sequences with many empty-tree fallbacks this dominates HSD wall time:
/// per-call cost drops from ~21 ms (D2H of `vocab × 4 B` bytes) to ~1 ms
/// (kernel launch + 4-byte D2H). nsys: `cuMemcpyDtoHAsync_v2` was 87 % of
/// HSD time on long-output pages before this change.
fn argmax_on_device(t: &Tensor) -> CandleResult<u32> {
    t.argmax(candle_core::D::Minus1)?
        .to_dtype(DType::U32)?
        .to_scalar::<u32>()
}

/// Greedy traversal of the prefix tree under the τ-tolerance acceptance test.
///
/// Returns
/// - `path`: tree-node indices visited along the accepted path, root → leaf.
/// - `terminal_lp`: log-prob distribution at the final node (used to sample
///   the greedy bonus token û). Pre-materialised on the host so the caller can
///   `argmax_host` it without another GPU→CPU sync.
///
/// We pull the entire `(num_nodes, vocab)` log-prob matrix to the host in a
/// single transfer at the top, plus the `(vocab,)` root distribution. Per
/// nsys, the previous per-step `cuMemcpyDtoHAsync` calls (one per traversal
/// step, each of size `4 × vocab`) accounted for ~87 % of HSD wall time on
/// long-output pages because each transfer paid the full GPU sync latency. A
/// single bulk transfer is dominated by PCIe bandwidth regardless of how many
/// traversal steps follow.
fn greedy_traverse(
    tree: &PrefixTree,
    node_logprobs: &Tensor,
    root_logprobs: &Tensor,
    tau: f32,
) -> CandleResult<(Vec<usize>, Vec<f32>)> {
    let log_tau = tau.ln();
    let trace = std::env::var_os("OAROCR_HSD_TRACE_ACCEPT").is_some();
    let mut path: Vec<usize> = Vec::new();
    let mut s: Option<usize> = None;

    // Bulk D2H copies: root's (vocab,) and the full (num_nodes, vocab) matrix.
    let root_host: Vec<f32> = lp_to_host(root_logprobs)?;
    let nodes_host: Vec<f32> = if node_logprobs.dims().len() >= 2 {
        node_logprobs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?
    } else {
        Vec::new()
    };
    let vocab = root_host.len();

    let row = |node_idx: usize| -> &[f32] {
        let start = node_idx * vocab;
        &nodes_host[start..start + vocab]
    };

    let mut cur_lp_view: &[f32] = &root_host[..];
    let mut terminal_lp_owned: Vec<f32> = root_host.clone();

    loop {
        let children = tree.children_of(s);
        if children.is_empty() {
            break;
        }
        let (u_hat, u_hat_lp) = argmax_host(cur_lp_view);

        let mut best_node: Option<usize> = None;
        let mut best_lp = f32::NEG_INFINITY;
        for &c in &children {
            let tok = tree.tokens[c] as usize;
            let lp = cur_lp_view.get(tok).copied().unwrap_or(f32::NEG_INFINITY);
            if lp > best_lp {
                best_lp = lp;
                best_node = Some(c);
            }
        }
        let best = best_node.expect("non-empty children list");
        let best_tok = tree.tokens[best];
        let margin = best_lp - u_hat_lp;

        if trace {
            eprintln!(
                "HSD_ACCEPT depth={} best_tok={} best_lp={:.6} argmax={} argmax_lp={:.6} margin={:.6} log_tau={:.6}",
                tree.depths[best], best_tok, best_lp, u_hat, u_hat_lp, margin, log_tau
            );
        }

        if margin >= log_tau {
            path.push(best);
            cur_lp_view = row(best);
            s = Some(best);
        } else {
            break;
        }
    }

    // Save the terminal distribution for the bonus-token sample. If we
    // accepted at least one tree node, the terminal distribution lives in
    // `nodes_host`; otherwise it's the root.
    if let Some(last) = path.last().copied() {
        terminal_lp_owned = row(last).to_vec();
    }
    Ok((path, terminal_lp_owned))
}

/// `SpecDecode(p_θ, z, Ỹ)` — the page-/region-level verification driver.
///
/// `initial_logprobs` is the prefill's last-position distribution (shape
/// `(vocab,)`). `drafts` may contain Stage-1 region drafts (one verify per
/// region) or Stage-2 page drafts (one verify per page).
pub fn spec_decode<B: SpecBackend>(
    backend: &mut B,
    drafts: &[Draft],
    initial_logprobs: Tensor,
    max_new_tokens: usize,
    cfg: &DsvConfig,
    stats: &mut AcceptStats,
    timings: &mut SpecDecodeStats,
) -> CandleResult<Vec<u32>> {
    if cfg.tau >= 1.0 {
        return spec_decode_strict(
            backend,
            drafts,
            initial_logprobs,
            max_new_tokens,
            cfg,
            stats,
            timings,
        );
    }

    let mut accepted: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut cur_logprobs = initial_logprobs;

    while accepted.len() < max_new_tokens {
        nvtx::mark("dsv.iter");

        // 1. Build candidates from the most recent accepted-token window.
        let n = accepted.len().min(cfg.window_len);
        let tail = &accepted[accepted.len() - n..];
        let t_build = Instant::now();
        let tree = {
            let _r = nvtx::Range::new("dsv.candidate_build");
            let candidates = collect_candidates(tail, drafts, cfg);
            let candidate_count = candidates.len() as u32;
            timings.candidate_steps += 1;
            timings.candidates_total += candidate_count as u64;
            timings.candidates_max = timings.candidates_max.max(candidate_count);
            build_prefix_tree(&candidates)
        };
        timings.candidate_build += t_build.elapsed();

        // 2. Empty tree → fall back to a single-token greedy step.
        //    Argmax on-device so we only D2H a single scalar instead of the
        //    full vocab. On pages where most steps are fallbacks this saves
        //    ~20 ms / iteration vs the previous host-side argmax path.
        if tree.is_empty() {
            timings.empty_tree_calls += 1;
            let t_argmax = Instant::now();
            let u_hat = {
                let _r = nvtx::Range::new("dsv.fallback_argmax");
                argmax_on_device(&cur_logprobs)?
            };
            timings.fallback_argmax += t_argmax.elapsed();
            timings.fallback_argmax_calls += 1;
            accepted.push(u_hat);
            stats.record_fallback();
            if backend.is_eos(u_hat) {
                break;
            }
            if accepted.len() >= max_new_tokens {
                break;
            }
            let t_step = Instant::now();
            cur_logprobs = {
                let _r = nvtx::Range::new("dsv.step_one");
                backend.step_one(u_hat)?
            };
            timings.step_one += t_step.elapsed();
            timings.step_one_calls += 1;
            continue;
        }

        // 3. Verify the tree in one packed forward pass.
        let nodes = tree.num_nodes() as u32;
        let t_verify = Instant::now();
        let node_logprobs = {
            let _r = nvtx::Range::new("dsv.verify_tree");
            backend.verify_tree(&tree)?
        };
        timings.verify_tree += t_verify.elapsed();
        timings.verify_tree_calls += 1;
        timings.tree_nodes_total += nodes as u64;
        timings.tree_nodes_max = timings.tree_nodes_max.max(nodes);

        // 4. Greedy traversal under the τ test (returns the accepted path
        //    and the *host-side* terminal distribution — only one D2H copy
        //    per verify step now, vs one per traversal step before).
        let t_traverse = Instant::now();
        let (path, term_host) = {
            let _r = nvtx::Range::new("dsv.traverse");
            greedy_traverse(&tree, &node_logprobs, &cur_logprobs, cfg.tau)?
        };
        timings.traverse += t_traverse.elapsed();

        // 5. Commit the path: gather the KV cache so it keeps only the
        //    accepted-path positions.
        let t_commit = Instant::now();
        {
            let _r = nvtx::Range::new("dsv.commit");
            backend.commit_verify(&path)?;
        }
        timings.commit += t_commit.elapsed();

        // 6. Append accepted-path tokens. Note: AAL is recorded as the count of
        //    *draft* tokens accepted in this step (paper §4.2 / Leviathan et
        //    al. 2023), excluding the bonus û.
        let mut path_eos = false;
        for &node_idx in &path {
            let tok = tree.tokens[node_idx];
            accepted.push(tok);
            if backend.is_eos(tok) {
                path_eos = true;
                break;
            }
        }
        stats.record(path.len() as u32);
        if path.is_empty() {
            timings.rejected_tree_calls += 1;
        } else {
            timings.accepted_tree_calls += 1;
        }
        if path_eos || accepted.len() >= max_new_tokens {
            break;
        }

        // 7. Take the greedy bonus token û from the terminal distribution
        //    (already on host from greedy_traverse).
        let (u_hat, _) = argmax_host(&term_host);
        accepted.push(u_hat);
        if backend.is_eos(u_hat) || accepted.len() >= max_new_tokens {
            break;
        }

        // 8. Step once on û to populate KV and seed the next iteration.
        let t_step = Instant::now();
        cur_logprobs = {
            let _r = nvtx::Range::new("dsv.step_one");
            backend.step_one(u_hat)?
        };
        timings.step_one += t_step.elapsed();
        timings.step_one_calls += 1;
    }

    Ok(accepted)
}

fn spec_decode_strict<B: SpecBackend>(
    backend: &mut B,
    drafts: &[Draft],
    initial_logprobs: Tensor,
    max_new_tokens: usize,
    cfg: &DsvConfig,
    stats: &mut AcceptStats,
    timings: &mut SpecDecodeStats,
) -> CandleResult<Vec<u32>> {
    let mut accepted: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut cur_logprobs = initial_logprobs;

    while accepted.len() < max_new_tokens {
        nvtx::mark("dsv.iter_strict");

        let n = accepted.len().min(cfg.window_len);
        let tail = &accepted[accepted.len() - n..];
        let t_build = Instant::now();
        let tree = {
            let _r = nvtx::Range::new("dsv.candidate_build");
            let candidates = collect_candidates(tail, drafts, cfg);
            let candidate_count = candidates.len() as u32;
            timings.candidate_steps += 1;
            timings.candidates_total += candidate_count as u64;
            timings.candidates_max = timings.candidates_max.max(candidate_count);
            build_prefix_tree(&candidates)
        };
        timings.candidate_build += t_build.elapsed();

        let mut step_accepted = 0u32;
        let mut s: Option<usize> = None;
        loop {
            let children = tree.children_of(s);
            if children.is_empty() || accepted.len() >= max_new_tokens {
                break;
            }

            let t_traverse = Instant::now();
            let best_child = {
                let _r = nvtx::Range::new("dsv.traverse_strict");
                let cur_host = lp_to_host(&cur_logprobs)?;
                let (u_hat, _) = argmax_host(&cur_host);
                children
                    .iter()
                    .copied()
                    .find(|&child| tree.tokens[child] == u_hat)
            };
            timings.traverse += t_traverse.elapsed();

            let Some(child) = best_child else {
                break;
            };

            let tok = tree.tokens[child];
            accepted.push(tok);
            step_accepted += 1;
            if backend.is_eos(tok) || accepted.len() >= max_new_tokens {
                break;
            }
            let t_step = Instant::now();
            cur_logprobs = {
                let _r = nvtx::Range::new("dsv.step_one");
                backend.step_one(tok)?
            };
            timings.step_one += t_step.elapsed();
            timings.step_one_calls += 1;
            s = Some(child);
        }

        if step_accepted > 0 {
            timings.accepted_tree_calls += 1;
            stats.record(step_accepted);
            if accepted
                .last()
                .copied()
                .is_some_and(|tok| backend.is_eos(tok))
                || accepted.len() >= max_new_tokens
            {
                break;
            }
            continue;
        }

        if tree.is_empty() {
            timings.empty_tree_calls += 1;
        } else {
            timings.rejected_tree_calls += 1;
        }
        let t_argmax = Instant::now();
        let u_hat = {
            let _r = nvtx::Range::new("dsv.fallback_argmax");
            argmax_on_device(&cur_logprobs)?
        };
        timings.fallback_argmax += t_argmax.elapsed();
        timings.fallback_argmax_calls += 1;
        accepted.push(u_hat);
        stats.record_fallback();
        if backend.is_eos(u_hat) || accepted.len() >= max_new_tokens {
            break;
        }
        let t_step = Instant::now();
        cur_logprobs = {
            let _r = nvtx::Range::new("dsv.step_one");
            backend.step_one(u_hat)?
        };
        timings.step_one += t_step.elapsed();
        timings.step_one_calls += 1;
    }

    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Deterministic oracle backend used to validate the driver in isolation
    /// from any model. Logprobs are concentrated (~1.0) at the next "true"
    /// token, so:
    /// - Drafts that match the oracle are fully accepted.
    /// - Drafts that diverge are rejected and the driver falls through to
    ///   single-token decoding.
    struct OracleBackend {
        vocab: usize,
        eos: u32,
        oracle: Vec<u32>,
        /// Number of post-prefill tokens already "decoded" (== KV positions
        /// past the prefill prefix). The next step_one / verify_tree result
        /// derives from `oracle[position + d]` for some depth d ≥ 1.
        position: usize,
        /// Tally of `step_one` and `verify_tree` calls — useful in tests to
        /// confirm parallel verification actually saved forward passes.
        n_step_one: usize,
        n_verify_tree: usize,
    }

    impl OracleBackend {
        fn new(vocab: usize, eos: u32, oracle: Vec<u32>) -> Self {
            Self {
                vocab,
                eos,
                oracle,
                position: 0,
                n_step_one: 0,
                n_verify_tree: 0,
            }
        }

        fn lp_for(&self, tok: u32) -> Tensor {
            // Concentrated distribution: target token ≈ log 1, others ≈ log 0.
            let mut v = vec![-100.0f32; self.vocab];
            v[tok as usize] = 0.0;
            Tensor::from_vec(v, (self.vocab,), &Device::Cpu).unwrap()
        }

        fn next_oracle(&self, offset: usize) -> u32 {
            self.oracle
                .get(self.position + offset)
                .copied()
                .unwrap_or(self.eos)
        }
    }

    impl SpecBackend for OracleBackend {
        fn step_one(&mut self, _token: u32) -> CandleResult<Tensor> {
            self.n_step_one += 1;
            self.position += 1;
            let nxt = self.next_oracle(0);
            Ok(self.lp_for(nxt))
        }

        fn verify_tree(&mut self, tree: &PrefixTree) -> CandleResult<Tensor> {
            self.n_verify_tree += 1;
            let n = tree.num_nodes();
            let mut buf = vec![-100.0f32; n * self.vocab];
            for i in 0..n {
                let depth = tree.depths[i] as usize;
                let nxt = self.next_oracle(depth);
                buf[i * self.vocab + nxt as usize] = 0.0;
            }
            Tensor::from_vec(buf, (n, self.vocab), &Device::Cpu)
        }

        fn commit_verify(&mut self, path: &[usize]) -> CandleResult<()> {
            self.position += path.len();
            Ok(())
        }

        fn is_eos(&self, tok: u32) -> bool {
            tok == self.eos
        }
    }

    fn cfg() -> DsvConfig {
        DsvConfig::default()
    }

    fn run_spec_decode<B: SpecBackend>(
        backend: &mut B,
        drafts: &[Draft],
        initial_logprobs: Tensor,
        max_new_tokens: usize,
        cfg: &DsvConfig,
        stats: &mut AcceptStats,
    ) -> CandleResult<Vec<u32>> {
        let mut timings = SpecDecodeStats::default();
        spec_decode(
            backend,
            drafts,
            initial_logprobs,
            max_new_tokens,
            cfg,
            stats,
            &mut timings,
        )
    }

    #[test]
    fn perfect_match_accepts_full_path() {
        let oracle = vec![10u32, 20, 30, 40, 50, 99];
        let mut backend = OracleBackend::new(128, 99, oracle.clone());
        let init_lp = backend.lp_for(oracle[0]);
        let drafts = vec![Draft::new(oracle.clone())];
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 64, &cfg(), &mut stats).unwrap();

        assert_eq!(out, oracle);
        // One verify call accepted the entire chain (including EOS).
        assert_eq!(backend.n_verify_tree, 1);
        // step_one is never invoked because EOS was reached inside the path.
        assert_eq!(backend.n_step_one, 0);
        assert!(stats.aal() > 0.0);
    }

    #[test]
    fn no_match_falls_back_to_step_one() {
        let oracle = vec![10u32, 20, 30, 40, 99];
        let mut backend = OracleBackend::new(128, 99, oracle.clone());
        let init_lp = backend.lp_for(oracle[0]);
        // Drafts that share no tokens with the oracle.
        let drafts = vec![Draft::new(vec![1, 2, 3, 4])];
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 64, &cfg(), &mut stats).unwrap();

        assert_eq!(out, oracle);
        // Step 0: empty-tail tree from the draft → all rejected → fallback.
        // Subsequent steps: tail = [oracle_i], no match in [1,2,3,4] →
        // tree empty → fallback, so every subsequent token also falls back.
        assert!(backend.n_step_one >= oracle.len() - 1);
        assert!(stats.num_fallbacks >= (oracle.len() as u32) - 1);
    }

    #[test]
    fn partial_match_accepts_prefix_then_diverges() {
        // Oracle starts the same as the draft, then diverges.
        let oracle = vec![10u32, 20, 30, 77, 88, 99];
        let draft_tokens = vec![10u32, 20, 30, 40, 50];
        let mut backend = OracleBackend::new(128, 99, oracle.clone());
        let init_lp = backend.lp_for(oracle[0]);
        let drafts = vec![Draft::new(draft_tokens)];
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 64, &cfg(), &mut stats).unwrap();
        assert_eq!(out, oracle);
        // First verify accepts [10, 20, 30] (depths 1..=3 match the oracle),
        // then the depth-4 child carries token 40 which the oracle rejects.
        // After that, the accepted-tail [_,_,30] no longer matches the draft
        // (the draft has 30 at index 2, but its trailing window [10,20,30]
        // only appears once with suffix [40,50]; reject again at first child).
        assert!(backend.n_verify_tree >= 1);
        // AAL across all steps should reflect the early acceptance.
        assert!(stats.aal() > 0.0);
    }

    #[test]
    fn max_new_tokens_caps_output() {
        let oracle = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut backend = OracleBackend::new(64, 63, oracle.clone());
        let init_lp = backend.lp_for(oracle[0]);
        let drafts = vec![Draft::new(oracle.clone())];
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 4, &cfg(), &mut stats).unwrap();
        assert!(out.len() >= 4 && out.len() <= 10);
        assert_eq!(&out[..4], &oracle[..4]);
    }

    #[test]
    fn eos_in_initial_logprobs_short_circuits_via_fallback() {
        // Initial logprobs concentrated at EOS: with a draft starting at EOS,
        // the first verify accepts EOS as path[0] and we exit immediately.
        let mut backend = OracleBackend::new(32, 31, vec![31]);
        let init_lp = backend.lp_for(31);
        let drafts = vec![Draft::new(vec![31])];
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 64, &cfg(), &mut stats).unwrap();
        assert_eq!(out, vec![31]);
    }

    #[test]
    fn multiple_drafts_get_merged_in_tree() {
        // Two drafts sharing a prefix; the longer one should be accepted.
        let oracle = vec![10u32, 20, 30, 40, 50, 99];
        let mut backend = OracleBackend::new(128, 99, oracle.clone());
        let init_lp = backend.lp_for(oracle[0]);
        let drafts = vec![
            Draft::new(vec![10, 20, 30]),
            Draft::new(vec![10, 20, 30, 40, 50, 99]),
        ];
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 64, &cfg(), &mut stats).unwrap();
        assert_eq!(out, oracle);
        assert_eq!(backend.n_verify_tree, 1);
    }

    #[test]
    fn tau_one_is_strict() {
        // With τ = 1.0 and a draft containing the oracle, acceptance still
        // succeeds because cur_lp(u*) == cur_lp(û) (both 0.0) so the test
        // (0 - 0 >= log 1 = 0) just barely passes.
        let oracle = vec![5u32, 6, 7, 99];
        let mut backend = OracleBackend::new(128, 99, oracle.clone());
        let init_lp = backend.lp_for(oracle[0]);
        let drafts = vec![Draft::new(oracle.clone())];
        let cfg = DsvConfig {
            tau: 1.0,
            ..DsvConfig::default()
        };
        let mut stats = AcceptStats::default();
        let out = run_spec_decode(&mut backend, &drafts, init_lp, 32, &cfg, &mut stats).unwrap();
        assert_eq!(out, oracle);
    }
}
