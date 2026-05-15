//! Draft-target matching with a sliding reference window (paper §3.2).
//!
//! Given the target VLM's accepted token sequence `ŷ_{1:t}` and a set of fixed
//! drafts `Ỹ`, this module finds every position in every draft where the last
//! `n` accepted tokens reappear, and extracts the suffix that strictly follows
//! each such match. The collected suffixes form the candidate set `C` consumed
//! by the [`super::prefix_tree`] builder.

use super::types::{Draft, DsvConfig};

/// A candidate suffix extracted from one draft after a successful window match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Index of the source draft inside the input slice (debug / introspection).
    pub draft_idx: usize,
    /// Position in the source draft where the *suffix* starts (i.e. `j + n` in
    /// the paper's notation). Mostly useful for diagnostics.
    pub suffix_start: usize,
    /// Token sequence following the matched window.
    pub tokens: Vec<u32>,
}

/// Slide a length-`n` reference window over each draft and collect every suffix
/// that follows a match.
///
/// The effective window length is `min(accepted_tail.len(), cfg.window_len)`,
/// so this also works during the first few generation steps when the accepted
/// history is shorter than `n`. If `accepted_tail` is empty, every draft's
/// leading prefix becomes a candidate (interpreted as a zero-length window
/// matching the empty string at position 0).
///
/// Each candidate is truncated to at most `cfg.max_suffix_len` tokens. The
/// total number of candidates returned is capped at
/// `cfg.max_candidates_per_step`. When the cap is hit, half of the budget keeps
/// early scan-order candidates and the rest keeps the longest remaining
/// suffixes. This preserves some draft/position diversity while still favoring
/// paths that provide more parallel verification headroom.
pub fn collect_candidates(
    accepted_tail: &[u32],
    accepted_len: usize,
    drafts: &[Draft],
    cfg: &DsvConfig,
) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();

    let win_len = accepted_tail.len().min(cfg.window_len);
    let window = &accepted_tail[accepted_tail.len() - win_len..];

    for (di, draft) in drafts.iter().enumerate() {
        if draft.tokens.len() <= win_len {
            // Even with a perfect match there'd be no suffix to extract.
            continue;
        }

        if win_len == 0 {
            // No accepted history. Paper's formulation has no match with an
            // empty window → no candidates at step 0 (driver falls back to
            // step_one for the first token). `cold_start_full_draft = true`
            // (default) softens this by emitting each draft's leading prefix
            // as a single candidate so the very first step still has tree
            // material to verify.
            if cfg.cold_start_full_draft {
                let take = cfg.max_suffix_len.min(draft.tokens.len());
                if take > 0 {
                    out.push(Candidate {
                        draft_idx: di,
                        suffix_start: 0,
                        tokens: draft.tokens[..take].to_vec(),
                    });
                }
            }
            continue;
        }

        // Naive O(|draft| * n) scan. The drafts here are page-level Markdown,
        // typically a few thousand tokens, so this is comfortably fast. If a
        // profile ever shows it as a hotspot, switch to a rolling-hash search.
        let last_start = draft.tokens.len() - win_len;
        let mut j = 0;
        while j <= last_start {
            if &draft.tokens[j..j + win_len] == window {
                let suf_start = j + win_len;
                if suf_start < draft.tokens.len() {
                    let take = cfg.max_suffix_len.min(draft.tokens.len() - suf_start);
                    if take > 0 {
                        out.push(Candidate {
                            draft_idx: di,
                            suffix_start: suf_start,
                            tokens: draft.tokens[suf_start..suf_start + take].to_vec(),
                        });
                    }
                }
            }
            j += 1;
        }
    }

    cap_candidates(out, accepted_len, cfg.max_candidates_per_step)
}

fn cap_candidates(
    out: Vec<Candidate>,
    accepted_len: usize,
    max_candidates: usize,
) -> Vec<Candidate> {
    if out.len() <= max_candidates {
        return out;
    }
    if max_candidates == 0 {
        return Vec::new();
    }

    // Position-aware cap: prefer candidates whose match position is closest to
    // the **current decode point** (`accepted_len`). For an oracle / aligned
    // draft, `suffix_start == accepted_len` exactly — and that's the only
    // candidate whose suffix is the correct continuation. Both the previous
    // "earliest by scan order" and the (briefly-shipped) "latest by scan
    // order" policies were wrong in opposite ways: earliest dropped the
    // current-position match in favor of stale repeats from before; latest
    // dropped it in favor of stale repeats from after. Distance to
    // `accepted_len` is the only signal that works across patterns.
    //
    // Reserve a smaller "long-suffix diversity" slot in case the ideal-position
    // match happens to have a short or empty suffix (e.g. the 3-gram appears
    // near the end of the draft and there's not much past it). With a 128-cap,
    // that's 96 by position + 32 by suffix length.
    let position_quota = max_candidates.saturating_sub(max_candidates / 4).max(1);
    let long_quota = max_candidates - position_quota;
    let mut selected = vec![false; out.len()];
    let mut capped: Vec<Candidate> = Vec::with_capacity(max_candidates);

    let mut by_distance: Vec<usize> = (0..out.len()).collect();
    by_distance.sort_by(|&a, &b| {
        let da = out[a].suffix_start.abs_diff(accepted_len);
        let db = out[b].suffix_start.abs_diff(accepted_len);
        da.cmp(&db)
            .then_with(|| out[b].tokens.len().cmp(&out[a].tokens.len()))
            .then_with(|| a.cmp(&b))
    });
    for &idx in by_distance.iter().take(position_quota) {
        selected[idx] = true;
        capped.push(out[idx].clone());
    }

    let mut remaining: Vec<usize> = (0..out.len()).filter(|&idx| !selected[idx]).collect();
    remaining.sort_by(|&a, &b| {
        out[b]
            .tokens
            .len()
            .cmp(&out[a].tokens.len())
            .then_with(|| a.cmp(&b))
    });

    for idx in remaining.into_iter().take(long_quota) {
        capped.push(out[idx].clone());
    }

    capped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(tokens: &[u32]) -> Draft {
        Draft::new(tokens.to_vec())
    }

    fn cfg() -> DsvConfig {
        DsvConfig {
            window_len: 3,
            tau: 0.75,
            max_candidates_per_step: 32,
            max_suffix_len: 256,
            ..Default::default()
        }
    }

    #[test]
    fn empty_inputs() {
        assert!(collect_candidates(&[], 0, &[], &cfg()).is_empty());
        assert!(collect_candidates(&[1, 2, 3], 3, &[], &cfg()).is_empty());
    }

    #[test]
    fn empty_draft_skipped() {
        let drafts = vec![d(&[])];
        assert!(collect_candidates(&[1, 2, 3], 3, &drafts, &cfg()).is_empty());
    }

    #[test]
    fn single_match_extracts_suffix() {
        let drafts = vec![d(&[10, 20, 30, 1, 2, 3, 40, 50, 60])];
        let got = collect_candidates(&[1, 2, 3], 3, &drafts, &cfg());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].draft_idx, 0);
        assert_eq!(got[0].suffix_start, 6);
        assert_eq!(got[0].tokens, vec![40, 50, 60]);
    }

    #[test]
    fn no_match_no_candidates() {
        let drafts = vec![d(&[10, 20, 30, 40, 50])];
        assert!(collect_candidates(&[1, 2, 3], 3, &drafts, &cfg()).is_empty());
    }

    #[test]
    fn match_at_end_yields_no_suffix() {
        let drafts = vec![d(&[10, 1, 2, 3])];
        // window matches at position 1, but there's no token after it.
        assert!(collect_candidates(&[1, 2, 3], 3, &drafts, &cfg()).is_empty());
    }

    #[test]
    fn multiple_matches_same_draft() {
        let drafts = vec![d(&[1, 2, 3, 7, 1, 2, 3, 8, 9])];
        let got = collect_candidates(&[1, 2, 3], 3, &drafts, &cfg());
        // Two matches at positions 0 and 4, two distinct suffixes.
        let suffixes: Vec<_> = got.iter().map(|c| c.tokens.clone()).collect();
        assert!(suffixes.contains(&vec![7, 1, 2, 3, 8, 9]));
        assert!(suffixes.contains(&vec![8, 9]));
    }

    #[test]
    fn matches_across_multiple_drafts() {
        let drafts = vec![d(&[1, 2, 3, 4, 5]), d(&[9, 1, 2, 3, 6])];
        let got = collect_candidates(&[1, 2, 3], 3, &drafts, &cfg());
        assert_eq!(got.len(), 2);
        let by_draft: Vec<_> = got
            .iter()
            .map(|c| (c.draft_idx, c.tokens.clone()))
            .collect();
        assert!(by_draft.contains(&(0, vec![4, 5])));
        assert!(by_draft.contains(&(1, vec![6])));
    }

    #[test]
    fn shorter_window_when_history_short() {
        // accepted_tail shorter than cfg.window_len: window shrinks gracefully.
        let drafts = vec![d(&[7, 1, 2, 3, 4])];
        let got = collect_candidates(&[1], 1, &drafts, &cfg());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].tokens, vec![2, 3, 4]);
    }

    #[test]
    fn empty_history_uses_each_draft_prefix() {
        let cfg = DsvConfig {
            max_suffix_len: 3,
            ..cfg()
        };
        let drafts = vec![d(&[10, 11, 12, 13]), d(&[20, 21])];
        let got = collect_candidates(&[], 0, &drafts, &cfg);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].tokens, vec![10, 11, 12]); // truncated by max_suffix_len
        assert_eq!(got[1].tokens, vec![20, 21]);
    }

    #[test]
    fn suffix_length_capped() {
        let cfg = DsvConfig {
            max_suffix_len: 2,
            ..cfg()
        };
        let drafts = vec![d(&[1, 2, 3, 7, 8, 9, 10, 11])];
        let got = collect_candidates(&[1, 2, 3], 3, &drafts, &cfg);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].tokens, vec![7, 8]);
    }

    #[test]
    fn cap_prefers_match_at_current_decode_position() {
        // Position-aware cap: with `accepted_len = 100`, the candidate whose
        // `suffix_start` is closest to 100 wins the position-quota slot, even
        // if it's not the longest or first/last in scan order. This is the
        // 2026-05-14 fix that replaced the earlier "head" and "latest" cap
        // heuristics — both were wrong for opposite reasons (each dropped
        // the current-position match in favor of stale repeats on the wrong
        // side). For oracle / well-aligned drafts, the current decode point
        // is the only signal that picks the correct continuation across
        // patterns.
        //
        // Setup: a single 200-token draft where the 3-gram `[1, 2, 3]`
        // appears at positions 0, 50, 100, 150 — each followed by a unique
        // marker (10, 50, 100, 150). With cap=1 and accepted_len=103
        // (suffix would start at draft position 103), the match at
        // suffix_start=103 (closest to 103) must win.
        let cfg = DsvConfig {
            max_candidates_per_step: 1,
            max_suffix_len: 10,
            ..cfg()
        };
        let mut tokens = vec![99u32; 200];
        // place markers at positions: window starts at 0, 50, 100, 150
        // so suffix_starts are 3, 53, 103, 153.
        for (i, &start) in [0usize, 50, 100, 150].iter().enumerate() {
            tokens[start] = 1;
            tokens[start + 1] = 2;
            tokens[start + 2] = 3;
            // distinctive marker as the first suffix token at start+3.
            tokens[start + 3] = 1000 + (i as u32);
        }
        let drafts = vec![d(&tokens)];
        // accepted_len = 103 means the current decode point is draft pos 103.
        // Closest match suffix_start = 103 → marker 1002.
        let got = collect_candidates(&[1, 2, 3], 103, &drafts, &cfg);
        assert_eq!(got.len(), 1, "cap=1 should keep exactly one candidate");
        assert_eq!(
            got[0].tokens.first(),
            Some(&1002),
            "must keep match closest to accepted_len=103, got tokens={:?}",
            got[0].tokens
        );
    }

    #[test]
    fn cap_at_repeated_3gram_keeps_current_position() {
        // Regression test (refined 2026-05-14). Pattern `[X, Y, Z]` repeats
        // 20 times in a row, each followed by a unique marker. With
        // `accepted_len` set to the position WHERE WE ARE in the draft (here
        // simulating "we just accepted up to and including marker 9, so we
        // expect marker 10 next"), the position-aware cap must keep the
        // marker-10 match, not the earliest (marker 0 — original head bug)
        // or the latest (marker 19 — the intermediate "latest" bug).
        let cfg = DsvConfig {
            max_candidates_per_step: 1,
            ..cfg()
        };
        let mut tokens: Vec<u32> = Vec::new();
        for i in 0..20u32 {
            tokens.extend_from_slice(&[100, 101, 102, 200 + i]);
        }
        let drafts = vec![d(&tokens)];
        // Each unit is 4 tokens. The 3-gram at unit i starts at position
        // 4*i; its suffix begins at position 4*i + 3 (marker 200 + i).
        // Simulating "we have accepted up to position 4*10 + 3 = 43, so the
        // next correct token is marker 210 at suffix_start = 43."
        let got = collect_candidates(&[100, 101, 102], 43, &drafts, &cfg);
        assert_eq!(got.len(), 1);
        assert_eq!(
            got[0].tokens.first(),
            Some(&210),
            "must keep current-position match (marker 210), got {:?}",
            got[0].tokens
        );
    }

    #[test]
    fn candidate_count_zero_drops_all_candidates() {
        let cfg = DsvConfig {
            max_candidates_per_step: 0,
            ..cfg()
        };
        let drafts = vec![d(&[1, 2, 3, 9])];
        assert!(collect_candidates(&[1, 2, 3], 3, &drafts, &cfg).is_empty());
    }

    #[test]
    fn cold_start_disabled_yields_no_candidates() {
        // With cold_start_full_draft = false and an empty accepted tail the
        // matcher must produce *no* candidates.
        let cfg = DsvConfig {
            cold_start_full_draft: false,
            ..cfg()
        };
        let drafts = vec![d(&[10, 11, 12, 13])];
        assert!(collect_candidates(&[], 0, &drafts, &cfg).is_empty());
    }
}
