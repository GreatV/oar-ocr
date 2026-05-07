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
/// `cfg.max_candidates_per_step`. When the cap is hit, *longer* candidates are
/// preferred since they offer more parallel verification headroom.
pub fn collect_candidates(
    accepted_tail: &[u32],
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
            // No accepted history: emit the entire draft prefix once.
            let take = cfg.max_suffix_len.min(draft.tokens.len());
            if take > 0 {
                out.push(Candidate {
                    draft_idx: di,
                    suffix_start: 0,
                    tokens: draft.tokens[..take].to_vec(),
                });
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

    if out.len() > cfg.max_candidates_per_step {
        // Prefer longer candidates: longer = more parallel headroom per step.
        // Stable sort keeps draft-order ties deterministic.
        out.sort_by(|a, b| b.tokens.len().cmp(&a.tokens.len()));
        out.truncate(cfg.max_candidates_per_step);
    }

    out
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
            max_suffix_len: 64,
        }
    }

    #[test]
    fn empty_inputs() {
        assert!(collect_candidates(&[], &[], &cfg()).is_empty());
        assert!(collect_candidates(&[1, 2, 3], &[], &cfg()).is_empty());
    }

    #[test]
    fn empty_draft_skipped() {
        let drafts = vec![d(&[])];
        assert!(collect_candidates(&[1, 2, 3], &drafts, &cfg()).is_empty());
    }

    #[test]
    fn single_match_extracts_suffix() {
        let drafts = vec![d(&[10, 20, 30, 1, 2, 3, 40, 50, 60])];
        let got = collect_candidates(&[1, 2, 3], &drafts, &cfg());
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].draft_idx, 0);
        assert_eq!(got[0].suffix_start, 6);
        assert_eq!(got[0].tokens, vec![40, 50, 60]);
    }

    #[test]
    fn no_match_no_candidates() {
        let drafts = vec![d(&[10, 20, 30, 40, 50])];
        assert!(collect_candidates(&[1, 2, 3], &drafts, &cfg()).is_empty());
    }

    #[test]
    fn match_at_end_yields_no_suffix() {
        let drafts = vec![d(&[10, 1, 2, 3])];
        // window matches at position 1, but there's no token after it.
        assert!(collect_candidates(&[1, 2, 3], &drafts, &cfg()).is_empty());
    }

    #[test]
    fn multiple_matches_same_draft() {
        let drafts = vec![d(&[1, 2, 3, 7, 1, 2, 3, 8, 9])];
        let got = collect_candidates(&[1, 2, 3], &drafts, &cfg());
        // Two matches at positions 0 and 4, two distinct suffixes.
        let suffixes: Vec<_> = got.iter().map(|c| c.tokens.clone()).collect();
        assert!(suffixes.contains(&vec![7, 1, 2, 3, 8, 9]));
        assert!(suffixes.contains(&vec![8, 9]));
    }

    #[test]
    fn matches_across_multiple_drafts() {
        let drafts = vec![d(&[1, 2, 3, 4, 5]), d(&[9, 1, 2, 3, 6])];
        let got = collect_candidates(&[1, 2, 3], &drafts, &cfg());
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
        let got = collect_candidates(&[1], &drafts, &cfg());
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
        let got = collect_candidates(&[], &drafts, &cfg);
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
        let got = collect_candidates(&[1, 2, 3], &drafts, &cfg);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].tokens, vec![7, 8]);
    }

    #[test]
    fn candidate_count_capped_prefers_longer() {
        let cfg = DsvConfig {
            max_candidates_per_step: 2,
            ..cfg()
        };
        let drafts = vec![
            d(&[1, 2, 3, 9]),                // suffix len 1
            d(&[1, 2, 3, 8, 7]),             // suffix len 2
            d(&[1, 2, 3, 6, 5, 4, 3, 2, 1]), // suffix len 6
        ];
        let got = collect_candidates(&[1, 2, 3], &drafts, &cfg);
        assert_eq!(got.len(), 2);
        let lens: Vec<_> = got.iter().map(|c| c.tokens.len()).collect();
        assert!(lens.contains(&6));
        assert!(lens.contains(&2));
    }
}
