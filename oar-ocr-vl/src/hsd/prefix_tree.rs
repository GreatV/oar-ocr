//! Prefix tree for parallel verification of multiple candidate suffixes
//! (paper §3.2 / Fig. 2b).
//!
//! Each candidate suffix produced by [`super::matching::collect_candidates`] is
//! inserted into a tree that shares common prefixes. Common prefixes are merged
//! so a single packed forward pass through the model can verify all candidates
//! at once (under a tree-ancestry attention mask, built in [`super::verify`]).
//!
//! ## Indexing convention
//!
//! - The root is implicit: it carries no token, has no numeric node id, and is
//!   represented as parent `None`.
//! - All `Vec`s in [`PrefixTree`] are indexed by non-root node id starting at
//!   0 — i.e. `tokens[0]` is the first inserted child of the root.
//! - `parents[i] = None` means the parent is the root.

use super::matching::Candidate;
use std::collections::HashMap;

/// A flattened prefix tree, ready to be turned into a packed token sequence
/// plus a tree-ancestry attention mask.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixTree {
    /// Token id at each non-root node.
    pub tokens: Vec<u32>,
    /// Parent index for each non-root node. `None` ⇒ child of root.
    pub parents: Vec<Option<usize>>,
    /// Distance from root, counted in tokens (root-children have depth 1).
    pub depths: Vec<u32>,
    /// `leaf_for[i] = Some((cand_idx, depth))` if a candidate ends exactly at
    /// node `i`. If multiple candidates end here, the deepest candidate wins;
    /// equal-depth ties keep the first inserted candidate.
    pub leaf_for: Vec<Option<(usize, u32)>>,
}

impl PrefixTree {
    pub fn num_nodes(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Walk from `node` up to the root, collecting tokens in root → node order.
    /// Useful for diagnostics and for reconstructing accepted segments after
    /// greedy traversal.
    pub fn path_tokens(&self, node: usize) -> Vec<u32> {
        let mut path = Vec::with_capacity(self.depths[node] as usize);
        let mut cur = Some(node);
        while let Some(i) = cur {
            path.push(self.tokens[i]);
            cur = self.parents[i];
        }
        path.reverse();
        path
    }

    /// Indices of `node`'s direct children.
    ///
    /// This is a linear scan over the flattened tree. Tree sizes are bounded by
    /// `DsvConfig::{max_candidates_per_step,max_suffix_len}`; if larger trees
    /// become common, cache adjacency lists during construction.
    pub fn children_of(&self, node: Option<usize>) -> Vec<usize> {
        self.parents
            .iter()
            .enumerate()
            .filter_map(|(i, p)| if *p == node { Some(i) } else { None })
            .collect()
    }
}

/// Build a [`PrefixTree`] from candidate suffixes.
///
/// Candidates are inserted in the order given. Duplicate paths collapse onto
/// the same set of nodes. A candidate that is a prefix of another marks its
/// terminal node as a leaf without breaking the longer candidate's path.
pub fn build_prefix_tree(candidates: &[Candidate]) -> PrefixTree {
    let mut tree = PrefixTree::default();
    // (parent_node, token) -> child_node, where parent_node == None means root.
    let mut child_map: HashMap<(Option<usize>, u32), usize> = HashMap::new();

    for (cand_idx, cand) in candidates.iter().enumerate() {
        let mut parent: Option<usize> = None;
        let mut depth: u32 = 0;
        for &tok in &cand.tokens {
            depth += 1;
            let node = match child_map.get(&(parent, tok)) {
                Some(&existing) => existing,
                None => {
                    let new_idx = tree.tokens.len();
                    tree.tokens.push(tok);
                    tree.parents.push(parent);
                    tree.depths.push(depth);
                    tree.leaf_for.push(None);
                    child_map.insert((parent, tok), new_idx);
                    new_idx
                }
            };
            parent = Some(node);
        }
        // Record this candidate's terminal node — deepest wins, equal-depth
        // duplicates keep the first inserted candidate.
        if let Some(end) = parent {
            let prev = tree.leaf_for[end];
            let new_depth = tree.depths[end];
            tree.leaf_for[end] = match prev {
                Some((_, d)) if d >= new_depth => prev,
                _ => Some((cand_idx, new_depth)),
            };
        }
    }

    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(idx: usize, toks: &[u32]) -> Candidate {
        Candidate {
            draft_idx: idx,
            suffix_start: 0,
            tokens: toks.to_vec(),
        }
    }

    #[test]
    fn empty_input() {
        let t = build_prefix_tree(&[]);
        assert!(t.is_empty());
        assert_eq!(t.num_nodes(), 0);
    }

    #[test]
    fn single_candidate_is_a_chain() {
        let t = build_prefix_tree(&[c(0, &[7, 8, 9])]);
        assert_eq!(t.tokens, vec![7, 8, 9]);
        assert_eq!(t.parents, vec![None, Some(0), Some(1)]);
        assert_eq!(t.depths, vec![1, 2, 3]);
        assert_eq!(t.leaf_for, vec![None, None, Some((0, 3))]);
    }

    #[test]
    fn shared_prefix_merges() {
        let t = build_prefix_tree(&[c(0, &[1, 2, 3]), c(1, &[1, 2, 4])]);
        // Nodes (in insertion order): 1 (root child), 2, 3 (leaf for cand 0), 4 (leaf for cand 1)
        assert_eq!(t.tokens, vec![1, 2, 3, 4]);
        assert_eq!(t.parents, vec![None, Some(0), Some(1), Some(1)]);
        assert_eq!(t.depths, vec![1, 2, 3, 3]);
        assert_eq!(t.leaf_for, vec![None, None, Some((0, 3)), Some((1, 3))]);
    }

    #[test]
    fn candidate_that_is_prefix_of_another() {
        let t = build_prefix_tree(&[c(0, &[1, 2]), c(1, &[1, 2, 3])]);
        // Node 1 (token 2) is a leaf for cand 0; node 2 (token 3) is a leaf for cand 1.
        assert_eq!(t.tokens, vec![1, 2, 3]);
        assert_eq!(t.parents, vec![None, Some(0), Some(1)]);
        assert_eq!(t.leaf_for[1], Some((0, 2)));
        assert_eq!(t.leaf_for[2], Some((1, 3)));
    }

    #[test]
    fn duplicate_candidate_collapses() {
        let t = build_prefix_tree(&[c(0, &[5, 6]), c(1, &[5, 6])]);
        assert_eq!(t.tokens, vec![5, 6]);
        // Longest-leaf rule keeps cand 0 (first inserted, equal depth).
        assert_eq!(t.leaf_for[1], Some((0, 2)));
    }

    #[test]
    fn path_tokens_reconstructs() {
        let t = build_prefix_tree(&[c(0, &[1, 2, 3])]);
        assert_eq!(t.path_tokens(2), vec![1, 2, 3]);
        assert_eq!(t.path_tokens(0), vec![1]);
    }

    #[test]
    fn children_of_root_and_internal() {
        let t = build_prefix_tree(&[c(0, &[1, 2]), c(1, &[3, 4])]);
        let mut roots = t.children_of(None);
        roots.sort();
        assert_eq!(roots, vec![0, 2]); // first child of each chain
        let mut c1 = t.children_of(Some(0));
        c1.sort();
        assert_eq!(c1, vec![1]);
    }

    #[test]
    fn duplicate_path_keeps_first_leaf() {
        // Same path with two candidates keeps the first id at the shared
        // terminal because the terminal depth is equal.
        let t = build_prefix_tree(&[c(0, &[1, 2]), c(1, &[1, 2])]);
        assert_eq!(t.leaf_for[1], Some((0, 2)));
    }
}
