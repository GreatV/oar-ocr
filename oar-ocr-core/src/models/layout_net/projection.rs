//! Grid-projection signals for LayoutNet.
//!
//! LayoutNet was trained on text items that LiteParse had already projected onto a page grid: items are grouped into lines and blocks, items sharing a left, right or centre edge form anchor groups, and each item records its block, the size of its anchor groups, the edge it snaps to and whether its block is flowing prose. This module computes those signals. Adapted from LiteParse's `projection.rs` (Apache-2.0). See `processors/NOTICE`.
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default)]
pub struct TextItem {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub rotation: f32,
    pub font_name: Option<String>,
    pub font_size: Option<f32>,
    pub font_height: Option<f32>,
    pub font_weight: Option<i32>,
    pub font_flags: Option<i32>,
    pub font_is_buggy: bool,
    pub has_unicode_map_error: bool,
    pub followed_by_space: bool,
    pub confidence: Option<f32>,
    pub link: Option<String>,
    pub strike: bool,
    pub underline: bool,
    pub sup: bool,
    pub sub: bool,
    pub style_pieces: Vec<TextItem>,
    pub script_geom: Option<(f32, f32)>,
    pub math: bool,
    pub ocr_faint: bool,
}

#[derive(Debug)]
pub struct ProjectedTextItem {
    pub item: TextItem,
    pub is_margin_line_number: bool,
    pub rotated: bool,
    pub d: f32,
    pub orig_x: f32,
    pub orig_y: f32,
    pub orig_width: f32,
    pub orig_height: f32,
}

pub type AnchorMap = HashMap<i32, Vec<(usize, usize)>>;

// Flowing text detection constants
const FLOWING_MAX_TOTAL_ANCHORS: usize = 4;
const FLOWING_MAX_LEFT_ANCHORS: usize = 3;
const FLOWING_MIN_LINES: usize = 3;
const FLOWING_WIDE_LINE_RATIO: f32 = 0.5;
const FLOWING_WIDE_LINE_THRESHOLD: f32 = 0.6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnapKind {
    Left,
    Right,
    Center,
}

struct LineRange {
    start: usize,
    end: usize,
}

fn compute_median_textbox_size(items: &[ProjectedTextItem]) -> (f32, f32) {
    if items.is_empty() {
        return (0.0, 0.0);
    }

    // Match TS behavior: median width is computed as average character width.
    let mut widths: Vec<f32> = items
        .iter()
        .filter_map(|item| {
            if item.item.width <= 0.0 {
                return None;
            }
            let char_len = item.item.text.chars().count();
            if char_len == 0 {
                return None;
            }
            Some(item.item.width / char_len as f32)
        })
        .collect();
    let mut heights: Vec<f32> = items
        .iter()
        .filter_map(|item| {
            if item.item.height > 0.0 {
                Some(item.item.height)
            } else {
                None
            }
        })
        .collect();

    if widths.is_empty() {
        widths.push(1.0);
    }
    if heights.is_empty() {
        heights.push(1.0);
    }

    widths.sort_by(|a, b| a.total_cmp(b));
    heights.sort_by(|a, b| a.total_cmp(b));

    let width_mid = widths.len() / 2;
    let height_mid = heights.len() / 2;

    let median_width = if widths.len().is_multiple_of(2) {
        (widths[width_mid - 1] + widths[width_mid]) / 2.0
    } else {
        widths[width_mid]
    };

    let median_height = if heights.len().is_multiple_of(2) {
        (heights[height_mid - 1] + heights[height_mid]) / 2.0
    } else {
        heights[height_mid]
    };

    (median_width, median_height)
}

fn canonical_rotation(rotation: f32) -> i32 {
    let r = rotation.rem_euclid(360.0);

    let candidates = [0.0f32, 90.0, 180.0, 270.0];
    let mut best = 0.0f32;
    let mut best_delta = f32::INFINITY;
    for c in candidates {
        // Circular angular distance, so rotations just under 360° are treated
        // as near 0° (e.g. 359° is 1° from upright, not 89° from 270°).
        let raw = (r - c).abs();
        let delta = raw.min(360.0 - raw);
        if delta < best_delta {
            best_delta = delta;
            best = c;
        }
    }

    if best_delta <= 2.0 {
        best as i32
    } else {
        r.round() as i32
    }
}

/// Append `right` (the item to the right on the page) to `left` in reading order: after it for
/// a left-to-right pair, before it when the pair reads right to left (Hebrew / Arabic words
/// PDFium emitted as separate runs). Joining in x order reversed RTL word order.
fn join_in_reading_order(left: &mut String, right: &str, sep: &str) {
    if bidi::is_rtl_pieces([left.as_str(), right]) {
        let mut joined = right.trim_end().to_string();
        if !sep.is_empty() && !joined.is_empty() {
            joined.push_str(sep);
        }
        joined.push_str(left.trim_start());
        *left = joined;
    } else {
        if !sep.is_empty() && !left.ends_with(' ') {
            left.push_str(sep);
        }
        left.push_str(right);
    }
}

fn handle_rotation_reading_order(items: &mut [ProjectedTextItem], page_height: f32) {
    if !items
        .iter()
        .any(|b| canonical_rotation(b.item.rotation) != 0)
    {
        return;
    }

    // Group all items by rotation value.
    let mut groups_by_rotation: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, bbox) in items.iter().enumerate() {
        let r = canonical_rotation(bbox.item.rotation);
        groups_by_rotation.entry(r).or_default().push(idx);
    }

    // For 90/270° groups, further split into spatially distinct clusters by y proximity.
    // This prevents e.g. top and bottom pin labels from being merged into one group.
    let mut bbox_groups: Vec<Vec<usize>> = Vec::new();
    for (rot, mut group) in groups_by_rotation {
        group.sort_by(|a, b| items[*a].item.y.total_cmp(&items[*b].item.y));
        if (rot == 90 || rot == 270) && group.len() > 1 {
            // Split when y gap between consecutive items exceeds a threshold
            let max_h = group
                .iter()
                .map(|idx| items[*idx].item.height)
                .fold(0.0f32, |a, b| a.max(b));
            let gap_threshold = max_h * 3.0;
            let mut cluster: Vec<usize> = vec![group[0]];
            for i in 1..group.len() {
                let prev_bottom = items[group[i - 1]].item.y + items[group[i - 1]].item.height;
                let cur_top = items[group[i]].item.y;
                if cur_top - prev_bottom > gap_threshold {
                    bbox_groups.push(std::mem::take(&mut cluster));
                }
                cluster.push(group[i]);
            }
            if !cluster.is_empty() {
                bbox_groups.push(cluster);
            }
        } else {
            bbox_groups.push(group);
        }
    }

    // Sort each subgroup by y.
    for group in &mut bbox_groups {
        group.sort_by(|a, b| items[*a].item.y.total_cmp(&items[*b].item.y));
    }

    // Groups are already y-sorted. Break x ties by their topmost item,
    // then by that item's input index for coincident groups.
    bbox_groups.sort_by(|a, b| {
        let min_x_a = a
            .iter()
            .map(|idx| items[*idx].item.x)
            .fold(f32::INFINITY, |acc, v| acc.min(v));
        let min_x_b = b
            .iter()
            .map(|idx| items[*idx].item.x)
            .fold(f32::INFINITY, |acc, v| acc.min(v));
        min_x_a
            .total_cmp(&min_x_b)
            .then_with(|| items[a[0]].item.y.total_cmp(&items[b[0]].item.y))
            .then_with(|| a[0].cmp(&b[0]))
    });

    for group_idx in 0..bbox_groups.len() {
        let group = bbox_groups[group_idx].clone();
        if group.is_empty() {
            continue;
        }

        let group_rotation = canonical_rotation(items[group[0]].item.rotation);
        if group_rotation != 90 && group_rotation != 270 {
            continue;
        }

        // Long-span guard: a rotated group whose screen-space extent along its
        // reading direction spans a large fraction of the page is almost
        // always sidebar/marginalia (page footers, signature stamps, vertical
        // disclaimers on legal/scanned docs) — never an inline label. Force the
        // separate-block branch so it doesn't get flattened onto body lines.
        //
        // For 90/270° text the reading direction is screen-vertical, so the
        // group's vertical screen span is the relevant axis.
        let group_min_y = group
            .iter()
            .map(|idx| items[*idx].item.y)
            .fold(f32::INFINITY, f32::min);
        let group_max_y = group
            .iter()
            .map(|idx| items[*idx].item.y + items[*idx].item.height)
            .fold(f32::NEG_INFINITY, f32::max);
        let group_vspan = (group_max_y - group_min_y).max(0.0);
        let long_span = page_height > 0.0 && group_vspan > page_height * 0.4;

        // Check if non-rotated/other-rotated items visually overlap or are near this group.
        // Use a proximity margin so rotated labels in diagrams (e.g. pin diagrams)
        // that are close to but don't strictly overlap non-rotated items are kept inline.
        let mut global_overlap = false;
        'outer: for (other_idx, other_bbox) in items.iter().enumerate() {
            let other_rot = canonical_rotation(other_bbox.item.rotation);
            if other_rot == group_rotation {
                continue;
            }

            for group_item_idx in &group {
                if *group_item_idx == other_idx {
                    continue;
                }
                let b = &items[*group_item_idx].item;
                let o = &other_bbox.item;
                // Proximity margin: use the max height of both items
                let margin = b.height.max(o.height);
                // Proper range overlap with margin on y-axis
                let x_overlap = b.x < o.x + o.width && b.x + b.width > o.x;
                let y_overlap = b.y < o.y + o.height + margin && b.y + b.height + margin > o.y;
                if x_overlap && y_overlap {
                    global_overlap = true;
                    break 'outer;
                }
            }
        }

        if global_overlap && !long_span {
            // For 90/270° groups kept inline, compute a common y from the
            // group's average vertical midpoint so all labels land on one row.
            // Keep original w/h to preserve x-spacing between labels.
            if group_rotation == 90 || group_rotation == 270 {
                let avg_cy: f32 = group
                    .iter()
                    .map(|idx| items[*idx].item.y + items[*idx].item.height / 2.0)
                    .sum::<f32>()
                    / group.len() as f32;
                let avg_w: f32 = group.iter().map(|idx| items[*idx].item.width).sum::<f32>()
                    / group.len() as f32;
                let common_y = avg_cy - avg_w / 2.0;

                for idx in &group {
                    if items[*idx].d != 0.0 {
                        items[*idx].item.y += items[*idx].d;
                        items[*idx].d = 0.0;
                    }
                    items[*idx].item.y = common_y;
                    items[*idx].item.height = avg_w;
                    items[*idx].item.rotation = 0.0;
                    items[*idx].rotated = true;
                }
            } else {
                for idx in &group {
                    if items[*idx].d != 0.0 {
                        items[*idx].item.y += items[*idx].d;
                        items[*idx].d = 0.0;
                    }
                    items[*idx].item.rotation = 0.0;
                    items[*idx].rotated = true;
                }
            }
        } else {
            let group_max_x = group
                .iter()
                .map(|idx| items[*idx].item.x + items[*idx].item.width)
                .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));

            let mut delta_y = 0.0f32;
            if group_idx != 0 {
                let previous_group = &bbox_groups[group_idx - 1];
                let previous_group_max_y = previous_group
                    .iter()
                    .map(|idx| items[*idx].item.y + items[*idx].item.height)
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
                delta_y = previous_group_max_y + page_height;
            }

            if group_rotation == 90 {
                for idx in &group {
                    let new_x = items[*idx].item.y.round();
                    let new_y = items[*idx].item.x + delta_y;
                    let new_w = items[*idx].item.height;
                    let new_h = items[*idx].item.width;
                    items[*idx].item.x = new_x;
                    items[*idx].item.y = new_y;
                    items[*idx].item.width = new_w;
                    items[*idx].item.height = new_h;
                    items[*idx].item.rotation = 0.0;
                    items[*idx].rotated = true;
                }
            }

            if group_rotation == 270 {
                let max_y = group
                    .iter()
                    .map(|idx| items[*idx].item.y + items[*idx].item.height)
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(v));
                for idx in &group {
                    let new_x = (max_y - items[*idx].item.y - items[*idx].item.height).round();
                    let new_y = items[*idx].item.x + delta_y;
                    let new_w = items[*idx].item.height;
                    let new_h = items[*idx].item.width;
                    items[*idx].item.x = new_x;
                    items[*idx].item.y = new_y;
                    items[*idx].item.width = new_w;
                    items[*idx].item.height = new_h;
                    items[*idx].item.rotation = 0.0;
                    items[*idx].rotated = true;
                }
            }

            // Push later groups just below the block. `delta_y` already includes
            // earlier pushes, so pushing by the block's absolute bottom compounds them.
            let block_bottom = delta_y + group_max_x;
            let later_top = bbox_groups[(group_idx + 1)..]
                .iter()
                .flatten()
                .map(|idx| items[*idx].item.y + items[*idx].d)
                .fold(f32::INFINITY, f32::min);
            let global_delta = if later_top.is_finite() {
                (block_bottom + page_height - later_top).max(0.0)
            } else {
                0.0
            };
            for other_group in &bbox_groups[(group_idx + 1)..] {
                for idx in other_group {
                    let rot = canonical_rotation(items[*idx].item.rotation);
                    if rot == 90 || rot == 270 {
                        items[*idx].d += global_delta;
                    } else {
                        items[*idx].item.y += global_delta;
                    }
                }
            }
        }
    }

    // Handle 180-degree rotation conservatively.
    // Unlike TS, we don't have extractor-provided rx/ry fields, so normalize to unrotated
    // and preserve local ordering by x.
    for group in &bbox_groups {
        if group.is_empty() {
            continue;
        }
        let rotation = canonical_rotation(items[group[0]].item.rotation);
        if rotation == 180 {
            let mut sorted = group.clone();
            sorted.sort_by(|a, b| items[*a].item.x.total_cmp(&items[*b].item.x));
            for idx in sorted {
                items[idx].item.rotation = 0.0;
                items[idx].rotated = true;
            }
        }
    }

    items.sort_by(|a, b| a.item.y.total_cmp(&b.item.y));
}

// Three distinct baselines support native gutter numbering. This recurrence
// threshold is a layout heuristic. Isolated footers are exempted independently.
const MIN_GUTTER_ROWS: usize = 3;

/// Candidate numbers use the existing two-column gutter tolerances: their center
/// is strictly between 5 pt left and 20 pt right of the page midpoint, and their
/// width is below 15 pt. These are layout heuristics, not PDF format limits.
///
/// Keep one or two ASCII digits, optionally followed by an OCR `O` for zero.
/// The three-character limit allows that suffix after two digits. Footer
/// detection below uses the same horizontal band but requires actual digits.
fn is_gutter_number_candidate(item: &ProjectedTextItem, page_width: f32) -> bool {
    let center = item.item.x + item.item.width * 0.5;
    let chars: Vec<_> = item.item.text.trim().chars().collect();
    if !(center > page_width * 0.5 - 5.0
        && center < page_width * 0.5 + 20.0
        && item.item.width < 15.0)
        || chars.is_empty()
        || chars.len() > 3
    {
        return false;
    }
    let mut digits = 0;
    for (idx, c) in chars.iter().enumerate() {
        if c.is_ascii_digit() {
            digits += 1;
        } else if *c != 'O' || idx != chars.len() - 1 {
            return false;
        }
    }
    (1..=2).contains(&digits)
}

/// Text on both sides of a separated number is direct evidence that it occupies
/// the space between columns, even without a repeated vertical sequence.
/// Half the smaller glyph height is the existing same-baseline tolerance; using
/// the smaller height prevents a tall font box from spanning adjacent rows.
fn between_text_on_baseline(items: &[ProjectedTextItem], idx: usize) -> bool {
    let number = &items[idx].item;
    let mut left = false;
    let mut right = false;
    for (j, other) in items.iter().enumerate() {
        let text = &other.item;
        if j == idx
            || text.text.trim().is_empty()
            || (text.y - number.y).abs() > number.height.min(text.height) * 0.5
        {
            continue;
        }
        left |= text.x + text.width <= number.x;
        right |= text.x >= number.x + number.width;
    }
    left && right
}

fn clean_projected_items(items: &mut Vec<ProjectedTextItem>, page_width: f32, page_height: f32) {
    // Apply the native recurrence requirement after grid rendering. Preserve
    // the earlier OCR/baseline flags and the existing same-line body-text guard.

    let mut has_non_margin_by_line: HashMap<i32, bool> = HashMap::new();
    for item in items.iter() {
        let line_key = item.item.y.round() as i32;
        if !item.is_margin_line_number {
            has_non_margin_by_line.insert(line_key, true);
        }
    }

    let likely_margin = |item: &ProjectedTextItem| {
        item.is_margin_line_number || is_gutter_number_candidate(item, page_width)
    };
    // Keep the gutter heuristic's row count unchanged for body numbers.
    // Isolated footers are exempted at removal, regardless of that count.
    let rows: HashSet<_> = items
        .iter()
        .filter(|item| likely_margin(item))
        .map(|item| item.item.y.round() as i32)
        .collect();
    let footers: HashSet<_> = (0..items.len())
        .filter(|&i| isolated_numeric_footer(items, i, page_width, page_height))
        .collect();
    let mut idx = 0;
    items.retain(|item| {
        let footer = footers.contains(&idx);
        idx += 1;
        let line_has_content = has_non_margin_by_line
            .get(&(item.item.y.round() as i32))
            .copied()
            .unwrap_or(false);
        footer
            || (rows.len() < MIN_GUTTER_ROWS && !item.is_margin_line_number)
            || !likely_margin(item)
            || line_has_content
    });
}

fn form_lines(
    items: &mut Vec<ProjectedTextItem>,
    median_width: f32,
    median_height: f32,
    page_width: f32,
    page_height: f32,
    style_runs: bool,
) -> Vec<Vec<ProjectedTextItem>> {
    // Y-tolerance for sorting: items within this threshold are considered same line
    let y_sort_tolerance: f32 = (median_height * 0.5).max(5.0);

    // A lone centered footer has the same shape as a gutter number. Require
    // a repeated column or text on both sides before marking it for removal.
    if page_width > 0.0 {
        // A real gutter line number is isolated on its baseline. An item that
        // abuts a neighbour is a fragment of a larger token, e.g. the "7" of a
        // table cell PDFium emits as "7" + ".7" (issue #465).
        /// Preserve the existing token-fragment guard: a quarter glyph height
        /// admits a tight join, and half a height restricts it to one baseline.
        /// The 0.5 pt overlap tolerance accommodates rounded box widths.
        fn abuts_neighbour(items: &[ProjectedTextItem], idx: usize) -> bool {
            let it = &items[idx].item;
            let max_gap = it.height * 0.25;
            items.iter().enumerate().any(|(j, other)| {
                if j == idx {
                    return false;
                }
                let o = &other.item;
                if (o.y - it.y).abs() > it.height * 0.5 {
                    return false;
                }
                let gap_right = o.x - (it.x + it.width);
                let gap_left = it.x - (o.x + o.width);
                (-0.5..=max_gap).contains(&gap_right) || (-0.5..=max_gap).contains(&gap_left)
            })
        }

        let margin_idxs: Vec<usize> = (0..items.len())
            .filter(|&i| {
                is_gutter_number_candidate(&items[i], page_width) && !abuts_neighbour(items, i)
            })
            .collect();
        let rows: HashSet<_> = margin_idxs
            .iter()
            .map(|&i| items[i].item.y.round() as i32)
            .collect();
        for i in margin_idxs {
            // Exempt the footer before flagging. Preserve existing OCR gutter
            // filtering: short OCR numerals may be recognition noise. Native
            // digits need repeated rows or text on both sides.
            if !isolated_numeric_footer(items, i, page_width, page_height)
                && (items[i].item.confidence.is_some()
                    || rows.len() >= MIN_GUTTER_ROWS
                    || between_text_on_baseline(items, i))
            {
                items[i].is_margin_line_number = true;
            }
        }
    }

    // Sort by y then x. Snap y to a grid so items on the same visual line
    // get identical keys — this keeps the comparator transitive (total order).
    let snap_y = |y: f32| -> i64 {
        if y_sort_tolerance > 0.0 {
            (y / y_sort_tolerance).round() as i64
        } else {
            (y * 1000.0).round() as i64
        }
    };
    items.sort_by(|a, b| {
        let ya = snap_y(a.item.y);
        let yb = snap_y(b.item.y);
        ya.cmp(&yb).then_with(|| a.item.x.total_cmp(&b.item.x))
    });

    // Keep bidirectional ordering and punctuation on the established grid path.
    let use_native_gaps = !items
        .iter()
        .any(|p| p.item.text.chars().any(bidi::is_rtl_char));

    const WORD_GAP_ALIGNMENT_TOLERANCE: f32 = 0.1;
    // Sheared boxes can hide small word spaces. Only subpixel residuals bridge
    // those gaps, while upright text keeps the existing 0.1 pt merge allowance.
    const GLYPH_GAP_HEIGHT_RATIO: f32 = 0.001;
    const GLYPH_OVERLAP_HEIGHT_RATIO: f32 = 0.04;

    fn gap_height(a: &ProjectedTextItem, b: &ProjectedTextItem) -> f32 {
        let height = |p: &ProjectedTextItem| p.item.font_height.unwrap_or(p.orig_height);
        height(a).min(height(b)).max(0.0)
    }

    /// Measure aligned native Latin runs before whole-point width rounding.
    /// OCR, scripts and unreliable font mappings retain their grid checks.
    fn native_gap(a: &ProjectedTextItem, b: &ProjectedTextItem) -> Option<f32> {
        let latin = |p: &ProjectedTextItem| {
            !p.rotated
                && p.item.confidence.is_none()
                && !p.item.font_is_buggy
                && !p.item.has_unicode_map_error
                && !(p.item.sup || p.item.sub || p.item.script_geom.is_some())
                && p.item.text.chars().filter(|c| c.is_alphabetic()).all(
                    |c| matches!(c as u32, 0x0000..=0x024F | 0x1E00..=0x1EFF | 0xFB00..=0xFB06),
                )
        };
        if !latin(a)
            || !latin(b)
            || a.item.font_size != b.item.font_size
            || a.item.font_height != b.item.font_height
            || (a.orig_y - b.orig_y).abs() > WORD_GAP_ALIGNMENT_TOLERANCE
            || (a.orig_height - b.orig_height).abs() > WORD_GAP_ALIGNMENT_TOLERANCE
        {
            return None;
        }
        let gap = b.orig_x - (a.orig_x + a.orig_width);
        // Monospace and sheared boxes can overlap across genuine word spaces.
        if gap < 0.0
            && (is_mono_item(&a.item)
                || is_mono_item(&b.item)
                || canonical_rotation(a.item.rotation) != 0
                || canonical_rotation(b.item.rotation) != 0)
        {
            return None;
        }
        (gap >= -(gap_height(a, b) * GLYPH_OVERLAP_HEIGHT_RATIO).min(0.5)).then_some(gap)
    }

    fn can_merge(
        prev: &ProjectedTextItem,
        cur: &ProjectedTextItem,
        y_tolerance: f32,
        h_tolerance: f32,
        use_native_gaps: bool,
    ) -> bool {
        // A script run shares its host's line band. Keep it separate until script flagging
        // can preserve it as a styled piece, even when its grid gap is zero.
        if prev.item.script_geom.is_some() || cur.item.script_geom.is_some() {
            return false;
        }
        if (cur.item.y - prev.item.y).abs() <= y_tolerance
            && (cur.item.height - prev.item.height).abs() <= h_tolerance
        {
            // This early merge keeps one style. Preserve its grid behavior across style changes.
            let measured = (use_native_gaps && same_emphasis(&prev.item, &cur.item))
                .then(|| native_gap(prev, cur))
                .flatten();
            let delta_x = measured.unwrap_or(cur.item.x - (prev.item.x + prev.item.width));
            let max_gap = if measured.is_some()
                && (canonical_rotation(prev.item.rotation) != 0
                    || canonical_rotation(cur.item.rotation) != 0)
            {
                (gap_height(prev, cur) * GLYPH_GAP_HEIGHT_RATIO).min(0.1)
            } else {
                0.1
            };
            return (-0.5..0.0).contains(&delta_x) || (0.0..max_gap).contains(&delta_x);
        }

        false
    }

    fn merge_bbox(prev: &ProjectedTextItem, cur: &ProjectedTextItem) -> (f32, f32, f32, f32) {
        let x1 = prev.item.x.min(cur.item.x);
        let y1 = prev.item.y.min(cur.item.y);
        let x2 = (prev.item.x + prev.item.width).max(cur.item.x + cur.item.width);
        let y2 = (prev.item.y + prev.item.height).max(cur.item.y + cur.item.height);
        (x1, y1, x2 - x1, y2 - y1)
    }

    fn merge_orig_bbox(prev: &mut ProjectedTextItem, cur: &ProjectedTextItem) {
        let x1 = prev.orig_x.min(cur.orig_x);
        let y1 = prev.orig_y.min(cur.orig_y);
        let x2 = (prev.orig_x + prev.orig_width).max(cur.orig_x + cur.orig_width);
        let y2 = (prev.orig_y + prev.orig_height).max(cur.orig_y + cur.orig_height);

        prev.orig_x = x1;
        prev.orig_y = y1;
        prev.orig_width = x2 - x1;
        prev.orig_height = y2 - y1;
    }

    // Every joiner must leave the merged item with `next`'s trailing-space
    // state; a stale flag would add a space later. The line pass joins
    // through here; the bbox pass below appends in stream order.
    fn append_text(prev: &mut ProjectedTextItem, next: &ProjectedTextItem, sep: &str) {
        join_in_reading_order(&mut prev.item.text, &next.item.text, sep);
        prev.item.followed_by_space = next.item.followed_by_space;
    }

    fn trailing_sep(prev: &ProjectedTextItem) -> &'static str {
        if prev.item.followed_by_space { " " } else { "" }
    }

    // Merge continuous bbox items in a single linear pass.
    let merge_y_tolerance = 0.1;
    let merge_h_tolerance = 0.1;

    let mut merged_items: Vec<ProjectedTextItem> = Vec::with_capacity(items.len());
    for cur in items.drain(..) {
        let should_merge = merged_items
            .last()
            .map(|prev| {
                can_merge(
                    prev,
                    &cur,
                    merge_y_tolerance,
                    merge_h_tolerance,
                    use_native_gaps,
                )
            })
            .unwrap_or(false);

        if should_merge {
            if let Some(prev) = merged_items.last_mut() {
                let merged = merge_bbox(prev, &cur);
                if prev.item.followed_by_space && !prev.item.text.ends_with(' ') {
                    prev.item.text.push(' ');
                }
                prev.item.text.push_str(&cur.item.text);
                prev.item.followed_by_space = cur.item.followed_by_space;
                prev.item.x = merged.0;
                prev.item.y = merged.1;
                prev.item.width = merged.2;
                prev.item.height = merged.3;
                merge_orig_bbox(prev, &cur);
            }
        } else {
            merged_items.push(cur);
        }
    }

    *items = merged_items;

    // try to find the bounding box that forms a line and group items by line
    let mut lines: Vec<Vec<ProjectedTextItem>> = Vec::new();
    let mut current_line: Vec<ProjectedTextItem> = Vec::new();
    let mut current_line_min_y = f32::INFINITY;
    let mut current_line_max_y = f32::NEG_INFINITY;
    for item in items.drain(..) {
        if !current_line.is_empty() {
            let mut line_collide = false;
            for line_item in current_line.iter() {
                let overlap_length = (line_item.item.x + line_item.item.width)
                    .min(item.item.x + item.item.width)
                    - line_item.item.x.max(item.item.x);

                if overlap_length > f32::max(5.0, median_width / 3.0) {
                    line_collide = true;
                    break;
                }
            }

            // Don't merge margin line numbers with regular content
            let cur_line_has_margin = current_line.iter().any(|i| i.is_margin_line_number);
            let cur_item_has_margin = item.is_margin_line_number;
            let margin_mismatch = cur_line_has_margin != cur_item_has_margin;

            // For rotated text, use y-tolerance based merging since heights may be inconsistent
            let y_tolerance_merge = if item.rotated {
                (median_height * 2.0).max(20.0)
            } else {
                0.0
            };
            let y_within_tolerance =
                item.rotated && (item.item.y - current_line_min_y).abs() < y_tolerance_merge;

            // Prevent "snowball" effect: when two columns have slightly offset
            // y-values, the line range keeps expanding as items from alternating
            // columns are added, merging multiple visual rows into one mega-line.
            // Cap the line height to a reasonable multiple of median text height.
            let proposed_min_y = current_line_min_y.min(item.item.y);
            let proposed_max_y = current_line_max_y.max(item.item.y + item.item.height);
            // too_tall threshold: prefer the smallest *trusted* item height
            // already in the current line as the line-height baseline. PDFium
            // reports inflated em-box heights for items with matrix-baked
            // sizing (fs=1.0); using the global `median_height` then permits
            // a real heading row to absorb the next visual row below it.
            // Falls back to median_height when no smaller anchor is available.
            let line_anchor_h = current_line
                .iter()
                .filter_map(|i| {
                    if matches!(i.item.font_size, Some(fs) if fs > 1.5)
                        || i.item.height < median_height
                    {
                        Some(i.item.height)
                    } else {
                        None
                    }
                })
                .fold(f32::INFINITY, f32::min);
            let too_tall_threshold = if line_anchor_h.is_finite() {
                (line_anchor_h * 1.8).max(median_height * 0.6)
            } else {
                median_height * 1.8
            };
            let too_tall = (proposed_max_y - proposed_min_y) > too_tall_threshold;

            if !line_collide
                && !margin_mismatch
                && !too_tall
                && (y_within_tolerance
                    || (item.item.y + item.item.height * 0.5 >= current_line_min_y
                        && item.item.y + item.item.height * 0.5 <= current_line_max_y)
                    || (item.item.y >= current_line_min_y && item.item.y <= current_line_max_y))
            {
                current_line_min_y = current_line_min_y.min(item.item.y);
                current_line_max_y = current_line_max_y.max(item.item.y + item.item.height);
                current_line.push(item);
            } else {
                lines.push(std::mem::take(&mut current_line));
                current_line_min_y = item.item.y;
                current_line_max_y = item.item.y + item.item.height;
                current_line.push(item);
            }
        } else {
            current_line_min_y = item.item.y;
            current_line_max_y = item.item.y + item.item.height;
            current_line.push(item);
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    // sort each line by x
    for line in lines.iter_mut() {
        line.sort_by(|a, b| a.item.x.total_cmp(&b.item.x));
    }

    for line in lines.iter_mut() {
        mark_script_items(line);
    }

    // sort lines by y
    lines.sort_by(|a, b| {
        let ay = a.first().map(|v| v.item.y).unwrap_or(f32::INFINITY);
        let by = b.first().map(|v| v.item.y).unwrap_or(f32::INFINITY);
        ay.total_cmp(&by)
    });

    // merge 'words'
    const MERGE_THRESHOLD: f32 = 1.0;

    fn looks_like_table_number(text: &str) -> bool {
        // Strip trailing statistical-significance markers ("0.77**", "0.31 *")
        // so decorated correlation/p-value cells still count as numbers. Without
        // this, the `both_are_numbers` guard below misfires and two adjacent
        // stat cells get merged into one, destroying the table's column tracks.
        let trimmed = text
            .trim()
            .trim_end_matches(|c: char| c == '*' || c == '†' || c == '‡' || c.is_whitespace());
        if trimmed.chars().count() < 2 {
            return false;
        }

        let mut chars = trimmed.chars().peekable();
        if matches!(chars.peek(), Some('$')) {
            chars.next();
        }
        if matches!(chars.peek(), Some('-')) {
            chars.next();
        }

        let mut has_digit = false;
        let mut has_decimal = false;
        for c in chars {
            if c.is_ascii_digit() {
                has_digit = true;
            } else if c == ',' {
                continue;
            } else if c == '.' {
                if has_decimal {
                    return false;
                }
                has_decimal = true;
            } else if c == '%' {
                return has_digit && trimmed.ends_with('%');
            } else {
                return false;
            }
        }

        has_digit
    }

    for line in lines.iter_mut() {
        let mut merged_line: Vec<ProjectedTextItem> = Vec::with_capacity(line.len());
        for item in line.drain(..) {
            if let Some(prev) = merged_line.last_mut() {
                let both_are_numbers = looks_like_table_number(&prev.item.text)
                    && looks_like_table_number(&item.item.text);

                let delta_x = item.item.x - prev.item.x - prev.item.width;
                // Don't merge items with noticeably different y positions (>1.5px).
                // Items on the same baseline typically differ by <0.5px.
                let y_diff = (item.item.y - prev.item.y).abs();
                let y_compatible = y_diff <= 1.5;

                // Decimal-fragment continuation: PDFium sometimes emits the
                // fractional part of a number ("15.6", "200.4") as a separate,
                // x-contiguous run beginning with the decimal point (".6").
                // `both_are_numbers` below treats them as two distinct table
                // values and keeps them apart, leaving "15 .6" in the rendered
                // cell and injecting a spurious column track between them. A
                // real pair of adjacent table numbers always has a visible
                // inter-cell gap, never a digit butted directly against a
                // leading-decimal run, so this is safe to join silently.
                let is_decimal_continuation = y_compatible
                    && (-2.0..=1.0).contains(&delta_x)
                    && prev
                        .item
                        .text
                        .chars()
                        .last()
                        .is_some_and(|c| c.is_ascii_digit())
                    && {
                        let mut cs = item.item.text.chars();
                        cs.next() == Some('.') && cs.next().is_some_and(|c| c.is_ascii_digit())
                    };
                if is_decimal_continuation {
                    // Markdown renders the pieces, so a numeric suffix must join both views.
                    if style_runs
                        && (!prev.item.style_pieces.is_empty()
                            || !same_emphasis(&prev.item, &item.item))
                    {
                        record_style_piece(&mut prev.item, &item.item, "");
                    }
                    prev.item.width = item.item.x + item.item.width - prev.item.x;
                    append_text(prev, &item, "");
                    merge_orig_bbox(prev, &item);
                    continue;
                }

                // Word-boundary guard: when prev ends with an alphabetic
                // character and item starts with one, a positive delta_x is
                // very likely a missing-space gap (PDFium omitted the space
                // glyph between two words), not a deliberate silent join.
                // Silent-merging here fuses "of the" → "ofthe". Skip the
                // silent path so the items stay separate and the line
                // builder inserts a space based on `gap > space_threshold`.
                // Punctuation joins (`Standards`+`.`, `width`+`)`, etc.)
                // and overlapping/negative-gap merges still go through.
                let gap = use_native_gaps
                    .then(|| native_gap(prev, &item))
                    .flatten()
                    .unwrap_or(delta_x);
                let alpha_alpha_gap = gap > 0.0
                    && prev
                        .item
                        .text
                        .chars()
                        .last()
                        .is_some_and(|c| c.is_alphabetic())
                    && item
                        .item
                        .text
                        .chars()
                        .next()
                        .is_some_and(|c| c.is_alphabetic());
                // Items merge whatever their emphasis, so lines, tables and blocks see the same
                // items in every output; for markdown a merge across an emphasis change keeps
                // the pieces, so each renders in its own style (`record_style_piece`).
                let keep_pieces = style_runs
                    && (!prev.item.style_pieces.is_empty()
                        || !same_emphasis(&prev.item, &item.item));
                if y_compatible
                    && !both_are_numbers
                    && !alpha_alpha_gap
                    && delta_x <= MERGE_THRESHOLD
                {
                    let sep = trailing_sep(prev);
                    if keep_pieces {
                        record_style_piece(&mut prev.item, &item.item, sep);
                    }
                    prev.item.width = item.item.x + item.item.width - prev.item.x;
                    append_text(prev, &item, sep);
                    merge_orig_bbox(prev, &item);
                    continue;
                }

                let prev_len = prev.item.text.chars().count().max(1) as f32;
                let avg_char_width = prev.item.width / prev_len;
                if y_compatible && !both_are_numbers && delta_x < avg_char_width {
                    if keep_pieces {
                        record_style_piece(&mut prev.item, &item.item, " ");
                    }
                    prev.item.width = item.item.x + item.item.width - prev.item.x;
                    append_text(prev, &item, " ");
                    merge_orig_bbox(prev, &item);
                    continue;
                }
            }

            merged_line.push(item);
        }

        *line = merged_line;
    }

    // Merge overlapping lines when there is no horizontal bbox overlap,
    // folding into an accumulator whose last line keeps absorbing successors.
    // Ribbon pages reach 10⁵+ lines, so this must stay linear: the x-sort of
    // a merged line is deferred until it stops absorbing (no check here
    // depends on item order) and its y-range is maintained incrementally.
    let y_range = |line: &[ProjectedTextItem]| -> (f32, f32) {
        line.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, v| {
                (acc.0.min(v.item.y), acc.1.max(v.item.y + v.item.height))
            })
    };
    let mut merged: Vec<Vec<ProjectedTextItem>> = Vec::with_capacity(lines.len());
    let mut prev_range = (f32::INFINITY, f32::NEG_INFINITY);
    let mut prev_absorbed = false;
    for mut current in lines {
        let (current_min_y, current_max_y) = y_range(&current);
        if let Some(previous) = merged.last_mut() {
            let (previous_min_y, previous_max_y) = prev_range;

            // Do the two lines overlap vertically?
            let lines_overlap = previous_max_y > current_min_y && previous_min_y < current_max_y;

            if lines_overlap {
                let bbox_overlap = current.iter().any(|bbox| {
                    previous.iter().any(|prev_bbox| {
                        (bbox.item.x >= prev_bbox.item.x
                            && bbox.item.x <= prev_bbox.item.x + prev_bbox.item.width)
                            || (prev_bbox.item.x >= bbox.item.x
                                && prev_bbox.item.x <= bbox.item.x + bbox.item.width)
                    })
                });

                if !bbox_overlap {
                    previous.append(&mut current);
                    prev_range = (
                        previous_min_y.min(current_min_y),
                        previous_max_y.max(current_max_y),
                    );
                    prev_absorbed = true;
                    continue;
                }
            }
            if prev_absorbed {
                previous.sort_by(|a, b| a.item.x.total_cmp(&b.item.x));
            }
        }
        merged.push(current);
        prev_range = (current_min_y, current_max_y);
        prev_absorbed = false;
    }
    if prev_absorbed && let Some(previous) = merged.last_mut() {
        previous.sort_by(|a, b| a.item.x.total_cmp(&b.item.x));
    }
    let lines = merged;

    // Insert blank lines for vertical gaps between lines. Gaps are measured
    // between consecutive content lines only; inserted blanks never
    // participate.
    let mut with_blanks: Vec<Vec<ProjectedTextItem>> = Vec::with_capacity(lines.len());
    let mut prev_metrics: Option<(f32, f32, f32)> = None;
    for line in lines {
        let cur_metrics = representative_line_metrics(&line, median_height);
        if let Some(prev_metrics) = prev_metrics {
            let y_delta = cur_metrics.0 - prev_metrics.1; // cur_top - prev_bottom
            let reference_height = median_height.max(prev_metrics.2.min(cur_metrics.2));

            if y_delta > reference_height {
                let num_blank = ((y_delta / reference_height).round() as usize).saturating_sub(1);
                for _ in 0..num_blank.clamp(1, 10) {
                    with_blanks.push(Vec::new());
                }
            }
        }
        prev_metrics = Some(cur_metrics);
        with_blanks.push(line);
    }

    with_blanks
}

/// Returns (top, bottom, height) for representative items in a line,
/// filtering out items much shorter than median height.
fn representative_line_metrics(
    line: &[ProjectedTextItem],
    global_median_height: f32,
) -> (f32, f32, f32) {
    if line.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let min_representative = global_median_height * 0.5;
    let has_representative = line.iter().any(|b| b.item.height >= min_representative);

    let top = line
        .iter()
        .filter(|b| !has_representative || b.item.height >= min_representative)
        .map(|b| b.item.y)
        .fold(f32::INFINITY, f32::min);
    let bottom = line
        .iter()
        .filter(|b| !has_representative || b.item.height >= min_representative)
        .map(|b| b.item.y + b.item.height)
        .fold(f32::NEG_INFINITY, f32::max);
    (top, bottom, bottom - top)
}

#[derive(Clone, Debug, Default)]
struct BoxMeta {
    left_anchor: Option<i32>,
    right_anchor: Option<i32>,
    center_anchor: Option<i32>,
    snap: Option<SnapKind>,
    force_unsnapped: bool,
    // Recorded only when projection signals are collected: which block the item sits in,
    // whether that block rendered as flowing prose, and how many items share each of its
    // anchors (0 = no anchor of that kind).
    block_idx: usize,
    flowing: bool,
    left_count: usize,
    right_count: usize,
    center_count: usize,
}

fn anchor_key(x: f32) -> i32 {
    (x * 4.0).round() as i32
}

fn anchor_to_x(key: i32) -> f32 {
    key as f32 / 4.0
}

fn merge_nearby_anchor_groups(collection: &mut HashMap<i32, Vec<(usize, usize)>>) {
    const MERGE_TOLERANCE: i32 = 8; // 2 units in quarter-point anchor key space

    let sorted_keys: Vec<i32> = {
        let mut keys: Vec<i32> = collection.keys().copied().collect();
        keys.sort_unstable();
        keys
    };

    for (i, anchor) in sorted_keys.iter().enumerate() {
        if !collection.contains_key(anchor) {
            continue;
        }
        for next_anchor in sorted_keys.iter().skip(i + 1) {
            if !collection.contains_key(next_anchor) {
                continue;
            }
            if next_anchor - anchor > MERGE_TOLERANCE {
                break;
            }

            let current_len = collection.get(anchor).map(|v| v.len()).unwrap_or(0);
            let next_len = collection.get(next_anchor).map(|v| v.len()).unwrap_or(0);

            if next_len > current_len {
                if let Some(cur_items) = collection.remove(anchor)
                    && let Some(next_items) = collection.get_mut(next_anchor)
                {
                    next_items.extend(cur_items);
                }
                break;
            } else if let Some(next_items) = collection.remove(next_anchor)
                && let Some(cur_items) = collection.get_mut(anchor)
            {
                cur_items.extend(next_items);
            }
        }
    }
}

/// Segments page lines into blocks separated by double blank lines.
fn segment_blocks(lines: &[Vec<ProjectedTextItem>]) -> Vec<LineRange> {
    let mut blocks = Vec::new();
    let mut empty_count = 0usize;
    let mut start: Option<usize> = None;

    for (line_idx, line) in lines.iter().enumerate() {
        if line.is_empty() {
            empty_count += 1;
            if empty_count > 1 {
                if let Some(s) = start {
                    // Include the trailing double-blank at the end of the block
                    blocks.push(LineRange {
                        start: s,
                        end: line_idx + 1,
                    });
                }
                start = None;
            }
        } else {
            empty_count = 0;
            if start.is_none() {
                start = Some(line_idx);
            }
        }
    }

    if let Some(s) = start {
        blocks.push(LineRange {
            start: s,
            end: lines.len(),
        });
    }

    // If no blocks found, treat entire page as one block
    if blocks.is_empty() && !lines.is_empty() {
        blocks.push(LineRange {
            start: 0,
            end: lines.len(),
        });
    }

    blocks
}

/// Extract anchor maps for a single block of lines.
/// Returns (anchor_left, anchor_right, anchor_center) with absolute line indices.
fn extract_block_anchors(
    lines: &[Vec<ProjectedTextItem>],
    block: &LineRange,
) -> (AnchorMap, AnchorMap, AnchorMap) {
    let mut anchor_left: AnchorMap = HashMap::new();
    let mut anchor_right: AnchorMap = HashMap::new();
    let mut anchor_center: AnchorMap = HashMap::new();

    for (line_idx, line) in lines.iter().enumerate().take(block.end).skip(block.start) {
        for (box_idx, bbox) in line.iter().enumerate() {
            // Skip rotated items from anchor detection — their coordinates are
            // transformed and would create spurious column anchors that stretch
            // the layout. They'll render as floating items at their natural positions.
            if bbox.rotated {
                continue;
            }
            let left_key = anchor_key(bbox.item.x);
            let right_key = anchor_key(bbox.item.x + bbox.item.width);
            let center_key = anchor_key(bbox.item.x + bbox.item.width * 0.5);
            anchor_left
                .entry(left_key)
                .or_default()
                .push((line_idx, box_idx));
            anchor_right
                .entry(right_key)
                .or_default()
                .push((line_idx, box_idx));
            anchor_center
                .entry(center_key)
                .or_default()
                .push((line_idx, box_idx));
        }
    }

    (anchor_left, anchor_right, anchor_center)
}

/// Remove vertically isolated items from anchor groups.
/// Items must have a neighbor within `page_height * delta` to survive.
fn delta_min_filter(
    collection: &mut HashMap<i32, Vec<(usize, usize)>>,
    lines: &[Vec<ProjectedTextItem>],
    page_height: f32,
    delta: f32,
) {
    let max_delta = page_height * delta;

    for members in collection.values_mut() {
        // Sort members by y coordinate
        members.sort_by(|a, b| {
            let ya = lines[a.0][a.1].item.y;
            let yb = lines[b.0][b.1].item.y;
            ya.total_cmp(&yb)
        });

        let mut keep = vec![false; members.len()];
        for i in 0..members.len() {
            let y_cur = lines[members[i].0][members[i].1].item.y;
            if i > 0 {
                let y_prev = lines[members[i - 1].0][members[i - 1].1].item.y;
                if y_cur - y_prev < max_delta {
                    keep[i] = true;
                    keep[i - 1] = true;
                }
            }
            if i + 1 < members.len() {
                let y_next = lines[members[i + 1].0][members[i + 1].1].item.y;
                if y_next - y_cur < max_delta {
                    keep[i] = true;
                }
            }
        }

        let mut idx = 0;
        members.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
    }

    collection.retain(|_, v| !v.is_empty());
}

/// Remove anchors where text from other items visually crosses the anchor x-position
/// between every consecutive pair of anchor members.
fn intercept_filter(
    collection: &mut HashMap<i32, Vec<(usize, usize)>>,
    lines: &[Vec<ProjectedTextItem>],
) {
    let anchors_to_remove: Vec<i32> = collection
        .iter()
        .filter_map(|(anchor_key_val, members)| {
            if members.len() < 2 {
                return None;
            }

            let anchor_x = anchor_to_x(*anchor_key_val);
            let mut any_pair_clear = false;

            for i in 1..members.len() {
                let y1 = lines[members[i - 1].0][members[i - 1].1].item.y;
                let y2 = lines[members[i].0][members[i].1].item.y;
                let (y_min, y_max) = if y1 < y2 { (y1, y2) } else { (y2, y1) };

                let mut intercepted = false;
                for line in lines.iter() {
                    if line.is_empty() {
                        continue;
                    }
                    let line_y = line[0].item.y;
                    if line_y > y_min && line_y < y_max {
                        for item in line {
                            if item.item.x < anchor_x && item.item.x + item.item.width > anchor_x {
                                intercepted = true;
                                break;
                            }
                        }
                        if intercepted {
                            break;
                        }
                    }
                }

                if !intercepted {
                    any_pair_clear = true;
                    break;
                }
            }

            if !any_pair_clear {
                Some(*anchor_key_val)
            } else {
                None
            }
        })
        .collect();

    for key in anchors_to_remove {
        collection.remove(&key);
    }
}

/// Try to align floating bboxes (not in any surviving anchor) to nearby anchors
/// on adjacent lines within the given margin.
fn try_align_floating(
    target: &mut HashMap<i32, Vec<(usize, usize)>>,
    lines: &[Vec<ProjectedTextItem>],
    block: &LineRange,
    anchored: &HashSet<(usize, usize)>,
    margin: f32,
    ref_x_fn: fn(&TextItem) -> f32,
    anchor_key_fn: fn(&TextItem) -> i32,
) {
    let mut additions: Vec<(i32, (usize, usize))> = Vec::new();

    for line_idx in block.start..block.end {
        for box_idx in 0..lines[line_idx].len() {
            if anchored.contains(&(line_idx, box_idx)) {
                continue;
            }
            // Skip rotated items — they are excluded from anchor extraction
            // and should remain floating to avoid spurious snap assignments.
            if lines[line_idx][box_idx].rotated {
                continue;
            }

            let ref_x = ref_x_fn(&lines[line_idx][box_idx].item);

            // Check adjacent lines for candidate anchors
            let mut candidate_anchor: Option<i32> = None;
            let mut prev_diff = margin + 1.0;

            let adj_lines: [Option<usize>; 2] = [
                if line_idx > block.start {
                    Some(line_idx - 1)
                } else {
                    None
                },
                if line_idx + 1 < block.end {
                    Some(line_idx + 1)
                } else {
                    None
                },
            ];

            for adj_opt in &adj_lines {
                let Some(adj_line_idx) = adj_opt else {
                    continue;
                };
                for adj_box in &lines[*adj_line_idx] {
                    let cand_key = anchor_key_fn(&adj_box.item);
                    if !target.contains_key(&cand_key) {
                        continue;
                    }
                    let x_diff = (anchor_to_x(cand_key) - ref_x).abs();
                    if x_diff <= margin && x_diff < prev_diff {
                        candidate_anchor = Some(cand_key);
                        prev_diff = x_diff;
                    }
                }
            }

            if let Some(key) = candidate_anchor {
                additions.push((key, (line_idx, box_idx)));
            }
        }
    }

    // Apply additions after iteration to avoid borrow conflicts
    for (key, item) in additions {
        if let Some(members) = target.get_mut(&key)
            && !members.contains(&item)
        {
            members.push(item);
        }
    }
}

/// Check if a line has a column-like gap: one gap that is much larger than
/// the typical inter-word gaps on the same line. This catches two-column
/// layouts where the absolute column gap is below the threshold but is
/// clearly an outlier relative to other gaps on the line.
fn line_has_column_gap(line: &[ProjectedTextItem], median_width: f32, page_width: f32) -> bool {
    if line.len() < 2 {
        return false;
    }
    let midpoint = page_width * 0.5;
    for i in 1..line.len() {
        let prev_end = line[i - 1].item.x + line[i - 1].item.width;
        let cur_start = line[i].item.x;
        let gap = cur_start - prev_end;
        // A gap is a column separator if it's large enough (>2x median char width)
        // AND items on either side span across the page midpoint.
        if gap > median_width * 2.0 && prev_end < midpoint && cur_start > midpoint {
            return true;
        }
    }
    false
}

/// Check if a block of lines is flowing paragraph text (vs structured/tabular).
fn is_flowing_text_block(
    lines: &[Vec<ProjectedTextItem>],
    block: &LineRange,
    anchor_left: &HashMap<i32, Vec<(usize, usize)>>,
    anchor_right: &HashMap<i32, Vec<(usize, usize)>>,
    anchor_center: &HashMap<i32, Vec<(usize, usize)>>,
    page_width: f32,
    median_width: f32,
) -> bool {
    let total_anchors = anchor_left.len() + anchor_right.len() + anchor_center.len();
    if total_anchors > FLOWING_MAX_TOTAL_ANCHORS {
        return false;
    }
    if anchor_left.len() > FLOWING_MAX_LEFT_ANCHORS {
        return false;
    }

    let mut non_empty_lines = 0usize;
    let mut wide_lines = 0usize;
    let mut column_gap_lines = 0usize;

    for line in lines.iter().take(block.end).skip(block.start) {
        if line.is_empty() {
            continue;
        }
        non_empty_lines += 1;

        let line_start = line[0].item.x;
        let line_end = line.last().map(|b| b.item.x + b.item.width).unwrap_or(0.0);
        if line_end - line_start > page_width * FLOWING_WIDE_LINE_RATIO {
            wide_lines += 1;
        }
        if line_has_column_gap(line, median_width, page_width) {
            column_gap_lines += 1;
        }
    }

    if non_empty_lines < FLOWING_MIN_LINES {
        return false;
    }

    // If multiple lines have column gaps, this is a multi-column block,
    // not flowing text.
    if column_gap_lines >= 2 {
        return false;
    }

    (wide_lines as f32 / non_empty_lines as f32) > FLOWING_WIDE_LINE_THRESHOLD
}

/// PDF font descriptor flag bits we care about.
/// See PDF spec §9.8.2 (table 123).
const PDF_FONT_FLAG_FIXED_PITCH: i32 = 1; // bit 1
const PDF_FONT_FLAG_ITALIC: i32 = 64; // bit 7
const PDF_FONT_FLAG_FORCE_BOLD: i32 = 262144; // bit 19

/// Font-name substrings that indicate bold weight.
const BOLD_NAME_HINTS: &[&str] = &["Bold", "Black", "Heavy", "Semibold", "Demibold"];
/// Font-name substrings that indicate italic / oblique slant.
const ITALIC_NAME_HINTS: &[&str] = &["Italic", "Oblique"];
/// Font-name substrings that indicate a monospaced typeface.
const MONO_NAME_HINTS: &[&str] = &[
    "Courier",
    "Mono",
    "Consolas",
    "Menlo",
    "Source Code",
    "Inconsolata",
    "Hack",
    "Fira Code",
];

fn name_contains_any(name: &str, hints: &[&str]) -> bool {
    hints.iter().any(|h| name.contains(h))
}

/// Flag the line's short items set clearly smaller than its body text and raised above its
/// baseline (superscripts: footnote markers, ordinals) or dropped below it (subscripts). The
/// body is the line's tallest-run text by character count; OCR and math items are left alone
/// (OCR boxes are too loose to tell a raised glyph from jitter).
fn mark_script_items(line: &mut [ProjectedTextItem]) {
    let native = |p: &ProjectedTextItem| p.item.confidence.is_none() && !p.item.math;
    // an item's own glyph box: a run split off at a size step carries its host's band
    let geom = |it: &TextItem| it.script_geom.unwrap_or((it.y, it.height));
    let mut hs: Vec<(f32, usize)> = line
        .iter()
        .filter(|p| native(p) && geom(&p.item).1 > 0.0)
        .map(|p| {
            (
                geom(&p.item).1,
                p.item.text.chars().filter(|c| !c.is_whitespace()).count(),
            )
        })
        .collect();
    let total: usize = hs.iter().map(|h| h.1).sum();
    if total < 4 {
        return;
    }
    hs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut acc, mut body_h) = (0, 0.0);
    for (h, n) in hs {
        acc += n;
        if acc * 2 >= total {
            body_h = h;
            break;
        }
    }
    let mut bottoms: Vec<f32> = line
        .iter()
        .filter(|p| native(p) && geom(&p.item).1 >= body_h * 0.9)
        .map(|p| geom(&p.item).0 + geom(&p.item).1)
        .collect();
    if bottoms.is_empty() || body_h <= 0.0 {
        return;
    }
    bottoms.sort_by(f32::total_cmp);
    let base = bottoms[bottoms.len() / 2];
    for p in line.iter_mut().filter(|p| native(p)) {
        let it = &mut p.item;
        let n = it.text.trim().chars().count();
        let (gy, gh) = geom(it);
        if n == 0 || n > 6 || gh > body_h * 0.78 {
            continue;
        }
        let bottom = gy + gh;
        if bottom <= base - body_h * 0.2 {
            it.sup = true;
        } else if bottom >= base + body_h * 0.1 && gy >= base - body_h * 0.6 {
            it.sub = true;
        }
    }
    // A script hugs the text it marks ("Taha¹", "CO₂", "substrate⁴¹"): a flagged run of three or
    // more items or over 8 characters is a smaller-type phrase on a mixed line (a byline next
    // to a drop cap), and a flagged item with no unflagged neighbour within half the body height
    // is no script either.
    let mut order: Vec<usize> = (0..line.len()).collect();
    order.sort_by(|&a, &b| line[a].item.x.total_cmp(&line[b].item.x));
    let script = |p: &ProjectedTextItem| p.item.sup || p.item.sub;
    let mut k = 0;
    while k < order.len() {
        if !script(&line[order[k]]) {
            k += 1;
            continue;
        }
        let mut e = k;
        while e + 1 < order.len() && script(&line[order[e + 1]]) {
            e += 1;
        }
        let chars: usize = order[k..=e]
            .iter()
            .map(|&i| {
                line[i]
                    .item
                    .text
                    .chars()
                    .filter(|c| !c.is_whitespace())
                    .count()
            })
            .sum();
        let near = |i: Option<&usize>, left: bool| {
            i.is_some_and(|&i| {
                let (a, b) = (&line[order[k]].item, &line[order[e]].item);
                let n = &line[i].item;
                let gap = if left {
                    a.x - (n.x + n.width)
                } else {
                    n.x - (b.x + b.width)
                };
                !script(&line[i]) && gap <= body_h * 0.5
            })
        };
        let hugs = near(k.checked_sub(1).and_then(|j| order.get(j)), true)
            || near(order.get(e + 1), false);
        if e - k + 1 >= 3 || chars > 8 || !hugs {
            for &i in &order[k..=e] {
                line[i].item.sup = false;
                line[i].item.sub = false;
            }
        }
        k = e + 1;
    }
}

/// Keep every merged character in the pieces Markdown renders. Leading-dot
/// continuations with the same style stay in one piece, preserving URLs and numbers.
fn record_style_piece(prev: &mut TextItem, next: &TextItem, sep: &str) {
    if prev.style_pieces.is_empty() {
        let first = prev.clone();
        prev.style_pieces.push(first);
    }
    let mut pieces = if next.style_pieces.is_empty() {
        let piece = next.clone();
        vec![piece]
    } else {
        next.style_pieces.clone()
    };
    if let (Some(last), Some(first)) = (prev.style_pieces.last_mut(), pieces.first())
        && sep.is_empty()
        && first.text.starts_with('.')
        && same_emphasis(last, first)
        && last.link == first.link
        && !bidi::is_rtl_pieces([last.text.as_str(), first.text.as_str()])
    {
        let right = (last.x + last.width).max(first.x + first.width);
        let bottom = (last.y + last.height).max(first.y + first.height);
        last.x = last.x.min(first.x);
        last.y = last.y.min(first.y);
        last.width = right - last.x;
        last.height = bottom - last.y;
        last.text.push_str(&first.text);
        pieces.remove(0);
    }
    prev.style_pieces.extend(pieces);
}

/// Whether two items render with the same inline emphasis.
fn same_emphasis(a: &TextItem, b: &TextItem) -> bool {
    is_bold_item(a) == is_bold_item(b)
        && is_italic_item(a) == is_italic_item(b)
        && is_strike_item(a) == is_strike_item(b)
        && is_mono_item(a) == is_mono_item(b)
        && a.underline == b.underline
        && a.sup == b.sup
        && a.sub == b.sub
}

/// Bold: for an item merged from differently-styled pieces, only when every piece is (a bold
/// run-in lead merged with the plain sentence after it is not a bold line).
pub(crate) fn is_bold_item(item: &TextItem) -> bool {
    if !item.style_pieces.is_empty() {
        return item.style_pieces.iter().all(is_bold_item);
    }
    font_is_bold(item.font_name.as_deref(), item.font_weight, item.font_flags)
}

pub(crate) fn is_italic_item(item: &TextItem) -> bool {
    if !item.style_pieces.is_empty() {
        return item.style_pieces.iter().all(is_italic_item);
    }
    font_is_italic(item.font_name.as_deref(), item.font_flags)
}

/// Bold from a font's flags (ForceBold), weight (600+) or name.
pub(crate) fn font_is_bold(name: Option<&str>, weight: Option<i32>, flags: Option<i32>) -> bool {
    flags.is_some_and(|f| f & PDF_FONT_FLAG_FORCE_BOLD != 0)
        || weight.is_some_and(|w| w >= 600)
        || name.is_some_and(|n| name_contains_any(n, BOLD_NAME_HINTS))
}

/// Italic from a font's flags or name.
pub(crate) fn font_is_italic(name: Option<&str>, flags: Option<i32>) -> bool {
    flags.is_some_and(|f| f & PDF_FONT_FLAG_ITALIC != 0)
        || name.is_some_and(|n| name_contains_any(n, ITALIC_NAME_HINTS))
}

pub(crate) fn is_strike_item(item: &TextItem) -> bool {
    item.strike
}

pub(crate) fn is_mono_item(item: &TextItem) -> bool {
    if let Some(flags) = item.font_flags
        && flags & PDF_FONT_FLAG_FIXED_PITCH != 0
    {
        return true;
    }
    if let Some(n) = &item.font_name
        && name_contains_any(n, MONO_NAME_HINTS)
    {
        return true;
    }
    false
}

/// A footer is a centered one- to three-digit item below every other text item.
/// Use original glyph geometry so grid snapping cannot change this decision.
///
/// The lower half limits the search to the page end. In the bottom tenth, the
/// half-height gap only requires a distinct baseline. Above that band, require
/// two smaller-glyph heights, leaving a blank line between body and footer.
/// This stronger gap handles large blank bottom margins without detaching a
/// nearby numeric table value. These fractions are conservative layout rules.
fn isolated_numeric_footer(
    items: &[ProjectedTextItem],
    idx: usize,
    page_width: f32,
    page_height: f32,
) -> bool {
    let item = &items[idx];
    let text = item.item.text.trim();
    let center = item.orig_x + item.orig_width * 0.5;
    if item.is_margin_line_number
        || !(1..=3).contains(&text.len())
        || !text.bytes().all(|b| b.is_ascii_digit())
        || item.orig_y < page_height * 0.5
        || center <= page_width * 0.5 - 5.0
        || center >= page_width * 0.5 + 20.0
    {
        return false;
    }
    let gap_ratio = if item.orig_y < page_height * 0.9 {
        2.0
    } else {
        0.5
    };
    items.iter().enumerate().all(|(other_idx, other)| {
        other_idx == idx
            || other.item.text.trim().is_empty()
            || other.orig_y + item.orig_height.min(other.orig_height) * gap_ratio < item.orig_y
    })
}

/// A projected item with its signals: box as left, top, width and height in page points.
#[derive(Debug, Clone)]
pub(super) struct Projected {
    pub text: String,
    pub bx: [f32; 4],
    pub faint: bool,
    pub block: usize,
    pub flow: bool,
    pub snap: char,
    pub lc: usize,
    pub rc: usize,
    pub cc: usize,
    pub leaf: f32,
}

mod bidi {

    /// Strong right-to-left character (Unicode bidi class R or AL). Covers Hebrew,
    /// Arabic, Syriac, Thaana, N'Ko, Samaritan, Mandaic, Adlam and the Arabic
    /// presentation-form blocks that subset fonts commonly map to.
    pub(crate) fn is_rtl_char(c: char) -> bool {
        matches!(c as u32,
            0x0590..=0x05FF   // Hebrew
            | 0x0600..=0x07BF // Arabic, Syriac, Thaana, N'Ko
            | 0x0800..=0x085F // Samaritan, Mandaic
            | 0x08A0..=0x08FF // Arabic Extended-A
            | 0xFB1D..=0xFDFF // Hebrew/Arabic presentation forms A
            | 0xFE70..=0xFEFF // Arabic presentation forms B
            | 0x10800..=0x10FFF // Cypriot..Old Hungarian etc.
            | 0x1E800..=0x1EFFF // Mende Kikakui, Adlam, Arabic Math
        )
    }

    /// Strong left-to-right character. Deliberately *not* the complement of
    /// [`is_rtl_char`]: digits, punctuation and whitespace are bidi-neutral and
    /// must not vote on a line's base direction, or every "12.34 SAR" amount on an
    /// Arabic invoice would drag its line back to LTR.
    fn is_strong_ltr_char(c: char) -> bool {
        c.is_alphabetic() && !is_rtl_char(c)
    }

    /// Base direction of a run of characters, decided by majority of
    /// strong-direction characters. Neutral-only text (pure numbers, punctuation)
    /// is LTR, which keeps every left-to-right document on exactly the path it took
    /// before.
    fn is_rtl_chars(chars: impl Iterator<Item = char>) -> bool {
        let mut rtl = 0usize;
        let mut ltr = 0usize;
        for c in chars {
            if is_rtl_char(c) {
                rtl += 1;
            } else if is_strong_ltr_char(c) {
                ltr += 1;
            }
        }
        rtl > ltr
    }

    /// Base direction of a line still held as separate pieces, without joining
    /// them first — line assembly needs the direction *before* it picks a join
    /// order, and the pieces can be numerous.
    ///
    /// Every RTL character is non-ASCII, so a line whose pieces are all ASCII has
    /// an RTL count of zero and cannot be RTL whatever its LTR count. `is_ascii` is
    /// a byte-wise scan rather than per-`char` classification, which keeps this off
    /// the hot path for left-to-right documents — the overwhelmingly common case.
    pub(crate) fn is_rtl_pieces<'a>(pieces: impl IntoIterator<Item = &'a str> + Clone) -> bool {
        if pieces.clone().into_iter().all(|p| p.is_ascii()) {
            return false;
        }
        is_rtl_chars(pieces.into_iter().flat_map(str::chars))
    }
}

/// Per-item projection signals of a page: lines, blocks, anchor groups, snap and flow, as the first half of LiteParse's `project_to_grid` computes them.
pub(super) fn project_signals(
    page_width: f32,
    page_height: f32,
    mut projection_boxes: Vec<ProjectedTextItem>,
) -> Vec<Projected> {
    if projection_boxes.is_empty() {
        return Vec::new();
    }

    // Filter out items that are purely dots
    let mut dot_count = 0usize;
    projection_boxes.iter().for_each(|item| {
        if item
            .item
            .text
            .chars()
            .all(|c| c == '.' || c == '·' || c == '•')
        {
            dot_count += 1;
        }
    });

    if dot_count > 100 && (dot_count as f64) > (projection_boxes.len() as f64) * 0.05 {
        projection_boxes.retain(|item| {
            !item
                .item
                .text
                .chars()
                .all(|c| c == '.' || c == '·' || c == '•')
        });
    }

    // Round dimensions
    for item in projection_boxes.iter_mut() {
        item.item.width = item.item.width.round();
        item.item.height = item.item.height.round();
    }

    // Compute median distances
    let (median_width, median_height) = compute_median_textbox_size(&projection_boxes);

    // Handle reading order rotations
    handle_rotation_reading_order(&mut projection_boxes, page_height);

    // Form lines of boxes
    let lines = form_lines(
        &mut projection_boxes,
        median_width,
        median_height,
        page_width,
        page_height,
        false,
    );
    if lines.is_empty() {
        return Vec::new();
    }

    // Segment into blocks
    let blocks = segment_blocks(&lines);

    let mut meta: Vec<Vec<BoxMeta>> = lines
        .iter()
        .map(|line| vec![BoxMeta::default(); line.len()])
        .collect();

    for (block_idx, block) in blocks.iter().enumerate() {
        for meta_line in meta.iter_mut().take(block.end).skip(block.start) {
            for m in meta_line {
                m.block_idx = block_idx;
            }
        }
        // --- Anchor extraction (per block) ---
        let (mut anchor_left, mut anchor_right, mut anchor_center) =
            extract_block_anchors(&lines, block);

        merge_nearby_anchor_groups(&mut anchor_left);
        merge_nearby_anchor_groups(&mut anchor_right);
        merge_nearby_anchor_groups(&mut anchor_center);

        // Isolation filtering: remove vertically isolated anchor members
        delta_min_filter(&mut anchor_left, &lines, page_height, 0.25);
        delta_min_filter(&mut anchor_right, &lines, page_height, 0.17);
        delta_min_filter(&mut anchor_center, &lines, page_height, 0.05);

        // Intercept filtering: remove anchors crossed by other text
        intercept_filter(&mut anchor_left, &lines);
        intercept_filter(&mut anchor_right, &lines);
        intercept_filter(&mut anchor_center, &lines);

        // Try to align floating bboxes to nearby existing anchors
        // Align floating items to nearby anchors. Use type-specific skip sets
        // so items with e.g. only a left anchor can still be aligned to a right anchor.
        let left_anchored: HashSet<(usize, usize)> = anchor_left
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        try_align_floating(
            &mut anchor_left,
            &lines,
            block,
            &left_anchored,
            4.0,
            |item| item.x,
            |item| anchor_key(item.x),
        );
        let right_anchored: HashSet<(usize, usize)> = anchor_right
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        try_align_floating(
            &mut anchor_right,
            &lines,
            block,
            &right_anchored,
            4.0,
            |item| item.x + item.width,
            |item| anchor_key(item.x + item.width),
        );
        let center_anchored: HashSet<(usize, usize)> = anchor_center
            .values()
            .flat_map(|v| v.iter().copied())
            .collect();
        try_align_floating(
            &mut anchor_center,
            &lines,
            block,
            &center_anchored,
            4.0,
            |item| item.x + item.width * 0.5,
            |item| anchor_key(item.x + item.width * 0.5),
        );

        // Remove singletons
        anchor_left.retain(|_, v| v.len() >= 2);
        anchor_right.retain(|_, v| v.len() >= 2);
        anchor_center.retain(|_, v| v.len() >= 2);

        // --- Flowing block detection ---
        if is_flowing_text_block(
            &lines,
            block,
            &anchor_left,
            &anchor_right,
            &anchor_center,
            page_width,
            median_width,
        ) {
            for meta_line in meta.iter_mut().take(block.end).skip(block.start) {
                for m in meta_line {
                    m.flowing = true;
                }
            }
            continue;
        }

        // --- Assign anchors to items in this block ---
        for (anchor, members) in &anchor_left {
            for &(li, bi) in members {
                meta[li][bi].left_anchor = Some(*anchor);
            }
        }
        for (anchor, members) in &anchor_right {
            for &(li, bi) in members {
                meta[li][bi].right_anchor = Some(*anchor);
            }
        }
        for (anchor, members) in &anchor_center {
            for &(li, bi) in members {
                meta[li][bi].center_anchor = Some(*anchor);
            }
        }

        // Resolve snap kind (strongest anchor wins; tie-break: left > right > center)
        for meta_line in meta.iter_mut().take(block.end).skip(block.start) {
            for m in meta_line {
                let left_count = m
                    .left_anchor
                    .and_then(|k| anchor_left.get(&k).map(|v| v.len()))
                    .unwrap_or(0);
                let right_count = m
                    .right_anchor
                    .and_then(|k| anchor_right.get(&k).map(|v| v.len()))
                    .unwrap_or(0);
                let center_count = m
                    .center_anchor
                    .and_then(|k| anchor_center.get(&k).map(|v| v.len()))
                    .unwrap_or(0);
                m.left_count = left_count;
                m.right_count = right_count;
                m.center_count = center_count;

                if left_count == 0 && right_count == 0 && center_count == 0 {
                    continue;
                }

                // Prefer left alignment when left and right counts are close.
                // In justified text, the right margin often collects slightly more
                // members than the left (due to indented paragraphs breaking the
                // left margin but keeping the right). A left-bias prevents justified
                // body text from right-snapping, which causes ragged left margins.
                let left_biased = left_count > 0
                    && right_count > 0
                    && left_count as f64 >= right_count as f64 * 0.8;

                let kind =
                    if (left_count >= right_count || left_biased) && left_count >= center_count {
                        SnapKind::Left
                    } else if right_count >= left_count && right_count >= center_count {
                        SnapKind::Right
                    } else {
                        SnapKind::Center
                    };
                m.snap = Some(kind);
            }
        }

        // Fixup pass: In justified text columns, most items share both a left
        // and right anchor. Items that lost their left anchor (e.g., indented
        // first lines) get right-snapped, but they should render at their
        // natural position. If a right anchor's members are overwhelmingly
        // also left-anchored, right-only members are likely justified body text
        // and should be unsnapped (rendered floating at natural x).
        //
        // Additional guard: the item's left x must be near the left x of the
        // left-anchored members (within half a page width). This prevents
        // unsnapping items in a different column that happen to share a right margin.
        {
            // For each right anchor: (total, has_left, median_left_x of left-anchored members)
            let mut right_anchor_info: HashMap<i32, (usize, usize, f32)> = HashMap::new();
            for (anchor_key, members) in &anchor_right {
                let total = members.len();
                let mut left_xs: Vec<f32> = Vec::new();
                let mut has_left = 0usize;
                for &(li, bi) in members {
                    if meta[li][bi].left_anchor.is_some() {
                        has_left += 1;
                        left_xs.push(lines[li][bi].item.x);
                    }
                }
                left_xs.sort_by(|a, b| a.total_cmp(b));
                let median_x = if left_xs.is_empty() {
                    0.0
                } else {
                    left_xs[left_xs.len() / 2]
                };
                right_anchor_info.insert(*anchor_key, (total, has_left, median_x));
            }

            for line_idx in block.start..block.end {
                for (box_idx, m) in meta[line_idx].iter_mut().enumerate() {
                    if m.snap != Some(SnapKind::Right) {
                        continue;
                    }
                    // Only fix items with no left anchor
                    if m.left_anchor.is_some() {
                        continue;
                    }
                    let Some(right_key) = m.right_anchor else {
                        continue;
                    };
                    let (total, has_left, median_left_x) = right_anchor_info
                        .get(&right_key)
                        .copied()
                        .unwrap_or((0, 0, 0.0));
                    // Require overwhelming majority (90%+) of members to have left anchors
                    if total < 10 || (has_left as f64 / total as f64) < 0.9 {
                        continue;
                    }
                    // Check that the item is near the same column as the left-anchored members
                    let item_x = lines[line_idx][box_idx].item.x;
                    let x_distance = (item_x - median_left_x).abs();
                    if x_distance > page_width * 0.25 {
                        continue;
                    }
                    m.snap = None;
                    m.right_anchor = None;
                }
            }
        }

        // Symmetric fixup: In multi-column layouts, short right-column lines
        // may only have a left anchor (no right anchor) because their right
        // edge doesn't align with full-width lines. Meanwhile, the full-width
        // lines in the same left anchor group get right-snapped. The short
        // lines get left-snapped and placed too far left.
        //
        // Detection: if a left anchor has many right-snapped members AND the
        // anchor position is past the page midpoint (indicating right column),
        // unsnap the remaining left-only members so they float and inherit
        // correct positioning from forward anchors.
        {
            let page_mid_key = anchor_key(page_width * 0.4);

            let mut left_anchor_info: HashMap<i32, (usize, usize)> = HashMap::new();
            for (left_ak, members) in &anchor_left {
                let total = members.len();
                let mut right_snapped = 0usize;
                for &(li, bi) in members {
                    if meta[li][bi].snap == Some(SnapKind::Right) {
                        right_snapped += 1;
                    }
                }
                left_anchor_info.insert(*left_ak, (total, right_snapped));
            }

            for meta_line in meta.iter_mut().take(block.end).skip(block.start) {
                for m in meta_line {
                    if m.snap != Some(SnapKind::Left) {
                        continue;
                    }
                    // Only fix items with no right anchor (short lines)
                    if m.right_anchor.is_some() {
                        continue;
                    }
                    let Some(left_key) = m.left_anchor else {
                        continue;
                    };
                    // Only apply to items past the page midpoint (right column)
                    if left_key < page_mid_key {
                        continue;
                    }
                    let (total, right_snapped) =
                        left_anchor_info.get(&left_key).copied().unwrap_or((0, 0));
                    // Require a meaningful group where most members are right-snapped
                    if total < 4 || (right_snapped as f64 / total as f64) < 0.5 {
                        continue;
                    }
                    m.snap = None;
                    m.left_anchor = None;
                    m.force_unsnapped = true;
                }
            }
        }
    }

    let mut out = Vec::new();
    let mut flattened = Vec::new();
    for (line_idx, line) in lines.into_iter().enumerate() {
        for (box_idx, item) in line.into_iter().enumerate() {
            let m = &meta[line_idx][box_idx];
            out.push(Projected {
                bx: [item.orig_x, item.orig_y, item.orig_width, item.orig_height],
                faint: item.item.ocr_faint,
                text: item.item.text.clone(),
                block: m.block_idx,
                flow: m.flowing,
                snap: match m.snap {
                    Some(SnapKind::Left) => 'L',
                    Some(SnapKind::Right) => 'R',
                    Some(SnapKind::Center) => 'C',
                    None => ' ',
                },
                lc: m.left_count,
                rc: m.right_count,
                cc: m.center_count,
                leaf: -1.0,
            });
            flattened.push(item);
        }
    }

    // XY-cut leaves of the page: an item's leaf id numbers the leaves in the order items first reach them.
    clean_projected_items(&mut flattened, page_width, page_height);
    let region = xy_cut(&flattened, page_width, page_height, &[]);
    let mut leaves: Vec<(Vec<u16>, Vec<usize>)> = Vec::new();
    xy_walk_leaves(&region, &mut Vec::new(), &mut leaves);
    let mut leaf_boxes: Vec<([f32; 2], &Vec<u16>)> = Vec::new();
    for (path, indices) in &leaves {
        for &i in indices {
            if let Some(it) = flattened.get(i) {
                leaf_boxes.push(([it.orig_x, it.orig_y], path));
            }
        }
    }
    let r1 = |v: f32| (v * 10.0).round() as i64;
    let mut ids: Vec<&Vec<u16>> = Vec::new();
    for item in &mut out {
        // Items without text or area are not units and take no id.
        if item.text.trim().is_empty() || item.bx[2] <= 0.0 || item.bx[3] <= 0.0 {
            continue;
        }
        let path = leaf_boxes
            .iter()
            .find(|(b, _)| r1(b[0]) == r1(item.bx[0]) && r1(b[1]) == r1(item.bx[1]))
            .map(|l| l.1);
        if let Some(path) = path {
            item.leaf = match ids.iter().position(|q| *q == path) {
                Some(i) => i as f32,
                None => {
                    ids.push(path);
                    (ids.len() - 1) as f32
                }
            };
        }
    }
    out
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Region {
    // The region tree keeps its geometry; only leaf membership is read here.
    #[allow(dead_code)]
    pub bbox: Rect,
    pub kind: RegionKind,
}

#[derive(Debug, Clone)]
pub enum RegionKind {
    Leaf {
        item_indices: Vec<usize>,
    },
    Split {
        #[allow(dead_code)]
        axis: CutAxis,
        children: Vec<Region>,
    },
}

impl Default for RegionKind {
    fn default() -> Self {
        RegionKind::Leaf {
            item_indices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutAxis {
    Horizontal,
    Vertical,
}

/// Bucket size for the 1D density projection (points).
const XY_BUCKET_PT: f32 = 2.0;
/// Upper bound on density-histogram buckets per axis; longer extents get wider
/// buckets instead of a multi-GB histogram.
const XY_MAX_BUCKETS: usize = 1 << 16;

/// Bucket width for an axis of `length` points, capped at `XY_MAX_BUCKETS` buckets.
fn xy_bucket_pt(length: f32) -> f32 {
    XY_BUCKET_PT.max(length / XY_MAX_BUCKETS as f32)
}
/// Density threshold for a "valley", expressed as a fraction of the local
/// median non-zero bucket density.
const XY_T_DENS: f32 = 0.10;
/// Minimum vertical-cut (column gutter) valley width.
const XY_MIN_V_VALLEY_PT: f32 = 8.0;
/// Minimum horizontal-cut (row gap) valley width, expressed as a multiple of
/// the median line height.
const XY_MIN_H_VALLEY_FACTOR: f32 = 1.4;
/// Hard recursion cap.
// Backstop only — every cut still passes the banner/density/histogram gates.
// 6 starved pages with several stacked floats (tables + captions above a
// 2-col body): each banner peel burns a level, and the body band leafs out
// un-split at the cap.
const XY_MAX_DEPTH: u32 = 12;
/// Below this item count we stop trying to cut.
const XY_MIN_ITEMS_TO_CUT: usize = 4;
/// A horizontal cut must leave at least this many distinct text lines on each
/// side; otherwise we don't slice prose into sliver bands.
const XY_MIN_LINES_PER_H_SIDE: usize = 2;
/// Vertical-cut score must beat the horizontal score by this factor to be
/// chosen. Tie-breaks in favor of horizontal so pages are sliced into bands
/// before columns — matches natural reading order.
const XY_V_PREFERENCE_MARGIN: f32 = 1.1;

/// Bucket width used by the column-start histogram fallback. 5pt is finer
/// than typical inter-column gutters (8–20pt) and coarser than baseline
/// noise / kerning drift, so column edges land in distinct buckets.
const XY_COLUMN_BUCKET_PT: f32 = 5.0;
/// Minimum lines stacked at the same left-edge cluster for the cluster to
/// count as a column peak. A real column has many lines at one x (15+);
/// indented paragraph-starts and footnote markers produce a handful (3–6).
/// We set this above the typical "indented paragraph" count so those drop
/// out and only true column-left edges remain.
const XY_COLUMN_MIN_LINES_PER_PEAK: usize = 10;
/// Minimum horizontal distance between two column peaks, as a fraction of
/// `bbox.width`. Sub-paragraph indents and list nesting produce
/// closely-spaced peaks; a real column gutter is much further out.
const XY_COLUMN_MIN_GAP_FRACTION: f32 = 0.25;
/// Minimum fraction of items that must start at one of the two detected
/// column peaks. Real 2-column layouts cluster nearly every line at one of
/// two left edges; tables, lists, and scattered prose don't.
const XY_COLUMN_PEAK_DOMINANCE: f32 = 0.55;
/// Minimum smaller/larger ratio between the two column peaks' line counts.
/// Balanced 2-column layouts have ~equal line counts in each column; a
/// 90/10 split is almost certainly a single column with one off-x bullet
/// stack, not a real two-column structure.
const XY_COLUMN_PEAK_BALANCE_RATIO: f32 = 0.4;
/// Minimum per-column horizontal fill for an N-column (3+ peak) split.
/// Genuine N-column prose fills each inter-peak slot nearly edge-to-edge
/// with text; a data table leaves wide intra-row whitespace, so its cells
/// cover only a fraction of each slot. Gate the N-column split on the
/// least-filled column so we keep splitting prose but bail (→ table
/// detector) on sparse tabular layouts.
const XY_COLUMN_MIN_FILL: f32 = 0.55;

/// An equation number set at a column edge: "(22)", "(3.14)", "(A.2)".
fn is_equation_number(text: &str) -> bool {
    let t = text.trim();
    let Some(inner) = t.strip_prefix('(').and_then(|r| r.strip_suffix(')')) else {
        return false;
    };
    !inner.is_empty()
        && inner.len() <= 6
        && inner.chars().any(|c| c.is_ascii_digit())
        && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '.')
}

/// Text with no run of three letters: a matrix entry, a bracket piece or an
/// accent the math pass left behind ("0", "¯", ", B =").
fn is_formula_fragment(text: &str) -> bool {
    let mut run = 0;
    for c in text.chars() {
        run = if c.is_alphabetic() { run + 1 } else { 0 };
        if run >= 3 {
            return false;
        }
    }
    !text.trim().is_empty()
}

/// A peak whose band count is below this fraction of the most-banded column's
/// is treated as a figure-scatter "phantom" (axis tick labels, a point cloud
/// spread across x) rather than a real column. Real table columns have one
/// cell per row, so their band counts are comparable to their siblings; a
/// plot embedded in 2-column prose contributes a sparse, low-fill third peak
/// that would otherwise flip the whole region to `tabular` and suppress the
/// genuine column split. See the phantom-drop in `xy_find_column_cut`.
const XY_COLUMN_PHANTOM_BAND_FRACTION: f32 = 0.25;

#[derive(Debug, Clone, Copy)]
struct CutCandidate {
    axis: CutAxis,
    /// Coordinate of the split (y for horizontal cut, x for vertical).
    position: f32,
    score: f32,
}

/// Character-weighted contribution of an item to the projection. Headings get
/// roughly the same weight as the body text they're competing with, instead of
/// being amplified by their bbox area.
fn xy_item_weight(it: &TextItem) -> f32 {
    it.text.chars().count().max(1) as f32
}

fn xy_root_bbox(
    items: &[ProjectedTextItem],
    indices: &[usize],
    page_width: f32,
    page_height: f32,
) -> Rect {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &idx in indices {
        let it = &items[idx];
        if it.item.text.is_empty() {
            continue;
        }
        min_x = min_x.min(it.item.x);
        min_y = min_y.min(it.item.y);
        max_x = max_x.max(it.item.x + it.item.width.max(0.0));
        max_y = max_y.max(it.item.y + it.item.height.max(0.0));
    }
    if !min_x.is_finite() {
        return Rect {
            x: 0.0,
            y: 0.0,
            width: page_width.max(1.0),
            height: page_height.max(1.0),
        };
    }
    let x = min_x.max(0.0);
    let y = min_y.max(0.0);
    let w = (max_x - x).max(1.0);
    let h = (max_y - y).max(1.0);
    // Span the actual item extent. The page dimensions are only used as a
    // fallback (no items). We must NOT clamp the bbox down to them:
    // `handle_rotation_reading_order` lays rotated marginalia (e.g. an arXiv
    // sidebar stamp) out on a virtual canvas taller/wider than the physical
    // page and pushes the non-rotated body below it. Clamping to page_height
    // would drop those displaced body items outside the density buckets, so no
    // horizontal valley between a full-width header and the 2-column body is
    // ever found and the column V-cut shreds the centered title/author band.
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

fn xy_median_line_height(items: &[ProjectedTextItem], idxs: &[usize]) -> f32 {
    let mut heights: Vec<f32> = idxs
        .iter()
        .filter_map(|&i| {
            let h = items[i].item.height;
            if h > 0.5 { Some(h) } else { None }
        })
        .collect();
    if heights.is_empty() {
        return 10.0;
    }
    heights.sort_by(|a, b| a.total_cmp(b));
    heights[heights.len() / 2]
}

/// Approximate distinct-line count by y-banding, used only to enforce
/// `XY_MIN_LINES_PER_H_SIDE`.
fn xy_distinct_lines(items: &[ProjectedTextItem], idxs: &[usize], median_h: f32) -> usize {
    if idxs.is_empty() {
        return 0;
    }
    let band = (median_h * 0.5).max(2.0);
    let mut ys: Vec<f32> = idxs.iter().map(|&i| items[i].item.y).collect();
    ys.sort_by(|a, b| a.total_cmp(b));
    let mut lines = 1usize;
    let mut last = ys[0];
    for &y in &ys[1..] {
        if (y - last).abs() > band {
            lines += 1;
            last = y;
        }
    }
    lines
}

/// Find the best valley along `axis` inside `bbox` for the given items. The
/// returned position is in absolute page coordinates.
///
/// `figures` is the set of figure-region rectangles on the page. Any figure
/// that intersects `bbox` is stamped into the density projection like a giant
/// opaque text block — its perpendicular extent (clipped to `bbox`) is added
/// to each bucket it covers along `axis`. This makes the recursion prefer
/// cuts *around* figures rather than slicing through them (which is the
/// motivating fix for academic-paper page-1 layouts where a figure straddles
/// both columns).
fn xy_find_best_cut(
    items: &[ProjectedTextItem],
    idxs: &[usize],
    bbox: &Rect,
    axis: CutAxis,
    median_h: f32,
    figures: &[Rect],
) -> Option<CutCandidate> {
    let (origin, length) = match axis {
        CutAxis::Horizontal => (bbox.y, bbox.height),
        CutAxis::Vertical => (bbox.x, bbox.width),
    };
    if length <= 0.0 {
        return None;
    }
    let bucket_pt = xy_bucket_pt(length);
    let n_buckets = ((length / bucket_pt).ceil() as usize).max(1);
    let mut density = vec![0.0f32; n_buckets];

    for &i in idxs {
        let it = &items[i].item;
        let (c0, c1) = match axis {
            CutAxis::Horizontal => (it.y, it.y + it.height.max(0.0)),
            CutAxis::Vertical => (it.x, it.x + it.width.max(0.0)),
        };
        let weight = xy_item_weight(it);
        let b0_f = ((c0 - origin) / bucket_pt).floor();
        let b1_f = ((c1 - origin) / bucket_pt).ceil();
        let b0 = b0_f.max(0.0) as usize;
        let b1 = (b1_f.max(b0_f + 1.0) as usize).min(n_buckets);
        for slot in &mut density[b0..b1] {
            *slot += weight;
        }
    }

    // Stamp figures into the projection as opaque obstacles. A figure that
    // intersects `bbox` adds density to each axis-parallel bucket it spans;
    // weight = its perpendicular extent (clipped to bbox) so figures behave
    // like very wide/tall pseudo-items relative to text. The valley search
    // naturally finds cuts adjacent to figures rather than through them.
    for fig in figures {
        let fx0 = fig.x.max(bbox.x);
        let fy0 = fig.y.max(bbox.y);
        let fx1 = (fig.x + fig.width).min(bbox.x + bbox.width);
        let fy1 = (fig.y + fig.height).min(bbox.y + bbox.height);
        if fx1 <= fx0 || fy1 <= fy0 {
            continue;
        }
        let (c0, c1, perp) = match axis {
            CutAxis::Horizontal => (fy0, fy1, fx1 - fx0),
            CutAxis::Vertical => (fx0, fx1, fy1 - fy0),
        };
        let weight = perp.max(1.0);
        let b0_f = ((c0 - origin) / bucket_pt).floor();
        let b1_f = ((c1 - origin) / bucket_pt).ceil();
        let b0 = b0_f.max(0.0) as usize;
        let b1 = (b1_f.max(b0_f + 1.0) as usize).min(n_buckets);
        for slot in &mut density[b0..b1] {
            *slot += weight;
        }
    }

    let mut nonzero: Vec<f32> = density.iter().copied().filter(|d| *d > 0.0).collect();
    if nonzero.is_empty() {
        return None;
    }
    nonzero.sort_by(|a, b| a.total_cmp(b));
    let median = nonzero[nonzero.len() / 2];
    if median <= 0.0 {
        return None;
    }
    let threshold = median * XY_T_DENS;

    // Don't cut at leading/trailing whitespace — only consider valleys interior
    // to the content's span on this axis.
    let first_dense = density.iter().position(|d| *d > threshold)?;
    let last_dense = density.iter().rposition(|d| *d > threshold)?;
    if last_dense <= first_dense + 1 {
        return None;
    }

    let min_valley_pt = match axis {
        CutAxis::Vertical => XY_MIN_V_VALLEY_PT,
        CutAxis::Horizontal => median_h * XY_MIN_H_VALLEY_FACTOR,
    };
    let min_valley_buckets = ((min_valley_pt / bucket_pt).ceil() as usize).max(1);

    let mut best: Option<CutCandidate> = None;
    let mut i = first_dense + 1;
    while i <= last_dense {
        if density[i] <= threshold {
            let s = i;
            while i <= last_dense && density[i] <= threshold {
                i += 1;
            }
            let e = i; // exclusive
            let width = e - s;
            if width >= min_valley_buckets {
                let mean: f32 = density[s..e].iter().sum::<f32>() / width as f32;
                let depth = (1.0 - mean / median).max(0.0);
                let score = width as f32 * depth;
                let mid = (s as f32 + e as f32) * 0.5;
                let position = origin + mid * bucket_pt;
                // Density-stamping alone isn't always enough to push the
                // gutter inside a table region below threshold — the
                // inter-cell gaps win on tables with sparsely-filled cells
                // (docs 120/180/150). Hard-reject any cut line that passes
                // strictly through an obstacle rect intersecting `bbox`.
                if !cut_line_passes_through_obstacle(axis, position, bbox, figures)
                    && best.as_ref().is_none_or(|b| score > b.score)
                {
                    best = Some(CutCandidate {
                        axis,
                        position,
                        score,
                    });
                }
            }
        } else {
            i += 1;
        }
    }
    best
}

/// Hard-reject companion to the density-stamping in `xy_find_best_cut`.
/// Density-stamping makes the recursion *prefer* cuts around obstacles, but
/// sparsely-filled tables (docs 120/180/150) have inter-cell gaps that still
/// score better than the stamped obstacle band. Returns true iff the cut
/// line at `position` strictly passes through any obstacle that intersects
/// `bbox`.
fn cut_line_passes_through_obstacle(
    axis: CutAxis,
    position: f32,
    bbox: &Rect,
    obstacles: &[Rect],
) -> bool {
    // Tolerance: a cut whose position lands within ~1pt of an obstacle's
    // edge is allowed (the obstacle's edge is itself a natural cut line).
    const EDGE_TOLERANCE_PT: f32 = 1.0;
    for ob in obstacles {
        // Skip obstacles that don't intersect bbox at all.
        let ox0 = ob.x.max(bbox.x);
        let oy0 = ob.y.max(bbox.y);
        let ox1 = (ob.x + ob.width).min(bbox.x + bbox.width);
        let oy1 = (ob.y + ob.height).min(bbox.y + bbox.height);
        if ox1 <= ox0 || oy1 <= oy0 {
            continue;
        }
        let through = match axis {
            CutAxis::Vertical => {
                position > ox0 + EDGE_TOLERANCE_PT && position < ox1 - EDGE_TOLERANCE_PT
            }
            CutAxis::Horizontal => {
                position > oy0 + EDGE_TOLERANCE_PT && position < oy1 - EDGE_TOLERANCE_PT
            }
        };
        if through {
            return true;
        }
    }
    false
}

fn xy_split_bbox(bbox: &Rect, cut: &CutCandidate) -> (Rect, Rect) {
    match cut.axis {
        CutAxis::Horizontal => {
            let top_h = (cut.position - bbox.y).max(0.0);
            let top = Rect {
                x: bbox.x,
                y: bbox.y,
                width: bbox.width,
                height: top_h,
            };
            let bot = Rect {
                x: bbox.x,
                y: cut.position,
                width: bbox.width,
                height: (bbox.height - top_h).max(0.0),
            };
            (top, bot)
        }
        CutAxis::Vertical => {
            let left_w = (cut.position - bbox.x).max(0.0);
            let left = Rect {
                x: bbox.x,
                y: bbox.y,
                width: left_w,
                height: bbox.height,
            };
            let right = Rect {
                x: cut.position,
                y: bbox.y,
                width: (bbox.width - left_w).max(0.0),
                height: bbox.height,
            };
            (left, right)
        }
    }
}

/// Minimum fraction of region width a banner band must cover.
const XY_BANNER_WIDTH_FRACTION: f32 = 0.6;
/// Banner clearance gap above/below in multiples of median line height.
const XY_BANNER_CLEARANCE_FACTOR: f32 = 1.5;

/// Detect a "banner" y-band — one or more vertically-adjacent wide text bands
/// (≥ `XY_BANNER_WIDTH_FRACTION` of bbox width) flanked by a clear vertical
/// gap from the rest of the content. Returns an H-cut that isolates the banner
/// from neighboring content.
///
/// Motivating case: a full-width centered title + authors block above a 2-
/// column body where a figure straddles both columns. Density-based XY-cut
/// can't separate them because the figure prevents any clean V-cut and there's
/// no obvious H-valley between the banner and the body. The banner detector
/// runs first and force-cuts just below the banner stack, freeing the
/// recursion to find normal column gutters in the lower region.
///
/// Also helps mid-document section headings that span the full content width:
/// a wide centered "5 Conclusion" line above a 2-column body would otherwise
/// get pulled into one column.
/// Fallback column-gutter detector for cases where the density-based V-cut
/// search finds no valley — typically because a column-spanning element
/// (wide heading, figure, table) fills the gutter at one y-band and inflates
/// the projection at the gutter's x. Detects 2+ column structure by clustering
/// the left edges of "narrow" lines (lines that don't span the full region
/// width). When two clusters with ≥`XY_COLUMN_MIN_LINES_PER_PEAK` lines each
/// are separated by ≥`XY_COLUMN_MIN_GAP_FRACTION × bbox.width`, returns a
/// V-cut at the midpoint between them with a high score so it beats any
/// density-based cut except banner.
/// Per-column fill for a candidate peak set: per y-band covered x-span /
/// slot width, averaged per column. Returns `(min_fill, band-count-weighted
/// avg_fill)`, the same statistics as `xy_find_column_cut`'s main fill gate.
/// `sorted` must be y-then-x sorted item indices; `peaks` left→right.
fn xy_peak_fill_stats(
    items: &[ProjectedTextItem],
    sorted: &[usize],
    peaks: &[(f32, usize)],
    band_tol: f32,
    bucket_pt: f32,
    max_x: f32,
) -> (f32, f32) {
    let n = peaks.len();
    let mut covered_sum = vec![0.0f32; n];
    let mut band_count = vec![0usize; n];
    let mut kk = 0;
    while kk < sorted.len() {
        let by = items[sorted[kk]].item.y;
        let mut jj = kk;
        while jj < sorted.len() && (items[sorted[jj]].item.y - by).abs() <= band_tol {
            jj += 1;
        }
        let mut lo_x = vec![f32::INFINITY; n];
        let mut hi_x = vec![f32::NEG_INFINITY; n];
        for &idx in &sorted[kk..jj] {
            let it = &items[idx].item;
            if !it.x.is_finite() {
                continue;
            }
            let mut c = 0;
            while c + 1 < n && it.x >= peaks[c + 1].0 - bucket_pt {
                c += 1;
            }
            lo_x[c] = lo_x[c].min(it.x);
            hi_x[c] = hi_x[c].max(it.x + it.width.max(0.0));
        }
        for c in 0..n {
            if hi_x[c] > lo_x[c] {
                let col_w = if c + 1 < n {
                    peaks[c + 1].0 - peaks[c].0
                } else {
                    max_x - peaks[c].0
                };
                if col_w > 1.0 {
                    covered_sum[c] += ((hi_x[c] - lo_x[c]) / col_w).min(1.0);
                    band_count[c] += 1;
                }
            }
        }
        kk = jj;
    }
    let mut min_fill = f32::INFINITY;
    let mut weighted_sum = 0.0f32;
    let mut weighted_n = 0.0f32;
    for c in 0..n {
        if band_count[c] > 0 {
            let f = covered_sum[c] / band_count[c] as f32;
            min_fill = min_fill.min(f);
            weighted_sum += f * band_count[c] as f32;
            weighted_n += band_count[c] as f32;
        }
    }
    let avg_fill = if weighted_n > 0.0 {
        weighted_sum / weighted_n
    } else {
        f32::INFINITY
    };
    (min_fill, avg_fill)
}

fn xy_find_column_cut(
    items: &[ProjectedTextItem],
    idxs: &[usize],
    bbox: &Rect,
    median_h: f32,
    tabular: &mut bool,
) -> Option<CutCandidate> {
    if bbox.width <= 1.0 || idxs.len() < 8 {
        return None;
    }

    // Identify "line-start" items: within each y-band, the first item of
    // each horizontal run is a line-start. PDFium routinely emits mid-line
    // word-cluster fragments as separate items; counting all of them in
    // the histogram denominator dilutes the column-dominance ratio so the
    // wide-side probe never trips after a footer-peel H-cut. A 2-column
    // line at the same baseline contributes TWO line-starts (one per
    // column) because the inter-column gutter exceeds `gutter_gap`, so we
    // don't collapse a band to its single leftmost item.
    let band_tol = (median_h * 0.5).max(2.0);
    // OCR items are whole detected lines, not PDF word fragments: two of them in one band
    // are split at a column gutter (or a cell boundary, which the fill gate below handles),
    // never at a word space. Scans' tight gutters (~0.6-0.8 line heights on newspaper pages)
    // fail the native one-line-height test, and the near-start rescue needs sub-point edge
    // alignment that OCR boxes don't have, so N-column scans were read row by row.
    let ocr_items = idxs
        .iter()
        .filter(|&&i| items[i].item.confidence.is_some())
        .count();
    let gutter_gap = if ocr_items * 2 > idxs.len() {
        (median_h * 0.5).max(4.0)
    } else {
        median_h.max(4.0)
    };

    let mut sorted: Vec<usize> = idxs.to_vec();
    sorted.sort_by(|&a, &b| {
        items[a]
            .item
            .y
            .total_cmp(&items[b].item.y)
            .then(items[a].item.x.total_cmp(&items[b].item.x))
    });

    let mut line_starts: Vec<f32> = Vec::with_capacity(idxs.len() / 2);
    // Items whose gap to the previous item falls just under `gutter_gap` but
    // clearly above word-gap range. A tight column gutter (e.g. ~9pt at
    // median_h=10) fails the strict test on justified lines whose left column
    // runs right up to the gutter; these get a second chance below if their x
    // aligns with a column peak that has strict evidence.
    let mut near_starts: Vec<f32> = Vec::new();
    let near_gap_floor = gutter_gap * 0.6;
    let mut k = 0;
    while k < sorted.len() {
        let band_y = items[sorted[k]].item.y;
        let mut j = k;
        while j < sorted.len() && (items[sorted[j]].item.y - band_y).abs() <= band_tol {
            j += 1;
        }
        let mut band_items: Vec<usize> = sorted[k..j].to_vec();
        band_items.sort_by(|&a, &b| items[a].item.x.total_cmp(&items[b].item.x));
        let mut prev_right = f32::NEG_INFINITY;
        for &idx in &band_items {
            let it = &items[idx].item;
            if it.x.is_finite() {
                let gap = it.x - prev_right;
                if gap > gutter_gap {
                    line_starts.push(it.x);
                } else if gap > near_gap_floor {
                    near_starts.push(it.x);
                }
            }
            let right = it.x + it.width.max(0.0);
            if right > prev_right {
                prev_right = right;
            }
        }
        k = j;
    }

    if line_starts.len() < 8 {
        return None;
    }

    // Histogram extent uses full item bounds so `XY_COLUMN_MIN_GAP_FRACTION`
    // calibration (fraction of layout width) is preserved. Only line-starts
    // contribute to the bucket counts and the dominance denominator.
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    for &i in idxs {
        let it = &items[i].item;
        if it.x.is_finite() {
            min_x = min_x.min(it.x);
            max_x = max_x.max(it.x + it.width.max(0.0));
        }
    }
    if !min_x.is_finite() || max_x <= min_x + 1.0 {
        return None;
    }
    let total_w = max_x - min_x;
    let bucket_pt = XY_COLUMN_BUCKET_PT;
    let n_buckets = ((total_w / bucket_pt).ceil() as usize).max(1);
    let mut hist = vec![0usize; n_buckets];
    for &x in &line_starts {
        let b_f = ((x - min_x) / bucket_pt).floor();
        let b = b_f.max(0.0) as usize;
        if b < n_buckets {
            hist[b] += 1;
        }
    }

    // Peak-aligned rescue for tight gutters: a near-gap start only counts
    // when it lands at (nearly) the same x as existing strict-evidence
    // starts. A column left edge aligns to sub-point precision across
    // bands; word-gap "rivers" in justified text jitter by several points —
    // a looser bucket-width match here let rivers stack onto stray strict
    // starts and manufacture a fake second column inside narrow justified
    // prose. Require near-exact agreement with at least 2 strict starts.
    let mut rescued = 0usize;
    let mut counted_starts: Vec<f32> = line_starts.clone();
    if !near_starts.is_empty() {
        const RESCUE_ALIGN_TOL_PT: f32 = 1.5;
        for &x in &near_starts {
            let support = line_starts
                .iter()
                .filter(|&&s| (s - x).abs() <= RESCUE_ALIGN_TOL_PT)
                .count();
            if support >= 2 {
                let b = (((x - min_x) / bucket_pt).floor()).max(0.0) as usize;
                if b < n_buckets {
                    hist[b] += 1;
                    rescued += 1;
                    counted_starts.push(x);
                }
            }
        }
    }
    let n_starts = line_starts.len() + rescued;

    // Cluster adjacent occupied buckets into peaks. A peak is a contiguous
    // run of buckets with ≥1 count, merged through gaps of ≤2 zero buckets
    // (~10pt) to absorb slight intra-column drift.
    let mut peaks: Vec<(f32, usize)> = Vec::new(); // (x_center, item_count)
    let mut i = 0;
    while i < n_buckets {
        if hist[i] == 0 {
            i += 1;
            continue;
        }
        let s = i;
        let mut total = 0usize;
        let mut last_nonzero = i;
        let mut j = i;
        while j < n_buckets {
            if hist[j] > 0 {
                total += hist[j];
                last_nonzero = j;
                j += 1;
            } else if j - last_nonzero <= 2 {
                j += 1;
            } else {
                break;
            }
        }
        let x_center = min_x + bucket_pt * (s as f32 + last_nonzero as f32 + 1.0) * 0.5;
        peaks.push((x_center, total));
        i = j;
    }

    // Keep only "strong" peaks. A real column edge is sat-stacked by many
    // lines at the same left x; spurious peaks (a stray indented quote, a
    // numbered marker, etc.) carry only a few lines.
    let raw_counts: Vec<(f32, usize)> = peaks.clone();
    // Adaptive minimum: scale with how many line-starts the band actually has.
    // The static `XY_COLUMN_MIN_LINES_PER_PEAK = 10` floor is meant to drop
    // indented-paragraph false peaks (a handful of lines) on a large region,
    // but on smaller regions it can drop legitimate column peaks too. For a
    // body region with ~36 lines split across 2 columns, each peak only
    // carries 5–6 line-starts after the histogram spreads them across the
    // page width — well under 10. Use `max(line_starts / 6, 4)`: still rules
    // out the tiny indent peaks, but a 2-column region with ~5 lines/peak
    // qualifies. Capped by `XY_COLUMN_MIN_LINES_PER_PEAK` so large regions
    // keep the stricter floor.
    let adaptive_min = (n_starts / 6).clamp(4, XY_COLUMN_MIN_LINES_PER_PEAK);
    peaks.retain(|(_, c)| *c >= adaptive_min);
    if peaks.len() < 2 {
        // A table with a sparse row drops most peaks just under
        // `adaptive_min`, so the fill gate below never runs and density
        // V-cuts are free to slice the table's cell gutters (transposing it
        // column-major). 3+ balanced, dominant, low-fill raw peaks are still
        // tabular evidence — classify only, never cut.
        if raw_counts.len() >= 3 {
            let raw_sum: usize = raw_counts.iter().map(|(_, c)| *c).sum();
            let raw_min = raw_counts.iter().map(|(_, c)| *c).min().unwrap_or(0);
            let raw_max = raw_counts.iter().map(|(_, c)| *c).max().unwrap_or(0);
            let balanced = raw_max > 0
                && raw_min >= 2
                && raw_min as f32 / raw_max as f32 >= XY_COLUMN_PEAK_BALANCE_RATIO;
            let dominance = raw_sum as f32 / n_starts.max(1) as f32;
            if balanced && dominance >= XY_COLUMN_PEAK_DOMINANCE {
                // The low-fill requirement is load-bearing: N-column prose
                // whose short columns fell under `adaptive_min` (newspaper
                // pages, multi-column TOCs) matches the peak shape but fills
                // its slots nearly edge-to-edge.
                let mut raw_peaks = raw_counts.clone();
                raw_peaks.sort_by(|a, b| a.0.total_cmp(&b.0));
                let (min_fill, avg_fill) =
                    xy_peak_fill_stats(items, &sorted, &raw_peaks, band_tol, bucket_pt, max_x);
                let sparse = (min_fill.is_finite() && min_fill < 0.38)
                    || (avg_fill.is_finite() && avg_fill < XY_COLUMN_MIN_FILL);
                if sparse {
                    *tabular = true;
                }
            }
        }
        return None;
    }
    peaks.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Tabular layouts produce 3+ strong peaks (one per cell column). Real
    // page columns have exactly 2 dominant peaks (left edge of col 1, left
    // edge of col 2) — even multi-paragraph indents are sub-peaks weak
    // enough to drop out under the min_per_peak filter. Bail when we see
    // more than 2 strong peaks so we don't slice tables.
    // Tabular layouts produce 3+ strong peaks with comparable counts. But a
    // real 2-column page can also produce 3+ peaks when a centered float
    // (caption, sub-heading stack, pull-quote) sits between the columns and
    // contributes line-starts at a mid x. Distinguish: if the top-2 peaks
    // by count dominate the remainder, keep them and drop the rest;
    // otherwise (top-3 are comparable → tabular) bail.
    if peaks.len() > 2 {
        let mut by_count = peaks.clone();
        by_count.sort_by_key(|p| std::cmp::Reverse(p.1));
        let third = by_count[2].1 as f32;
        let second = by_count[1].1 as f32;
        let largest = by_count[0].1 as f32;
        if second <= 0.0 || third / second >= XY_COLUMN_PEAK_BALANCE_RATIO {
            // 3+ comparable peaks. Two shapes produce this: a genuine
            // N-column prose layout (newspaper / magazine / multi-col
            // reference list) or a data table. We can't bail on both —
            // bailing leaves N-column prose joined line-by-line, which the
            // borderless table detector then shreds into a fake table.
            // Keep every peak balanced against the largest and let the
            // gap logic cut the leftmost gutter; recursion re-runs on each
            // side and peels the remaining columns one at a time. The
            // downstream dominance + min-gap guards still reject tables
            // whose cells don't tightly left-cluster (numeric columns are
            // typically right-aligned / centered, so they never form
            // strong left-edge peaks and fail dominance here).
            peaks.retain(|(_, c)| {
                largest <= 0.0 || (*c as f32) / largest >= XY_COLUMN_PEAK_BALANCE_RATIO
            });
            if peaks.len() < 2 {
                return None;
            }
        } else {
            // Top-2 dominate → assume real columns + weak floats. Keep
            // top-2 (re-sorted left→right) and continue.
            let mut top2 = vec![by_count[0], by_count[1]];
            top2.sort_by(|a, b| a.0.total_cmp(&b.0));
            peaks = top2;
        }
    }
    // Fill-density gate: prose columns fill each inter-peak slot nearly
    // edge-to-edge; table cells leave wide whitespace. Applies to ALL peak
    // counts including 2 — a 2-col label/value table (description list)
    // produces the same two left-edge peaks as 2-col prose, and slicing it
    // strands each half in its own leaf where no table detector can see
    // the rows whole. Per band, measure each column's covered span / slot
    // width; if too sparse, treat as a table and bail.
    {
        let mut n = peaks.len();
        let mut covered_sum = vec![0.0f32; n];
        let mut band_count = vec![0usize; n];
        // Bands where everything in the column is display math or an
        // equation number (`math_bands`), or that plus formula fragments the
        // math pass left as text (`formula_bands`).
        let mut math_bands = vec![0usize; n];
        let mut formula_bands = vec![0usize; n];
        let mut kk = 0;
        while kk < sorted.len() {
            let by = items[sorted[kk]].item.y;
            let mut jj = kk;
            while jj < sorted.len() && (items[sorted[jj]].item.y - by).abs() <= band_tol {
                jj += 1;
            }
            let mut lo_x = vec![f32::INFINITY; n];
            let mut hi_x = vec![f32::NEG_INFINITY; n];
            let mut all_math = vec![true; n];
            let mut all_formula = vec![true; n];
            for &idx in &sorted[kk..jj] {
                let it = &items[idx].item;
                if !it.x.is_finite() {
                    continue;
                }
                // Peak centers are bucket-quantized, so a column's items can
                // start up to a bucket left of its peak center; without the
                // slack they all bin into the previous slot and that column
                // reads as empty (its sparse cells never get fill-checked).
                let mut c = 0;
                while c + 1 < n && it.x >= peaks[c + 1].0 - bucket_pt {
                    c += 1;
                }
                lo_x[c] = lo_x[c].min(it.x);
                hi_x[c] = hi_x[c].max(it.x + it.width.max(0.0));
                let math = it.math || is_equation_number(&it.text);
                all_math[c] &= math;
                all_formula[c] &= math || is_formula_fragment(&it.text);
            }
            for c in 0..n {
                if hi_x[c] > lo_x[c] {
                    let col_w = if c + 1 < n {
                        peaks[c + 1].0 - peaks[c].0
                    } else {
                        max_x - peaks[c].0
                    };
                    if col_w > 1.0 {
                        covered_sum[c] += ((hi_x[c] - lo_x[c]) / col_w).min(1.0);
                        band_count[c] += 1;
                        if all_math[c] {
                            math_bands[c] += 1;
                        }
                        if all_formula[c] {
                            formula_bands[c] += 1;
                        }
                    }
                }
            }
            kk = jj;
        }
        // Two-part gate that distinguishes prose columns from tables:
        //
        //   1. min_fill floor (MIN_FILL_HARD_FLOOR = 0.38) — rejects
        //      pages where any column has narrow text in many bands
        //      (e.g. table with numeric column). Real tables have
        //      at least one column with fill ≤ 0.36.
        //   2. Band-count-weighted avg_fill ≥ XY_COLUMN_MIN_FILL —
        //      catches table layouts where ALL columns are uniformly
        //      narrow (no column dominates the weighted average).
        //
        // Both must pass. The min-only gate was too brittle: a real
        // 3-column prose page with one page-bottom column having ~4
        // bands of short text dragged the min below 0.55 even though
        // every meaningful column was clearly prose. The avg-only
        // gate let through tables where the narrow column had enough
        // bands to be informative but its weight was diluted by
        // wider neighboring columns.
        let mut min_fill = f32::INFINITY;
        let mut weighted_sum = 0.0f32;
        let mut weighted_n = 0.0f32;
        let mut per_col_fill: Vec<f32> = Vec::with_capacity(n);
        for c in 0..n {
            if band_count[c] > 0 {
                let f = covered_sum[c] / band_count[c] as f32;
                per_col_fill.push(f);
                min_fill = min_fill.min(f);
                weighted_sum += f * band_count[c] as f32;
                weighted_n += band_count[c] as f32;
            }
        }
        let mut avg_fill = if weighted_n > 0.0 {
            weighted_sum / weighted_n
        } else {
            f32::INFINITY
        };
        const MIN_FILL_HARD_FLOOR: f32 = 0.38;
        // Figure-scatter phantom column: on a 2-column page with an embedded
        // plot (scatter cloud, axis tick labels), the plot's x-spread items
        // form a third peak that is BOTH sparsely-banded (far fewer text rows
        // than the real columns) and low-fill. Left in, it drags min_fill
        // under the floor and flips the whole region to `tabular`, suppressing
        // the genuine column split. A real narrow table column instead has a
        // band count comparable to its siblings (one cell per row), so the
        // band-count ratio cleanly separates the two. Drop phantom peaks when
        // ≥2 well-filled columns survive, then re-run the gate on them.
        if n > 2 {
            let max_bands = band_count.iter().copied().max().unwrap_or(0);
            if max_bands > 0 {
                let band_floor =
                    (max_bands as f32 * XY_COLUMN_PHANTOM_BAND_FRACTION).ceil() as usize;
                let col_fill = |c: usize| -> f32 {
                    if band_count[c] > 0 {
                        covered_sum[c] / band_count[c] as f32
                    } else {
                        0.0
                    }
                };
                // Note: display equations and their numbers inside a prose
                // column line up into the same sparse, low-fill peak (a
                // two-column paper read as a table); a column made mostly
                // of math bands is a phantom at any band count.
                // Dropping still needs the surviving columns to read as
                // prose, so a numeric table column keeps the region tabular.
                let display_math = |c: usize| {
                    math_bands[c] * 3 >= band_count[c] && formula_bands[c] * 3 >= band_count[c] * 2
                };
                let phantom = |c: usize| {
                    col_fill(c) < MIN_FILL_HARD_FLOOR
                        && (band_count[c] < band_floor || display_math(c))
                };
                let survivors: Vec<usize> = (0..n).filter(|&c| !phantom(c)).collect();
                let survivors_filled = survivors
                    .iter()
                    .all(|&c| col_fill(c) >= MIN_FILL_HARD_FLOOR);
                if survivors.len() >= 2 && survivors.len() < n && survivors_filled {
                    let kept: Vec<(f32, usize)> = survivors.iter().map(|&c| peaks[c]).collect();
                    let mut new_min = f32::INFINITY;
                    let mut wsum = 0.0f32;
                    let mut wn = 0.0f32;
                    for &c in &survivors {
                        let f = col_fill(c);
                        new_min = new_min.min(f);
                        wsum += f * band_count[c] as f32;
                        wn += band_count[c] as f32;
                    }
                    peaks = kept;
                    n = peaks.len();
                    min_fill = new_min;
                    avg_fill = if wn > 0.0 { wsum / wn } else { f32::INFINITY };
                }
            }
        }
        // The min-floor catches ≥3-col tables whose narrow numeric column is
        // diluted out of the weighted average by wide neighbors. With only 2
        // columns no dilution is possible — a sparse column of two drags the
        // average itself — and a short second prose column (a page-bottom
        // column with a few bands) would false-positive here, suppressing a
        // needed column split AND the density V via the tabular flag.
        if n > 2 && min_fill.is_finite() && min_fill < MIN_FILL_HARD_FLOOR {
            *tabular = true;
            return None;
        }
        if avg_fill.is_finite() && avg_fill < XY_COLUMN_MIN_FILL {
            *tabular = true;
            return None;
        }
    }
    // Require the two peaks to together dominate the histogram — a real
    // 2-column layout has the great majority of items starting at one of
    // the two column-left edges. Tables, lists, and prose with scattered
    // indents distribute their item-starts across many x's and won't pass.
    let peak_sum: usize = peaks.iter().map(|(_, c)| *c).sum();
    let dominance = (peak_sum as f32) / (n_starts as f32);
    if dominance < XY_COLUMN_PEAK_DOMINANCE {
        return None;
    }
    // Require the peaks to be roughly balanced in size. A real multi-column
    // layout has comparable line counts in each column; a long-single-column
    // page that happens to have one bullet stack at a different x produces
    // a tall + tiny peak pair that should not trigger a column split. Check
    // the global min/max across all kept peaks so an N-column layout with one
    // short column is also rejected.
    let smallest = peaks.iter().map(|(_, c)| *c).min().unwrap() as f32;
    let largest = peaks.iter().map(|(_, c)| *c).max().unwrap() as f32;
    let balance = if largest > 0.0 {
        smallest / largest
    } else {
        0.0
    };
    if balance < XY_COLUMN_PEAK_BALANCE_RATIO {
        return None;
    }

    // Cut at the leftmost gutter that clears `XY_COLUMN_MIN_GAP_FRACTION ×
    // total_w` (list-item / block-quote indents produce closely spaced peaks;
    // a real column gutter is much further apart). For a 2-column layout this
    // is the only gap. For N columns, peeling the leftmost column first and
    // letting the recursion re-run on the remainder splits the rest one
    // gutter at a time.
    let min_gap = total_w * XY_COLUMN_MIN_GAP_FRACTION;
    let mut best: Option<(f32, usize)> = None;
    for w_idx in 0..peaks.len() - 1 {
        let gap = peaks[w_idx + 1].0 - peaks[w_idx].0;
        if gap >= min_gap {
            best = Some((gap, w_idx));
            break;
        }
    }
    let li = match best {
        Some((_, li)) => li,
        None => return None,
    };
    // Place the cut at the actual gutter, not midway between peak centers.
    // Peaks mark column LEFT edges, so the center midpoint lands inside the
    // left column's text span and misroutes its mid-line items under
    // partition_by_item. The gutter spans from the rightmost left-side item
    // extent to the right column's leftmost counted start — NOT the peak
    // center: a right-aligned numeric column scatters its starts left of
    // the bucket-quantized center, and a cut placed at the center would
    // route those values into the left column.
    let mid_centers = (peaks[li].0 + peaks[li + 1].0) * 0.5;
    let right_min_start = counted_starts
        .iter()
        .copied()
        .filter(|&x| x > mid_centers)
        .fold(f32::INFINITY, f32::min);
    let boundary = if right_min_start.is_finite() {
        right_min_start
    } else {
        peaks[li + 1].0
    };
    let mut gutter_left = f32::NEG_INFINITY;
    for &i in idxs {
        let it = &items[i].item;
        if it.x.is_finite() {
            let r = it.x + it.width.max(0.0);
            if r <= boundary - 1.0 && r > gutter_left {
                gutter_left = r;
            }
        }
    }
    let cut_x = if gutter_left.is_finite() && gutter_left > peaks[li].0 {
        (gutter_left + boundary) * 0.5
    } else {
        mid_centers
    };
    // Validate cut lands inside the bbox; if bbox-clamping moved the bbox
    // off the items (some PDFs have content with negative x), fall back to
    // an item-relative midpoint check.
    if cut_x <= min_x + 1.0 || cut_x >= max_x - 1.0 {
        return None;
    }
    // High finite score so this beats every density cut but stays below
    // banner's infinity.
    Some(CutCandidate {
        axis: CutAxis::Vertical,
        position: cut_x,
        score: 1.0e9,
    })
}

/// X position of the vertical gutter that splits `idxs` (within `bbox`) into
/// columns — from the density V-cut or the column-start histogram fallback.
/// `None` when the region is single-column. Used to decide whether peeling a
/// wide spanning line above this region is productive.
fn column_cut_x(
    items: &[ProjectedTextItem],
    idxs: &[usize],
    bbox: &Rect,
    median_h: f32,
    figures: &[Rect],
) -> Option<f32> {
    xy_find_best_cut(items, idxs, bbox, CutAxis::Vertical, median_h, figures)
        .or_else(|| {
            let mut tabular = false;
            xy_find_column_cut(items, idxs, bbox, median_h, &mut tabular)
        })
        .map(|c| c.position)
}

/// Union horizontal extent (min x, max x) of `idxs`.
fn x_extent(items: &[ProjectedTextItem], idxs: &[usize]) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &i in idxs {
        let it = &items[i].item;
        lo = lo.min(it.x);
        hi = hi.max(it.x + it.width.max(0.0));
    }
    (lo, hi)
}

/// Horizontal cut that isolates lines crossing a vertical gutter at `gutter_x` (see
/// `xy_cut_rec`). A band crosses when one of its items reaches at least three line heights past
/// the gutter on both sides (a heading or float spanning the columns, not a line of the left
/// column overhanging a misplaced gutter), and at most a third of the bands cross. Cuts above the first crossing band that has non-crossing bands
/// above it; when the crossing bands only open the region, cuts below them.
fn xy_find_crossing_line_cut(
    items: &[ProjectedTextItem],
    idxs: &[usize],
    gutter_x: f32,
    median_h: f32,
) -> Option<CutCandidate> {
    let mut sorted: Vec<usize> = idxs.to_vec();
    sorted.sort_by(|&a, &b| items[a].item.y.total_cmp(&items[b].item.y));
    let band_tol = (median_h * 0.5).max(2.0);
    let reach = 3.0 * median_h.max(4.0);
    // (y_min, y_max, crosses)
    let mut bands: Vec<(f32, f32, bool)> = Vec::new();
    for &i in &sorted {
        let it = &items[i].item;
        let (y0, y1) = (it.y, it.y + it.height.max(0.0));
        let crosses = it.x < gutter_x - reach && it.x + it.width > gutter_x + reach;
        match bands.last_mut() {
            Some(last)
                if y0 <= last.1 - band_tol || (y0 - (last.0 + last.1) * 0.5).abs() <= band_tol =>
            {
                last.0 = last.0.min(y0);
                last.1 = last.1.max(y1);
                last.2 |= crosses;
            }
            _ => bands.push((y0, y1, crosses)),
        }
    }
    // Spanning lines are a minority of a region's lines; when most lines cross, the gutter is
    // misplaced (inside a column), not crossed.
    let crossing = bands.iter().filter(|b| b.2).count();
    if bands.len() < 3 || crossing == 0 || crossing * 3 > bands.len() {
        return None;
    }
    let between = |a: usize| CutCandidate {
        axis: CutAxis::Horizontal,
        position: (bands[a].1 + bands[a + 1].0) * 0.5,
        score: f32::INFINITY,
    };
    let first = bands.iter().position(|b| b.2)?;
    if first > 0 {
        return Some(between(first - 1));
    }
    let last_opening = bands.iter().position(|b| !b.2)? - 1;
    Some(between(last_opening))
}

fn xy_find_banner_cut(
    items: &[ProjectedTextItem],
    idxs: &[usize],
    bbox: &Rect,
    median_h: f32,
) -> Option<CutCandidate> {
    if bbox.width <= 1.0 || idxs.len() < 2 {
        return None;
    }
    let mut sorted: Vec<usize> = idxs.to_vec();
    sorted.sort_by(|&a, &b| items[a].item.y.total_cmp(&items[b].item.y));
    let band_tol = (median_h * 0.5).max(2.0);
    let min_clearance = (median_h * XY_BANNER_CLEARANCE_FACTOR).max(8.0);
    let width_threshold = XY_BANNER_WIDTH_FRACTION * bbox.width;

    // Group items into y-bands. A new item joins the current band when its top
    // is within `band_tol` of either the band's center or its existing bottom
    // (handles slight baseline variation).
    let mut bands: Vec<(f32, f32, f32, f32)> = Vec::new(); // (y_min, y_max, x_min, x_max)
    for &i in &sorted {
        let it = &items[i].item;
        let y0 = it.y;
        let y1 = it.y + it.height.max(0.0);
        let x0 = it.x;
        let x1 = it.x + it.width.max(0.0);
        let mut merged = false;
        if let Some(last) = bands.last_mut() {
            let center = (last.0 + last.1) * 0.5;
            if (y0 - center).abs() <= band_tol || y0 <= last.1 + band_tol {
                last.0 = last.0.min(y0);
                last.1 = last.1.max(y1);
                last.2 = last.2.min(x0);
                last.3 = last.3.max(x1);
                merged = true;
            }
        }
        if !merged {
            bands.push((y0, y1, x0, x1));
        }
    }
    if bands.is_empty() {
        return None;
    }
    let is_wide = |b: &(f32, f32, f32, f32)| -> bool { (b.3 - b.2) >= width_threshold };

    // Walk top→bottom. For each run of consecutive wide bands (close enough to
    // each other to count as a single banner stack), check above/below clearance
    // and emit a cut if isolated.
    let mut i = 0;
    while i < bands.len() {
        if !is_wide(&bands[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        // Extend through consecutive wide bands separated by less than the
        // clearance threshold — multi-line title + authors counts as one
        // banner stack.
        while end + 1 < bands.len()
            && is_wide(&bands[end + 1])
            && bands[end + 1].0 - bands[end].1 < min_clearance
        {
            end += 1;
        }
        let banner_top = bands[start].0;
        let banner_bot = bands[end].1;
        let above_y = if start > 0 {
            Some(bands[start - 1].1)
        } else {
            None
        };
        let below_y = if end + 1 < bands.len() {
            Some(bands[end + 1].0)
        } else {
            None
        };
        let above_clear = above_y.is_none_or(|ay| banner_top - ay >= min_clearance);
        let below_clear = below_y.is_none_or(|by| by - banner_bot >= min_clearance);

        if above_clear && below_clear {
            // Prefer cutting BELOW the banner so it becomes the top region of
            // the split (matches natural reading order). Fall back to cutting
            // above only when there's nothing below to cut against.
            let cut_y = if let Some(by) = below_y {
                (banner_bot + by) * 0.5
            } else if let Some(ay) = above_y {
                (ay + banner_top) * 0.5
            } else {
                // Banner spans the whole region — no productive cut.
                i = end + 1;
                continue;
            };
            // Skip degenerate cuts that would leave one side empty.
            if cut_y > bbox.y + 1.0 && cut_y < bbox.y + bbox.height - 1.0 {
                return Some(CutCandidate {
                    axis: CutAxis::Horizontal,
                    position: cut_y,
                    score: f32::INFINITY,
                });
            }
        }
        i = end + 1;
    }
    None
}

fn xy_cut_rec(
    items: &[ProjectedTextItem],
    idxs: Vec<usize>,
    bbox: Rect,
    depth: u32,
    inherited_tabular: bool,
    median_h: f32,
    figures: &[Rect],
) -> Region {
    if idxs.len() <= XY_MIN_ITEMS_TO_CUT || depth >= XY_MAX_DEPTH {
        return Region {
            bbox,
            kind: RegionKind::Leaf { item_indices: idxs },
        };
    }
    // Banner cut wins over density-based valleys. Score is f32::INFINITY so
    // the existing comparison logic naturally selects it.
    let banner = xy_find_banner_cut(items, &idxs, &bbox, median_h);
    let h = xy_find_best_cut(items, &idxs, &bbox, CutAxis::Horizontal, median_h, figures);
    let v = xy_find_best_cut(items, &idxs, &bbox, CutAxis::Vertical, median_h, figures);

    // Fallback column-start histogram. Only consulted when BOTH density
    // cuts returned None — those are the cases (wide-heading filled gutter,
    // figure-straddle layouts) where the column structure is real but the
    // density projection can't see it. If either density cut found a
    // valley, trust it instead of firing the histogram fallback.
    // Run the histogram probe unconditionally: besides being a fallback cut
    // source, its fill gate is the region-level table detector — when it
    // rejects with "tabular", density V-cuts must not slice the table's
    // inter-column gutters either (the dominant column-major-table failure).
    let mut region_tabular = false;
    let column_full = xy_find_column_cut(items, &idxs, &bbox, median_h, &mut region_tabular);
    let effective_tabular = region_tabular || (inherited_tabular && column_full.is_none());
    let column = if v.is_none() && h.is_none() {
        column_full
    } else {
        None
    };
    // Build a priority-ordered candidate list.
    //
    // **Primary** is the cut chosen by the main resolution order
    // (banner → column → H/V resolved by score margin). **Fallback** is
    // the column-histogram probe — but ONLY that, not density V or the
    // other density axis. The column histogram has strong discriminators
    // (2 strong dominant balanced peaks); a density V valley that lost to
    // H by the score margin is often a layout coincidence (mid-line word
    // clustering) we shouldn't trust as a fallback. Falling through to
    // the histogram-only fallback — rather than collapsing to a leaf —
    // is what unblocks 2-column pages whose density H-cut peels a
    // footer/header but trips the min-lines guard.
    let (h_score, v_score) = (h.as_ref().map(|c| c.score), v.as_ref().map(|c| c.score));
    let prefer_h_over_v = match (h_score, v_score) {
        (Some(hs), Some(vs)) => vs <= hs * XY_V_PREFERENCE_MARGIN,
        _ => true,
    };
    let mut candidates: Vec<CutCandidate> = Vec::new();
    if let Some(bc) = banner {
        candidates.push(bc);
    }
    // A line that crosses the would-be gutter (a section heading, a spanning float) separates
    // column runs; the gutter is the density valley's when there is one (nothing crosses it, so
    // the check stays quiet), else the column histogram's, which crossing lines don't block.
    // Cut above it before cutting columns, or the columns of every section join into one
    // page-long column and a heading is sliced in two.
    if !effective_tabular
        && let Some(x) = v.as_ref().or(column_full.as_ref()).map(|c| c.position)
        && let Some(hc) = xy_find_crossing_line_cut(items, &idxs, x, median_h)
    {
        candidates.push(hc);
    }
    if let Some(cc) = column {
        candidates.push(cc);
    }
    // Primary density cut, then the other axis as fallback. Without the
    // alternate-axis fallback, a low-impact H-cut that wins by the score
    // margin (e.g. peeling a 1-line footer) trips the min-lines guard and
    // collapses the whole region to a single leaf — masking the real
    // column structure the V-cut would have revealed.
    // When the histogram fill gate classified this region as tabular, a
    // density V valley here is a table gutter, not a column boundary —
    // suppress it so the table reaches the table detectors row-major.
    let v = if effective_tabular { None } else { v };
    if prefer_h_over_v {
        if let Some(hc) = h {
            candidates.push(hc);
        }
        if let Some(vc) = v {
            candidates.push(vc);
        }
    } else {
        if let Some(vc) = v {
            candidates.push(vc);
        }
        if let Some(hc) = h {
            candidates.push(hc);
        }
    }
    // Final fallback: re-use the unconditional column-histogram probe.
    // Skipped when an existing candidate is already a histogram-derived
    // column cut (score = 1.0e9 from `xy_find_column_cut`).
    if !candidates
        .iter()
        .any(|c| c.axis == CutAxis::Vertical && (c.score - 1.0e9).abs() < 1.0)
        && let Some(cc) = column_full
    {
        candidates.push(cc);
    }

    for cut in candidates {
        let mut left: Vec<usize> = Vec::new();
        let mut right: Vec<usize> = Vec::new();
        // Column-histogram V-cuts (score = 1.0e9) split between two column
        // left-edge clusters. The cut sits at the midpoint between the two
        // peak x's, which is typically far inside the left column. PDFium
        // routinely emits a single visual line as multiple items (line-start
        // + mid-line emphasis fragments + late-line clusters); each item has
        // its own x, so an item-by-item partition by `it.x` will split one
        // logical line across both sides whenever a continuation fragment
        // starts past `cut.position`. Verified on `031a888e…page_1` where
        // a histogram V-cut at x=360.8 inside column 2 routed the body-start
        // fragments at x=312 LEFT but their same-baseline continuation
        // "In Section 2 the" at x=474 RIGHT, orphaning the snippet from its
        // paragraph and producing a stranded block downstream.
        //
        // Fix: for histogram V-cuts, first measure how many y-bands have items
        // on *both* sides of the cut ("straddling rows"). A real column
        // boundary has every body row straddling (text in left + right column
        // on the same baseline) — partition by `it.x` so each column gets its
        // items. An intra-column histogram split (031a888e case: a single line
        // wrapped into multiple items at different x's) has almost no
        // straddling rows — partition by row leftmost-x so a wrapped line
        // stays whole. The threshold (>~25% of rows straddle) cleanly
        // separates: 0145bc47 (every body row straddles, ~100%) gets per-item
        // partition; 031a888e (1 of ~45 rows straddles, ~2%) gets per-row.
        //
        // Density V-cuts (white-space valleys) still use centroid: they only
        // fire where a real text-free vertical band exists, so an item that
        // crosses a density cut is genuinely wider than the column.
        let is_histogram_v_cut = cut.axis == CutAxis::Vertical && (cut.score - 1.0e9).abs() < 1.0;
        if is_histogram_v_cut {
            // Sort by y then x so items on one baseline cluster together.
            let mut sorted_idxs: Vec<usize> = idxs.clone();
            sorted_idxs.sort_by(|&a, &b| {
                items[a]
                    .item
                    .y
                    .total_cmp(&items[b].item.y)
                    .then(items[a].item.x.total_cmp(&items[b].item.x))
            });
            let band_tol = (median_h * 0.5).max(2.0);

            // First pass: count rows with items entirely-left, entirely-right,
            // or straddling the cut. Decides the partition strategy.
            let (mut rows_total, mut rows_straddle) = (0usize, 0usize);
            let mut k = 0;
            while k < sorted_idxs.len() {
                let band_y = items[sorted_idxs[k]].item.y;
                let mut j = k;
                let mut has_left = false;
                let mut has_right = false;
                while j < sorted_idxs.len()
                    && (items[sorted_idxs[j]].item.y - band_y).abs() <= band_tol
                {
                    let ix = items[sorted_idxs[j]].item.x;
                    if ix.is_finite() {
                        if ix < cut.position {
                            has_left = true;
                        } else {
                            has_right = true;
                        }
                    }
                    j += 1;
                }
                if has_left || has_right {
                    rows_total += 1;
                    if has_left && has_right {
                        rows_straddle += 1;
                    }
                }
                k = j;
            }
            // 25% threshold: when a quarter or more of rows have content on
            // both sides of the cut, it's a real column boundary (every body
            // row has text in both columns). Below that, the cut sits inside
            // a single column and the rare straddler is an intra-column
            // wrapped line that must not be torn apart.
            const STRADDLE_PARTITION_FRACTION: f32 = 0.25;
            let partition_by_item = rows_total > 0
                && (rows_straddle as f32) / (rows_total as f32) >= STRADDLE_PARTITION_FRACTION;

            if partition_by_item {
                for &i in &idxs {
                    if items[i].item.x < cut.position {
                        left.push(i);
                    } else {
                        right.push(i);
                    }
                }
            } else {
                let mut k = 0;
                while k < sorted_idxs.len() {
                    let band_y = items[sorted_idxs[k]].item.y;
                    let mut j = k;
                    let mut min_x = f32::INFINITY;
                    while j < sorted_idxs.len()
                        && (items[sorted_idxs[j]].item.y - band_y).abs() <= band_tol
                    {
                        let ix = items[sorted_idxs[j]].item.x;
                        if ix.is_finite() && ix < min_x {
                            min_x = ix;
                        }
                        j += 1;
                    }
                    let side_left = min_x < cut.position;
                    for &i in &sorted_idxs[k..j] {
                        if side_left {
                            left.push(i);
                        } else {
                            right.push(i);
                        }
                    }
                    k = j;
                }
            }
        } else {
            for &i in &idxs {
                let it = &items[i].item;
                let split_at = match cut.axis {
                    CutAxis::Horizontal => it.y + it.height.max(0.0) * 0.5,
                    CutAxis::Vertical => it.x + it.width.max(0.0) * 0.5,
                };
                if split_at < cut.position {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
        }
        if left.is_empty() || right.is_empty() {
            continue;
        }
        let (left_bbox, right_bbox) = xy_split_bbox(&bbox, &cut);

        // Banner cuts (score = ∞) at the page root are explicitly meant to
        // isolate single-line wide headers like titles, so skip the
        // min-lines guard for them. At deeper recursion levels we keep the
        // guard — an aggressive single-line banner cut mid-recursion
        // fragments paragraphs around emphasized lines (e.g. a centered
        // "Figure N caption" sandwiched between body paragraphs would
        // otherwise carve out its own sliver region and break paragraph
        // continuation).
        let allow_single_line =
            cut.axis == CutAxis::Horizontal && cut.score.is_infinite() && depth == 0;
        if cut.axis == CutAxis::Horizontal && !allow_single_line {
            let lc = xy_distinct_lines(items, &left, median_h);
            let rc = xy_distinct_lines(items, &right, median_h);
            // Rescue: a single wide spanning line (e.g. a centered
            // authors / affiliation line above a two-column body) is
            // worth isolating even mid-recursion when peeling it reveals
            // the column gutter underneath. Otherwise the spanning line
            // straddles the gutter, the V-cut can never fire, and the
            // whole body interleaves column-by-column. The density H-cut
            // that peels it has a finite score (the line is often just
            // under the banner width threshold), so this is NOT gated on
            // a banner cut. Discriminator vs. carving a single-column
            // caption out of running prose: the thin side must be one
            // line whose x-extent crosses the gutter revealed on the
            // other (multi-line) side.
            let spanning_rescue = {
                let thin_wide = if lc < XY_MIN_LINES_PER_H_SIDE && rc >= XY_MIN_LINES_PER_H_SIDE {
                    Some(((&left, lc), (&right, &right_bbox)))
                } else if rc < XY_MIN_LINES_PER_H_SIDE && lc >= XY_MIN_LINES_PER_H_SIDE {
                    Some(((&right, rc), (&left, &left_bbox)))
                } else {
                    None
                };
                thin_wide.is_some_and(|((thin, thin_lines), (wide, wide_bbox))| {
                    thin_lines <= 1
                        && column_cut_x(items, wide, wide_bbox, median_h, figures).is_some_and(
                            |gx| {
                                let (tx0, tx1) = x_extent(items, thin);
                                tx0 < gx - 1.0 && tx1 > gx + 1.0
                            },
                        )
                })
            };
            if (lc < XY_MIN_LINES_PER_H_SIDE || rc < XY_MIN_LINES_PER_H_SIDE) && !spanning_rescue {
                continue;
            }
        }
        // Vertical (column) min-lines guard. A real column gutter is sustained
        // across multiple stacked lines on both sides. A vertical cut that
        // isolates a single-line "column" is a false gutter: in single-column
        // justified prose, a short final line plus a wide inter-word gap in the
        // full-width line above it align an empty vertical band that is not a
        // column boundary. Without this, that band splits the physical line
        // into two sibling regions and the markdown emitter renders them out of
        // order (e.g. "…and its share repurchases are 100%" → the tail is
        // sliced off into its own region). The initial true 2-column split has
        // many lines per side, so this never blocks real columns.
        if cut.axis == CutAxis::Vertical {
            let lc = xy_distinct_lines(items, &left, median_h);
            let rc = xy_distinct_lines(items, &right, median_h);
            if lc < XY_MIN_LINES_PER_H_SIDE || rc < XY_MIN_LINES_PER_H_SIDE {
                continue;
            }
        }

        let child_inherited_tabular = effective_tabular && cut.axis == CutAxis::Horizontal;
        let left_region = xy_cut_rec(
            items,
            left,
            left_bbox,
            depth + 1,
            child_inherited_tabular,
            median_h,
            figures,
        );
        let right_region = xy_cut_rec(
            items,
            right,
            right_bbox,
            depth + 1,
            child_inherited_tabular,
            median_h,
            figures,
        );
        return Region {
            bbox,
            kind: RegionKind::Split {
                axis: cut.axis,
                children: vec![left_region, right_region],
            },
        };
    }

    Region {
        bbox,
        kind: RegionKind::Leaf { item_indices: idxs },
    }
}

/// Build the page's XY-cut body region tree.
///
/// Exclude isolated footer numbers from both density/banner cuts and the root
/// bounds. One extra bottom line can change which column split is accepted and
/// alter reading order. Retain the original items for a detached footer line.
///
/// `figures` is an optional set of figure-region bounding rectangles (from
/// `figure_cluster::detect_figure_rects`); pass `&[]` to disable obstacle
/// seeding. Figures cause the recursion to favor cuts around them.
pub(crate) fn xy_cut(
    items: &[ProjectedTextItem],
    page_width: f32,
    page_height: f32,
    figures: &[Rect],
) -> Region {
    let all: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(i, it)| {
            !it.item.text.is_empty() && !isolated_numeric_footer(items, *i, page_width, page_height)
        })
        .map(|(i, _)| i)
        .collect();
    if all.is_empty() {
        return Region {
            bbox: Rect {
                x: 0.0,
                y: 0.0,
                width: page_width.max(1.0),
                height: page_height.max(1.0),
            },
            kind: RegionKind::Leaf {
                item_indices: Vec::new(),
            },
        };
    }
    let bbox = xy_root_bbox(items, &all, page_width, page_height);
    let median_h = xy_median_line_height(items, &all);
    xy_cut_rec(items, all, bbox, 0, false, median_h, figures)
}

fn xy_walk_leaves(region: &Region, path: &mut Vec<u16>, out: &mut Vec<(Vec<u16>, Vec<usize>)>) {
    match &region.kind {
        RegionKind::Leaf { item_indices } => {
            if !item_indices.is_empty() {
                out.push((path.clone(), item_indices.clone()));
            }
        }
        RegionKind::Split { children, .. } => {
            for (i, child) in children.iter().enumerate() {
                path.push(i as u16);
                xy_walk_leaves(child, path, out);
                path.pop();
            }
        }
    }
}

/// One recognized line in page points: left, top, width and height.
pub(super) struct OcrLine {
    pub bx: [f32; 4],
    pub text: String,
    /// Quarter-turn rotation in degrees counter-clockwise.
    pub rotation: f32,
    pub font_size: f32,
    pub confidence: f32,
    pub faint: bool,
}

fn canonical_quarter(rot: f32) -> u32 {
    ((rot.rem_euclid(360.0) / 90.0).round() as u32 % 4) * 90
}

/// The quarter turn a line's polygon reads at, from its longer side. Near-square polygons have no reliable reading axis and count as upright.
pub(super) fn polygon_rotation(poly: &[[f32; 2]; 4]) -> f32 {
    let e0 = [poly[1][0] - poly[0][0], poly[1][1] - poly[0][1]];
    let e1 = [poly[2][0] - poly[1][0], poly[2][1] - poly[1][1]];
    let len0 = (e0[0] * e0[0] + e0[1] * e0[1]).sqrt();
    let len1 = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt();
    if len0.max(len1) < 1.0 {
        return 0.0;
    }
    let (longer, shorter) = if len0 >= len1 {
        (len0, len1)
    } else {
        (len1, len0)
    };
    if shorter > 0.0 && longer / shorter < 1.3 {
        return 0.0;
    }
    let reading = if len0 >= len1 { e0 } else { e1 };
    // atan2 with screen-down y, negated for the counter-clockwise angle.
    canonical_quarter(-reading[1].atan2(reading[0]).to_degrees()) as f32
}

/// Strip table border misreads (`|`, brackets) from around numeric cell text.
fn clean_ocr_table_artifacts(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let without_artifacts: &str = trimmed
        .trim_start_matches(['|', '[', ']', '(', ')', '{', '}'])
        .trim_end_matches(['|', '[', ']', '(', ')', '{', '}'])
        .trim();
    if without_artifacts.is_empty() {
        return trimmed.to_string();
    }
    // Only numeric-looking content is cleaned, so "(note)" keeps its brackets.
    let is_numeric_ish = without_artifacts
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, ',' | '.' | ' ' | '%' | '-' | '+' | '*' | '/'))
        || without_artifacts == "N/A"
        || without_artifacts == "Z"
        || without_artifacts == "-";
    if is_numeric_ish {
        // Parentheses directly around a number mark it negative, so that pair stays.
        if let Some((before, after)) = trimmed.split_once(without_artifacts)
            && before.ends_with('(')
            && after.starts_with(')')
        {
            return format!("({without_artifacts})");
        }
        without_artifacts.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Share of a page's text, by characters, that must share one quarter turn for the page to be read as a scan on its side or upside down.
const ROTATED_SCAN_MIN_SHARE: f32 = 0.7;

/// Project the recognized lines of a page without a text layer.
///
/// Returns the projected items, the quarter turns (0..=3) the page was read at, and the size of the frame the items are in. A page whose text mostly shares one rotation is turned upright first, and a quarter turn swaps the frame's sides.
pub(super) fn project_ocr_page(
    page: (f32, f32),
    lines: Vec<OcrLine>,
) -> (Vec<Projected>, usize, (f32, f32)) {
    let mut items: Vec<TextItem> = lines
        .into_iter()
        .filter(|line| line.confidence > 0.1)
        .filter_map(|line| {
            let text = clean_ocr_table_artifacts(&line.text);
            (!text.is_empty()).then(|| TextItem {
                text,
                x: line.bx[0],
                y: line.bx[1],
                width: line.bx[2],
                height: line.bx[3],
                rotation: line.rotation,
                font_name: Some("OCR".to_string()),
                font_size: Some(line.font_size),
                confidence: Some((line.confidence * 1000.0).round() / 1000.0),
                ocr_faint: line.faint,
                ..Default::default()
            })
        })
        .collect();
    let mut chars = [0usize; 4];
    for it in &items {
        chars[(canonical_quarter(it.rotation) / 90) as usize] += it.text.chars().count();
    }
    let total: usize = chars.iter().sum();
    let (mut turn, &top) = chars.iter().enumerate().max_by_key(|&(_, c)| *c).unwrap();
    if turn == 0 || total < 20 || (top as f32) < total as f32 * ROTATED_SCAN_MIN_SHARE {
        turn = 0;
    }
    let (w, h) = page;
    let mut frame = page;
    if turn != 0 {
        for it in &mut items {
            [it.x, it.y, it.width, it.height] =
                turn_box([it.x, it.y, it.width, it.height], turn, page);
            it.rotation = 0.0;
        }
        if turn != 2 {
            frame = (h, w);
        }
    }
    let boxes = items
        .into_iter()
        .map(|item| ProjectedTextItem {
            orig_x: item.x,
            orig_y: item.y,
            orig_width: item.width,
            orig_height: item.height,
            item,
            is_margin_line_number: false,
            rotated: false,
            d: 0.0,
        })
        .collect();
    (project_signals(frame.0, frame.1, boxes), turn, frame)
}

/// A box on the page, as left, top, width and height, in the reading frame of a page read at `turn` quarter turns.
pub(super) fn turn_box(b: [f32; 4], turn: usize, page: (f32, f32)) -> [f32; 4] {
    let [x, y, bw, bh] = b;
    let (w, h) = page;
    match turn {
        // Reading bottom to top: the page's top edge is on the left.
        1 => [h - (y + bh), x, bh, bw],
        2 => [w - (x + bw), h - (y + bh), bw, bh],
        // Reading top to bottom: the page's top edge is on the right.
        3 => [y, w - (x + bw), bh, bw],
        _ => b,
    }
}

/// A box of a turned page's reading frame, back on the page as left, top, width and height.
pub(super) fn unturn(b: [f32; 4], turn: usize, page: (f32, f32)) -> [f32; 4] {
    let [x, y, bw, bh] = b;
    let (w, h) = page;
    match turn {
        1 => [y, h - x - bw, bh, bw],
        2 => [w - x - bw, h - y - bh, bw, bh],
        3 => [w - y - bh, x, bh, bw],
        _ => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(x: f32, y: f32, w: f32, text: &str) -> OcrLine {
        OcrLine {
            bx: [x, y, w, 12.0],
            text: text.to_string(),
            rotation: 0.0,
            font_size: 12.0,
            confidence: 0.95,
            faint: false,
        }
    }

    #[test]
    fn a_column_of_cells_forms_anchor_groups_on_its_shared_edges() {
        // A label column with ragged right edges and a value column whose cells share both edges.
        let mut lines = Vec::new();
        for (row, label) in ["Revenue", "Cost of sales", "Margin"]
            .into_iter()
            .enumerate()
        {
            let y = 100.0 + 30.0 * row as f32;
            lines.push(line(50.0, y, 8.0 * label.len() as f32, label));
            lines.push(line(300.0, y, 40.0, "1,234"));
        }
        let (items, turn, frame) = project_ocr_page((600.0, 800.0), lines);
        assert_eq!((turn, frame), (0, (600.0, 800.0)));
        assert_eq!(items.len(), 6);
        for item in &items {
            let value = item.text == "1,234";
            assert_eq!(
                (item.lc, item.rc, item.snap),
                (3, if value { 3 } else { 0 }, 'L')
            );
            assert!(!item.flow);
        }
    }

    #[test]
    fn a_page_read_sideways_is_projected_upright_and_boxes_map_back() {
        let page = (600.0, 800.0);
        // Lines running bottom to top, as on a page scanned a quarter turn clockwise.
        let lines: Vec<OcrLine> = (0..4)
            .map(|i| OcrLine {
                bx: [100.0 + 30.0 * i as f32, 200.0, 12.0, 300.0],
                rotation: 90.0,
                ..line(0.0, 0.0, 0.0, "sideways text line")
            })
            .collect();
        let (items, turn, frame) = project_ocr_page(page, lines);
        assert_eq!((turn, frame), (1, (800.0, 600.0)));
        // In the reading frame the first line is 300 wide and 12 tall.
        assert_eq!(items[0].bx, [300.0, 100.0, 300.0, 12.0]);
        assert_eq!(unturn(items[0].bx, turn, page), [100.0, 200.0, 12.0, 300.0]);
        for turn in 0..4 {
            let b = [40.0, 90.0, 120.0, 16.0];
            assert_eq!(unturn(turn_box(b, turn, page), turn, page), b);
        }
    }
}
