//! LayoutNet: tables and layout blocks from the text units of a page.
//!
//! The model reads every unit's geometry and text statistics, the page's ruling lines and a 1024-pixel page image, and scores units and unit pairs: in a table, a header, the same table, row, column, cell or block, and one of 13 block classes. Tables and blocks are decoded from those scores. Adapted from LiteParse's `table_model` (Apache-2.0). See `processors/NOTICE`.
mod decode;
mod features;

use crate::core::OCRError;
use crate::core::config::{ModelInferenceConfig, OrtSessionConfig};
use crate::core::inference::{ModelSource, OrtInfer};
use decode::{DUnit, DecodeOptions, Grid, GridCell};
use features::{FUnit, PageImage};
use image::RgbImage;

/// Side of the square canvas the image branch reads.
const IMG_SIDE: usize = 1024;
/// Pages with more units skip the model: attention and decoding grow with the square of the count.
const MAX_UNITS: usize = 1500;
/// Resolution a page image is taken to have, as LiteParse assumes for images without resolution metadata. Unit geometry and ruling lines are measured in points at this resolution.
const PAGE_DPI: f32 = 150.0;
const POINTS_PER_PIXEL: f64 = 72.0 / PAGE_DPI as f64;

/// Block classes of the class head, indexed by [`LayoutNetBlock::class`].
pub const BLOCK_CLASSES: [&str; 13] = [
    "Text",
    "Heading",
    "Title",
    "PageHeader",
    "PageFooter",
    "List",
    "Table",
    "Caption",
    "Footnote",
    "Formula",
    "Code",
    "FigureText",
    "KeyValue",
];

/// One page's input to the image branch: grey ink in [0, 1] on the square canvas, unit boxes and the image extent as fractions of the canvas.
struct ImgInput {
    canvas: Vec<f32>,
    boxes: Vec<[f32; 4]>,
    valid: (f32, f32),
}

/// Node logits `[n, 2]`, pair logits `[n_pair, n, n]` and block class logits `[n, n_cls]`, row-major.
struct Logits {
    n: usize,
    node: Vec<f32>,
    pair: Vec<f32>,
    cls: Vec<f32>,
    n_cls: usize,
    n_pair: usize,
}

impl Logits {
    #[inline]
    fn node(&self, i: usize, k: usize) -> f32 {
        self.node[i * 2 + k]
    }
    #[inline]
    fn pair(&self, k: usize, i: usize, j: usize) -> f32 {
        self.pair[(k * self.n + i) * self.n + j]
    }
}

/// A line or cell of text on the page.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayoutUnit {
    /// Left, top, right and bottom in image pixels.
    pub bbox: [f32; 4],
    /// The unit's text.
    pub text: String,
    /// Hide the text's content from the model, which then sees only its length.
    pub opaque: bool,
}

impl LayoutUnit {
    /// Creates a unit from its box in image pixels and its text.
    pub fn new(bbox: [f32; 4], text: impl Into<String>) -> Self {
        Self {
            bbox,
            text: text.into(),
            opaque: false,
        }
    }

    /// Hide the text's content from the model, for a token whose reading should not count as table evidence.
    pub fn opaque(mut self, value: bool) -> Self {
        self.opaque = value;
        self
    }
}

/// A layout block: units the model grouped, with their class.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayoutNetBlock {
    /// Left, top, right and bottom in image pixels.
    pub bbox: [f32; 4],
    /// Index into [`BLOCK_CLASSES`].
    pub class: usize,
    /// Mean probability of the class over the block's units.
    pub score: f32,
    /// The units' text in reading order, joined with spaces.
    pub text: String,
    /// Indices of the block's units in reading order.
    pub units: Vec<usize>,
}

/// A cell of a decoded table.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayoutNetCell {
    /// Row and column of the cell's top-left slot.
    pub row: usize,
    pub col: usize,
    /// Rows and columns the cell spans.
    pub row_span: usize,
    pub col_span: usize,
    /// The cell's text.
    pub text: String,
    /// Left, top, right and bottom in image pixels.
    pub bbox: [f32; 4],
}

/// A decoded table.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LayoutNetTable {
    /// Left, top, right and bottom in image pixels.
    pub bbox: [f32; 4],
    /// Mean in-table probability of the table's units.
    pub score: f32,
    /// Grid size.
    pub rows: usize,
    pub cols: usize,
    /// Rows the model marked as headers.
    pub header_rows: Vec<usize>,
    /// Cells in placement order. Empty slots have no cell.
    pub cells: Vec<LayoutNetCell>,
    /// The table as HTML with `th` header cells and row and column spans.
    pub html: String,
    /// Indices of the table's units.
    pub units: Vec<usize>,
}

/// What LayoutNet found on a page.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LayoutNetOutput {
    /// Tables with at least two columns that are not tables of contents, charts or multi-column prose.
    pub tables: Vec<LayoutNetTable>,
    /// Layout blocks over every unit, including the units inside tables.
    pub blocks: Vec<LayoutNetBlock>,
}

/// LayoutNet over ONNX Runtime.
#[derive(Debug)]
pub struct LayoutNetModel {
    inference: OrtInfer,
}

impl LayoutNetModel {
    /// Load the model, which takes `feats`, `mask`, `pairf`, `image`, `boxes` and `valid` and returns `node`, `pair` and `cls` logits.
    pub fn new(
        model: impl Into<ModelSource>,
        ort_config: Option<OrtSessionConfig>,
    ) -> Result<Self, OCRError> {
        let inference = if ort_config.is_some() {
            let common = ModelInferenceConfig {
                ort_session: ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common, model, Some("feats"))?
        } else {
            OrtInfer::new(model, Some("feats"))?
        };
        Ok(Self { inference })
    }

    fn forward(
        &self,
        feats: Vec<f32>,
        pairf: Vec<f32>,
        n: usize,
        img: ImgInput,
    ) -> Result<Logits, OCRError> {
        use ort::value::Tensor;
        let failed = |context: &str, e: ort::Error| OCRError::Inference {
            model_name: "LayoutNet".to_string(),
            context: context.to_string(),
            source: Box::new(e),
        };
        let input = |e| failed("failed to build an input tensor", e);
        let width = feats.len() / n;
        let boxes: Vec<f32> = img.boxes.iter().flatten().copied().collect();
        let inputs = ort::inputs![
            "feats" => Tensor::from_array(([1, n, width], feats)).map_err(input)?,
            "mask" => Tensor::from_array(([1, n], vec![true; n])).map_err(input)?,
            "pairf" => Tensor::from_array(([1, n, n, 2], pairf)).map_err(input)?,
            "image" => Tensor::from_array(([1, 1, IMG_SIDE, IMG_SIDE], img.canvas)).map_err(input)?,
            "boxes" => Tensor::from_array(([1, n, 4], boxes)).map_err(input)?,
            "valid" => Tensor::from_array(([1, 2], vec![img.valid.0, img.valid.1])).map_err(input)?,
        ];
        let mut session = self.inference.get_session(0)?;
        let outputs = session
            .run(inputs)
            .map_err(|e| failed(&format!("failed to run on {n} units"), e))?;
        let read = |name: &str| {
            outputs
                .get(name)
                .ok_or_else(|| {
                    OCRError::invalid_input(format!("LayoutNet has no `{name}` output"))
                })?
                .try_extract_tensor::<f32>()
                .map(|(shape, data)| (shape.to_vec(), data.to_vec()))
                .map_err(|e| failed("failed to read an output", e))
        };
        let (_, node) = read("node")?;
        let (pair_shape, pair) = read("pair")?;
        let (cls_shape, cls) = read("cls")?;
        let (n_pair, n_cls) = (pair_shape[1] as usize, cls_shape[2] as usize);
        if node.len() != n * 2 || pair.len() != n_pair * n * n || cls.len() != n * n_cls {
            return Err(OCRError::invalid_input(format!(
                "unexpected LayoutNet outputs {pair_shape:?} and {cls_shape:?} for {n} units"
            )));
        }
        Ok(Logits {
            n,
            node,
            pair,
            cls,
            n_cls,
            n_pair,
        })
    }

    /// Find the tables and layout blocks of a page from its text units.
    ///
    /// The page is taken to be a 150 dpi scan. A page with fewer than 2 or more than 1500 units, or under 20 characters of text, returns nothing.
    pub fn predict(
        &self,
        image: &RgbImage,
        units: &[LayoutUnit],
    ) -> Result<LayoutNetOutput, OCRError> {
        let n = units.len();
        let (width, height) = image.dimensions();
        if !(2..=MAX_UNITS).contains(&n)
            || units.iter().map(|u| u.text.chars().count()).sum::<usize>() < 20
            || width == 0
            || height == 0
        {
            return Ok(LayoutNetOutput::default());
        }
        // Boxes in points, in f64 for decoding and f32 for the features.
        let boxes: Vec<[f64; 4]> = units
            .iter()
            .map(|u| u.bbox.map(|v| f64::from(v) * POINTS_PER_PIXEL))
            .collect();
        let narrow = |b: &[f64; 4]| b.map(|v| v as f32);
        // A line box is padded around its text: the text height is the box height over 1.4.
        let heights: Vec<f64> = boxes.iter().map(|b| (b[3] - b[1]) / 1.4).collect();
        let feature_units: Vec<FUnit> = units
            .iter()
            .zip(boxes.iter().zip(&heights))
            .map(|(u, (b, &h))| FUnit {
                bx: narrow(b),
                pbox: narrow(b),
                text: u.text.clone(),
                h: h as f32,
                bold: None,
                feat_text: u.opaque.then(|| "x".repeat(u.text.chars().count())),
            })
            .collect();
        let page_boxes: Vec<[f32; 4]> = boxes.iter().map(narrow).collect();
        let page_size = (
            (f64::from(width) * POINTS_PER_PIXEL) as f32,
            (f64::from(height) * POINTS_PER_PIXEL) as f32,
        );

        // The grey page follows the training renders: OpenCV's integer RGB-to-grey weights applied in BGR order, so blue takes the red weight.
        let (w, h) = (width as usize, height as usize);
        let gray: Vec<u8> = image
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0.map(u32::from);
                ((4899 * b + 9617 * g + 1868 * r + (1 << 13)) >> 14) as u8
            })
            .collect();
        let scale = PAGE_DPI / 72.0;
        let rules = features::detect_rules(&gray, w, h, scale);
        let shrink = IMG_SIDE as f32 / w.max(h) as f32;
        let (tw, th) = (
            ((w as f32 * shrink).round() as usize).max(8),
            ((h as f32 * shrink).round() as usize).max(8),
        );
        let page_image = PageImage {
            w: tw,
            h: th,
            px: features::resample_area(&gray, w, h, tw, th),
            k: scale * shrink,
        };

        let feats = features::unit_features(&feature_units, page_size, Some(&rules));
        let pairf = features::pair_rule_features(&page_boxes, &rules);
        let logits = self.forward(feats, pairf, n, page_image.input(&page_boxes))?;

        let decode_units: Vec<DUnit> = units
            .iter()
            .zip(boxes.iter().zip(&heights))
            .map(|(u, (&bx, &h))| DUnit {
                bx,
                text: u.text.clone(),
                h,
            })
            .collect();
        let to_pixels = |b: [f64; 4]| b.map(|v| (v / POINTS_PER_PIXEL) as f32);
        let extent = |group: &[usize]| {
            group.iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |b, &i| {
                    let p = boxes[i];
                    [
                        b[0].min(p[0]),
                        b[1].min(p[1]),
                        b[2].max(p[2]),
                        b[3].max(p[3]),
                    ]
                },
            )
        };
        let options = DecodeOptions::default();
        let mut tables = Vec::new();
        for group in decode::decode_tables(&decode_units, &logits, &options) {
            if group.len() < 2 {
                continue;
            }
            let grid = decode::decode_grid(&decode_units, &group, &logits, &options);
            // One column is a list or a stack of lines. Tables of contents, charts and multi-column prose read as grids are no tables either.
            if grid.n_cols < 2 || is_toc(&grid) || is_chart(&grid) || is_prose(&grid) {
                continue;
            }
            let in_table = |&i: &usize| 1.0 / (1.0 + (-logits.node(i, 0)).exp());
            tables.push(LayoutNetTable {
                bbox: to_pixels(extent(&group)),
                score: group.iter().map(in_table).sum::<f32>() / group.len() as f32,
                rows: grid.n_rows,
                cols: grid.n_cols,
                html: grid.to_html(),
                cells: grid
                    .cells
                    .iter()
                    .map(|c| LayoutNetCell {
                        row: c.r0,
                        col: c.c0,
                        row_span: c.r1 - c.r0 + 1,
                        col_span: c.c1 - c.c0 + 1,
                        text: c.text.clone(),
                        bbox: to_pixels(c.bx),
                    })
                    .collect(),
                header_rows: grid.head_rows,
                units: group,
            });
        }

        let mut blocks = Vec::new();
        if logits.n_pair == 5 && logits.n_cls > 0 {
            let classes = logits.n_cls;
            let class_of = |group: &[usize]| {
                let mut sum = vec![0f32; classes];
                for &i in group {
                    for (k, v) in logits.cls[i * classes..(i + 1) * classes]
                        .iter()
                        .enumerate()
                    {
                        sum[k] += v;
                    }
                }
                // The last maximum, as `Iterator::max_by` returns.
                (0..classes)
                    .max_by(|&a, &b| sum[a].total_cmp(&sum[b]))
                    .unwrap_or(0)
            };
            let all: Vec<usize> = (0..n).collect();
            for group in block_groups(&logits, &all) {
                // The same_block head joins running headers, side-by-side blocks, and a heading with the paragraph under it, so groups are split at wide horizontal gaps, paragraph breaks and heading changes.
                let parts = split_at_gaps(&group, &boxes)
                    .into_iter()
                    .flat_map(|part| split_paragraphs(&part, units, &boxes, &logits.cls, classes));
                for part in parts {
                    let part_boxes: Vec<[f64; 4]> = part.iter().map(|&i| boxes[i]).collect();
                    let part: Vec<usize> = decode::reading_order(&part_boxes)
                        .into_iter()
                        .map(|k| part[k])
                        .collect();
                    let text = part
                        .iter()
                        .map(|&i| units[i].text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let class = class_of(&part);
                    let probability = |&i: &usize| {
                        let row = &logits.cls[i * classes..(i + 1) * classes];
                        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        (row[class] - max).exp() / row.iter().map(|v| (v - max).exp()).sum::<f32>()
                    };
                    blocks.push(LayoutNetBlock {
                        bbox: to_pixels(extent(&part)),
                        class,
                        score: part.iter().map(probability).sum::<f32>() / part.len() as f32,
                        text,
                        units: part,
                    });
                }
            }
        }
        Ok(LayoutNetOutput { tables, blocks })
    }
}

/// Units grouped into blocks from same_block logits: connected components of positive links, then average linkage inside each component.
fn block_groups(l: &Logits, keep: &[usize]) -> Vec<Vec<usize>> {
    let m = keep.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn root(p: &mut [usize], mut k: usize) -> usize {
        while p[k] != k {
            p[k] = p[p[k]];
            k = p[k];
        }
        k
    }
    for a in 0..m {
        for b in a + 1..m {
            if l.pair(4, keep[a], keep[b]) > 0.0 {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }
    let mut components: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for a in 0..m {
        let r = root(&mut parent, a);
        components.entry(r).or_default().push(a);
    }
    let mut out = Vec::new();
    for c in components.into_values() {
        let k = c.len();
        let mut p = vec![0f64; k * k];
        for x in 0..k {
            for y in 0..k {
                p[x * k + y] = f64::from(l.pair(4, keep[c[x]], keep[c[y]]));
            }
        }
        for g in decode::avg_link(&p, k, 0.0) {
            out.push(g.into_iter().map(|x| keep[c[x]]).collect());
        }
    }
    out
}

/// A block's units split where their horizontal extents leave a gap wider than the units' median height, keeping the original order inside each part.
fn split_at_gaps(g: &[usize], boxes: &[[f64; 4]]) -> Vec<Vec<usize>> {
    let mut hs: Vec<f64> = g.iter().map(|&i| boxes[i][3] - boxes[i][1]).collect();
    hs.sort_by(f64::total_cmp);
    let gap = hs.get(hs.len() / 2).copied().unwrap_or(0.0).max(1.0);
    let mut by_x: Vec<usize> = g.to_vec();
    by_x.sort_by(|&a, &b| boxes[a][0].total_cmp(&boxes[b][0]));
    let mut parts: Vec<Vec<usize>> = Vec::new();
    let mut right = f64::NEG_INFINITY;
    for i in by_x {
        let p = boxes[i];
        if parts.is_empty() || p[0] - right > gap {
            parts.push(Vec::new());
        }
        parts.last_mut().unwrap().push(i);
        right = right.max(p[2]);
    }
    parts
}

/// A block's units split into runs of lines.
///
/// A new part starts where the vertical gap to the next line is more than the block's median line height, or where the lines change between heading-like (the Heading and Title share of the class head over half, by characters) and body text.
fn split_paragraphs(
    g: &[usize],
    units: &[LayoutUnit],
    boxes: &[[f64; 4]],
    cls: &[f32],
    c: usize,
) -> Vec<Vec<usize>> {
    if g.len() < 2 || c < 3 {
        return vec![g.to_vec()];
    }
    let mut by_y: Vec<usize> = g.to_vec();
    by_y.sort_by(|&a, &b| {
        let (pa, pb) = (boxes[a], boxes[b]);
        ((pa[1] + pa[3]) / 2.0).total_cmp(&((pb[1] + pb[3]) / 2.0))
    });
    // Lines: units overlapping the current line by more than half the smaller height.
    let mut lines: Vec<(f64, f64, Vec<usize>)> = Vec::new();
    for i in by_y {
        let p = boxes[i];
        match lines.last_mut() {
            Some(l) if (l.1.min(p[3]) - l.0.max(p[1])) > 0.5 * (l.1 - l.0).min(p[3] - p[1]) => {
                l.0 = l.0.min(p[1]);
                l.1 = l.1.max(p[3]);
                l.2.push(i);
            }
            _ => lines.push((p[1], p[3], vec![i])),
        }
    }
    if lines.len() < 2 {
        return vec![g.to_vec()];
    }
    let mut hs: Vec<f64> = lines.iter().map(|l| l.1 - l.0).collect();
    hs.sort_by(f64::total_cmp);
    let line_h = hs[hs.len() / 2].max(1.0);
    let headingish = |l: &[usize]| {
        let (mut head, mut all) = (0f32, 0f32);
        for &i in l {
            let row = &cls[i * c..(i + 1) * c];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let chars = units[i].text.chars().filter(|ch| !ch.is_whitespace());
            let n = chars.count().max(1) as f32;
            head += n * (e[1] + e[2]) / z;
            all += n;
        }
        head * 2.0 > all
    };
    let mut parts: Vec<Vec<usize>> = vec![lines[0].2.clone()];
    for k in 1..lines.len() {
        let gap = lines[k].0 - lines[k - 1].1;
        if gap > line_h || headingish(&lines[k].2) != headingish(&lines[k - 1].2) {
            parts.push(Vec::new());
        }
        parts.last_mut().unwrap().extend_from_slice(&lines[k].2);
    }
    parts
}

/// A table of contents: at most three columns, and most rows end in a page number, arabic or roman, with the arabic ones non-decreasing down the table.
fn is_toc(grid: &Grid) -> bool {
    if grid.n_cols > 3 {
        return false;
    }
    let mut rows: Vec<Vec<(usize, &str)>> = vec![Vec::new(); grid.n_rows];
    for c in &grid.cells {
        let t = c.text.trim();
        if !t.is_empty() {
            rows[c.r0].push((c.c0, t));
        }
    }
    let rows: Vec<_> = rows.into_iter().filter(|r| !r.is_empty()).collect();
    let mut hits = 0;
    let mut arabic = Vec::new();
    for r in &rows {
        let last = r.iter().max_by_key(|(q, _)| *q).unwrap().1;
        let roman = last.len() <= 7 && last.chars().all(|ch| "ivxlcdmIVXLCDM".contains(ch));
        let num = last.len() <= 4 && last.chars().all(|ch| ch.is_ascii_digit());
        if r.len() >= 2 && (roman || num) {
            hits += 1;
            if num {
                arabic.push(last.parse::<u32>().unwrap_or(0));
            }
        }
    }
    let rising = arabic.windows(2).filter(|w| w[1] < w[0]).count() <= arabic.len() / 10;
    hits >= 3 && hits * 5 >= rows.len() * 3 && rising
}

/// Multi-column prose read as a table: long, mostly non-numeric cells whose text runs on down each column, a cell that doesn't end a sentence followed by one that starts lowercase.
fn is_prose(grid: &Grid) -> bool {
    let words = |t: &str| t.split_whitespace().count();
    let cells: Vec<&GridCell> = grid
        .cells
        .iter()
        .filter(|c| !c.text.trim().is_empty())
        .collect();
    if cells.len() < 6 {
        return false;
    }
    let mut wc: Vec<usize> = cells.iter().map(|c| words(&c.text)).collect();
    wc.sort_unstable();
    if wc[wc.len() / 2] < 4 {
        return false;
    }
    let numeric = cells
        .iter()
        .filter(|c| words(&c.text) <= 3 && c.text.chars().any(|ch| ch.is_ascii_digit()))
        .count();
    if numeric * 10 >= cells.len() * 3 {
        return false;
    }
    let mut cols: std::collections::BTreeMap<usize, Vec<&GridCell>> = Default::default();
    for c in &cells {
        if c.c0 == c.c1 {
            cols.entry(c.c0).or_default().push(c);
        }
    }
    let (mut pairs, mut flow) = (0usize, 0usize);
    for col in cols.values_mut() {
        col.sort_by_key(|c| c.r0);
        for w in col.windows(2) {
            // Not adjacent: more than one empty row between them.
            if w[1].r0 > w[0].r1 + 2 {
                continue;
            }
            pairs += 1;
            let open = !w[0].text.trim_end().ends_with(['.', '!', '?', ':', ';']);
            let lower = w[1]
                .text
                .trim_start()
                .chars()
                .next()
                .is_some_and(|ch| ch.is_lowercase());
            flow += usize::from(open && lower);
        }
    }
    pairs >= 3 && flow * 10 >= pairs * 4
}

/// A chart read as a table: a sparse, mostly numeric grid with an axis, some column holding at least three lone numbers in an evenly stepped run.
fn is_chart(grid: &Grid) -> bool {
    let slots = grid.n_rows * grid.n_cols;
    if slots == 0 {
        return false;
    }
    let parse = |t: &str| -> Option<f64> {
        let t = t.trim().trim_end_matches('%').replace(',', "");
        if t.is_empty()
            || !t
                .chars()
                .all(|ch| ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+')
        {
            return None;
        }
        t.parse::<f64>().ok()
    };
    let (mut filled, mut nonempty, mut numeric) = (0, 0, 0);
    let mut cols: std::collections::BTreeMap<usize, Vec<(usize, f64)>> = Default::default();
    let mut flow: Vec<((usize, usize), f64)> = Vec::new();
    for c in &grid.cells {
        if c.text.trim().is_empty() {
            continue;
        }
        nonempty += 1;
        filled += (c.r1 - c.r0 + 1) * (c.c1 - c.c0 + 1);
        if let Some(v) = parse(&c.text) {
            numeric += 1;
            if c.c0 == c.c1 {
                cols.entry(c.c0).or_default().push((c.r0, v));
            }
            flow.push(((c.r0, c.c0), v));
        }
    }
    if filled * 2 >= slots || numeric * 2 < nonempty {
        return false;
    }
    let stepped = |vs: &[f64]| {
        vs.len() >= 3 && {
            let d: Vec<f64> = vs.windows(2).map(|w| w[1] - w[0]).collect();
            d[0] != 0.0
                && d.iter()
                    .all(|&x| (x - d[0]).abs() <= 1e-6 * d[0].abs().max(1.0))
        }
    };
    // A column of ticks (a y axis), or every number on the grid in reading order (an x axis of years strewn diagonally, one per row).
    flow.sort_by_key(|&(rc, _)| rc);
    let all: Vec<f64> = flow.iter().map(|&(_, v)| v).collect();
    stepped(&all)
        || cols.values_mut().any(|vs| {
            vs.sort_by_key(|&(r, _)| r);
            stepped(&vs.iter().map(|&(_, v)| v).collect::<Vec<_>>())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(r0: usize, c0: usize, r1: usize, c1: usize, text: &str) -> GridCell {
        GridCell {
            r0,
            c0,
            r1,
            c1,
            text: text.to_string(),
            bx: [0.0; 4],
        }
    }

    #[test]
    fn grid_html_marks_header_rows_spans_and_empty_slots() {
        let grid = Grid {
            cells: vec![
                cell(0, 0, 0, 1, "Totals <2024>"),
                cell(1, 0, 2, 0, "A"),
                cell(1, 1, 1, 1, "1"),
            ],
            head_rows: vec![0],
            n_rows: 3,
            n_cols: 2,
        };
        assert_eq!(
            grid.to_html(),
            "<table>\n<tr><th colspan=\"2\">Totals &lt;2024&gt;</th></tr>\n<tr><td rowspan=\"2\">A</td><td>1</td></tr>\n<tr><td></td></tr>\n</table>"
        );
    }

    #[test]
    fn a_grid_of_numbered_entries_is_a_table_of_contents_not_a_table() {
        let grid = Grid {
            cells: (0..4)
                .flat_map(|r| {
                    [
                        cell(r, 0, r, 0, "Chapter"),
                        cell(r, 1, r, 1, &(r * 12 + 1).to_string()),
                    ]
                })
                .collect(),
            head_rows: Vec::new(),
            n_rows: 4,
            n_cols: 2,
        };
        assert!(is_toc(&grid));
        assert!(!is_chart(&grid) && !is_prose(&grid));
    }
}
