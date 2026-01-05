use super::config::PaddleOcrVlImageProcessorConfig;
use crate::core::OCRError;
use candle_core::{DType, Device, Tensor};
use image::{RgbImage, imageops::FilterType};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PaddleOcrVlImageInputs {
    pub pixel_values: Tensor,
    pub image_grid_thw: Vec<(usize, usize, usize)>,
}

pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), OCRError> {
    if factor == 0 {
        return Err(OCRError::InvalidInput {
            message: "smart_resize: factor must be > 0".to_string(),
        });
    }

    let mut height = height as f64;
    let mut width = width as f64;
    let factor_f = factor as f64;

    if height < factor_f {
        width = ((width * factor_f) / height).round();
        height = factor_f;
    }
    if width < factor_f {
        height = ((height * factor_f) / width).round();
        width = factor_f;
    }

    let max_dim = height.max(width);
    let min_dim = height.min(width);
    if min_dim > 0.0 && (max_dim / min_dim) > 200.0 {
        return Err(OCRError::InvalidInput {
            message: format!(
                "smart_resize: absolute aspect ratio must be <= 200, got {:.3}",
                max_dim / min_dim
            ),
        });
    }

    let mut h_bar = (height / factor_f).round() * factor_f;
    let mut w_bar = (width / factor_f).round() * factor_f;

    let area = h_bar * w_bar;
    if area > max_pixels as f64 {
        let beta = ((height * width) / max_pixels as f64).sqrt();
        h_bar = ((height / beta) / factor_f).floor() * factor_f;
        w_bar = ((width / beta) / factor_f).floor() * factor_f;
    } else if area < min_pixels as f64 {
        let beta = (min_pixels as f64 / (height * width)).sqrt();
        h_bar = ((height * beta) / factor_f).ceil() * factor_f;
        w_bar = ((width * beta) / factor_f).ceil() * factor_f;
    }

    let h_out = h_bar.max(factor_f) as u32;
    let w_out = w_bar.max(factor_f) as u32;
    Ok((h_out, w_out))
}

pub fn preprocess_images(
    images: &[RgbImage],
    cfg: &PaddleOcrVlImageProcessorConfig,
    device: &Device,
    dtype: DType,
) -> Result<PaddleOcrVlImageInputs, OCRError> {
    cfg.validate()?;
    if images.is_empty() {
        return Err(OCRError::InvalidInput {
            message: "PaddleOCR-VL: no images provided".to_string(),
        });
    }

    let factor = (cfg.patch_size * cfg.merge_size) as u32;
    let patch = cfg.patch_size as u32;
    let resize_filter = cfg
        .resample
        .and_then(pil_resample_to_filter_type)
        .unwrap_or(FilterType::CatmullRom);

    let mut all_patches: Vec<Tensor> = Vec::with_capacity(images.len());
    let mut grids = Vec::with_capacity(images.len());

    for img in images {
        let (h, w) = (img.height(), img.width());
        let (rh, rw) = if cfg.do_resize {
            smart_resize(h, w, factor, cfg.min_pixels, cfg.max_pixels)?
        } else {
            (h, w)
        };

        let resized = if rh != h || rw != w {
            image::imageops::resize(img, rw, rh, resize_filter)
        } else {
            img.clone()
        };

        if rh % patch != 0 || rw % patch != 0 {
            return Err(OCRError::config_error(format!(
                "PaddleOCR-VL preprocess produced non-divisible dims: {rh}x{rw} not divisible by patch_size={patch}"
            )));
        }

        let grid_h = (rh / patch) as usize;
        let grid_w = (rw / patch) as usize;
        let grid_t = 1usize;

        let num_patches = grid_t * grid_h * grid_w;
        let mut patch_data = Vec::with_capacity(num_patches * 3 * cfg.patch_size * cfg.patch_size);

        let mean = &cfg.image_mean;
        let std = &cfg.image_std;

        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let base_x = (gw as u32) * patch;
                let base_y = (gh as u32) * patch;
                for c in 0..3usize {
                    for dy in 0..cfg.patch_size as u32 {
                        for dx in 0..cfg.patch_size as u32 {
                            let p = resized.get_pixel(base_x + dx, base_y + dy);
                            let mut v = p.0[c] as f32;
                            if cfg.do_rescale {
                                v *= cfg.rescale_factor;
                            }
                            if cfg.do_normalize {
                                v = (v - mean[c]) / std[c];
                            }
                            patch_data.push(v);
                        }
                    }
                }
            }
        }

        let patches = Tensor::from_vec(
            patch_data,
            (num_patches, 3usize, cfg.patch_size, cfg.patch_size),
            device,
        )
        .map_err(|e| OCRError::Processing {
            kind: crate::core::errors::ProcessingStage::TensorOperation,
            context: "PaddleOCR-VL: failed to create pixel_values tensor".to_string(),
            source: Box::new(e),
        })?;

        all_patches.push(patches);
        grids.push((grid_t, grid_h, grid_w));
    }

    let pixel_values = Tensor::cat(&all_patches, 0).map_err(|e| OCRError::Processing {
        kind: crate::core::errors::ProcessingStage::TensorOperation,
        context: "PaddleOCR-VL: failed to concatenate pixel_values tensors".to_string(),
        source: Box::new(e),
    })?;

    // Convert to the target dtype (e.g., BF16 for model inference)
    let pixel_values = pixel_values
        .to_dtype(dtype)
        .map_err(|e| OCRError::Processing {
            kind: crate::core::errors::ProcessingStage::TensorOperation,
            context: "PaddleOCR-VL: failed to convert pixel_values to target dtype".to_string(),
            source: Box::new(e),
        })?;

    Ok(PaddleOcrVlImageInputs {
        pixel_values,
        image_grid_thw: grids,
    })
}

fn pil_resample_to_filter_type(resample: u32) -> Option<FilterType> {
    // Match PIL / transformers `PILImageResampling` integer values:
    // 0=NEAREST, 1=LANCZOS, 2=BILINEAR, 3=BICUBIC, 4=BOX, 5=HAMMING.
    match resample {
        0 => Some(FilterType::Nearest),
        1 => Some(FilterType::Lanczos3),
        2 => Some(FilterType::Triangle),
        3 => Some(FilterType::CatmullRom),
        4 => Some(FilterType::Triangle),
        5 => Some(FilterType::CatmullRom),
        _ => None,
    }
}

pub fn strip_math_wrappers(input: &str) -> &str {
    let mut trimmed = input.trim();
    trimmed = trimmed
        .strip_prefix("$$")
        .and_then(|s| s.strip_suffix("$$"))
        .unwrap_or(trimmed);
    trimmed = trimmed
        .strip_prefix('$')
        .and_then(|s| s.strip_suffix('$'))
        .unwrap_or(trimmed);
    trimmed.trim()
}

pub fn postprocess_table_output(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains("<table") {
        return trimmed.to_string();
    }
    if !looks_like_table_tokens(trimmed) {
        return trimmed.to_string();
    }

    if let Some(html) = try_convert_table_tokens_to_html(trimmed) {
        return html;
    }
    strip_table_tokens_fallback(trimmed)
}

fn looks_like_table_tokens(input: &str) -> bool {
    input.contains("<fcel>")
        || input.contains("<lcel>")
        || input.contains("<ucel>")
        || input.contains("<xcel>")
        || input.contains("<ecel>")
        || input.contains("<nl>")
}

fn try_convert_table_tokens_to_html(input: &str) -> Option<String> {
    static TOKEN_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"(<fcel>|<lcel>|<ucel>|<xcel>|<ecel>|<nl>)")
            .expect("table token regex must compile")
    });

    let first = TOKEN_RE.find(input)?;
    let prefix = input[..first.start()].trim();
    let tokenized = &input[first.start()..];

    let mut rows = parse_table_token_rows(tokenized, &TOKEN_RE);
    if rows.is_empty() {
        return None;
    }
    let col_count = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if col_count == 0 {
        return None;
    }

    for row in rows.iter_mut() {
        if row.len() < col_count {
            row.extend(std::iter::repeat_n(
                (TableCellMarker::F, String::new()),
                col_count - row.len(),
            ));
        }
    }

    let row_count = rows.len();
    let cell_count = row_count * col_count;
    let mut uf = UnionFind::new(cell_count);

    for (r, row) in rows.iter().enumerate().take(row_count) {
        for (c, cell) in row.iter().enumerate().take(col_count) {
            let idx = r * col_count + c;
            match cell.0 {
                TableCellMarker::F => {} // No merging needed
                TableCellMarker::L => {
                    if c > 0 {
                        uf.union(idx, idx - 1);
                    }
                }
                TableCellMarker::U => {
                    if r > 0 {
                        uf.union(idx, idx - col_count);
                    }
                }
                TableCellMarker::X => {
                    if c > 0 {
                        uf.union(idx, idx - 1);
                    }
                    if r > 0 {
                        uf.union(idx, idx - col_count);
                    }
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    struct CellInfo {
        min_r: usize,
        min_c: usize,
        max_r: usize,
        max_c: usize,
        text: String,
        text_priority: u8,
        text_r: usize,
        text_c: usize,
        has_text: bool,
    }

    let mut infos: HashMap<usize, CellInfo> = HashMap::new();
    for (r, row) in rows.iter().enumerate().take(row_count) {
        for (c, cell) in row.iter().enumerate().take(col_count) {
            let idx = r * col_count + c;
            let root = uf.find(idx);
            let entry = infos.entry(root).or_insert(CellInfo {
                min_r: r,
                min_c: c,
                max_r: r,
                max_c: c,
                text: String::new(),
                text_priority: u8::MAX,
                text_r: r,
                text_c: c,
                has_text: false,
            });

            entry.min_r = entry.min_r.min(r);
            entry.min_c = entry.min_c.min(c);
            entry.max_r = entry.max_r.max(r);
            entry.max_c = entry.max_c.max(c);

            let text = cell.1.trim();
            if !text.is_empty() {
                let priority = if cell.0 == TableCellMarker::F { 0 } else { 1 };
                let better = !entry.has_text
                    || priority < entry.text_priority
                    || (priority == entry.text_priority && (r, c) < (entry.text_r, entry.text_c));
                if better {
                    entry.text = text.to_string();
                    entry.text_priority = priority;
                    entry.text_r = r;
                    entry.text_c = c;
                    entry.has_text = true;
                }
            }
        }
    }

    let mut html = String::new();
    html.push_str("<table>");
    if !prefix.is_empty() {
        html.push_str("<caption>");
        html.push_str(&escape_html_text(prefix));
        html.push_str("</caption>");
    }
    html.push_str("<tbody>");

    for r in 0..row_count {
        html.push_str("<tr>");
        for c in 0..col_count {
            let idx = r * col_count + c;
            let root = uf.find(idx);
            let info = infos.get(&root)?;
            if info.min_r == r && info.min_c == c {
                let rowspan = info.max_r - info.min_r + 1;
                let colspan = info.max_c - info.min_c + 1;
                html.push_str("<td");
                if rowspan > 1 {
                    html.push_str(&format!(" rowspan=\"{rowspan}\"",));
                }
                if colspan > 1 {
                    html.push_str(&format!(" colspan=\"{colspan}\"",));
                }
                html.push('>');
                html.push_str(&escape_html_text(info.text.trim()));
                html.push_str("</td>");
            }
        }
        html.push_str("</tr>");
    }

    html.push_str("</tbody></table>");
    Some(html)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableCellMarker {
    F,
    L,
    U,
    X,
}

impl TableCellMarker {
    fn from_token(token: &str) -> Option<Self> {
        match token {
            "<fcel>" => Some(Self::F),
            "<lcel>" => Some(Self::L),
            "<ucel>" => Some(Self::U),
            "<xcel>" => Some(Self::X),
            _ => None,
        }
    }
}

fn parse_table_token_rows(input: &str, token_re: &Regex) -> Vec<Vec<(TableCellMarker, String)>> {
    let mut rows = Vec::new();
    let mut current_row: Vec<(TableCellMarker, String)> = Vec::new();
    let mut current_marker: Option<TableCellMarker> = None;
    let mut current_text = String::new();
    let mut cursor = 0usize;

    for mat in token_re.find_iter(input) {
        let between = &input[cursor..mat.start()];
        if current_marker.is_some() {
            current_text.push_str(between);
        }
        cursor = mat.end();

        match mat.as_str() {
            "<fcel>" | "<lcel>" | "<ucel>" | "<xcel>" => {
                if let Some(marker) = current_marker.take() {
                    current_row.push((marker, current_text.trim().to_string()));
                    current_text.clear();
                }
                current_marker = TableCellMarker::from_token(mat.as_str());
            }
            "<ecel>" | "<nl>" => {
                if let Some(marker) = current_marker.take() {
                    current_row.push((marker, current_text.trim().to_string()));
                    current_text.clear();
                }
                if !current_row.is_empty() {
                    rows.push(std::mem::take(&mut current_row));
                }
            }
            _ => {} // Should not happen with the regex
        }
    }

    // Handle any remaining text after the last token
    if let Some(marker) = current_marker.take() {
        current_text.push_str(&input[cursor..]);
        current_row.push((marker, current_text.trim().to_string()));
    }
    if !current_row.is_empty() {
        rows.push(current_row);
    }
    rows
}

fn strip_table_tokens_fallback(input: &str) -> String {
    let mut out = input.replace("<ecel>", "\n").replace("<nl>", "\n");
    out = out
        .replace("<fcel>", "\t")
        .replace("<lcel>", "")
        .replace("<ucel>", "")
        .replace("<xcel>", "");
    out.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn escape_html_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"), // Numeric HTML entity for apostrophe
            _ => out.push(ch),
        }
    }
    out
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] == x {
            return x;
        }
        let root = self.find(self.parent[x]);
        self.parent[x] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PaddleOcrVlImageProcessorConfig;
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_smart_resize_factor_divisibility() {
        let (h, w) = smart_resize(100, 200, 28, 147_384, 2_822_400).unwrap();
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn test_preprocess_outputs_expected_shapes() {
        let cfg = PaddleOcrVlImageProcessorConfig {
            do_resize: true,
            do_rescale: true,
            do_normalize: true,
            do_convert_rgb: true,
            rescale_factor: 1.0 / 255.0,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
            min_pixels: 28 * 28 * 130,
            max_pixels: 28 * 28 * 1280,
            resample: None,
            patch_size: 14,
            temporal_patch_size: 1,
            merge_size: 2,
        };

        let mut img = RgbImage::new(64, 64);
        for p in img.pixels_mut() {
            *p = Rgb([127, 127, 127]);
        }

        let device = Device::Cpu;
        let out = preprocess_images(&[img], &cfg, &device, DType::F32).unwrap();
        assert_eq!(out.image_grid_thw.len(), 1);
        let (t, h, w) = out.image_grid_thw[0];
        assert_eq!(t, 1);
        assert!(h > 0);
        assert!(w > 0);
        let shape = out.pixel_values.dims();
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 14);
        assert_eq!(shape[3], 14);
    }
}
