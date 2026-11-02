//! Figure tokens shared by table crops and Markdown projection.

use crate::document::structure::{LayoutElement, LayoutElementType};
use image::RgbImage;
use regex::{Captures, Regex};
use std::sync::LazyLock;

// PaddleX 3.7.1 paddleocr_vl/uilts.py (Apache-2.0): fixed-seed figure
// numbering, full containment, the 25-pixel cutoff and token restoration.
struct Figure {
    index: Option<usize>,
    token: String,
    bounds: [f32; 4],
}

pub(super) struct TableFigures {
    figures: Vec<Figure>,
    tables: Vec<Vec<usize>>,
}

fn bounds(element: &LayoutElement) -> [f32; 4] {
    [
        element.bbox.x_min(),
        element.bbox.y_min(),
        element.bbox.x_max(),
        element.bbox.y_max(),
    ]
}

fn contained(outer: [f32; 4], inner: [f32; 4]) -> bool {
    outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3]
}

impl TableFigures {
    pub(super) fn new(elements: &[LayoutElement]) -> Self {
        Self::from_pool(elements, elements)
    }

    pub(super) fn from_pool(elements: &[LayoutElement], pool: &[LayoutElement]) -> Self {
        let mut figures: Vec<_> = pool
            .iter()
            .filter(|e| {
                matches!(
                    e.label.as_deref().unwrap_or(e.element_type.as_str()),
                    "image" | "figure" | "seal"
                )
            })
            .filter_map(|element| {
                let coordinates = bounds(element).map(|value| value.trunc().max(0.));
                let label = element
                    .label
                    .as_deref()
                    .unwrap_or(element.element_type.as_str());
                let index = elements.iter().position(|candidate| {
                    candidate
                        .label
                        .as_deref()
                        .unwrap_or(candidate.element_type.as_str())
                        == label
                        && bounds(candidate).map(|value| value.trunc().max(0.)) == coordinates
                });
                (coordinates[2] > coordinates[0] && coordinates[3] > coordinates[1]).then_some(
                    Figure {
                        index,
                        token: String::new(),
                        bounds: coordinates,
                    },
                )
            })
            .collect();
        let mut numbers: Vec<_> = (0usize..)
            .filter(|value| {
                !value
                    .to_string()
                    .bytes()
                    .any(|c| matches!(c, b'0' | b'1' | b'9'))
            })
            .take(figures.len())
            .collect();
        let mut random = PythonRandom::new(1024);
        for i in (1..numbers.len()).rev() {
            let other = random.below(i + 1);
            numbers.swap(i, other);
        }
        for (figure, number) in figures.iter_mut().zip(numbers) {
            figure.token = format!("[F{number}]");
        }
        let tables = elements
            .iter()
            .map(|element| {
                if element.element_type != LayoutElementType::Table {
                    return Vec::new();
                }
                figures
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| contained(bounds(element), f.bounds))
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        Self { figures, tables }
    }

    pub(super) fn paint(&self, index: usize, image: &mut RgbImage, origin: [f64; 2]) {
        for &figure in &self.tables[index] {
            let figure = &self.figures[figure];
            if (figure.bounds[2] - figure.bounds[0]).min(figure.bounds[3] - figure.bounds[1]) < 25.
            {
                continue;
            }
            let relative = std::array::from_fn(|i| f64::from(figure.bounds[i]) - origin[i % 2]);
            super::figure_text::paint(image, relative, &figure.token);
        }
    }

    pub(super) fn restore(
        &self,
        index: usize,
        content: &str,
        elements: &[LayoutElement],
    ) -> String {
        static TOKEN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[F\d+\]").unwrap());
        TOKEN
            .replace_all(content, |capture: &Captures<'_>| {
                let Some(figure) = self.tables[index]
                    .iter()
                    .map(|&i| &self.figures[i])
                    .find(|f| {
                        f.token == capture[0]
                            && (f.bounds[2] - f.bounds[0]).min(f.bounds[3] - f.bounds[1]) >= 25.
                    })
                else {
                    return capture[0].to_string();
                };
                let Some(index) = figure.index else {
                    return capture[0].to_string();
                };
                let mut restored = format!("<img src=\"image_{}.png\" alt=\"Image\" />", index + 1);
                if let Some(text) = elements[index]
                    .text
                    .as_deref()
                    .filter(|text| !text.is_empty())
                {
                    restored.push_str("\n\n");
                    restored.push_str(text);
                    restored.push_str("\n\n");
                }
                restored
            })
            .into_owned()
    }

    pub(super) fn absorbed(&self, index: usize) -> bool {
        self.tables
            .iter()
            .flatten()
            .any(|&i| self.figures[i].index == Some(index))
    }
}

// CPython's integer-seeded MT19937 and getrandbits rejection sampling keep the
// token permutation aligned with random.Random(1024).shuffle.
struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = 19_650_218;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let mut i = 1;
        for _ in 0..624 {
            state[i] = (state[i] ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(seed);
            i += 1;
            if i == 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            state[i] = (state[i]
                ^ (state[i - 1] ^ (state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i == 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        state[0] = 0x8000_0000;
        Self { state, index: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let value =
                    (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (value >> 1)
                    ^ if value & 1 == 1 { 0x9908_b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    fn below(&mut self, count: usize) -> usize {
        let bits = usize::BITS - count.leading_zeros();
        loop {
            let value = (self.next() >> (32 - bits)) as usize;
            if value < count {
                return value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_painting_match_the_reference() {
        for (size, bounds, fixture) in [
            (
                48,
                [8., 8., 40., 40.],
                include_bytes!("../../tests/fixtures/table_figure_48.png").as_slice(),
            ),
            (
                160,
                [20., 20., 140., 140.],
                include_bytes!("../../tests/fixtures/table_figure_160.png").as_slice(),
            ),
        ] {
            let mut image = RgbImage::from_pixel(size, size, image::Rgb([33; 3]));
            super::super::figure_text::paint(&mut image, bounds, "[F2]");
            let expected = image::load_from_memory(fixture).unwrap().to_rgb8();
            assert_eq!(image, expected, "{size}-pixel figure");
        }
        let mut random = PythonRandom::new(1024);
        let mut values: Vec<_> = (0..10).collect();
        for i in (1..values.len()).rev() {
            values.swap(i, random.below(i + 1));
        }
        assert_eq!(values, [5, 1, 8, 3, 9, 4, 2, 6, 7, 0]);
    }
}
