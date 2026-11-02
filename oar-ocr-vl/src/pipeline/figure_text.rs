//! Hershey text painting for table figure tokens.
// Adapted from OpenCV 4.13.0 drawing.cpp and hershey_fonts.cpp. The glyphs,
// metrics, fixed-point strokes and antialiasing retain the OpenCV license below.

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

use image::{Rgb, RgbImage};

type Point = [i64; 2];
const ONE: i64 = 1 << 16;
const HALF: i64 = ONE / 2;

fn glyph(character: u8) -> &'static str {
    match character {
        b'[' => r#"KYOBOb OBVB ObVb"#,
        b'F' => r#"HZLFL[ LFYF LPTP"#,
        b'0' => r#"H\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF"#,
        b'1' => r#"H\NJPISFS["#,
        b'2' => r#"H\LKLJMHNGPFTFVGWHXJXLWNUQK[Y["#,
        b'3' => r#"H\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW"#,
        b'4' => r#"H\UFKTZT UFU["#,
        b'5' => r#"H\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW"#,
        b'6' => r#"H\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT"#,
        b'7' => r#"H\YFO[ KFYF"#,
        b'8' => r#"H\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF"#,
        b'9' => r#"H\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX"#,
        b']' => r#"KYUBUb NBUB NbUb"#,
        _ => "JZ",
    }
}

fn text_size(text: &str, scale: f64) -> (i32, i32) {
    let width: f64 = text
        .bytes()
        .map(|c| {
            let glyph = glyph(c).as_bytes();
            f64::from(glyph[1] as i32 - glyph[0] as i32) * scale
        })
        .sum();
    (
        (width + 1.).round_ties_even() as i32,
        (21. * scale + 1.).round_ties_even() as i32,
    )
}

pub(super) fn paint(image: &mut RgbImage, bounds: [f64; 4], token: &str) {
    let [x1, y1, x2, y2] = bounds.map(|v| v as i32);
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    for y in y1.max(0)..=y2.min(image.height() as i32 - 1) {
        for x in x1.max(0)..=x2.min(image.width() as i32 - 1) {
            image.put_pixel(x as u32, y as u32, Rgb([255; 3]));
        }
    }
    let mut low = 0.2;
    let mut high = 10.;
    let mut scale = low;
    let mut size = (0, 0);
    while high - low > 0.01 {
        let mid = (low + high) / 2.;
        size = text_size(token, mid);
        if f64::from(size.0.max(size.1)) < f64::from((x2 - x1).min(y2 - y1)) * 0.9 {
            scale = mid;
            low = mid;
        } else {
            high = mid;
        }
    }
    let thickness = (scale * 4.).floor().max(1.) as i64;
    let origin = [
        i64::from(x1 + (x2 - x1 - size.0).div_euclid(2)),
        i64::from(y1 + (y2 - y1 + size.1).div_euclid(2)),
    ];
    let unit = (scale * ONE as f64).round_ties_even() as i64;
    let mut cursor = origin[0] * ONE;
    let baseline = origin[1] * ONE - 9 * unit;
    for c in token.bytes() {
        let data = glyph(c).as_bytes();
        let left = i64::from(data[0]) - 82;
        let right = i64::from(data[1]) - 82;
        cursor -= left * unit;
        for stroke in data[2..].split(|&b| b == b' ') {
            let points: Vec<_> = stroke
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[x, y]| {
                    [
                        cursor + (i64::from(x) - 82) * unit,
                        baseline + (i64::from(y) - 82) * unit,
                    ]
                })
                .collect();
            for (index, pair) in points.windows(2).enumerate() {
                thick_line(image, pair[0], pair[1], thickness, index == 0);
            }
        }
        cursor += right * unit;
    }
}

fn blend(image: &mut RgbImage, x: i64, y: i64, alpha: i64) {
    if x < 0 || y < 0 || x >= i64::from(image.width()) || y >= i64::from(image.height()) {
        return;
    }
    let pixel = image.get_pixel_mut(x as u32, y as u32);
    for channel in &mut pixel.0 {
        let mut value = i64::from(*channel);
        value += (-value * alpha + 127) >> 8;
        value += (-value * alpha + 127) >> 8;
        *channel = value as u8;
    }
}

fn line_aa(image: &mut RgbImage, mut a: Point, mut b: Point) {
    const SLOPE: [i64; 32] = [
        181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201, 203, 206,
        209, 211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254,
    ];
    const FILTER: [i64; 64] = [
        168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254, 254, 254,
        252, 249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168, 158, 149, 140, 131,
        122, 114, 105, 97, 89, 82, 75, 68, 62, 56, 50, 45, 40, 36, 32, 28, 25, 22, 19, 16, 14, 12,
        11, 9, 8, 7, 5, 5,
    ];
    let axis = usize::from((b[0] - a[0]).abs() <= (b[1] - a[1]).abs());
    if a[axis] > b[axis] {
        std::mem::swap(&mut a, &mut b);
    }
    let other = 1 - axis;
    let major = b[axis] - a[axis];
    let step = ((b[other] - a[other]) * ONE) / (major | 1);
    b[axis] += ONE;
    let count = (b[axis] >> 16) - (a[axis] >> 16);
    let offset = -(a[axis] & (ONE - 1));
    a[other] += ((step * offset) >> 16) + HALF;
    let mut slope = (step >> 11) & 63;
    if step < 0 {
        slope ^= 63;
    }
    let slope = if slope & 32 != 0 {
        256
    } else {
        SLOPE[slope as usize]
    };
    let i = (a[axis] >> 9) & 0x78;
    let j = (b[axis] >> 9) & 0x78;
    let t0 = slope << 7;
    let t1 = ((0x78 - i) | 4) * slope;
    let t2 = (j | 4) * slope;
    let end = (((((j - i) & 0x78) | 4) * slope) >> 8) & 0x1ff;
    let correction = [
        0,
        end,
        (t1 >> 8) & 0x1ff,
        end,
        ((((j - i + 0x80) | 4) * slope) >> 8) & 0x1ff,
        ((t1 + t0) >> 8) & 0x1ff,
        (t2 >> 8) & 0x1ff,
        ((t2 + t0) >> 8) & 0x1ff,
        slope,
    ];
    for n in 0..=count {
        let remaining = count - n;
        let start_index = ((i64::from(n >= 2) + 1) & (n | 2)) as usize;
        let end_index = ((i64::from(remaining >= 2) + 1) & (remaining | 2)) as usize;
        let ep = correction[start_index * 3 + end_index];
        let dist = ((a[other] >> 11) & 31) as usize;
        let weights = [FILTER[dist + 32], FILTER[dist], FILTER[63 - dist]];
        for (k, weight) in weights.into_iter().enumerate() {
            let mut pixel = [0, 0];
            pixel[axis] = (a[axis] >> 16) + n;
            pixel[other] = (a[other] >> 16) - 1 + k as i64;
            blend(image, pixel[0], pixel[1], ((ep * weight) >> 8) & 255);
        }
        a[other] += step;
    }
}

fn fill_convex(image: &mut RgbImage, points: &[Point]) {
    if points.len() < 2 {
        return;
    }
    let mut previous = points[points.len() - 1];
    for &point in points {
        line_aa(image, previous, point);
        previous = point;
    }
    #[derive(Clone, Copy)]
    struct Edge {
        index: usize,
        direction: usize,
        x: i64,
        dx: i64,
        end: i64,
    }
    let first = (0..points.len()).min_by_key(|&i| points[i][1]).unwrap_or(0);
    let min_y = (points[first][1] + HALF) >> 16;
    let max_y = ((points.iter().map(|p| p[1]).max().unwrap_or(0) + HALF) >> 16)
        .min(i64::from(image.height()) - 1);
    let mut edges = [
        Edge {
            index: first,
            direction: 1,
            x: -ONE,
            dx: 0,
            end: min_y,
        },
        Edge {
            index: first,
            direction: points.len() - 1,
            x: -ONE,
            dx: 0,
            end: min_y,
        },
    ];
    let mut remaining = points.len() as i32;
    for y in min_y..=max_y {
        if y < max_y || y == min_y {
            for edge in &mut edges {
                if y >= edge.end {
                    let mut from = edge.index;
                    let mut to = (from + edge.direction) % points.len();
                    loop {
                        remaining -= 1;
                        if remaining < 0 {
                            break;
                        }
                        let end = (points[to][1] + HALF) >> 16;
                        if end > y {
                            let xs = points[from][0];
                            let xe = points[to][0];
                            edge.end = end;
                            edge.dx = ((xe - xs) * 2 + end - y) / (2 * (end - y));
                            edge.x = xs;
                            edge.index = to;
                            break;
                        }
                        from = to;
                        to = (to + edge.direction) % points.len();
                    }
                }
            }
        }
        if remaining < 0 {
            break;
        }
        if y >= 0 {
            let low = (edges[0].x.min(edges[1].x) + ONE - 1) >> 16;
            let high = edges[0].x.max(edges[1].x) >> 16;
            for x in low.max(0)..=high.min(i64::from(image.width()) - 1) {
                image.put_pixel(x as u32, y as u32, Rgb([0; 3]));
            }
        }
        for edge in &mut edges {
            edge.x += edge.dx;
        }
    }
}

fn round_cap(image: &mut RgbImage, center: Point, radius: i64) {
    let pixels = (radius + HALF) >> 16;
    let delta = if pixels < 3 {
        90
    } else if pixels < 10 {
        30
    } else if pixels < 15 {
        18
    } else {
        5
    };
    let points: Vec<_> = (0..=360)
        .step_by(delta)
        .map(|angle| {
            let radians = (angle as f64).to_radians();
            let cos = f64::from(radians.cos() as f32);
            let sin = f64::from(radians.sin() as f32);
            [
                (center[0] as f64 + radius as f64 * cos).round_ties_even() as i64,
                (center[1] as f64 + radius as f64 * sin).round_ties_even() as i64,
            ]
        })
        .collect();
    fill_convex(image, &points);
}

fn thick_line(image: &mut RgbImage, a: Point, b: Point, thickness: i64, first: bool) {
    if thickness <= 1 {
        line_aa(image, a, b);
        return;
    }
    let dx = (a[0] - b[0]) as f64 / ONE as f64;
    let dy = (b[1] - a[1]) as f64 / ONE as f64;
    let radius = thickness * HALF;
    let distance = dx * dx + dy * dy;
    if distance > f64::EPSILON {
        let factor = (radius as f64 + (thickness & 1) as f64 * HALF as f64) / distance.sqrt();
        let offset = [
            (dy * factor).round_ties_even() as i64,
            (dx * factor).round_ties_even() as i64,
        ];
        fill_convex(
            image,
            &[
                [a[0] + offset[0], a[1] + offset[1]],
                [a[0] - offset[0], a[1] - offset[1]],
                [b[0] - offset[0], b[1] - offset[1]],
                [b[0] + offset[0], b[1] + offset[1]],
            ],
        );
    }
    if first {
        round_cap(image, a, radius);
    }
    round_cap(image, b, radius);
}
