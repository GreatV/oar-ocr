//! Float image resize and normalized sampling for page transforms.

use candle_core::{Device, Result, Tensor};
use image::RgbImage;

pub(crate) fn cv2_linear_rgb(
    image: &RgbImage,
    width: usize,
    height: usize,
    device: &Device,
) -> Result<Tensor> {
    let (source_w, source_h) = (image.width() as usize, image.height() as usize);
    if source_w == 0 || source_h == 0 || width == 0 || height == 0 {
        candle_core::bail!("linear resize requires nonempty images");
    }
    let coordinate = |index: usize, input: usize, output: usize| {
        let position = (index as f64 + 0.5) * input as f64 / output as f64 - 0.5;
        let lower = position.floor() as i64;
        if lower < 0 {
            (0, 0, 0f32)
        } else if lower as usize >= input - 1 {
            (input - 1, input - 1, 0.)
        } else {
            (
                lower as usize,
                lower as usize + 1,
                (position - lower as f64) as f32,
            )
        }
    };
    let mut values = vec![0f32; 3 * width * height];
    for y in 0..height {
        let (y0, y1, fy) = coordinate(y, source_h, height);
        for x in 0..width {
            let (x0, x1, fx) = coordinate(x, source_w, width);
            for c in 0..3 {
                let value =
                    |x: usize, y: usize| image.get_pixel(x as u32, y as u32)[c] as f32 / 255.;
                let a = value(x0, y0) * (1. - fx) + value(x1, y0) * fx;
                let b = value(x0, y1) * (1. - fx) + value(x1, y1) * fx;
                values[c * width * height + y * width + x] = a * (1. - fy) + b * fy;
            }
        }
    }
    Tensor::from_vec(values, (1, 3, height, width), device)
}

pub(crate) fn unwarp_rgb(image: &RgbImage, grid: &Tensor) -> Result<RgbImage> {
    let (batch, channels, _, _) = grid.dims4()?;
    if batch != 1 || channels != 2 || image.width() == 0 || image.height() == 0 {
        candle_core::bail!("unwarp requires one RGB image and an x/y grid");
    }
    let (width, height) = (image.width() as usize, image.height() as usize);
    let positions = grid
        .upsample_bilinear2d(height, width, true)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut output = RgbImage::new(image.width(), image.height());
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let gx = ((positions[i] + 1.) / 2.) * (width - 1) as f32;
            let gy = ((positions[width * height + i] + 1.) / 2.) * (height - 1) as f32;
            let (x0, y0) = (gx.floor() as i64, gy.floor() as i64);
            let (x1, y1) = (x0 + 1, y0 + 1);
            let weights = [
                ((x1 as f32 - gx) * (y1 as f32 - gy), x0, y0),
                ((gx - x0 as f32) * (y1 as f32 - gy), x1, y0),
                ((x1 as f32 - gx) * (gy - y0 as f32), x0, y1),
                ((gx - x0 as f32) * (gy - y0 as f32), x1, y1),
            ];
            let mut color = [0f32; 3];
            for (weight, xx, yy) in weights {
                if xx >= 0 && yy >= 0 && xx < width as i64 && yy < height as i64 {
                    let pixel = image.get_pixel(xx as u32, yy as u32);
                    for c in 0..3 {
                        color[c] += (pixel[c] as f32 / 255.) * weight;
                    }
                }
            }
            output.put_pixel(
                x as u32,
                y as u32,
                image::Rgb(color.map(|v| (v * 255.) as u8)),
            );
        }
    }
    Ok(output)
}
