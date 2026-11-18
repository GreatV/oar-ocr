//! The packed page tensor carries separate padded and visible image dimensions.
use crate::Error;
use crate::runtime::image::{
    image_to_merge_grouped_patches, pil_bilinear_resize, round_up_to_multiple,
};
use candle_core::{DType, Device, Tensor};
use image::{Rgb, RgbImage, imageops};

#[derive(serde::Deserialize)]
pub(super) struct ProcessorConfig {
    final_size: [u32; 2],
    image_mean: [f32; 3],
    image_std: [f32; 3],
    rescale_factor: f32,
}
pub(super) struct Page {
    pub patches: Tensor,
    pub rows: usize,
    pub cols: usize,
    pub height: usize,
    pub width: usize,
}

impl Page {
    pub fn prepare(
        image: &RgbImage,
        config: &ProcessorConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, Error> {
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return Err(Error::invalid_input(
                "Nemotron Parse requires a nonempty image",
            ));
        }
        let aspect = width as f64 / height as f64;
        let (mut w, mut h) = (width, height);
        let [max_height, max_width] = config.final_size;
        if h > max_height {
            h = max_height;
            w = (h as f64 * aspect) as u32;
        }
        if w > max_width {
            w = max_width;
            h = (w as f64 / aspect) as u32;
        }
        let (h, w) = (h.max(1), w.max(1));
        let (ph, pw) = (round_up_to_multiple(h, 16), round_up_to_multiple(w, 64));
        let mut canvas = RgbImage::from_pixel(pw, ph, Rgb([255; 3]));
        if (w, h) == image.dimensions() {
            imageops::replace(&mut canvas, image, 0, 0);
        } else {
            imageops::replace(&mut canvas, &pil_bilinear_resize(image, w, h)?, 0, 0);
        }
        let data = image_to_merge_grouped_patches(
            &canvas,
            &config.image_mean,
            &config.image_std,
            Some(config.rescale_factor),
            16,
            1,
            1,
        );
        let (rows, cols) = ((ph / 16) as usize, (pw / 16) as usize);
        let patches = Tensor::from_vec(data, (rows * cols, 768), device)
            .and_then(|p| p.to_dtype(dtype))
            .map_err(super::infer)?;
        Ok(Self {
            patches,
            rows,
            cols,
            height: h as usize,
            width: w as usize,
        })
    }
}
