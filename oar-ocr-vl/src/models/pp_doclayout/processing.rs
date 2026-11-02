//! PP-DocLayout image preprocessing.
//!
//! PaddleX uses OpenCV `INTER_CUBIC` to 800x800, a `1/255` rescale and an
//! identity normalization (mean 0, std 1). We reproduce its generic uint8
//! kernel; accelerated OpenCV paths can differ in final pixel rounding.

use super::err::proc_err;
use crate::error::Error;
use crate::runtime::image::cv2_cubic_resize;
use candle_core::{DType, Device, Tensor};
use image::RgbImage;

/// Input size both checkpoints are trained and exported at.
pub(super) const INPUT_SIZE: u32 = 800;

/// Model input plus the source dimensions needed to map boxes back.
pub(super) struct PreprocessedImage {
    /// `(1, 3, 800, 800)` float tensor.
    pub pixel_values: Tensor,
    pub original_width: u32,
    pub original_height: u32,
}

/// Resizes to the fixed 800x800 input and scales pixels into `[0, 1]`.
pub(super) fn preprocess(
    image: &RgbImage,
    device: &Device,
    dtype: DType,
) -> Result<PreprocessedImage, Error> {
    let (original_width, original_height) = (image.width(), image.height());
    if original_width == 0 || original_height == 0 {
        return Err(Error::Config {
            message: "PP-DocLayout: input image has zero width or height".to_string(),
        });
    }

    let resized = cv2_cubic_resize(image, INPUT_SIZE, INPUT_SIZE)?;

    let pixels = INPUT_SIZE as usize * INPUT_SIZE as usize;
    let mut data = vec![0f32; 3 * pixels];
    for (index, pixel) in resized.pixels().enumerate() {
        data[index] = pixel[0] as f32 / 255.0;
        data[pixels + index] = pixel[1] as f32 / 255.0;
        data[2 * pixels + index] = pixel[2] as f32 / 255.0;
    }

    let pixel_values = Tensor::from_vec(
        data,
        (1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize),
        device,
    )
    .and_then(|t| t.to_dtype(dtype))
    .map_err(|e| proc_err("build PP-DocLayout input tensor", e))?;

    Ok(PreprocessedImage {
        pixel_values,
        original_width,
        original_height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_a_normalized_fixed_size_tensor() {
        let image = RgbImage::from_pixel(37, 11, image::Rgb([255, 0, 128]));
        let prepared = preprocess(&image, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(
            prepared.pixel_values.dims(),
            &[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize]
        );
        assert_eq!(prepared.original_width, 37);
        assert_eq!(prepared.original_height, 11);

        let values = prepared
            .pixel_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!((values[0] - 1.0).abs() < 1e-6);

        // cv2 4.13.0 INTER_CUBIC, setUseOptimized(False): borders, clipping,
        // coefficient quantization and downsampling without an antialias pass.
        let sample = |width, height| {
            RgbImage::from_fn(width, height, |x, y| {
                image::Rgb(std::array::from_fn(|channel| {
                    let i = ((y * width + x) * 3) as usize + channel;
                    ((i * 37 + i / 11 * 19) % 256) as u8
                }))
            })
        };
        let up = cv2_cubic_resize(&sample(4, 3), 7, 5).unwrap();
        assert_eq!(
            up.as_raw(),
            &[
                0, 4, 68, 19, 74, 127, 103, 153, 191, 198, 68, 101, 223, 0, 21, 138, 61, 112, 67,
                130, 190, 83, 120, 27, 81, 128, 87, 90, 134, 164, 168, 109, 156, 202, 86, 123, 105,
                97, 141, 25, 108, 158, 221, 255, 14, 151, 188, 56, 65, 102, 124, 118, 153, 214,
                170, 208, 245, 84, 134, 171, 14, 72, 109, 208, 245, 153, 129, 165, 125, 33, 69, 99,
                67, 128, 177, 146, 193, 230, 162, 109, 145, 165, 38, 73, 168, 205, 255, 97, 134,
                187, 11, 48, 87, 36, 99, 134, 133, 154, 191, 229, 80, 117, 255, 18, 55,
            ]
        );
        let down = cv2_cubic_resize(&sample(12, 9), 3, 2).unwrap();
        assert_eq!(
            down.as_raw(),
            &[
                139, 142, 101, 91, 141, 177, 195, 55, 129, 115, 124, 161, 164, 90, 112, 117, 154,
                74,
            ]
        );
    }

    #[test]
    fn rejects_empty_images() {
        let image = RgbImage::new(0, 0);
        assert!(preprocess(&image, &Device::Cpu, DType::F32).is_err());
    }
}
