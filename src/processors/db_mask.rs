use image::GrayImage;
use imageproc::distance_transform::Norm;
use imageproc::morphology;

use super::DBPostProcess;

impl DBPostProcess {
    /// Applies dilation to a binary mask using a Chebyshev radius of 1.
    pub(super) fn dilate_mask(&self, mask: &[Vec<bool>]) -> Vec<Vec<bool>> {
        let height = mask.len();
        let width = if height > 0 { mask[0].len() } else { 0 };

        if height == 0 || width == 0 {
            return vec![vec![false; width]; height];
        }

        let mut gray_img = GrayImage::new(width as u32, height as u32);
        for (y, row) in mask.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let pixel_value = if value { 255 } else { 0 };
                gray_img.put_pixel(x as u32, y as u32, image::Luma([pixel_value]));
            }
        }

        let dilated_img = morphology::dilate(&gray_img, Norm::LInf, 1);

        let mut dilated = vec![vec![false; width]; height];
        for (y, dilated_row) in dilated.iter_mut().enumerate() {
            for (x, dilated_pixel) in dilated_row.iter_mut().enumerate() {
                let pixel = dilated_img.get_pixel(x as u32, y as u32);
                *dilated_pixel = pixel[0] > 0;
            }
        }

        dilated
    }
}
