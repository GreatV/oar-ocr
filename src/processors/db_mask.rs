use image::GrayImage;
use imageproc::distance_transform::Norm;
use imageproc::morphology;

use super::DBPostProcess;

impl DBPostProcess {
    /// Applies dilation to a binary mask using a Chebyshev radius of 1.
    ///
    /// This method works directly with GrayImage to avoid intermediate allocations.
    pub(super) fn dilate_mask_img(&self, mask_img: &GrayImage) -> GrayImage {
        morphology::dilate(mask_img, Norm::LInf, 1)
    }
}
