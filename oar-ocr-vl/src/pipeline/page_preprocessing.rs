//! Image preparation shared by complete-page parsing pipelines.

use crate::Error;
use image::RgbImage;
use std::borrow::Cow;

/// Prepare a page image before layout detection and region recognition.
///
/// The returned canvas defines the coordinates of the parsed regions.
pub trait PagePreprocessor: Send + Sync {
    /// Transform one page into its parsing canvas.
    fn preprocess(&self, image: &RgbImage) -> Result<RgbImage, Error>;
}

pub(crate) fn prepare_page<'a>(
    image: Cow<'a, RgbImage>,
    preprocessor: Option<&dyn PagePreprocessor>,
) -> Result<Cow<'a, RgbImage>, Error> {
    match preprocessor {
        Some(preprocessor) => preprocessor.preprocess(&image).map(Cow::Owned),
        None => Ok(image),
    }
}
