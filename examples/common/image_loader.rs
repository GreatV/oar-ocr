use image::{DynamicImage, ImageError, ImageReader, RgbImage};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Loads an image into RGB format while tolerating mismatched file extensions.
///
/// The `image` crate guesses the decoder from the file extension by default,
/// which fails when users store JPEG bytes in a `.png` file (or vice versa).
/// This helper retries with format sniffing so the actual content determines
/// the decoder.
pub fn load_rgb_image(path: &Path) -> Result<RgbImage, ImageError> {
    load_dynamic_image(path).map(|img| img.to_rgb8())
}

fn load_dynamic_image(path: &Path) -> Result<DynamicImage, ImageError> {
    match image::open(path) {
        Ok(img) => Ok(img),
        Err(err) if should_retry(&err) => {
            tracing::warn!(
                "Standard decode failed for {} ({err}). Retrying with format sniffing.",
                path.display()
            );
            decode_with_guessed_format(path)
        }
        Err(err) => Err(err),
    }
}

fn should_retry(err: &ImageError) -> bool {
    matches!(err, ImageError::Decoding(_) | ImageError::Unsupported(_))
}

fn decode_with_guessed_format(path: &Path) -> Result<DynamicImage, ImageError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let reader = ImageReader::new(reader).with_guessed_format()?;
    reader.decode()
}
