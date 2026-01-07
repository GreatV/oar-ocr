//! Copy-on-Write utilities for efficient image handling.
//!
//! This module provides utilities for implementing Copy-on-Write semantics
//! when working with Arc-wrapped images, minimizing unnecessary clones.

use image::RgbImage;
use std::sync::Arc;

/// Trait for Copy-on-Write operations on Arc-wrapped data.
pub trait ArcCow<T> {
    /// Gets a mutable reference to the inner value, cloning if necessary.
    ///
    /// If this is the only Arc pointing to the value, returns the owned value
    /// without cloning. Otherwise, clones the value.
    fn make_mut(arc: &mut Arc<T>) -> &mut T
    where
        T: Clone;

    /// Tries to unwrap the Arc to get owned data, or clones if there are other references.
    fn unwrap_or_clone(arc: Arc<T>) -> T
    where
        T: Clone;
}

impl ArcCow<RgbImage> for RgbImage {
    fn make_mut(arc: &mut Arc<RgbImage>) -> &mut RgbImage {
        Arc::make_mut(arc)
    }

    fn unwrap_or_clone(arc: Arc<RgbImage>) -> RgbImage {
        Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone())
    }
}

/// Clones an Arc-wrapped image only if it has multiple strong references.
///
/// This is useful for Copy-on-Write semantics where you want to modify an image
/// but only clone it if other parts of the code are also referencing it.
///
/// # Arguments
///
/// * `image` - Arc-wrapped image
///
/// # Returns
///
/// A new Arc with exclusive ownership of the image (cloning if necessary)
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use image::RgbImage;
/// use oar_ocr_core::utils::cow::clone_if_shared;
///
/// let image = Arc::new(RgbImage::new(100, 100));
/// let cloned = Arc::clone(&image); // Two references now
///
/// // This will clone because there are multiple references
/// let owned = clone_if_shared(cloned);
/// assert_eq!(Arc::strong_count(&owned), 1);
/// ```
pub fn clone_if_shared(image: Arc<RgbImage>) -> Arc<RgbImage> {
    match Arc::try_unwrap(image) {
        Ok(owned) => Arc::new(owned),
        Err(arc) => Arc::new((*arc).clone()),
    }
}

/// Modifies an Arc-wrapped image in place if possible, cloning only if necessary.
///
/// This function applies a transformation to an image, using Copy-on-Write semantics.
/// If the Arc has no other references, the image is modified in place.
/// Otherwise, it is cloned first.
///
/// # Arguments
///
/// * `image` - Arc-wrapped image to modify
/// * `f` - Function that modifies the image
///
/// # Returns
///
/// A new Arc containing the modified image
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use image::{RgbImage, Rgb};
/// use oar_ocr_core::utils::cow::modify_cow;
///
/// let image = Arc::new(RgbImage::new(100, 100));
///
/// // Modify the image (will clone if needed)
/// let modified = modify_cow(image, |img| {
///     // Fill with white
///     for pixel in img.pixels_mut() {
///         *pixel = Rgb([255, 255, 255]);
///     }
/// });
/// ```
pub fn modify_cow<F>(mut image: Arc<RgbImage>, f: F) -> Arc<RgbImage>
where
    F: FnOnce(&mut RgbImage),
{
    // make_mut will clone only if there are other references
    f(Arc::make_mut(&mut image));
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn test_clone_if_shared_unique() {
        // Unique reference - no clone should occur
        let image = Arc::new(RgbImage::new(10, 10));
        let result = clone_if_shared(image);
        assert_eq!(Arc::strong_count(&result), 1);
    }

    #[test]
    fn test_clone_if_shared_multiple_refs() {
        // Multiple references - should clone
        let image = Arc::new(RgbImage::new(10, 10));
        let clone1 = Arc::clone(&image);
        let _clone2 = Arc::clone(&image);

        let result = clone_if_shared(clone1);
        assert_eq!(Arc::strong_count(&result), 1);
        assert_eq!(Arc::strong_count(&image), 2); // Original still has 2 refs
    }

    #[test]
    fn test_modify_cow_unique() {
        // Unique reference - should modify in place
        let image = Arc::new(RgbImage::new(10, 10));
        let ptr_before = Arc::as_ptr(&image);

        let modified = modify_cow(image, |img| {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([255, 0, 0]);
            }
        });

        let ptr_after = Arc::as_ptr(&modified);
        // Same pointer means in-place modification
        assert_eq!(ptr_before, ptr_after);
        assert_eq!(modified.get_pixel(0, 0), &Rgb([255, 0, 0]));
    }

    #[test]
    fn test_modify_cow_shared() {
        // Multiple references - should clone before modifying
        let image = Arc::new(RgbImage::new(10, 10));
        let clone = Arc::clone(&image);
        let ptr_original = Arc::as_ptr(&image);

        let modified = modify_cow(clone, |img| {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([255, 0, 0]);
            }
        });

        let ptr_modified = Arc::as_ptr(&modified);
        // Different pointers means clone occurred
        assert_ne!(ptr_original, ptr_modified);

        // Original unchanged
        assert_eq!(image.get_pixel(0, 0), &Rgb([0, 0, 0]));
        // Modified has new value
        assert_eq!(modified.get_pixel(0, 0), &Rgb([255, 0, 0]));
    }

    #[test]
    fn test_unwrap_or_clone_unique() {
        let image = Arc::new(RgbImage::new(10, 10));
        let owned = RgbImage::unwrap_or_clone(image);
        assert_eq!(owned.dimensions(), (10, 10));
    }

    #[test]
    fn test_unwrap_or_clone_shared() {
        let image = Arc::new(RgbImage::new(10, 10));
        let _clone = Arc::clone(&image);
        let owned = RgbImage::unwrap_or_clone(image);
        assert_eq!(owned.dimensions(), (10, 10));
    }

    #[test]
    fn test_make_mut() {
        let mut image = Arc::new(RgbImage::new(10, 10));
        let img_mut = RgbImage::make_mut(&mut image);
        img_mut.put_pixel(0, 0, Rgb([255, 0, 0]));
        assert_eq!(image.get_pixel(0, 0), &Rgb([255, 0, 0]));
    }
}
