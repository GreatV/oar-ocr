//! Transform configuration types and registry.

use serde::{Deserialize, Serialize};

/// Enumeration of available transform types.
///
/// This enum represents the different types of transformations that can be applied
/// to images in the OCR pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformType {
    /// Resize an image to a specified size.
    ResizeImage,
    /// Crop an image to a specified size.
    CropImage,
    /// Normalize pixel values.
    Normalize,
    /// Convert image to grayscale.
    ToGrayscale,
    /// Apply Gaussian blur.
    GaussianBlur,
    /// Adjust brightness.
    Brightness,
    /// Adjust contrast.
    Contrast,
    /// Rotate image.
    Rotate,
    /// Flip image horizontally.
    FlipHorizontal,
    /// Flip image vertically.
    FlipVertical,
}

/// Configuration for different transform types.
///
/// This enum contains the configuration parameters for each transform type.
#[derive(Debug, Serialize, Deserialize)]
pub enum TransformConfig {
    /// Configuration for resizing an image.
    ResizeImage {
        /// The size to resize the shorter side to (optional).
        shorter_side: Option<u32>,
        /// The size to resize the longer side to (optional).
        longer_side: Option<u32>,
        /// The exact width to resize to (optional).
        width: Option<u32>,
        /// The exact height to resize to (optional).
        height: Option<u32>,
        /// Whether to maintain aspect ratio (default: true).
        maintain_aspect_ratio: Option<bool>,
        /// The interpolation method to use (default: "bilinear").
        interpolation: Option<String>,
    },
    /// Configuration for cropping an image.
    CropImage {
        /// The width of the crop.
        width: u32,
        /// The height of the crop.
        height: u32,
        /// The crop mode (e.g., "center", "top_left").
        mode: Option<String>,
    },
    /// Configuration for normalizing pixel values.
    Normalize {
        /// The mean values for each channel.
        mean: Vec<f32>,
        /// The standard deviation values for each channel.
        std: Vec<f32>,
    },
    /// Configuration for converting to grayscale (no parameters).
    ToGrayscale,
    /// Configuration for Gaussian blur.
    GaussianBlur {
        /// The sigma value for the blur.
        sigma: f32,
    },
    /// Configuration for brightness adjustment.
    Brightness {
        /// The brightness factor (-1.0 to 1.0).
        factor: f32,
    },
    /// Configuration for contrast adjustment.
    Contrast {
        /// The contrast factor (0.0 to 2.0).
        factor: f32,
    },
    /// Configuration for rotation.
    Rotate {
        /// The angle in degrees.
        angle: f32,
    },
    /// Configuration for horizontal flip (no parameters).
    FlipHorizontal,
    /// Configuration for vertical flip (no parameters).
    FlipVertical,
}

/// A registry for managing transform configurations.
///
/// This struct provides a way to register and manage multiple transform configurations
/// that can be applied to images in sequence.
#[derive(Debug)]
pub struct TransformRegistry {
    /// A vector of tuples containing transform types and their configurations.
    transforms: Vec<(TransformType, TransformConfig)>,
}

impl TransformRegistry {
    /// Creates a new empty TransformRegistry.
    ///
    /// # Returns
    ///
    /// A new empty TransformRegistry.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oar_ocr::core::config::transform::TransformRegistry;
    ///
    /// let registry = TransformRegistry::new();
    /// assert_eq!(registry.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Adds a transform to the registry.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to add.
    /// * `config` - The configuration for the transform.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oar_ocr::core::config::transform::{TransformRegistry, TransformType, TransformConfig};
    ///
    /// let mut registry = TransformRegistry::new();
    /// registry.add(TransformType::ResizeImage, TransformConfig::ResizeImage {
    ///     width: Some(224),
    ///     height: Some(224),
    ///     shorter_side: None,
    ///     longer_side: None,
    ///     maintain_aspect_ratio: Some(false),
    ///     interpolation: None,
    /// });
    /// assert_eq!(registry.len(), 1);
    /// ```
    pub fn add(&mut self, transform_type: TransformType, config: TransformConfig) {
        self.transforms.push((transform_type, config));
    }

    /// Removes all transforms of a specific type from the registry.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to remove.
    ///
    /// # Returns
    ///
    /// The number of transforms removed.
    pub fn remove(&mut self, transform_type: &TransformType) -> usize {
        let initial_len = self.transforms.len();
        self.transforms.retain(|(t, _)| t != transform_type);
        initial_len - self.transforms.len()
    }

    /// Gets the number of transforms in the registry.
    ///
    /// # Returns
    ///
    /// The number of transforms in the registry.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Checks if the registry is empty.
    ///
    /// # Returns
    ///
    /// True if the registry is empty, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Clears all transforms from the registry.
    pub fn clear(&mut self) {
        self.transforms.clear();
    }

    /// Gets an iterator over the transforms in the registry.
    ///
    /// # Returns
    ///
    /// An iterator over tuples of (TransformType, TransformConfig).
    pub fn iter(&self) -> std::slice::Iter<'_, (TransformType, TransformConfig)> {
        self.transforms.iter()
    }

    /// Gets a mutable iterator over the transforms in the registry.
    ///
    /// # Returns
    ///
    /// A mutable iterator over tuples of (TransformType, TransformConfig).
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, (TransformType, TransformConfig)> {
        self.transforms.iter_mut()
    }

    /// Checks if the registry contains a specific transform type.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to check for.
    ///
    /// # Returns
    ///
    /// True if the registry contains the transform type, false otherwise.
    pub fn contains(&self, transform_type: &TransformType) -> bool {
        self.transforms.iter().any(|(t, _)| t == transform_type)
    }

    /// Gets all transforms of a specific type.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to get.
    ///
    /// # Returns
    ///
    /// A vector of references to the configurations for the specified transform type.
    pub fn get_all(&self, transform_type: &TransformType) -> Vec<&TransformConfig> {
        self.transforms
            .iter()
            .filter_map(|(t, config)| {
                if t == transform_type {
                    Some(config)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Gets the first transform of a specific type.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to get.
    ///
    /// # Returns
    ///
    /// An optional reference to the configuration for the first transform of the specified type.
    pub fn get_first(&self, transform_type: &TransformType) -> Option<&TransformConfig> {
        self.transforms.iter().find_map(|(t, config)| {
            if t == transform_type {
                Some(config)
            } else {
                None
            }
        })
    }
}

impl IntoIterator for TransformRegistry {
    type Item = (TransformType, TransformConfig);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    /// This allows TransformRegistry to be used in for loops and other iterator contexts.
    fn into_iter(self) -> Self::IntoIter {
        self.transforms.into_iter()
    }
}

impl<'a> IntoIterator for &'a TransformRegistry {
    type Item = &'a (TransformType, TransformConfig);
    type IntoIter = std::slice::Iter<'a, (TransformType, TransformConfig)>;

    /// This allows &TransformRegistry to be used in for loops and other iterator contexts.
    fn into_iter(self) -> Self::IntoIter {
        self.transforms.iter()
    }
}

impl<'a> IntoIterator for &'a mut TransformRegistry {
    type Item = &'a mut (TransformType, TransformConfig);
    type IntoIter = std::slice::IterMut<'a, (TransformType, TransformConfig)>;

    /// This allows &mut TransformRegistry to be used in for loops and other iterator contexts.
    fn into_iter(self) -> Self::IntoIter {
        self.transforms.iter_mut()
    }
}

impl Default for TransformRegistry {
    /// This allows TransformRegistry to be created with default values.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_registry_new() {
        let registry = TransformRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_transform_registry_add() {
        let mut registry = TransformRegistry::new();
        registry.add(
            TransformType::ResizeImage,
            TransformConfig::ResizeImage {
                width: Some(224),
                height: Some(224),
                shorter_side: None,
                longer_side: None,
                maintain_aspect_ratio: Some(false),
                interpolation: None,
            },
        );
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_transform_registry_remove() {
        let mut registry = TransformRegistry::new();
        registry.add(TransformType::ResizeImage, TransformConfig::ToGrayscale);
        registry.add(TransformType::ToGrayscale, TransformConfig::ToGrayscale);

        let removed = registry.remove(&TransformType::ToGrayscale);
        assert_eq!(removed, 1);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_transform_registry_contains() {
        let mut registry = TransformRegistry::new();
        registry.add(TransformType::ResizeImage, TransformConfig::ToGrayscale);

        assert!(registry.contains(&TransformType::ResizeImage));
        assert!(!registry.contains(&TransformType::ToGrayscale));
    }

    #[test]
    fn test_transform_registry_clear() {
        let mut registry = TransformRegistry::new();
        registry.add(TransformType::ResizeImage, TransformConfig::ToGrayscale);
        registry.add(TransformType::ToGrayscale, TransformConfig::ToGrayscale);

        registry.clear();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }
}
