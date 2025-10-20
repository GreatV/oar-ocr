//! Data processors for task graph edges.
//!
//! This module provides processors that transform data between task nodes in the graph.
//! For example, cropping and perspective transformation between detection and recognition.

use crate::core::OCRError;
use crate::processors::BoundingBox;
use crate::utils::BBoxCrop;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Trait for processors that transform data between task nodes.
pub trait EdgeProcessor: Debug + Send + Sync {
    /// Input type for this processor
    type Input;

    /// Output type for this processor
    type Output;

    /// Process the input data and produce output
    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError>;

    /// Get the processor name for debugging
    fn name(&self) -> &str;
}

/// Configuration for edge processors in the task graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EdgeProcessorConfig {
    /// Crop text regions from image based on bounding boxes
    TextCropping {
        /// Whether to handle rotated bounding boxes
        #[serde(default = "default_true")]
        handle_rotation: bool,
    },

    /// Apply perspective transformation to correct text orientation
    PerspectiveTransform {
        /// Target width for transformed images
        target_width: Option<u32>,
        /// Target height for transformed images
        target_height: Option<u32>,
    },

    /// Rotate images based on orientation angles
    ImageRotation {
        /// Whether to rotate based on detected angles
        #[serde(default = "default_true")]
        auto_rotate: bool,
    },

    /// Resize images to specific dimensions
    ImageResize {
        /// Target width
        width: u32,
        /// Target height
        height: u32,
        /// Whether to maintain aspect ratio
        #[serde(default)]
        maintain_aspect_ratio: bool,
    },

    /// Chain multiple processors
    Chain {
        /// List of processors to apply in sequence
        processors: Vec<EdgeProcessorConfig>,
    },
}

fn default_true() -> bool {
    true
}

/// Processor that crops text regions from an image based on bounding boxes.
#[derive(Debug)]
pub struct TextCroppingProcessor {
    handle_rotation: bool,
}

impl TextCroppingProcessor {
    pub fn new(handle_rotation: bool) -> Self {
        Self { handle_rotation }
    }

    /// Crop a single bounding box from an image
    fn crop_single(&self, image: &RgbImage, bbox: &BoundingBox) -> Result<RgbImage, OCRError> {
        if self.handle_rotation && bbox.points.len() == 4 {
            // Rotated bounding box (quadrilateral) - use perspective transform
            BBoxCrop::crop_rotated_bounding_box(image, bbox)
        } else {
            // Regular axis-aligned bounding box
            BBoxCrop::crop_bounding_box(image, bbox)
        }
    }
}

impl EdgeProcessor for TextCroppingProcessor {
    type Input = (RgbImage, Vec<BoundingBox>);
    type Output = Vec<Option<RgbImage>>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
        let (image, bboxes) = input;

        let cropped_images: Vec<Option<RgbImage>> = bboxes
            .iter()
            .map(|bbox| {
                self.crop_single(&image, bbox)
                    .map(Some)
                    .unwrap_or_else(|_e| {
                        // Failed to crop, return None
                        None
                    })
            })
            .collect();

        Ok(cropped_images)
    }

    fn name(&self) -> &str {
        "TextCropping"
    }
}

/// Processor that rotates images based on orientation angles.
#[derive(Debug)]
pub struct ImageRotationProcessor {
    auto_rotate: bool,
}

impl ImageRotationProcessor {
    pub fn new(auto_rotate: bool) -> Self {
        Self { auto_rotate }
    }
}

impl EdgeProcessor for ImageRotationProcessor {
    type Input = (Vec<Option<RgbImage>>, Vec<Option<f32>>);
    type Output = Vec<Option<RgbImage>>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
        let (images, angles) = input;

        if !self.auto_rotate {
            return Ok(images);
        }

        let rotated_images: Vec<Option<RgbImage>> = images
            .into_iter()
            .zip(angles.iter())
            .map(|(img_opt, angle_opt)| {
                match (img_opt, angle_opt) {
                    (Some(img), Some(angle)) if angle.abs() > 0.1 => {
                        // Rotate image by the detected angle
                        // Note: For now, we just return the original image
                        // A full implementation would use image rotation utilities
                        Some(img)
                    }
                    (img_opt, _) => img_opt,
                }
            })
            .collect();

        Ok(rotated_images)
    }

    fn name(&self) -> &str {
        "ImageRotation"
    }
}

/// Processor that chains multiple processors together.
#[derive(Debug)]
#[allow(dead_code)]
pub struct ChainProcessor<I, O> {
    processors: Vec<Box<dyn EdgeProcessor<Input = I, Output = O>>>,
}

impl<I, O> ChainProcessor<I, O> {
    #[allow(dead_code)]
    pub fn new(processors: Vec<Box<dyn EdgeProcessor<Input = I, Output = O>>>) -> Self {
        Self { processors }
    }
}

impl<I, O> EdgeProcessor for ChainProcessor<I, O>
where
    I: Debug + Send + Sync,
    O: Debug + Send + Sync,
{
    type Input = I;
    type Output = O;

    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
        // For now, we just apply the first processor
        // A full implementation would need to handle chaining properly
        if let Some(first) = self.processors.first() {
            first.process(input)
        } else {
            Err(OCRError::ConfigError {
                message: "Empty processor chain".to_string(),
            })
        }
    }

    fn name(&self) -> &str {
        "Chain"
    }
}

/// Type alias for text cropping processor output
type TextCroppingOutput =
    Box<dyn EdgeProcessor<Input = (RgbImage, Vec<BoundingBox>), Output = Vec<Option<RgbImage>>>>;

/// Type alias for image rotation processor output
type ImageRotationOutput = Box<
    dyn EdgeProcessor<
            Input = (Vec<Option<RgbImage>>, Vec<Option<f32>>),
            Output = Vec<Option<RgbImage>>,
        >,
>;

/// Factory for creating edge processors from configuration.
pub struct EdgeProcessorFactory;

impl EdgeProcessorFactory {
    /// Create a text cropping processor
    pub fn create_text_cropping(handle_rotation: bool) -> TextCroppingOutput {
        Box::new(TextCroppingProcessor::new(handle_rotation))
    }

    /// Create an image rotation processor
    pub fn create_image_rotation(auto_rotate: bool) -> ImageRotationOutput {
        Box::new(ImageRotationProcessor::new(auto_rotate))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_cropping_processor_creation() {
        let processor = TextCroppingProcessor::new(true);
        assert_eq!(processor.name(), "TextCropping");
    }

    #[test]
    fn test_image_rotation_processor_creation() {
        let processor = ImageRotationProcessor::new(true);
        assert_eq!(processor.name(), "ImageRotation");
    }

    #[test]
    fn test_edge_processor_config_serialization() {
        let config = EdgeProcessorConfig::TextCropping {
            handle_rotation: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("TextCropping"));

        let deserialized: EdgeProcessorConfig = serde_json::from_str(&json).unwrap();
        match deserialized {
            EdgeProcessorConfig::TextCropping { handle_rotation } => {
                assert!(handle_rotation);
            }
            _ => panic!("Wrong variant"),
        }
    }
}
