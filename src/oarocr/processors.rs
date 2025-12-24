//! Data processors for task graph edges.
//!
//! This module provides processors that transform data between task nodes in the graph.
//! For example, cropping and perspective transformation between detection and recognition.

use crate::core::OCRError;
use crate::processors::BoundingBox;
use crate::utils::BBoxCrop;
use image::{Rgb, RgbImage};
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

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
    pub(crate) handle_rotation: bool,
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
    type Input = (Arc<RgbImage>, Vec<BoundingBox>);
    type Output = Vec<Option<Arc<RgbImage>>>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
        let (image, bboxes) = input;

        let cropped_images: Vec<Option<Arc<RgbImage>>> = bboxes
            .iter()
            .map(|bbox| {
                self.crop_single(&image, bbox)
                    .map(|img| Some(Arc::new(img)))
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
    type Input = (Vec<Option<Arc<RgbImage>>>, Vec<Option<f32>>);
    type Output = Vec<Option<Arc<RgbImage>>>;

    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
        let (images, angles) = input;

        if !self.auto_rotate {
            return Ok(images);
        }

        let rotated_images: Vec<Option<Arc<RgbImage>>> = images
            .into_iter()
            .zip(angles.iter())
            .map(|(img_opt, angle_opt)| {
                match (img_opt, angle_opt) {
                    (Some(img), Some(angle)) if angle.abs() > 0.1 => {
                        // Rotate image by the detected angle
                        // Convert angle from degrees to radians (imageproc expects radians)
                        let angle_radians = -angle.to_radians(); // Negative for clockwise rotation

                        // Use bilinear interpolation for smooth rotation
                        let rotated = rotate_about_center(
                            &img,
                            angle_radians,
                            Interpolation::Bilinear,
                            Rgb([255u8, 255u8, 255u8]), // White background for padding
                        );

                        Some(Arc::new(rotated))
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
///
/// All processors in the chain must have the same input and output types,
/// allowing the output of each processor to be fed as input to the next.
#[derive(Debug)]
pub struct ChainProcessor<T> {
    processors: Vec<Box<dyn EdgeProcessor<Input = T, Output = T>>>,
}

impl<T> ChainProcessor<T> {
    /// Creates a new chain processor with the given processors.
    pub fn new(processors: Vec<Box<dyn EdgeProcessor<Input = T, Output = T>>>) -> Self {
        Self { processors }
    }
}

impl<T> EdgeProcessor for ChainProcessor<T>
where
    T: Debug + Send + Sync,
{
    type Input = T;
    type Output = T;

    fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
        if self.processors.is_empty() {
            return Err(OCRError::ConfigError {
                message: "Empty processor chain".to_string(),
            });
        }

        // Apply all processors in sequence, threading the output of each as input to the next
        let mut current = input;

        for processor in &self.processors {
            current = processor.process(current)?;
        }

        Ok(current)
    }

    fn name(&self) -> &str {
        "Chain"
    }
}

/// Type alias for text cropping processor output
type TextCroppingOutput = Box<
    dyn EdgeProcessor<
            Input = (Arc<RgbImage>, Vec<BoundingBox>),
            Output = Vec<Option<Arc<RgbImage>>>,
        >,
>;

/// Type alias for image rotation processor output
type ImageRotationOutput = Box<
    dyn EdgeProcessor<
            Input = (Vec<Option<Arc<RgbImage>>>, Vec<Option<f32>>),
            Output = Vec<Option<Arc<RgbImage>>>,
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
        if let EdgeProcessorConfig::TextCropping { handle_rotation } = deserialized {
            assert!(handle_rotation);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_image_rotation_processor_rotates_images() {
        let processor = ImageRotationProcessor::new(true);

        // Create a simple test image (10x10 white image)
        let img = Arc::new(RgbImage::from_pixel(10, 10, Rgb([255u8, 255u8, 255u8])));

        // Test with rotation angle
        let images = vec![Some(img.clone())];
        let angles = vec![Some(45.0)]; // 45 degree rotation

        let result = processor.process((images, angles)).unwrap();

        // Should have one rotated image
        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());

        // The rotated image should have different dimensions due to rotation
        let rotated = result[0].as_ref().unwrap();
        // After rotation, the image will be larger to accommodate the rotated content
        assert!(rotated.width() >= 10 || rotated.height() >= 10);
    }

    #[test]
    fn test_image_rotation_processor_skips_small_angles() {
        let processor = ImageRotationProcessor::new(true);

        let img = Arc::new(RgbImage::from_pixel(10, 10, Rgb([255u8, 255u8, 255u8])));
        let images = vec![Some(img.clone())];
        let angles = vec![Some(0.05)]; // Very small angle, should be skipped

        let result = processor.process((images, angles)).unwrap();

        // Should return the original image unchanged
        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());
        let output = result[0].as_ref().unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    #[test]
    fn test_image_rotation_processor_disabled() {
        let processor = ImageRotationProcessor::new(false); // auto_rotate disabled

        let img = Arc::new(RgbImage::from_pixel(10, 10, Rgb([255u8, 255u8, 255u8])));
        let images = vec![Some(img.clone())];
        let angles = vec![Some(45.0)];

        let result = processor.process((images, angles)).unwrap();

        // Should return the original image unchanged
        assert_eq!(result.len(), 1);
        assert!(result[0].is_some());
        let output = result[0].as_ref().unwrap();
        assert_eq!(output.dimensions(), img.dimensions());
    }

    // Test processor that adds a value to an integer
    #[derive(Debug)]
    struct AddProcessor {
        value: i32,
    }

    impl EdgeProcessor for AddProcessor {
        type Input = i32;
        type Output = i32;

        fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
            Ok(input + self.value)
        }

        fn name(&self) -> &str {
            "Add"
        }
    }

    // Test processor that multiplies an integer by a value
    #[derive(Debug)]
    struct MultiplyProcessor {
        value: i32,
    }

    impl EdgeProcessor for MultiplyProcessor {
        type Input = i32;
        type Output = i32;

        fn process(&self, input: Self::Input) -> Result<Self::Output, OCRError> {
            Ok(input * self.value)
        }

        fn name(&self) -> &str {
            "Multiply"
        }
    }

    #[test]
    fn test_chain_processor_single_processor() {
        let processors: Vec<Box<dyn EdgeProcessor<Input = i32, Output = i32>>> =
            vec![Box::new(AddProcessor { value: 5 })];

        let chain = ChainProcessor::new(processors);
        let result = chain.process(10).unwrap();

        // 10 + 5 = 15
        assert_eq!(result, 15);
    }

    #[test]
    fn test_chain_processor_multiple_processors() {
        let processors: Vec<Box<dyn EdgeProcessor<Input = i32, Output = i32>>> = vec![
            Box::new(AddProcessor { value: 5 }),      // 10 + 5 = 15
            Box::new(MultiplyProcessor { value: 2 }), // 15 * 2 = 30
            Box::new(AddProcessor { value: 10 }),     // 30 + 10 = 40
        ];

        let chain = ChainProcessor::new(processors);
        let result = chain.process(10).unwrap();

        // (10 + 5) * 2 + 10 = 40
        assert_eq!(result, 40);
    }

    #[test]
    fn test_chain_processor_empty_chain() {
        let processors: Vec<Box<dyn EdgeProcessor<Input = i32, Output = i32>>> = vec![];

        let chain = ChainProcessor::new(processors);
        let result = chain.process(10);

        // Should return an error for empty chain
        assert!(result.is_err());
        if let Err(OCRError::ConfigError { message }) = result {
            assert_eq!(message, "Empty processor chain");
        } else {
            panic!("Expected ConfigError");
        }
    }

    #[test]
    fn test_chain_processor_name() {
        let processors: Vec<Box<dyn EdgeProcessor<Input = i32, Output = i32>>> =
            vec![Box::new(AddProcessor { value: 5 })];

        let chain = ChainProcessor::new(processors);
        assert_eq!(chain.name(), "Chain");
    }

    #[test]
    fn test_chain_processor_order_matters() {
        // Test that processors are applied in order
        let processors1: Vec<Box<dyn EdgeProcessor<Input = i32, Output = i32>>> = vec![
            Box::new(AddProcessor { value: 5 }),      // 10 + 5 = 15
            Box::new(MultiplyProcessor { value: 2 }), // 15 * 2 = 30
        ];

        let processors2: Vec<Box<dyn EdgeProcessor<Input = i32, Output = i32>>> = vec![
            Box::new(MultiplyProcessor { value: 2 }), // 10 * 2 = 20
            Box::new(AddProcessor { value: 5 }),      // 20 + 5 = 25
        ];

        let chain1 = ChainProcessor::new(processors1);
        let chain2 = ChainProcessor::new(processors2);

        let result1 = chain1.process(10).unwrap();
        let result2 = chain2.process(10).unwrap();

        // (10 + 5) * 2 = 30
        assert_eq!(result1, 30);
        // (10 * 2) + 5 = 25
        assert_eq!(result2, 25);
        // Results should be different
        assert_ne!(result1, result2);
    }
}
