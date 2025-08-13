//! Document transformation post-processing functionality.

use std::str::FromStr;

/// Post-processor for document transformation results.
///
/// The `DocTrPostProcess` struct handles the post-processing of document
/// transformation model outputs, converting normalized coordinates back
/// to pixel coordinates and applying various transformations.
#[derive(Debug)]
pub struct DocTrPostProcess {
    /// Scale factor to convert normalized values back to pixel values.
    pub scale: f32,
}

impl DocTrPostProcess {
    /// Creates a new DocTrPostProcess instance.
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale factor for converting normalized coordinates to pixels.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::doctr_postprocess::DocTrPostProcess;
    ///
    /// let postprocessor = DocTrPostProcess::new(1.0);
    /// ```
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }

    /// Gets the current scale factor.
    ///
    /// # Returns
    ///
    /// The scale factor used for coordinate conversion.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Sets a new scale factor.
    ///
    /// # Arguments
    ///
    /// * `scale` - New scale factor.
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Converts normalized coordinates to pixel coordinates.
    ///
    /// # Arguments
    ///
    /// * `normalized_coords` - Vector of normalized coordinates (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` - Vector of pixel coordinates.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::doctr_postprocess::DocTrPostProcess;
    ///
    /// let postprocessor = DocTrPostProcess::new(100.0);
    /// let normalized = vec![0.1, 0.2, 0.8, 0.9];
    /// let pixels = postprocessor.denormalize_coordinates(&normalized);
    /// assert_eq!(pixels, vec![10.0, 20.0, 80.0, 90.0]);
    /// ```
    pub fn denormalize_coordinates(&self, normalized_coords: &[f32]) -> Vec<f32> {
        normalized_coords
            .iter()
            .map(|&coord| coord * self.scale)
            .collect()
    }

    /// Converts pixel coordinates to normalized coordinates.
    ///
    /// # Arguments
    ///
    /// * `pixel_coords` - Vector of pixel coordinates.
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` - Vector of normalized coordinates (0.0 to 1.0).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::doctr_postprocess::DocTrPostProcess;
    ///
    /// let postprocessor = DocTrPostProcess::new(100.0);
    /// let pixels = vec![10.0, 20.0, 80.0, 90.0];
    /// let normalized = postprocessor.normalize_coordinates(&pixels);
    /// assert_eq!(normalized, vec![0.1, 0.2, 0.8, 0.9]);
    /// ```
    pub fn normalize_coordinates(&self, pixel_coords: &[f32]) -> Vec<f32> {
        if self.scale == 0.0 {
            return vec![0.0; pixel_coords.len()];
        }
        pixel_coords
            .iter()
            .map(|&coord| coord / self.scale)
            .collect()
    }

    /// Processes a bounding box from normalized to pixel coordinates.
    ///
    /// # Arguments
    ///
    /// * `bbox` - Bounding box as [x1, y1, x2, y2] in normalized coordinates.
    ///
    /// # Returns
    ///
    /// * `[f32; 4]` - Bounding box in pixel coordinates.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::doctr_postprocess::DocTrPostProcess;
    ///
    /// let postprocessor = DocTrPostProcess::new(100.0);
    /// let normalized_bbox = [0.1, 0.2, 0.8, 0.9];
    /// let pixel_bbox = postprocessor.process_bbox(&normalized_bbox);
    /// assert_eq!(pixel_bbox, [10.0, 20.0, 80.0, 90.0]);
    /// ```
    pub fn process_bbox(&self, bbox: &[f32; 4]) -> [f32; 4] {
        [
            bbox[0] * self.scale,
            bbox[1] * self.scale,
            bbox[2] * self.scale,
            bbox[3] * self.scale,
        ]
    }

    /// Processes multiple bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `bboxes` - Vector of bounding boxes in normalized coordinates.
    ///
    /// # Returns
    ///
    /// * `Vec<[f32; 4]>` - Vector of bounding boxes in pixel coordinates.
    pub fn process_bboxes(&self, bboxes: &[[f32; 4]]) -> Vec<[f32; 4]> {
        bboxes.iter().map(|bbox| self.process_bbox(bbox)).collect()
    }

    /// Processes a polygon from normalized to pixel coordinates.
    ///
    /// # Arguments
    ///
    /// * `polygon` - Vector of points as [x, y] pairs in normalized coordinates.
    ///
    /// # Returns
    ///
    /// * `Vec<[f32; 2]>` - Vector of points in pixel coordinates.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::doctr_postprocess::DocTrPostProcess;
    ///
    /// let postprocessor = DocTrPostProcess::new(100.0);
    /// let normalized_polygon = vec![[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]];
    /// let pixel_polygon = postprocessor.process_polygon(&normalized_polygon);
    /// assert_eq!(pixel_polygon[0], [10.0, 20.0]);
    /// ```
    pub fn process_polygon(&self, polygon: &[[f32; 2]]) -> Vec<[f32; 2]> {
        polygon
            .iter()
            .map(|&[x, y]| [x * self.scale, y * self.scale])
            .collect()
    }

    /// Clamps coordinates to valid ranges.
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of coordinates to clamp.
    /// * `min_val` - Minimum allowed value.
    /// * `max_val` - Maximum allowed value.
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` - Vector of clamped coordinates.
    pub fn clamp_coordinates(&self, coords: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
        coords
            .iter()
            .map(|&coord| coord.clamp(min_val, max_val))
            .collect()
    }

    /// Validates that coordinates are within expected ranges.
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of coordinates to validate.
    /// * `min_val` - Minimum expected value.
    /// * `max_val` - Maximum expected value.
    ///
    /// # Returns
    ///
    /// * `true` - If all coordinates are within range.
    /// * `false` - If any coordinate is out of range.
    pub fn validate_coordinates(&self, coords: &[f32], min_val: f32, max_val: f32) -> bool {
        coords
            .iter()
            .all(|&coord| coord >= min_val && coord <= max_val)
    }

    /// Rounds coordinates to integer values.
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of coordinates to round.
    ///
    /// # Returns
    ///
    /// * `Vec<i32>` - Vector of rounded integer coordinates.
    pub fn round_coordinates(&self, coords: &[f32]) -> Vec<i32> {
        coords.iter().map(|&coord| coord.round() as i32).collect()
    }

    /// Processes transformation matrix values.
    ///
    /// # Arguments
    ///
    /// * `matrix` - 3x3 transformation matrix as a flat vector.
    ///
    /// # Returns
    ///
    /// * `Vec<f32>` - Processed transformation matrix.
    pub fn process_transformation_matrix(&self, matrix: &[f32; 9]) -> [f32; 9] {
        // Apply scale to translation components (indices 2 and 5)
        let mut processed = *matrix;
        processed[2] *= self.scale; // tx
        processed[5] *= self.scale; // ty
        processed
    }

    /// Applies inverse transformation to coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - Vector of coordinates to transform.
    /// * `matrix` - 3x3 transformation matrix.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<[f32; 2]>, String>` - Transformed coordinates or error.
    pub fn apply_inverse_transform(
        &self,
        coords: &[[f32; 2]],
        matrix: &[f32; 9],
    ) -> Result<Vec<[f32; 2]>, String> {
        // Calculate determinant for matrix inversion
        let det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
            - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
            + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);

        if det.abs() < f32::EPSILON {
            return Err("Matrix is not invertible (determinant is zero)".to_string());
        }

        // For simplicity, this is a basic implementation
        // In practice, you might want to use a proper matrix library
        let mut transformed = Vec::new();
        for &[x, y] in coords {
            // Apply inverse transformation (simplified)
            let new_x = (x - matrix[2]) / matrix[0];
            let new_y = (y - matrix[5]) / matrix[4];
            transformed.push([new_x, new_y]);
        }

        Ok(transformed)
    }

    /// Applies batch processing to tensor output to produce rectified images.
    ///
    /// # Arguments
    ///
    /// * `output` - 4D tensor output from the model [batch, channels, height, width].
    ///
    /// # Returns
    ///
    /// * `Result<Vec<image::RgbImage>, String>` - Vector of rectified images or error.
    pub fn apply_batch(
        &self,
        output: &crate::core::Tensor4D,
    ) -> Result<Vec<image::RgbImage>, String> {
        use image::{Rgb, RgbImage};

        let shape = output.shape();
        if shape.len() != 4 {
            return Err("Expected 4D tensor [batch, channels, height, width]".to_string());
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        if channels != 3 {
            return Err("Expected 3 channels (RGB)".to_string());
        }

        let mut images = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut img = RgbImage::new(width as u32, height as u32);

            for y in 0..height {
                for x in 0..width {
                    // Extract RGB values and denormalize
                    let r = (output[[b, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
                    let g = (output[[b, 1, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
                    let b_val = (output[[b, 2, y, x]] * 255.0).clamp(0.0, 255.0) as u8;

                    img.put_pixel(x as u32, y as u32, Rgb([r, g, b_val]));
                }
            }

            images.push(img);
        }

        Ok(images)
    }
}

impl Default for DocTrPostProcess {
    /// Creates a default DocTrPostProcess with scale factor 1.0.
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl FromStr for DocTrPostProcess {
    type Err = std::num::ParseFloatError;

    /// Creates a DocTrPostProcess from a string representation of the scale factor.
    ///
    /// # Arguments
    ///
    /// * `s` - String representation of the scale factor.
    ///
    /// # Returns
    ///
    /// * `Ok(DocTrPostProcess)` - If the string can be parsed as a float.
    /// * `Err(ParseFloatError)` - If the string cannot be parsed.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let scale = s.parse::<f32>()?;
        Ok(Self::new(scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denormalize_coordinates() {
        let postprocessor = DocTrPostProcess::new(100.0);
        let normalized = vec![0.1, 0.2, 0.8, 0.9];
        let pixels = postprocessor.denormalize_coordinates(&normalized);
        assert_eq!(pixels, vec![10.0, 20.0, 80.0, 90.0]);
    }

    #[test]
    fn test_normalize_coordinates() {
        let postprocessor = DocTrPostProcess::new(100.0);
        let pixels = vec![10.0, 20.0, 80.0, 90.0];
        let normalized = postprocessor.normalize_coordinates(&pixels);
        assert_eq!(normalized, vec![0.1, 0.2, 0.8, 0.9]);
    }

    #[test]
    fn test_process_bbox() {
        let postprocessor = DocTrPostProcess::new(100.0);
        let normalized_bbox = [0.1, 0.2, 0.8, 0.9];
        let pixel_bbox = postprocessor.process_bbox(&normalized_bbox);
        assert_eq!(pixel_bbox, [10.0, 20.0, 80.0, 90.0]);
    }

    #[test]
    fn test_process_polygon() {
        let postprocessor = DocTrPostProcess::new(100.0);
        let normalized_polygon = vec![[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]];
        let pixel_polygon = postprocessor.process_polygon(&normalized_polygon);
        assert_eq!(pixel_polygon[0], [10.0, 20.0]);
        assert_eq!(pixel_polygon[1], [80.0, 20.0]);
    }

    #[test]
    fn test_clamp_coordinates() {
        let postprocessor = DocTrPostProcess::new(1.0);
        let coords = vec![-10.0, 50.0, 150.0];
        let clamped = postprocessor.clamp_coordinates(&coords, 0.0, 100.0);
        assert_eq!(clamped, vec![0.0, 50.0, 100.0]);
    }

    #[test]
    fn test_validate_coordinates() {
        let postprocessor = DocTrPostProcess::new(1.0);
        let valid_coords = vec![10.0, 50.0, 90.0];
        let invalid_coords = vec![10.0, 150.0, 90.0];

        assert!(postprocessor.validate_coordinates(&valid_coords, 0.0, 100.0));
        assert!(!postprocessor.validate_coordinates(&invalid_coords, 0.0, 100.0));
    }

    #[test]
    fn test_round_coordinates() {
        let postprocessor = DocTrPostProcess::new(1.0);
        let coords = vec![10.3, 20.7, 30.5];
        let rounded = postprocessor.round_coordinates(&coords);
        assert_eq!(rounded, vec![10, 21, 31]);
    }

    #[test]
    fn test_from_str() {
        let postprocessor: DocTrPostProcess = "2.5".parse().unwrap();
        assert_eq!(postprocessor.scale(), 2.5);

        assert!("invalid".parse::<DocTrPostProcess>().is_err());
    }

    #[test]
    fn test_zero_scale_normalize() {
        let postprocessor = DocTrPostProcess::new(0.0);
        let pixels = vec![10.0, 20.0];
        let normalized = postprocessor.normalize_coordinates(&pixels);
        assert_eq!(normalized, vec![0.0, 0.0]);
    }
}
