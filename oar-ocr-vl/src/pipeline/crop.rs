//! Cropping a page image down to a detected region.

use crate::api::error::{Error, ProcessingStage};
use crate::document::geometry::BoundingBox;
use image::RgbImage;

/// Rounding applied to page-pixel crop bounds.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub enum CropRounding {
    /// Truncate bounds toward zero.
    #[default]
    Truncate,
    /// Round bounds to the nearest integer, with ties to even.
    Nearest,
    /// Round lower bounds down and upper bounds up.
    Outward,
}

/// Geometry used when cropping a recognized region.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct CropOptions {
    /// Coordinate rounding rule.
    pub rounding: CropRounding,
    /// Keep at least one pixel along each axis after clipping.
    pub minimum_one_pixel: bool,
    /// Counterclockwise rotation in degrees.
    pub angle: Option<u16>,
    /// Geometry used when a detector provides a polygon.
    pub shape_mode: super::layout::LayoutShapeMode,
    /// Quantize crop bounds through this page-relative coordinate grid.
    pub coordinate_grid: Option<u32>,
}

impl CropOptions {
    /// Set the page-relative grid used before pixel rounding.
    pub fn with_coordinate_grid(mut self, grid: Option<u32>) -> Self {
        self.coordinate_grid = grid.filter(|&grid| grid > 0);
        self
    }

    /// Set the coordinate rounding.
    pub fn with_rounding(mut self, rounding: CropRounding) -> Self {
        self.rounding = rounding;
        self
    }

    /// Set the minimum one-pixel crop behavior.
    pub fn with_minimum_one_pixel(mut self, enabled: bool) -> Self {
        self.minimum_one_pixel = enabled;
        self
    }

    /// Set the counterclockwise rotation in degrees.
    pub fn with_angle(mut self, angle: Option<u16>) -> Self {
        self.angle = angle;
        self
    }

    /// Set the geometry used for detector polygons.
    pub fn with_shape_mode(mut self, mode: super::layout::LayoutShapeMode) -> Self {
        self.shape_mode = mode;
        self
    }
}

pub(crate) fn crop_with_options(
    image: &RgbImage,
    bounds: [f64; 4],
    options: &CropOptions,
) -> Result<RgbImage, Error> {
    crop_with_polygon(image, bounds, None, options)
}

pub(crate) fn crop_with_polygon(
    image: &RgbImage,
    bounds: [f64; 4],
    polygon: Option<&BoundingBox>,
    options: &CropOptions,
) -> Result<RgbImage, Error> {
    crop_with_polygon_and_paint(image, bounds, polygon, options, |_, _| {})
}

pub(crate) fn crop_with_polygon_and_paint(
    image: &RgbImage,
    bounds: [f64; 4],
    polygon: Option<&BoundingBox>,
    options: &CropOptions,
    paint: impl FnOnce(&mut RgbImage, [f64; 2]),
) -> Result<RgbImage, Error> {
    if image.width() == 0 || image.height() == 0 || bounds.iter().any(|v| !v.is_finite()) {
        return Err(crop_error(
            "crop requires a nonempty image and finite bounds".into(),
        ));
    }
    let bounds = if let Some(grid) = options.coordinate_grid.filter(|&grid| grid > 0) {
        std::array::from_fn(|i| {
            let size = if i % 2 == 0 {
                f64::from(image.width())
            } else {
                f64::from(image.height())
            };
            (bounds[i] / size * f64::from(grid)).trunc() * size / f64::from(grid)
        })
    } else {
        bounds
    };
    let rounded = std::array::from_fn::<_, 4, _>(|i| match options.rounding {
        CropRounding::Truncate => bounds[i].trunc(),
        CropRounding::Nearest => bounds[i].round_ties_even(),
        CropRounding::Outward if i < 2 => bounds[i].floor(),
        CropRounding::Outward => bounds[i].ceil(),
    });
    let x1 = rounded[0].clamp(0., (image.width() - 1) as f64) as u32;
    let y1 = rounded[1].clamp(0., (image.height() - 1) as f64) as u32;
    let minimum = u32::from(options.minimum_one_pixel);
    let x2 = rounded[2].clamp((x1 + minimum) as f64, image.width() as f64) as u32;
    let y2 = rounded[3].clamp((y1 + minimum) as f64, image.height() as f64) as u32;
    if x2 <= x1 || y2 <= y1 {
        return Err(crop_error(format!(
            "invalid crop region: ({x1}, {y1}) to ({x2}, {y2})"
        )));
    }
    let mut crop = image::imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
    if options.shape_mode != super::layout::LayoutShapeMode::Rect
        && let Some(polygon) = polygon
    {
        super::geometry::polygon::mask_crop(&mut crop, polygon, (x1, y1));
    }
    paint(&mut crop, [bounds[0], bounds[1]]);
    Ok(match options.angle {
        Some(90) => image::imageops::rotate270(&crop),
        Some(180) => image::imageops::rotate180(&crop),
        Some(270) => image::imageops::rotate90(&crop),
        _ => crop,
    })
}

/// Crops the axis-aligned extent of `bbox` out of `image`, clamped to the image
/// bounds.
pub fn crop_bounding_box(image: &RgbImage, bbox: &BoundingBox) -> Result<RgbImage, Error> {
    if bbox.points.is_empty() {
        return Err(crop_error("empty bounding box".to_string()));
    }

    crop_with_options(
        image,
        [bbox.x_min(), bbox.y_min(), bbox.x_max(), bbox.y_max()].map(f64::from),
        &CropOptions::default(),
    )
}

fn crop_error(context: String) -> Error {
    Error::Processing {
        kind: ProcessingStage::ImageProcessing,
        context,
        source: Box::new(Error::invalid_input("bounding box crop")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crops_the_requested_region() {
        let image = RgbImage::new(100, 80);
        let crop = crop_bounding_box(&image, &BoundingBox::from_coords(10.0, 20.0, 40.0, 60.0))
            .expect("crop succeeds");
        assert_eq!(crop.dimensions(), (30, 40));
        let page = RgbImage::from_fn(1003, 601, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 251) as u8, 0])
        });
        let quantized = crop_with_options(
            &page,
            [10., 10., 40., 60.],
            &CropOptions::default().with_coordinate_grid(Some(1000)),
        )
        .unwrap();
        assert_eq!(quantized.dimensions(), (30, 50));
        assert_eq!(quantized.get_pixel(0, 0).0, [9, 9, 0]);
        let polygon = BoundingBox::new(vec![
            crate::document::geometry::Point::new(0., 0.),
            crate::document::geometry::Point::new(4., 0.),
            crate::document::geometry::Point::new(0., 4.),
        ]);
        let painted = crop_with_polygon_and_paint(
            &image,
            [0., 0., 4., 4.],
            Some(&polygon),
            &CropOptions::default().with_shape_mode(super::super::layout::LayoutShapeMode::Poly),
            |crop, _| crop.put_pixel(3, 3, image::Rgb([0; 3])),
        )
        .unwrap();
        assert_eq!(painted.get_pixel(3, 3).0, [0; 3]);
    }

    /// Boxes reaching past the edges are clamped rather than rejected.
    #[test]
    fn clamps_to_the_image_bounds() {
        let image = RgbImage::new(50, 50);
        let crop = crop_bounding_box(
            &image,
            &BoundingBox::from_coords(-10.0, -10.0, 999.0, 999.0),
        )
        .expect("crop succeeds");
        assert_eq!(crop.dimensions(), (50, 50));
    }

    #[test]
    fn rejects_degenerate_boxes() {
        let image = RgbImage::new(50, 50);
        assert!(crop_bounding_box(&image, &BoundingBox::new(Vec::new())).is_err());
        assert!(
            crop_bounding_box(&image, &BoundingBox::from_coords(10.0, 10.0, 10.0, 10.0)).is_err()
        );
    }
}
