//! Geometric utilities for OCR processing.
//!
//! This module provides geometric primitives and algorithms commonly used in OCR systems,
//! such as point representations, bounding boxes, and algorithms for calculating areas,
//! perimeters, convex hulls, and minimum area rectangles.

use imageproc::contours::Contour;
use imageproc::point::Point as ImageProcPoint;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use std::f32::consts::PI;

/// A 2D point with floating-point coordinates.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    /// X-coordinate of the point.
    pub x: f32,
    /// Y-coordinate of the point.
    pub y: f32,
}

impl Point {
    /// Creates a new point with the given coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the point.
    /// * `y` - The y-coordinate of the point.
    ///
    /// # Returns
    ///
    /// A new `Point` instance.
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Creates a point from an imageproc point with integer coordinates.
    ///
    /// # Arguments
    ///
    /// * `p` - An imageproc point with integer coordinates.
    ///
    /// # Returns
    ///
    /// A new `Point` instance with floating-point coordinates.
    pub fn from_imageproc_point(p: ImageProcPoint<i32>) -> Self {
        Self {
            x: p.x as f32,
            y: p.y as f32,
        }
    }

    /// Converts this point to an imageproc point with integer coordinates.
    ///
    /// # Returns
    ///
    /// An imageproc point with coordinates rounded down to integers.
    pub fn to_imageproc_point(&self) -> ImageProcPoint<i32> {
        ImageProcPoint::new(self.x as i32, self.y as i32)
    }
}

/// A bounding box represented by a collection of points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// The points that define the bounding box.
    pub points: Vec<Point>,
}

impl BoundingBox {
    /// Creates a new bounding box from a vector of points.
    ///
    /// # Arguments
    ///
    /// * `points` - A vector of points that define the bounding box.
    ///
    /// # Returns
    ///
    /// A new `BoundingBox` instance.
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }

    /// Creates a bounding box from coordinates.
    ///
    /// # Arguments
    ///
    /// * `x1` - The x-coordinate of the top-left corner.
    /// * `y1` - The y-coordinate of the top-left corner.
    /// * `x2` - The x-coordinate of the bottom-right corner.
    /// * `y2` - The y-coordinate of the bottom-right corner.
    ///
    /// # Returns
    ///
    /// A new `BoundingBox` instance representing a rectangle.
    pub fn from_coords(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let points = vec![
            Point::new(x1, y1),
            Point::new(x2, y1),
            Point::new(x2, y2),
            Point::new(x1, y2),
        ];
        Self { points }
    }

    /// Creates a bounding box from a contour.
    ///
    /// # Arguments
    ///
    /// * `contour` - A reference to a contour from imageproc.
    ///
    /// # Returns
    ///
    /// A new `BoundingBox` instance with points converted from the contour.
    pub fn from_contour(contour: &Contour<u32>) -> Self {
        let points = contour
            .points
            .iter()
            .map(|p| Point::new(p.x as f32, p.y as f32))
            .collect();
        Self { points }
    }

    /// Calculates the area of the bounding box using the shoelace formula.
    ///
    /// # Returns
    ///
    /// The area of the bounding box. Returns 0.0 if the bounding box has fewer than 3 points.
    pub fn area(&self) -> f32 {
        if self.points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.points[i].x * self.points[j].y;
            area -= self.points[j].x * self.points[i].y;
        }
        area.abs() / 2.0
    }

    /// Calculates the perimeter of the bounding box.
    ///
    /// # Returns
    ///
    /// The perimeter of the bounding box.
    pub fn perimeter(&self) -> f32 {
        let mut perimeter = 0.0;
        let n = self.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.points[j].x - self.points[i].x;
            let dy = self.points[j].y - self.points[i].y;
            perimeter += (dx * dx + dy * dy).sqrt();
        }
        perimeter
    }

    /// Gets the minimum x-coordinate of all points in the bounding box.
    ///
    /// # Returns
    ///
    /// The minimum x-coordinate, or 0.0 if there are no points.
    pub fn x_min(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points
            .iter()
            .map(|p| p.x)
            .fold(f32::INFINITY, f32::min)
    }

    /// Gets the minimum y-coordinate of all points in the bounding box.
    ///
    /// # Returns
    ///
    /// The minimum y-coordinate, or 0.0 if there are no points.
    pub fn y_min(&self) -> f32 {
        if self.points.is_empty() {
            return 0.0;
        }
        self.points
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min)
    }

    /// Computes the convex hull of the bounding box using Graham's scan algorithm.
    ///
    /// # Returns
    ///
    /// A new `BoundingBox` representing the convex hull. If the bounding box has fewer than 3 points,
    /// returns a clone of the original bounding box.
    fn convex_hull(&self) -> BoundingBox {
        if self.points.len() < 3 {
            return self.clone();
        }

        let mut points = self.points.clone();

        // Find the point with the lowest y-coordinate (and leftmost if tied)
        let mut start_idx = 0;
        for i in 1..points.len() {
            if points[i].y < points[start_idx].y
                || (points[i].y == points[start_idx].y && points[i].x < points[start_idx].x)
            {
                start_idx = i;
            }
        }
        points.swap(0, start_idx);
        let start_point = points[0];

        // Sort points by polar angle with respect to the start point
        points[1..].sort_by(|a, b| {
            let cross = Self::cross_product(&start_point, a, b);
            if cross == 0.0 {
                // If points are collinear, sort by distance from start point
                let dist_a = (a.x - start_point.x).powi(2) + (a.y - start_point.y).powi(2);
                let dist_b = (b.x - start_point.x).powi(2) + (b.y - start_point.y).powi(2);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else if cross > 0.0 {
                // Counter-clockwise turn
                std::cmp::Ordering::Less
            } else {
                // Clockwise turn
                std::cmp::Ordering::Greater
            }
        });

        // Build the convex hull using a stack
        let mut hull = Vec::new();
        for point in points {
            // Remove points that make clockwise turns
            while hull.len() > 1
                && Self::cross_product(&hull[hull.len() - 2], &hull[hull.len() - 1], &point) <= 0.0
            {
                hull.pop();
            }
            hull.push(point);
        }

        BoundingBox::new(hull)
    }

    /// Computes the cross product of three points.
    ///
    /// # Arguments
    ///
    /// * `p1` - The first point.
    /// * `p2` - The second point.
    /// * `p3` - The third point.
    ///
    /// # Returns
    ///
    /// The cross product value. A positive value indicates a counter-clockwise turn,
    /// a negative value indicates a clockwise turn, and zero indicates collinearity.
    fn cross_product(p1: &Point, p2: &Point, p3: &Point) -> f32 {
        (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    }

    /// Computes the minimum area rectangle that encloses the bounding box.
    ///
    /// This method uses the rotating calipers algorithm on the convex hull of the bounding box
    /// to find the minimum area rectangle.
    ///
    /// # Returns
    ///
    /// A `MinAreaRect` representing the minimum area rectangle. If the bounding box has fewer than
    /// 3 points, returns a rectangle with zero dimensions.
    pub fn get_min_area_rect(&self) -> MinAreaRect {
        if self.points.len() < 3 {
            return MinAreaRect {
                center: Point::new(0.0, 0.0),
                width: 0.0,
                height: 0.0,
                angle: 0.0,
            };
        }

        // Get the convex hull of the bounding box
        let hull = self.convex_hull();
        let hull_points = &hull.points;

        // Handle degenerate cases
        if hull_points.len() < 3 {
            let (min_x, max_x) = match self.points.iter().map(|p| p.x).minmax().into_option() {
                Some((min, max)) => (min, max),
                None => {
                    return MinAreaRect {
                        center: Point::new(0.0, 0.0),
                        width: 0.0,
                        height: 0.0,
                        angle: 0.0,
                    };
                }
            };

            let (min_y, max_y) = match self.points.iter().map(|p| p.y).minmax().into_option() {
                Some((min, max)) => (min, max),
                None => {
                    return MinAreaRect {
                        center: Point::new(0.0, 0.0),
                        width: 0.0,
                        height: 0.0,
                        angle: 0.0,
                    };
                }
            };

            let center = Point::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
            let width = max_x - min_x;
            let height = max_y - min_y;

            return MinAreaRect {
                center,
                width,
                height,
                angle: 0.0,
            };
        }

        // Find the minimum area rectangle using rotating calipers
        let mut min_area = f32::MAX;
        let mut min_rect = MinAreaRect {
            center: Point::new(0.0, 0.0),
            width: 0.0,
            height: 0.0,
            angle: 0.0,
        };

        let n = hull_points.len();
        for i in 0..n {
            let j = (i + 1) % n;

            // Calculate the edge vector
            let edge_x = hull_points[j].x - hull_points[i].x;
            let edge_y = hull_points[j].y - hull_points[i].y;
            let edge_length = (edge_x * edge_x + edge_y * edge_y).sqrt();

            // Skip degenerate edges
            if edge_length < f32::EPSILON {
                continue;
            }

            // Normalize the edge vector
            let nx = edge_x / edge_length;
            let ny = edge_y / edge_length;

            // Calculate the perpendicular vector
            let px = -ny;
            let py = nx;

            // Project all points onto the edge and perpendicular vectors
            let mut min_n = f32::MAX;
            let mut max_n = f32::MIN;
            let mut min_p = f32::MAX;
            let mut max_p = f32::MIN;

            for k in 0..n {
                let point = &hull_points[k];

                let proj_n = nx * (point.x - hull_points[i].x) + ny * (point.y - hull_points[i].y);
                min_n = min_n.min(proj_n);
                max_n = max_n.max(proj_n);

                let proj_p = px * (point.x - hull_points[i].x) + py * (point.y - hull_points[i].y);
                min_p = min_p.min(proj_p);
                max_p = max_p.max(proj_p);
            }

            // Calculate the width, height, and area of the rectangle
            let width = max_n - min_n;
            let height = max_p - min_p;
            let area = width * height;

            // Update the minimum area rectangle if this one is smaller
            if area < min_area {
                min_area = area;

                let center_n = (min_n + max_n) / 2.0;
                let center_p = (min_p + max_p) / 2.0;

                let center_x = hull_points[i].x + center_n * nx + center_p * px;
                let center_y = hull_points[i].y + center_n * ny + center_p * py;

                let angle_rad = f32::atan2(ny, nx);
                let angle_deg = angle_rad * 180.0 / PI;

                min_rect = MinAreaRect {
                    center: Point::new(center_x, center_y),
                    width,
                    height,
                    angle: angle_deg,
                };
            }
        }

        min_rect
    }

    /// Approximates a polygon using the Douglas-Peucker algorithm.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The maximum distance between the original curve and the simplified curve.
    ///
    /// # Returns
    ///
    /// A new `BoundingBox` with simplified points. If the bounding box has 2 or fewer points,
    /// returns a clone of the original bounding box.
    pub fn approx_poly_dp(&self, epsilon: f32) -> BoundingBox {
        if self.points.len() <= 2 {
            return self.clone();
        }

        let mut simplified = Vec::new();
        self.douglas_peucker(&self.points, epsilon, &mut simplified);

        BoundingBox::new(simplified)
    }

    /// Implements the Douglas-Peucker algorithm for curve simplification.
    ///
    /// # Arguments
    ///
    /// * `points` - The points to simplify.
    /// * `epsilon` - The maximum distance between the original curve and the simplified curve.
    /// * `result` - A mutable reference to a vector where the simplified points will be stored.
    fn douglas_peucker(&self, points: &[Point], epsilon: f32, result: &mut Vec<Point>) {
        if points.len() <= 2 {
            result.extend_from_slice(points);
            return;
        }

        // Initialize a stack for iterative implementation
        let mut stack = Vec::new();
        stack.push((0, points.len() - 1));

        // Track which points to keep
        let mut keep = vec![false; points.len()];
        keep[0] = true;
        keep[points.len() - 1] = true;

        // Process the stack
        const MAX_ITERATIONS: usize = 10000;
        let mut iterations = 0;

        while let Some((start, end)) = stack.pop() {
            iterations += 1;
            // Prevent infinite loops
            if iterations > MAX_ITERATIONS {
                keep.iter_mut()
                    .take(end + 1)
                    .skip(start)
                    .for_each(|k| *k = true);
                break;
            }

            // Skip segments with only 2 points
            if end - start <= 1 {
                continue;
            }

            // Find the point with maximum distance from the line segment
            let mut max_dist = 0.0;
            let mut max_index = start;

            for i in (start + 1)..end {
                let dist = self.point_to_line_distance(&points[i], &points[start], &points[end]);
                if dist > max_dist {
                    max_dist = dist;
                    max_index = i;
                }
            }

            // If the maximum distance exceeds epsilon, split the segment
            if max_dist > epsilon {
                keep[max_index] = true;

                if max_index - start > 1 {
                    stack.push((start, max_index));
                }
                if end - max_index > 1 {
                    stack.push((max_index, end));
                }
            }
        }

        // Collect the points to keep
        for (i, &should_keep) in keep.iter().enumerate() {
            if should_keep {
                result.push(points[i]);
            }
        }
    }

    /// Calculates the perpendicular distance from a point to a line segment.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to calculate the distance for.
    /// * `line_start` - The start point of the line segment.
    /// * `line_end` - The end point of the line segment.
    ///
    /// # Returns
    ///
    /// The perpendicular distance from the point to the line segment.
    fn point_to_line_distance(&self, point: &Point, line_start: &Point, line_end: &Point) -> f32 {
        let a = line_end.y - line_start.y;
        let b = line_start.x - line_end.x;
        let c = line_end.x * line_start.y - line_start.x * line_end.y;

        let denominator = (a * a + b * b).sqrt();
        if denominator == 0.0 {
            return 0.0;
        }

        (a * point.x + b * point.y + c).abs() / denominator
    }
}

/// A rectangle with minimum area that encloses a shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinAreaRect {
    /// The center point of the rectangle.
    pub center: Point,
    /// The width of the rectangle.
    pub width: f32,
    /// The height of the rectangle.
    pub height: f32,
    /// The rotation angle of the rectangle in degrees.
    pub angle: f32,
}

impl MinAreaRect {
    /// Gets the four corner points of the rectangle.
    ///
    /// # Returns
    ///
    /// A vector containing the four corner points of the rectangle ordered as:
    /// top-left, top-right, bottom-right, bottom-left in the final image coordinate system.
    pub fn get_box_points(&self) -> Vec<Point> {
        let cos_a = (self.angle * PI / 180.0).cos();
        let sin_a = (self.angle * PI / 180.0).sin();

        let w_2 = self.width / 2.0;
        let h_2 = self.height / 2.0;

        let corners = [(-w_2, -h_2), (w_2, -h_2), (w_2, h_2), (-w_2, h_2)];

        let mut points: Vec<Point> = corners
            .iter()
            .map(|(x, y)| {
                let rotated_x = x * cos_a - y * sin_a + self.center.x;
                let rotated_y = x * sin_a + y * cos_a + self.center.y;
                Point::new(rotated_x, rotated_y)
            })
            .collect();

        // Sort points to ensure consistent ordering: top-left, top-right, bottom-right, bottom-left
        Self::sort_box_points(&mut points);
        points
    }

    /// Sorts four points to ensure consistent ordering for OCR bounding boxes.
    ///
    /// Orders points as: top-left, top-right, bottom-right, bottom-left
    /// based on their actual coordinates in the image space.
    ///
    /// This algorithm works by:
    /// 1. Finding the centroid of the four points
    /// 2. Classifying each point based on its position relative to the centroid
    /// 3. Assigning points to corners based on their quadrant
    ///
    /// # Arguments
    ///
    /// * `points` - A mutable reference to a vector of exactly 4 points
    fn sort_box_points(points: &mut [Point]) {
        if points.len() != 4 {
            return;
        }

        // Calculate the centroid of the four points
        let center_x = points.iter().map(|p| p.x).sum::<f32>() / 4.0;
        let center_y = points.iter().map(|p| p.y).sum::<f32>() / 4.0;

        // Create a vector to store points with their classifications
        let mut classified_points = Vec::with_capacity(4);

        for point in points.iter() {
            let is_left = point.x < center_x;
            let is_top = point.y < center_y;

            let corner_type = match (is_left, is_top) {
                (true, true) => 0,   // top-left
                (false, true) => 1,  // top-right
                (false, false) => 2, // bottom-right
                (true, false) => 3,  // bottom-left
            };

            classified_points.push((corner_type, *point));
        }

        // Sort by corner type to get the desired order
        classified_points.sort_by_key(|&(corner_type, _)| corner_type);

        // Handle the case where multiple points might be classified as the same corner
        // This can happen with very thin or rotated rectangles
        let mut corner_types = HashSet::new();
        for (corner_type, _) in &classified_points {
            corner_types.insert(*corner_type);
        }

        if corner_types.len() < 4 {
            // Fallback to a more robust method using angles from centroid
            Self::sort_box_points_by_angle(points, center_x, center_y);
        } else {
            // Update the original points vector with the sorted points
            for (i, (_, point)) in classified_points.iter().enumerate() {
                points[i] = *point;
            }
        }
    }

    /// Fallback sorting method using polar angles from the centroid.
    ///
    /// # Arguments
    ///
    /// * `points` - A mutable reference to a vector of exactly 4 points
    /// * `center_x` - X coordinate of the centroid
    /// * `center_y` - Y coordinate of the centroid
    fn sort_box_points_by_angle(points: &mut [Point], center_x: f32, center_y: f32) {
        // Calculate angle from centroid to each point
        let mut points_with_angles: Vec<(f32, Point)> = points
            .iter()
            .map(|p| {
                let angle = f32::atan2(p.y - center_y, p.x - center_x);
                // Normalize angle to [0, 2π) and adjust so that top-left is first
                let normalized_angle = if angle < -PI / 2.0 {
                    angle + 2.0 * PI
                } else {
                    angle
                };
                (normalized_angle, *p)
            })
            .collect();

        // Sort by angle (starting from top-left, going clockwise)
        points_with_angles
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find the starting point (closest to top-left quadrant)
        let mut start_idx = 0;
        let mut min_top_left_score = f32::MAX;

        for (i, (_, point)) in points_with_angles.iter().enumerate() {
            // Score based on distance from theoretical top-left position
            let top_left_score =
                (point.x - center_x + 100.0).powi(2) + (point.y - center_y + 100.0).powi(2);
            if top_left_score < min_top_left_score {
                min_top_left_score = top_left_score;
                start_idx = i;
            }
        }

        // Reorder starting from the identified top-left point
        for (i, point) in points.iter_mut().enumerate().take(4) {
            let src_idx = (start_idx + i) % 4;
            *point = points_with_angles[src_idx].1;
        }
    }

    /// Gets the length of the shorter side of the rectangle.
    ///
    /// # Returns
    ///
    /// The length of the shorter side.
    pub fn min_side(&self) -> f32 {
        self.width.min(self.height)
    }
}

/// A buffer for processing scanlines in polygon rasterization.
pub(crate) struct ScanlineBuffer {
    /// Intersections of the scanline with polygon edges.
    pub(crate) intersections: Vec<f32>,
}

impl ScanlineBuffer {
    /// Creates a new scanline buffer with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `max_polygon_points` - The maximum number of polygon points, used to pre-allocate memory.
    ///
    /// # Returns
    ///
    /// A new `ScanlineBuffer` instance.
    pub(crate) fn new(max_polygon_points: usize) -> Self {
        Self {
            intersections: Vec::with_capacity(max_polygon_points),
        }
    }

    /// Processes a scanline by finding intersections with polygon edges and accumulating scores.
    ///
    /// # Arguments
    ///
    /// * `y` - The y-coordinate of the scanline.
    /// * `bbox` - The bounding box representing the polygon.
    /// * `start_x` - The starting x-coordinate for processing.
    /// * `end_x` - The ending x-coordinate for processing.
    /// * `pred` - A 2D array of prediction scores.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * The accumulated line score
    /// * The number of pixels processed
    pub(crate) fn process_scanline(
        &mut self,
        y: f32,
        bbox: &BoundingBox,
        start_x: usize,
        end_x: usize,
        pred: &ndarray::Array2<f32>,
    ) -> (f32, usize) {
        // Clear previous intersections
        self.intersections.clear();

        // Find intersections of the scanline with polygon edges
        let n = bbox.points.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let p1 = &bbox.points[i];
            let p2 = &bbox.points[j];

            // Check if the edge crosses the scanline
            if ((p1.y <= y && y < p2.y) || (p2.y <= y && y < p1.y))
                && (p2.y - p1.y).abs() > f32::EPSILON
            {
                let x = p1.x + (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y);
                self.intersections.push(x);
            }
        }

        // Sort intersections by x-coordinate
        self.intersections
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut line_score = 0.0;
        let mut line_pixels = 0;

        // Process pairs of intersections (segments of the scanline inside the polygon)
        for chunk in self.intersections.chunks(2) {
            if chunk.len() == 2 {
                let x1 = chunk[0].max(start_x as f32) as usize;
                let x2 = chunk[1].min(end_x as f32) as usize;

                // Accumulate scores for pixels within the segment
                if x1 < x2 && x1 >= start_x && x2 <= end_x {
                    for x in x1..x2 {
                        if (y as usize) < pred.shape()[0] && x < pred.shape()[1] {
                            line_score += pred[[y as usize, x]];
                            line_pixels += 1;
                        }
                    }
                }
            }
        }

        (line_score, line_pixels)
    }
}
