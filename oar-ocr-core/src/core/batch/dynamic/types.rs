//! Data structures for dynamic batching

use image::RgbImage;
use std::collections::HashMap;
use std::time::Duration;

/// Speed metrics for batch operations
#[derive(Debug, Clone, Default)]
pub struct BatchPerformanceMetrics {
    /// Total batching time
    pub batching_time: Duration,
    /// Number of batches created
    pub batch_count: usize,
    /// Total number of images processed
    pub total_images: usize,
}

impl BatchPerformanceMetrics {
    /// Creates new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates metrics with new batch information
    pub fn update(&mut self, batch_size: usize, processing_time: Duration) {
        self.batch_count += 1;
        self.total_images += batch_size;
        self.batching_time += processing_time;
    }

    /// Gets the average processing time per batch
    pub fn average_batch_time(&self) -> Duration {
        if self.batch_count > 0 {
            self.batching_time / self.batch_count as u32
        } else {
            Duration::ZERO
        }
    }

    /// Gets the average processing time per image
    pub fn average_image_time(&self) -> Duration {
        if self.total_images > 0 {
            self.batching_time / self.total_images as u32
        } else {
            Duration::ZERO
        }
    }
}

/// A batch of compatible images that can be processed together
#[derive(Debug, Clone)]
pub struct CompatibleBatch {
    /// Images in this batch
    pub images: Vec<RgbImage>,
    /// Original indices of the images
    pub indices: Vec<usize>,
    /// Target dimensions for this batch (height, width)
    pub target_dimensions: (u32, u32),
    /// Batch identifier/name
    pub batch_id: String,
    /// Metadata for this batch
    pub metadata: HashMap<String, String>,
}

impl CompatibleBatch {
    /// Creates a new compatible batch
    pub fn new(batch_id: String, target_dimensions: (u32, u32)) -> Self {
        Self {
            images: Vec::new(),
            indices: Vec::new(),
            target_dimensions,
            batch_id,
            metadata: HashMap::new(),
        }
    }

    /// Adds an image to this batch
    pub fn add_image(&mut self, image: RgbImage, index: usize) {
        self.images.push(image);
        self.indices.push(index);
    }

    /// Gets the batch size
    pub fn size(&self) -> usize {
        self.images.len()
    }

    /// Checks if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Adds metadata to this batch
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

/// Result of dynamic batch processing
#[derive(Debug)]
pub struct DynamicBatchResult<T> {
    /// Batched results
    pub batches: Vec<T>,
    /// Mapping from original index to batch and position within batch
    pub index_mapping: HashMap<usize, (usize, usize)>,
    /// Speed metrics
    pub metrics: BatchPerformanceMetrics,
}

impl<T> DynamicBatchResult<T> {
    /// Creates a new batch result
    pub fn new() -> Self {
        Self {
            batches: Vec::new(),
            index_mapping: HashMap::new(),
            metrics: BatchPerformanceMetrics::new(),
        }
    }

    /// Adds a batch result
    pub fn add_batch(&mut self, batch: T, original_indices: Vec<usize>) {
        let batch_index = self.batches.len();
        self.batches.push(batch);

        for (position, original_index) in original_indices.into_iter().enumerate() {
            self.index_mapping
                .insert(original_index, (batch_index, position));
        }
    }

    /// Gets the result for a specific original index
    pub fn get_result_location(&self, original_index: usize) -> Option<(usize, usize)> {
        self.index_mapping.get(&original_index).copied()
    }

    /// Gets the number of batches
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Gets the total number of items processed
    pub fn total_items(&self) -> usize {
        self.index_mapping.len()
    }
}

impl<T> Default for DynamicBatchResult<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross-image item for batching (e.g., text regions from multiple images)
#[derive(Debug, Clone)]
pub struct CrossImageItem {
    /// Source image index
    pub source_image_index: usize,
    /// Item index within the source image
    pub item_index: usize,
    /// The actual image data
    pub image: RgbImage,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl CrossImageItem {
    /// Creates a new cross-image item
    pub fn new(source_image_index: usize, item_index: usize, image: RgbImage) -> Self {
        Self {
            source_image_index,
            item_index,
            image,
            metadata: HashMap::new(),
        }
    }

    /// Adds metadata to this item
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Gets metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Gets the image dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }
}

/// Batch of cross-image items
#[derive(Debug, Clone)]
pub struct CrossImageBatch {
    /// Items in this batch
    pub items: Vec<CrossImageItem>,
    /// Target dimensions for this batch
    pub target_dimensions: (u32, u32),
    /// Batch identifier
    pub batch_id: String,
}

impl CrossImageBatch {
    /// Creates a new cross-image batch
    pub fn new(batch_id: String, target_dimensions: (u32, u32)) -> Self {
        Self {
            items: Vec::new(),
            target_dimensions,
            batch_id,
        }
    }

    /// Adds an item to this batch
    pub fn add_item(&mut self, item: CrossImageItem) {
        self.items.push(item);
    }

    /// Gets the batch size
    pub fn size(&self) -> usize {
        self.items.len()
    }

    /// Checks if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Gets all images in this batch
    pub fn get_images(&self) -> Vec<&RgbImage> {
        self.items.iter().map(|item| &item.image).collect()
    }
}
