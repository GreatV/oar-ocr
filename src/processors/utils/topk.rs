//! Top-k classification result processing.

use std::collections::HashMap;

/// Result structure for top-k classification processing.
///
/// Contains the top-k class indexes and their corresponding confidence scores
/// for each prediction in a batch.
#[derive(Debug, Clone)]
pub struct TopkResult {
    /// Vector of vectors containing the class indexes for each prediction.
    /// Each inner vector contains the top-k class indexes for one prediction.
    pub indexes: Vec<Vec<usize>>,
    /// Vector of vectors containing the confidence scores for each prediction.
    /// Each inner vector contains the top-k scores corresponding to the indexes.
    pub scores: Vec<Vec<f32>>,
    /// Optional vector of vectors containing class names for each prediction.
    /// Only populated if class name mapping is provided.
    pub class_names: Option<Vec<Vec<String>>>,
}

/// A processor for extracting top-k results from classification outputs.
///
/// The `Topk` struct processes classification model outputs to extract the
/// top-k most confident predictions along with their class names (if available).
#[derive(Debug)]
pub struct Topk {
    /// Optional mapping from class IDs to class names.
    class_id_map: Option<HashMap<usize, String>>,
}

impl Topk {
    /// Creates a new Topk processor with optional class name mapping.
    ///
    /// # Arguments
    ///
    /// * `class_id_map` - Optional mapping from class IDs to human-readable class names.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::collections::HashMap;
    /// use oar_ocr::processors::utils::topk::Topk;
    ///
    /// let mut class_map = HashMap::new();
    /// class_map.insert(0, "cat".to_string());
    /// class_map.insert(1, "dog".to_string());
    ///
    /// let topk = Topk::new(Some(class_map));
    /// ```
    pub fn new(class_id_map: Option<HashMap<usize, String>>) -> Self {
        Self { class_id_map }
    }

    /// Creates a new Topk processor without class name mapping.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::topk::Topk;
    ///
    /// let topk = Topk::without_class_names();
    /// ```
    pub fn without_class_names() -> Self {
        Self::new(None)
    }

    /// Creates a new Topk processor with class names from a vector.
    ///
    /// The vector index corresponds to the class ID.
    ///
    /// # Arguments
    ///
    /// * `class_names` - Vector of class names where index = class ID.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::topk::Topk;
    ///
    /// let class_names = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()];
    /// let topk = Topk::from_class_names(class_names);
    /// ```
    pub fn from_class_names(class_names: Vec<String>) -> Self {
        let class_id_map: HashMap<usize, String> = class_names.into_iter().enumerate().collect();
        Self::new(Some(class_id_map))
    }

    /// Processes classification outputs to extract top-k results.
    ///
    /// # Arguments
    ///
    /// * `predictions` - 2D vector where each inner vector contains the confidence
    ///   scores for all classes for one prediction.
    /// * `k` - Number of top predictions to extract (must be > 0).
    ///
    /// # Returns
    ///
    /// * `Ok(TopkResult)` - The top-k results with indexes, scores, and optional class names.
    /// * `Err(String)` - If k is 0 or if the input is invalid.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr::processors::utils::topk::Topk;
    ///
    /// let topk = Topk::without_class_names();
    /// let predictions = vec![
    ///     vec![0.1, 0.8, 0.1],  // Prediction 1: class 1 has highest score
    ///     vec![0.7, 0.2, 0.1],  // Prediction 2: class 0 has highest score
    /// ];
    /// let result = topk.process(&predictions, 2).unwrap();
    /// ```
    pub fn process(&self, predictions: &[Vec<f32>], k: usize) -> Result<TopkResult, String> {
        if k == 0 {
            return Err("k must be greater than 0".to_string());
        }

        if predictions.is_empty() {
            return Ok(TopkResult {
                indexes: vec![],
                scores: vec![],
                class_names: None,
            });
        }

        let mut all_indexes = Vec::new();
        let mut all_scores = Vec::new();
        let mut all_class_names = if self.class_id_map.is_some() {
            Some(Vec::new())
        } else {
            None
        };

        for prediction in predictions {
            if prediction.is_empty() {
                return Err("Empty prediction vector".to_string());
            }

            let effective_k = k.min(prediction.len());
            let (top_indexes, top_scores) =
                self.extract_topk_from_prediction(prediction, effective_k);

            all_indexes.push(top_indexes.clone());
            all_scores.push(top_scores);

            // Add class names if mapping is available
            if let Some(ref mut class_names_vec) = all_class_names {
                let names = self.map_indexes_to_names(&top_indexes);
                class_names_vec.push(names);
            }
        }

        Ok(TopkResult {
            indexes: all_indexes,
            scores: all_scores,
            class_names: all_class_names,
        })
    }

    /// Extracts top-k indexes and scores from a single prediction.
    ///
    /// # Arguments
    ///
    /// * `prediction` - Vector of confidence scores for all classes.
    /// * `k` - Number of top predictions to extract.
    ///
    /// # Returns
    ///
    /// * `(Vec<usize>, Vec<f32>)` - Tuple of (top_indexes, top_scores).
    fn extract_topk_from_prediction(&self, prediction: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
        // Create pairs of (index, score) and sort by score in descending order
        let mut indexed_scores: Vec<(usize, f32)> = prediction
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        // Sort by score in descending order
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let top_k: Vec<(usize, f32)> = indexed_scores.into_iter().take(k).collect();

        // Separate indexes and scores
        let (indexes, scores): (Vec<usize>, Vec<f32>) = top_k.into_iter().unzip();

        (indexes, scores)
    }

    /// Maps class indexes to class names using the internal mapping.
    ///
    /// # Arguments
    ///
    /// * `indexes` - Vector of class indexes.
    ///
    /// # Returns
    ///
    /// * `Vec<String>` - Vector of class names. Unknown indexes are mapped to "Unknown".
    fn map_indexes_to_names(&self, indexes: &[usize]) -> Vec<String> {
        if let Some(ref class_map) = self.class_id_map {
            indexes
                .iter()
                .map(|&idx| {
                    class_map
                        .get(&idx)
                        .cloned()
                        .unwrap_or_else(|| format!("Unknown({})", idx))
                })
                .collect()
        } else {
            indexes.iter().map(|&idx| idx.to_string()).collect()
        }
    }

    /// Gets the class name for a given class ID.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The class ID to look up.
    ///
    /// # Returns
    ///
    /// * `Option<&String>` - The class name if available.
    pub fn get_class_name(&self, class_id: usize) -> Option<&String> {
        self.class_id_map.as_ref()?.get(&class_id)
    }

    /// Checks if class name mapping is available.
    ///
    /// # Returns
    ///
    /// * `true` - If class name mapping is available.
    /// * `false` - If no class name mapping is available.
    pub fn has_class_names(&self) -> bool {
        self.class_id_map.is_some()
    }

    /// Gets the number of classes in the mapping.
    ///
    /// # Returns
    ///
    /// * `Option<usize>` - Number of classes if mapping is available.
    pub fn num_classes(&self) -> Option<usize> {
        self.class_id_map.as_ref().map(|map| map.len())
    }

    /// Updates the class name mapping.
    ///
    /// # Arguments
    ///
    /// * `class_id_map` - New class ID to name mapping.
    pub fn set_class_mapping(&mut self, class_id_map: Option<HashMap<usize, String>>) {
        self.class_id_map = class_id_map;
    }

    /// Processes a single prediction vector.
    ///
    /// # Arguments
    ///
    /// * `prediction` - Vector of confidence scores for all classes.
    /// * `k` - Number of top predictions to extract.
    ///
    /// # Returns
    ///
    /// * `Ok(TopkResult)` - The top-k results for the single prediction.
    /// * `Err(String)` - If k is 0 or if the input is invalid.
    pub fn process_single(&self, prediction: &[f32], k: usize) -> Result<TopkResult, String> {
        self.process(&[prediction.to_vec()], k)
    }
}

impl Default for Topk {
    /// Creates a default Topk processor without class name mapping.
    fn default() -> Self {
        Self::without_class_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topk_without_class_names() {
        let topk = Topk::without_class_names();
        let predictions = vec![vec![0.1, 0.8, 0.1], vec![0.7, 0.2, 0.1]];

        let result = topk.process(&predictions, 2).unwrap();
        assert_eq!(result.indexes.len(), 2);
        assert_eq!(result.indexes[0], vec![1, 0]); // Class 1 (0.8), Class 0 (0.1)
        assert_eq!(result.indexes[1], vec![0, 1]); // Class 0 (0.7), Class 1 (0.2)
        assert!(result.class_names.is_none());
    }

    #[test]
    fn test_topk_with_class_names() {
        let mut class_map = HashMap::new();
        class_map.insert(0, "cat".to_string());
        class_map.insert(1, "dog".to_string());
        class_map.insert(2, "bird".to_string());

        let topk = Topk::new(Some(class_map));
        let predictions = vec![vec![0.1, 0.8, 0.1]];

        let result = topk.process(&predictions, 2).unwrap();
        assert_eq!(result.indexes[0], vec![1, 0]);
        assert_eq!(result.class_names.as_ref().unwrap()[0], vec!["dog", "cat"]);
    }

    #[test]
    fn test_topk_from_class_names() {
        let class_names = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()];
        let topk = Topk::from_class_names(class_names);

        assert!(topk.has_class_names());
        assert_eq!(topk.num_classes(), Some(3));
        assert_eq!(topk.get_class_name(0), Some(&"cat".to_string()));
    }

    #[test]
    fn test_topk_k_larger_than_classes() {
        let topk = Topk::without_class_names();
        let predictions = vec![vec![0.1, 0.8]]; // Only 2 classes

        let result = topk.process(&predictions, 5).unwrap(); // Ask for 5
        assert_eq!(result.indexes[0].len(), 2); // Should only get 2
    }

    #[test]
    fn test_topk_invalid_k() {
        let topk = Topk::without_class_names();
        let predictions = vec![vec![0.1, 0.8, 0.1]];

        assert!(topk.process(&predictions, 0).is_err());
    }

    #[test]
    fn test_topk_empty_predictions() {
        let topk = Topk::without_class_names();
        let predictions = vec![];

        let result = topk.process(&predictions, 2).unwrap();
        assert!(result.indexes.is_empty());
        assert!(result.scores.is_empty());
    }

    #[test]
    fn test_process_single() {
        let topk = Topk::without_class_names();
        let prediction = vec![0.1, 0.8, 0.1];

        let result = topk.process_single(&prediction, 2).unwrap();
        assert_eq!(result.indexes.len(), 1);
        assert_eq!(result.indexes[0], vec![1, 0]);
    }
}
