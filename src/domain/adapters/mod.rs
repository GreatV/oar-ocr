//! Task-level adapters.
//!
//! This module contains task-specific adapters that use models and adapt their
//! outputs to specific task formats. Adapters bridge the gap between pure model
//! implementations and task requirements.

// Detection task adapters
pub mod layout_detection_adapter;
pub mod seal_text_detection_adapter;
pub mod text_detection_adapter;

// Recognition task adapters
pub mod text_recognition_adapter;

// Classification adapters
pub mod document_orientation_adapter;
pub mod text_line_orientation_adapter;

// Formula recognition adapters
pub mod formula_recognition_adapter;

// Rectification adapters
pub mod document_rectification_adapter;

// Re-export detection adapters
pub use seal_text_detection_adapter::{SealTextDetectionAdapter, SealTextDetectionAdapterBuilder};
pub use text_detection_adapter::{TextDetectionAdapter, TextDetectionAdapterBuilder};

// Re-export layout detection adapters
pub use layout_detection_adapter::{
    LayoutDetectionAdapter, LayoutDetectionAdapterBuilder, LayoutModelConfig, PPDocLayoutAdapter,
    PPDocLayoutAdapterBuilder, PicoDetLayoutAdapter, PicoDetLayoutAdapterBuilder,
    RTDetrLayoutAdapter, RTDetrLayoutAdapterBuilder,
};

// Re-export recognition adapters
pub use text_recognition_adapter::{TextRecognitionAdapter, TextRecognitionAdapterBuilder};

// Re-export classification adapters
pub use document_orientation_adapter::{
    DocumentOrientationAdapter, DocumentOrientationAdapterBuilder,
};
pub use text_line_orientation_adapter::{
    TextLineOrientationAdapter, TextLineOrientationAdapterBuilder,
};

// Re-export formula recognition adapters
pub use formula_recognition_adapter::{
    FormulaModelConfig, FormulaRecognitionAdapter, FormulaRecognitionAdapterBuilder,
    PPFormulaNetAdapter, PPFormulaNetAdapterBuilder, UniMERNetFormulaAdapter,
    UniMERNetFormulaAdapterBuilder,
};

// Re-export rectification adapters
pub use document_rectification_adapter::{UVDocRectifierAdapter, UVDocRectifierAdapterBuilder};
