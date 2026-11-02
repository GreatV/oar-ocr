//! Complete-page adapter for the external-layout document pipeline.

use crate::api::error::Error;
use crate::api::page_parser::PageParser;
use crate::api::recognition::RecognitionBackend;
use crate::document::page::PageDocument;
use crate::document::structure::StructureResult;
use crate::pipeline::doc_parser::{DocParser, DocParserConfig, DocParserOptions};
use crate::pipeline::layout::LayoutSource;
use image::RgbImage;

/// Per-page overrides for an external-layout parser.
///
/// Defaults retain the parser's configuration and a region batch size of one,
/// unless changed with the parser's builder. The config override replaces the
/// whole configuration for this call without changing subsequent calls.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct LayoutPageParserOptions {
    /// Override crop padding, token limit, region filtering, and Markdown settings.
    pub config: Option<DocParserConfig>,
    /// Override the maximum same-task region batch size. Zero is treated as one.
    pub region_batch_size: Option<usize>,
    /// Document overrides applied after the profile and chosen config.
    pub doc_parser: DocParserOptions,
}

impl LayoutPageParserOptions {
    /// Set document overrides for this call.
    pub fn with_doc_parser_options(mut self, options: DocParserOptions) -> Self {
        self.doc_parser = options;
        self
    }

    /// Override the document settings for this page.
    pub fn with_config(mut self, config: DocParserConfig) -> Self {
        self.config = Some(config);
        self
    }

    /// Override the same-task region batch size. Zero is treated as one.
    pub fn with_region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = Some(size.max(1));
        self
    }
}

/// A complete-page parser composed from a layout source and region recognizer.
///
/// The components are held by value, so an owned parser has no model lifetime.
/// References also implement the component traits, allowing the same parser to
/// borrow existing models. Each call reuses the [`DocParser`] pipeline.
pub struct LayoutPageParser<L: LayoutSource, B: RecognitionBackend> {
    layout: L,
    backend: B,
    config: DocParserConfig,
    config_override: Option<DocParserConfig>,
    options: DocParserOptions,
    region_batch_size: usize,
    page_preprocessor:
        Option<std::panic::AssertUnwindSafe<std::sync::Arc<dyn crate::PagePreprocessor>>>,
}

impl<L: LayoutSource, B: RecognitionBackend> LayoutPageParser<L, B> {
    /// Compose a layout source and backend with default document settings.
    pub fn new(layout: L, backend: B) -> Self {
        let config = DocParser::new(&backend).config().clone();
        Self {
            layout,
            backend,
            config,
            config_override: None,
            options: DocParserOptions::default(),
            region_batch_size: 1,
            page_preprocessor: None,
        }
    }

    /// Compose a layout source and backend with custom document settings.
    pub fn with_config(layout: L, backend: B, config: DocParserConfig) -> Self {
        Self {
            layout,
            backend,
            config: config.clone(),
            config_override: Some(config),
            options: DocParserOptions::default(),
            region_batch_size: 1,
            page_preprocessor: None,
        }
    }

    /// Apply document overrides above the profile and explicit configuration.
    pub fn with_options(mut self, options: DocParserOptions) -> Self {
        self.options = self.options.merge(&options);
        self.config = self
            .doc_parser(&LayoutPageParserOptions::default())
            .config()
            .clone();
        self
    }

    /// Set an owned page transform applied before layout detection.
    pub fn with_page_preprocessor(
        mut self,
        preprocessor: std::sync::Arc<dyn crate::PagePreprocessor>,
    ) -> Self {
        self.page_preprocessor = Some(std::panic::AssertUnwindSafe(preprocessor));
        self
    }

    /// Set the default same-task region batch size. Zero is treated as one.
    pub fn with_region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = size.max(1);
        self
    }

    /// Returns the layout source.
    pub fn layout(&self) -> &L {
        &self.layout
    }

    /// Returns the recognition backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the default document settings.
    pub fn config(&self) -> &DocParserConfig {
        &self.config
    }

    /// Returns the default region batch size.
    pub fn region_batch_size(&self) -> usize {
        self.region_batch_size
    }

    /// Parse a page into the original layout-first structure representation.
    ///
    /// Retains pixel coordinates, confidence, reading order, source metadata,
    /// tables, and formulas. Crop diagnostics are exposed by
    /// [`parse_page`](PageParser::parse_page).
    pub fn parse_structure(
        &self,
        image: &RgbImage,
        options: &LayoutPageParserOptions,
    ) -> Result<StructureResult, Error> {
        self.doc_parser(options)
            .parse_image(&self.layout, image)
            .map(|parsed| parsed.structure)
    }

    fn doc_parser(&self, options: &LayoutPageParserOptions) -> DocParser<'_, B> {
        let config = options.config.as_ref().or(self.config_override.as_ref());
        let parser = match config {
            Some(config) => DocParser::with_config(&self.backend, config.clone()),
            None => DocParser::new(&self.backend),
        };
        let parser = match &self.page_preprocessor {
            Some(preprocessor) => parser.with_page_preprocessor(preprocessor.0.as_ref()),
            None => parser,
        };
        parser
            .with_options(self.options.clone().merge(&options.doc_parser))
            .with_region_batch_size(options.region_batch_size.unwrap_or(self.region_batch_size))
    }
}

impl<L: LayoutSource, B: RecognitionBackend> PageParser for LayoutPageParser<L, B> {
    type Options = LayoutPageParserOptions;

    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let parser = self.doc_parser(options);
        let parsed = parser.parse_image(&self.layout, image)?;
        let markdown = parser.render_markdown(&parsed.structure);
        let mut page = PageDocument::from_structure_with_markdown(
            parsed.structure,
            parsed.width,
            parsed.height,
            markdown,
        );
        page.diagnostics = parsed.diagnostics;
        page.raw_output = parsed.raw_output;
        Ok(page)
    }
}

/// Backwards-compatible borrowed adapter with unit per-page options.
///
/// Use [`LayoutPageParser`] to own the components or override settings per page.
#[deprecated(since = "0.10.1", note = "use LayoutPageParser")]
pub struct LayoutFirstPageParser<'a, L: LayoutSource + ?Sized, B: RecognitionBackend + ?Sized> {
    inner: LayoutPageParser<&'a L, &'a B>,
}

#[allow(deprecated)]
impl<'a, L: LayoutSource + ?Sized, B: RecognitionBackend + ?Sized> LayoutFirstPageParser<'a, L, B> {
    pub fn new(layout: &'a L, backend: &'a B) -> Self {
        Self {
            inner: LayoutPageParser::new(layout, backend),
        }
    }

    pub fn with_config(layout: &'a L, backend: &'a B, config: DocParserConfig) -> Self {
        Self {
            inner: LayoutPageParser::with_config(layout, backend, config),
        }
    }

    pub fn with_region_batch_size(mut self, size: usize) -> Self {
        self.inner = self.inner.with_region_batch_size(size);
        self
    }
}

#[allow(deprecated)]
impl<L: LayoutSource + ?Sized, B: RecognitionBackend + ?Sized> PageParser
    for LayoutFirstPageParser<'_, L, B>
{
    type Options = ();

    fn parse_page(
        &self,
        image: &RgbImage,
        _options: &Self::Options,
    ) -> Result<PageDocument, Error> {
        self.inner
            .parse_page(image, &LayoutPageParserOptions::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DocParserProfile;
    use crate::api::error::BatchResult;
    use crate::api::generation::GenerationOptions;
    use crate::api::recognition::{BackendCapabilities, RecognitionTask};
    use crate::document::geometry::BoundingBox;
    use crate::pipeline::layout::{LayoutDetectionElement, LayoutDetections};
    use std::cell::{Cell, RefCell};

    #[derive(Default)]
    struct MockLayout {
        elements: Vec<LayoutDetectionElement>,
        calls: Cell<usize>,
        fail: bool,
    }

    impl LayoutSource for MockLayout {
        fn detect(&self, _image: &RgbImage) -> Result<LayoutDetections, Error> {
            self.calls.set(self.calls.get() + 1);
            if self.fail {
                return Err(Error::invalid_input("layout failure"));
            }
            Ok(LayoutDetections::new(self.elements.clone()))
        }
    }

    #[derive(Debug, PartialEq)]
    struct BatchCall {
        tasks: Vec<RecognitionTask>,
        dimensions: Vec<(u32, u32)>,
        max_tokens: usize,
    }

    #[derive(Default)]
    struct MockBackend {
        batches: RefCell<Vec<BatchCall>>,
        keys: Cell<usize>,
        full_image_calls: Cell<usize>,
        fail: bool,
        profile: DocParserProfile,
        label_budgets: bool,
        preprocessor: Option<std::sync::Arc<dyn crate::PagePreprocessor>>,
    }

    impl RecognitionBackend for MockBackend {
        fn page_preprocessor(&self) -> Option<&dyn crate::PagePreprocessor> {
            self.preprocessor.as_deref()
        }
        fn doc_parser_profile(&self) -> DocParserProfile {
            self.profile.clone()
        }

        fn region_token_budget(&self, label: &str, task: RecognitionTask) -> Option<usize> {
            self.label_budgets
                .then_some(if task == RecognitionTask::Table {
                    8192
                } else if label == "doc_title" {
                    512
                } else {
                    4096
                })
        }

        fn recognize(
            &self,
            _image: RgbImage,
            _task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            panic!("options-aware methods must be forwarded");
        }

        fn recognize_with_options(
            &self,
            image: RgbImage,
            task: RecognitionTask,
            options: &GenerationOptions,
        ) -> Result<String, Error> {
            assert_eq!(task, RecognitionTask::Ocr);
            assert_eq!(image.dimensions(), (100, 100));
            assert_eq!(options.max_new_tokens, 4096);
            self.full_image_calls.set(self.full_image_calls.get() + 1);
            Ok("whole page".to_string())
        }

        fn recognize_batch_with_options(
            &self,
            images: Vec<RgbImage>,
            tasks: &[RecognitionTask],
            options: &GenerationOptions,
        ) -> BatchResult<String> {
            self.batches.borrow_mut().push(BatchCall {
                tasks: tasks.to_vec(),
                dimensions: images.iter().map(RgbImage::dimensions).collect(),
                max_tokens: options.max_new_tokens,
            });
            Ok(tasks
                .iter()
                .map(|task| {
                    if self.fail {
                        return Err(Error::invalid_input("recognition failure"));
                    }
                    Ok(match task {
                        RecognitionTask::Table => "<table><tr><td>cell</td></tr></table>",
                        RecognitionTask::Formula => "x^2",
                        _ => "recognized text",
                    }
                    .to_string())
                })
                .collect())
        }

        fn recognition_batch_key(&self, _image: &RgbImage, _task: RecognitionTask) -> u64 {
            self.keys.set(self.keys.get() + 1);
            7
        }

        fn capabilities(&self) -> BackendCapabilities {
            BackendCapabilities {
                supports_chart: false,
                ..Default::default()
            }
        }
    }

    fn region(label: &str, bbox: [f32; 4]) -> LayoutDetectionElement {
        LayoutDetectionElement {
            bbox: BoundingBox::from_coords(bbox[0], bbox[1], bbox[2], bbox[3]),
            element_type: label.to_string(),
            score: 0.9,
        }
    }

    fn layout() -> MockLayout {
        MockLayout {
            elements: vec![
                region("doc_title", [10.0, 10.0, 40.0, 20.0]),
                region("table", [50.0, 10.0, 90.0, 30.0]),
                region("table", [50.0, 40.0, 90.0, 60.0]),
                region("chart", [10.0, 70.0, 40.0, 90.0]),
                region("header", [10.0, 0.0, 40.0, 5.0]),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn owned_parser_matches_legacy_markdown_and_retains_structure() {
        struct BulletBackend;
        impl RecognitionBackend for BulletBackend {
            fn doc_parser_profile(&self) -> DocParserProfile {
                DocParserProfile::default()
                    .with_merge_labels(Vec::new())
                    .with_crop(
                        crate::pipeline::crop::CropOptions::default()
                            .with_coordinate_grid(Some(1000)),
                    )
                    .with_render_style(
                        crate::render::markdown::RenderStyle::default()
                            .with_normalize_bullets(true),
                    )
            }
            fn recognize(
                &self,
                image: RgbImage,
                _: RecognitionTask,
                _: usize,
            ) -> Result<String, Error> {
                Ok(if image.width() == 60 {
                    "- item"
                } else {
                    "middle"
                }
                .into())
            }
        }
        fn thread_safe<T: Send + Sync + std::panic::UnwindSafe + std::panic::RefUnwindSafe>() {}
        thread_safe::<DocParser<'_, BulletBackend>>();
        thread_safe::<LayoutPageParser<crate::StaticLayout, BulletBackend>>();
        let bullets = crate::StaticLayout::new(vec![
            region("text", [100., 0., 160., 30.]),
            region("text", [119., 40., 180., 70.]),
            region("text", [100., 80., 160., 110.]),
        ]);
        let page = LayoutPageParser::new(bullets, BulletBackend)
            .parse_page(&RgbImage::new(2000, 150), &Default::default())
            .unwrap();
        assert_eq!(
            page.markdown.as_deref(),
            Some("- item\n\n- middle\n\n- item")
        );

        struct Prepare(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl crate::PagePreprocessor for Prepare {
            fn preprocess(&self, image: &RgbImage) -> Result<RgbImage, Error> {
                assert_eq!(image.dimensions(), (100, 100));
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(RgbImage::new(200, 100))
            }
        }
        struct PreparedLayout(MockLayout);
        impl LayoutSource for PreparedLayout {
            fn detect(&self, image: &RgbImage) -> Result<LayoutDetections, Error> {
                assert_eq!(image.dimensions(), (200, 100));
                self.0.detect(image)
            }
        }
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let preprocessor = std::sync::Arc::new(Prepare(count.clone()));
        let backend = MockBackend {
            preprocessor: Some(preprocessor.clone()),
            ..Default::default()
        };
        let prepared = LayoutPageParser::new(PreparedLayout(layout()), &backend);
        let page = prepared
            .parse_page(&RgbImage::new(100, 100), &Default::default())
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(page.blocks[0].bbox, [0.05, 0.1, 0.2, 0.2]);
        let prepared = prepared.with_page_preprocessor(preprocessor);
        prepared
            .parse_page(&RgbImage::new(100, 100), &Default::default())
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);

        let parser =
            LayoutPageParser::new(layout(), MockBackend::default()).with_region_batch_size(2);
        let image = RgbImage::new(100, 100);
        let page = parser.parse_page(&image, &Default::default()).unwrap();
        let legacy = DocParser::with_config(parser.backend(), parser.config().clone())
            .with_region_batch_size(2)
            .parse(parser.layout(), image.clone())
            .unwrap();
        assert_eq!(page.markdown.unwrap(), legacy.to_markdown());
        assert_eq!(page.blocks.len(), 4);
        assert_eq!(page.blocks[0].block_type, "doc_title");
        assert_eq!(page.blocks[1].bbox, [0.5, 0.1, 0.9, 0.3]);
        assert_eq!(page.blocks[3].content, None);
        let structure = parser.parse_structure(&image, &Default::default()).unwrap();
        assert_eq!(
            serde_json::to_value(&structure).unwrap(),
            serde_json::to_value(&legacy).unwrap()
        );
        assert_eq!(structure.tables.len(), 2);
        assert_eq!(structure.layout_elements[0].order_index, Some(1));
        assert_eq!(structure.layout_elements[0].confidence, 0.9);
        assert!(page.diagnostics.is_empty());
        assert_eq!(parser.layout().calls.get(), 3);
        assert!(
            parser
                .backend()
                .batches
                .borrow()
                .iter()
                .any(|batch| batch.tasks.len() == 2)
        );
        let native = LayoutPageParser::new(
            layout(),
            MockBackend {
                profile: DocParserProfile::default().with_max_tokens(8192),
                label_budgets: true,
                ..Default::default()
            },
        )
        .with_region_batch_size(2);
        native.parse_page(&image, &Default::default()).unwrap();
        assert!(
            native
                .backend()
                .batches
                .borrow()
                .iter()
                .any(
                    |batch| batch.tasks == [RecognitionTask::Table, RecognitionTask::Table]
                        && batch.max_tokens == 8192
                )
        );
        assert!(
            native
                .backend()
                .batches
                .borrow()
                .iter()
                .any(|batch| batch.tasks == [RecognitionTask::Ocr] && batch.max_tokens == 512)
        );
        let explicit = LayoutPageParser::with_config(
            layout(),
            MockBackend {
                profile: DocParserProfile::default().with_max_tokens(8192),
                label_budgets: true,
                ..Default::default()
            },
            DocParserConfig::default(),
        );
        explicit.parse_page(&image, &Default::default()).unwrap();
        assert!(
            explicit
                .backend()
                .batches
                .borrow()
                .iter()
                .filter(|batch| batch.tasks == [RecognitionTask::Table])
                .all(|batch| batch.max_tokens == 4096)
        );
        let explicit = explicit.with_options(DocParserOptions::default().with_max_tokens(777));
        explicit.backend().batches.borrow_mut().clear();
        explicit
            .parse_page(
                &image,
                &LayoutPageParserOptions::default().with_config(DocParserConfig {
                    max_tokens: 1024,
                    ..Default::default()
                }),
            )
            .unwrap();
        assert!(
            explicit
                .backend()
                .batches
                .borrow()
                .iter()
                .filter(|batch| batch.tasks == [RecognitionTask::Table])
                .all(|batch| batch.max_tokens == 777)
        );
        explicit.backend().batches.borrow_mut().clear();
        explicit
            .parse_page(
                &image,
                &LayoutPageParserOptions::default()
                    .with_doc_parser_options(DocParserOptions::default().with_max_tokens(333)),
            )
            .unwrap();
        assert!(
            explicit
                .backend()
                .batches
                .borrow()
                .iter()
                .all(|batch| batch.max_tokens == 333)
        );
    }

    #[test]
    fn borrowed_parser_forwards_native_batching_and_per_page_overrides() {
        let layout = layout();
        let backend = MockBackend::default();
        let parser = LayoutPageParser::new(
            &layout as &dyn LayoutSource,
            &backend as &dyn RecognitionBackend,
        );
        let image = RgbImage::new(100, 100);
        let options = LayoutPageParserOptions::default()
            .with_config(DocParserConfig {
                max_tokens: 123,
                crop_pad_ratio: 0.1,
                skip_auxiliary_regions: false,
                skip_region_blocks: false,
                markdown_ignore_labels: vec!["doc_title".to_string()],
                markdown_pretty: false,
            })
            .with_region_batch_size(2);
        let page = parser.parse_page(&image, &options).unwrap();
        let legacy = DocParser::with_config(&backend, options.config.clone().unwrap())
            .with_region_batch_size(2)
            .parse_to_markdown(&layout, image.clone())
            .unwrap();
        assert_eq!(page.markdown.as_deref(), Some(legacy.as_str()));
        assert!(!page.markdown.unwrap().contains("# recognized text"));
        assert_eq!(page.blocks.len(), 5);
        assert_eq!(backend.keys.get(), 8);
        assert!(
            backend
                .batches
                .borrow()
                .iter()
                .all(|batch| batch.max_tokens == 123)
        );
        assert!(backend.batches.borrow().iter().any(|batch| batch.tasks
            == [RecognitionTask::Table, RecognitionTask::Table]
            && batch.dimensions == [(48, 24), (48, 24)]));
        backend.batches.borrow_mut().clear();
        let structure = parser.parse_structure(&image, &options).unwrap();
        assert_eq!(structure.layout_elements.len(), 5);
        assert_eq!(
            structure.layout_elements[0].bbox,
            BoundingBox::from_coords(10.0, 10.0, 40.0, 20.0)
        );
        assert!(
            backend
                .batches
                .borrow()
                .iter()
                .all(|batch| batch.max_tokens == 123)
        );
        assert!(
            backend
                .batches
                .borrow()
                .iter()
                .any(|batch| batch.tasks.len() == 2)
        );
        backend.batches.borrow_mut().clear();
        let page = parser.parse_page(&image, &Default::default()).unwrap();
        assert_eq!(page.blocks.len(), 4);
        assert!(
            backend
                .batches
                .borrow()
                .iter()
                .all(|batch| batch.max_tokens == 4096 && batch.tasks.len() == 1)
        );
        assert_eq!(parser.region_batch_size(), 1);
        assert_eq!(parser.config().max_tokens, 4096);
    }

    #[test]
    #[allow(deprecated)]
    fn empty_layout_falls_back_to_full_image_and_borrowed_adapter_still_works() {
        let layout = MockLayout::default();
        let backend = MockBackend::default();
        let parser = LayoutFirstPageParser::new(&layout, &backend);
        let page = parser.parse_page(&RgbImage::new(100, 100), &()).unwrap();
        assert_eq!(page.blocks[0].bbox, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(page.blocks[0].content.as_deref(), Some("whole page"));
        assert_eq!(backend.full_image_calls.get(), 1);
        assert!(backend.batches.borrow().is_empty());
        let backend = MockBackend {
            profile: DocParserProfile::default()
                .with_empty_layout(crate::api::recognition::EmptyLayoutBehavior::Empty),
            ..Default::default()
        };
        let parser = LayoutPageParser::new(&layout, &backend);
        assert!(
            parser
                .parse_page(&RgbImage::new(100, 100), &Default::default())
                .unwrap()
                .blocks
                .is_empty()
        );
        assert_eq!(backend.full_image_calls.get(), 0);
    }

    #[test]
    fn invalid_crop_is_reported_without_losing_its_block() {
        let layout = MockLayout {
            elements: vec![region("table", [10.0, 10.0, 10.0, 20.0])],
            ..Default::default()
        };
        let parser = LayoutPageParser::new(layout, MockBackend::default());
        let page = parser
            .parse_page(&RgbImage::new(100, 100), &Default::default())
            .unwrap();
        assert_eq!(page.blocks.len(), 1);
        assert!(page.blocks[0].content.is_none());
        assert_eq!(page.diagnostics.len(), 1);
        assert_eq!(page.diagnostics[0].block_index, Some(0));
        assert_eq!(page.diagnostics[0].stage, "crop");
        assert!(page.diagnostics[0].message.contains("invalid crop region"));
        let image = RgbImage::new(100, 100);
        let legacy = DocParser::new(parser.backend())
            .parse(parser.layout(), image.clone())
            .unwrap();
        let structure = parser.parse_structure(&image, &Default::default()).unwrap();
        assert_eq!(
            serde_json::to_value(structure).unwrap(),
            serde_json::to_value(legacy).unwrap()
        );
    }

    #[test]
    fn layout_and_recognition_errors_propagate() {
        let image = RgbImage::new(100, 100);
        let parser = LayoutPageParser::new(
            MockLayout {
                fail: true,
                ..Default::default()
            },
            MockBackend::default(),
        );
        assert!(
            parser
                .parse_page(&image, &Default::default())
                .unwrap_err()
                .to_string()
                .contains("layout failure")
        );
        let parser = LayoutPageParser::new(
            layout(),
            MockBackend {
                fail: true,
                ..Default::default()
            },
        );
        assert!(
            parser
                .parse_page(&image, &Default::default())
                .unwrap_err()
                .to_string()
                .contains("recognition failure")
        );
        let parser = parser.with_options(
            DocParserOptions::default()
                .with_region_errors(crate::api::recognition::RegionErrorBehavior::KeepRegion),
        );
        let page = parser.parse_page(&image, &Default::default()).unwrap();
        assert_eq!(page.blocks.len(), 4);
        assert_eq!(page.diagnostics.len(), 3);
    }
}
