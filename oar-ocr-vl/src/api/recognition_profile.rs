//! Parameter-based document defaults and caller overrides.

use crate::pipeline::crop::CropOptions;
use crate::pipeline::layout::LayoutPostprocessOptions;
use crate::render::markdown::RenderStyle;
use crate::render::table::OtslOptions;
use crate::render::text::{DelimiterOptions, FormulaOptions, RepetitionOptions};

/// Processing of an auxiliary layout label.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum AuxiliaryLabelAction {
    /// Recognize and render the region.
    #[default]
    Recognize,
    /// Recognize the region and hide it in Markdown.
    HideAtRender,
    /// Remove the region before recognition.
    Drop,
}

/// A processing action for one native layout label.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AuxiliaryLabelRule {
    /// Native layout label.
    pub label: String,
    /// Processing action.
    pub action: AuxiliaryLabelAction,
}

impl AuxiliaryLabelRule {
    /// Set the native layout label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set the processing action.
    pub fn with_action(mut self, action: AuxiliaryLabelAction) -> Self {
        self.action = action;
        self
    }
}

/// Composition of formula regions and recognized equation numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormulaNumberComposition {
    /// Use the configured Markdown label filtering.
    #[default]
    None,
    /// Attach a following number with an unparenthesized LaTeX tag.
    FollowingStarTag,
    /// Attach a preceding or following number with a LaTeX tag.
    AdjacentTag,
    /// Render each number as its own paragraph.
    SeparateParagraph,
}

/// Behavior when layout filtering leaves no regions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmptyLayoutBehavior {
    /// Recognize the whole page with the text prompt.
    #[default]
    FullPageOcr,
    /// Return an empty structured page.
    Empty,
}

/// Handling of per-region recognition errors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegionErrorBehavior {
    /// Return the recognition error.
    #[default]
    Propagate,
    /// Keep the region and add a diagnostic.
    KeepRegion,
}

/// Backend-provided document parsing parameters.
///
/// Default preserves the existing shared parser behavior.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DocParserProfile {
    /// Region generation limit.
    pub max_tokens: usize,
    /// Padding added around each region before cropping.
    pub crop_pad_ratio: f32,
    /// Skip page numbers, footnotes, headers, footers and margin text.
    pub skip_auxiliary_regions: bool,
    /// Skip hierarchical region blocks.
    pub skip_region_blocks: bool,
    /// Native labels hidden in Markdown.
    pub markdown_ignore_labels: Vec<String>,
    /// Center captions and tables with HTML.
    pub markdown_pretty: bool,
    /// Labels eligible for crop merging; an empty list disables merging.
    pub merge_labels: Vec<String>,
    /// Labels that block crop merging; None counts every other region.
    pub merge_blocker_labels: Option<Vec<String>>,
    /// Actions for individual native labels.
    pub auxiliary_labels: Vec<AuxiliaryLabelRule>,
    /// Placement of recognized formula numbers.
    pub formula_numbers: FormulaNumberComposition,
    /// Result when layout filtering leaves no regions.
    pub empty_layout: EmptyLayoutBehavior,
    /// Handling of per-region recognition errors.
    pub region_errors: RegionErrorBehavior,
    /// Rounding, rotation, and polygon crop settings.
    pub crop: CropOptions,
    /// Layout post-processing and document overlap filter.
    pub layout: Option<LayoutPostprocessOptions>,
    /// Formatting parameters for the page renderer.
    pub render_style: RenderStyle,
    /// Paint figure tokens in table crops and restore their image references.
    pub table_figure_tokens: bool,
    pub(crate) repetition: Option<RepetitionOptions>,
    pub(crate) formula: FormulaOptions,
    pub(crate) delimiters: Option<DelimiterOptions>,
    pub(crate) table: OtslOptions,
}

impl Default for DocParserProfile {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            crop_pad_ratio: 0.,
            skip_auxiliary_regions: true,
            skip_region_blocks: true,
            markdown_ignore_labels: crate::render::markdown::default_markdown_ignore_labels(),
            markdown_pretty: true,
            merge_labels: vec!["text".into()],
            merge_blocker_labels: None,
            auxiliary_labels: Vec::new(),
            formula_numbers: FormulaNumberComposition::None,
            empty_layout: EmptyLayoutBehavior::FullPageOcr,
            region_errors: RegionErrorBehavior::Propagate,
            crop: CropOptions::default(),
            layout: None,
            render_style: RenderStyle::default(),
            table_figure_tokens: false,
            repetition: None,
            formula: FormulaOptions::default(),
            delimiters: None,
            table: OtslOptions::default(),
        }
    }
}

impl DocParserProfile {
    /// Set whether to paint figure tokens in table crops and restore image references.
    pub fn with_table_figure_tokens(mut self, enabled: bool) -> Self {
        self.table_figure_tokens = enabled;
        self
    }

    /// Set the region generation limit.
    pub fn with_max_tokens(mut self, value: usize) -> Self {
        self.max_tokens = value;
        self
    }

    /// Set the padding added around each region before cropping.
    pub fn with_crop_pad_ratio(mut self, value: f32) -> Self {
        self.crop_pad_ratio = value;
        self
    }

    /// Set whether to skip page numbers, footnotes, headers, footers and margin text.
    pub fn with_skip_auxiliary_regions(mut self, value: bool) -> Self {
        self.skip_auxiliary_regions = value;
        self
    }

    /// Set whether to skip hierarchical region blocks.
    pub fn with_skip_region_blocks(mut self, value: bool) -> Self {
        self.skip_region_blocks = value;
        self
    }

    /// Set the native labels hidden in Markdown.
    pub fn with_markdown_ignore_labels(mut self, value: Vec<String>) -> Self {
        self.markdown_ignore_labels = value;
        self
    }

    /// Set whether to center captions and tables with HTML.
    pub fn with_markdown_pretty(mut self, value: bool) -> Self {
        self.markdown_pretty = value;
        self
    }

    /// Set the labels eligible for crop merging; an empty list disables merging.
    pub fn with_merge_labels(mut self, value: Vec<String>) -> Self {
        self.merge_labels = value;
        self
    }

    /// Set the labels that block crop merging; None counts every other region.
    pub fn with_merge_blocker_labels(mut self, value: Option<Vec<String>>) -> Self {
        self.merge_blocker_labels = value;
        self
    }

    /// Set the actions for individual native labels.
    pub fn with_auxiliary_labels(mut self, value: Vec<AuxiliaryLabelRule>) -> Self {
        self.auxiliary_labels = value;
        self
    }

    /// Set the placement of recognized formula numbers.
    pub fn with_formula_numbers(mut self, value: FormulaNumberComposition) -> Self {
        self.formula_numbers = value;
        self
    }

    /// Set the result when layout filtering leaves no regions.
    pub fn with_empty_layout(mut self, value: EmptyLayoutBehavior) -> Self {
        self.empty_layout = value;
        self
    }

    /// Set the handling of per-region recognition errors.
    pub fn with_region_errors(mut self, value: RegionErrorBehavior) -> Self {
        self.region_errors = value;
        self
    }

    /// Set the rounding, rotation, and polygon crop settings.
    pub fn with_crop(mut self, value: CropOptions) -> Self {
        self.crop = value;
        self
    }

    /// Set the layout post-processing and document overlap filter.
    pub fn with_layout(mut self, value: Option<LayoutPostprocessOptions>) -> Self {
        self.layout = value;
        self
    }

    /// Set the formatting parameters for the page renderer.
    pub fn with_render_style(mut self, value: RenderStyle) -> Self {
        self.render_style = value;
        self
    }
}

/// Parser-level or per-call overrides of the selected profile.
///
/// Unset fields inherit the backend profile and explicit DocParserConfig.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DocParserOptions {
    /// Replace the backend profile before applying configuration and overrides.
    pub profile: Option<DocParserProfile>,
    /// Override the region generation limit.
    pub max_tokens: Option<usize>,
    /// Override the padding added around each region before cropping.
    pub crop_pad_ratio: Option<f32>,
    /// Override whether to skip page numbers, footnotes, headers, footers and margin text.
    pub skip_auxiliary_regions: Option<bool>,
    /// Override whether to skip hierarchical region blocks.
    pub skip_region_blocks: Option<bool>,
    /// Override the native labels hidden in Markdown.
    pub markdown_ignore_labels: Option<Vec<String>>,
    /// Override whether to center captions and tables with HTML.
    pub markdown_pretty: Option<bool>,
    /// Override the labels eligible for crop merging; an empty list disables merging.
    pub merge_labels: Option<Vec<String>>,
    /// Override the labels that block crop merging; None counts every other region.
    pub merge_blocker_labels: Option<Vec<String>>,
    /// Override the actions for individual native labels.
    pub auxiliary_labels: Option<Vec<AuxiliaryLabelRule>>,
    /// Override the placement of recognized formula numbers.
    pub formula_numbers: Option<FormulaNumberComposition>,
    /// Override the result when layout filtering leaves no regions.
    pub empty_layout: Option<EmptyLayoutBehavior>,
    /// Override the handling of per-region recognition errors.
    pub region_errors: Option<RegionErrorBehavior>,
    /// Override the rounding, rotation, and polygon crop settings.
    pub crop: Option<CropOptions>,
    /// Override the layout post-processing and document overlap filter.
    pub layout: Option<LayoutPostprocessOptions>,
    /// Override the formatting parameters for the page renderer.
    pub render_style: Option<RenderStyle>,
}

impl DocParserOptions {
    /// Replace the backend profile.
    pub fn with_profile(mut self, profile: DocParserProfile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Override the region generation limit.
    pub fn with_max_tokens(mut self, value: usize) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Override the padding added around each region before cropping.
    pub fn with_crop_pad_ratio(mut self, value: f32) -> Self {
        self.crop_pad_ratio = Some(value);
        self
    }

    /// Override whether to skip page numbers, footnotes, headers, footers and margin text.
    pub fn with_skip_auxiliary_regions(mut self, value: bool) -> Self {
        self.skip_auxiliary_regions = Some(value);
        self
    }

    /// Override whether to skip hierarchical region blocks.
    pub fn with_skip_region_blocks(mut self, value: bool) -> Self {
        self.skip_region_blocks = Some(value);
        self
    }

    /// Override the native labels hidden in Markdown.
    pub fn with_markdown_ignore_labels(mut self, value: Vec<String>) -> Self {
        self.markdown_ignore_labels = Some(value);
        self
    }

    /// Override whether to center captions and tables with HTML.
    pub fn with_markdown_pretty(mut self, value: bool) -> Self {
        self.markdown_pretty = Some(value);
        self
    }

    /// Override the labels eligible for crop merging; an empty list disables merging.
    pub fn with_merge_labels(mut self, value: Vec<String>) -> Self {
        self.merge_labels = Some(value);
        self
    }

    /// Override the labels that block crop merging; None counts every other region.
    pub fn with_merge_blocker_labels(mut self, value: Vec<String>) -> Self {
        self.merge_blocker_labels = Some(value);
        self
    }

    /// Override the actions for individual native labels.
    pub fn with_auxiliary_labels(mut self, value: Vec<AuxiliaryLabelRule>) -> Self {
        self.auxiliary_labels = Some(value);
        self
    }

    /// Override the placement of recognized formula numbers.
    pub fn with_formula_numbers(mut self, value: FormulaNumberComposition) -> Self {
        self.formula_numbers = Some(value);
        self
    }

    /// Override the result when layout filtering leaves no regions.
    pub fn with_empty_layout(mut self, value: EmptyLayoutBehavior) -> Self {
        self.empty_layout = Some(value);
        self
    }

    /// Override the handling of per-region recognition errors.
    pub fn with_region_errors(mut self, value: RegionErrorBehavior) -> Self {
        self.region_errors = Some(value);
        self
    }

    /// Override the rounding, rotation, and polygon crop settings.
    pub fn with_crop(mut self, value: CropOptions) -> Self {
        self.crop = Some(value);
        self
    }

    /// Override the layout post-processing and document overlap filter.
    pub fn with_layout(mut self, value: LayoutPostprocessOptions) -> Self {
        self.layout = Some(value);
        self
    }

    /// Override the formatting parameters for the page renderer.
    pub fn with_render_style(mut self, value: RenderStyle) -> Self {
        self.render_style = Some(value);
        self
    }

    pub(crate) fn apply(&self, profile: &mut DocParserProfile) {
        if let Some(value) = &self.max_tokens {
            profile.max_tokens = *value;
        }
        if let Some(value) = &self.crop_pad_ratio {
            profile.crop_pad_ratio = *value;
        }
        if let Some(value) = &self.skip_auxiliary_regions {
            profile.skip_auxiliary_regions = *value;
        }
        if let Some(value) = &self.skip_region_blocks {
            profile.skip_region_blocks = *value;
        }
        if let Some(value) = &self.markdown_ignore_labels {
            profile.markdown_ignore_labels = value.clone();
        }
        if let Some(value) = &self.markdown_pretty {
            profile.markdown_pretty = *value;
        }
        if let Some(value) = &self.merge_labels {
            profile.merge_labels = value.clone();
        }
        if let Some(value) = &self.merge_blocker_labels {
            profile.merge_blocker_labels = Some(value.clone());
        }
        if let Some(value) = &self.auxiliary_labels {
            profile.auxiliary_labels = value.clone();
        }
        if let Some(value) = &self.formula_numbers {
            profile.formula_numbers = *value;
        }
        if let Some(value) = &self.empty_layout {
            profile.empty_layout = *value;
        }
        if let Some(value) = &self.region_errors {
            profile.region_errors = *value;
        }
        if let Some(value) = &self.crop {
            profile.crop = *value;
        }
        if let Some(value) = &self.layout {
            profile.layout = Some(value.clone());
        }
        if let Some(value) = &self.render_style {
            profile.render_style = value.clone();
        }
    }

    pub(crate) fn merge(mut self, overrides: &Self) -> Self {
        if overrides.profile.is_some() {
            self.profile = overrides.profile.clone();
        }
        if overrides.max_tokens.is_some() {
            self.max_tokens = overrides.max_tokens;
        }
        if overrides.crop_pad_ratio.is_some() {
            self.crop_pad_ratio = overrides.crop_pad_ratio;
        }
        if overrides.skip_auxiliary_regions.is_some() {
            self.skip_auxiliary_regions = overrides.skip_auxiliary_regions;
        }
        if overrides.skip_region_blocks.is_some() {
            self.skip_region_blocks = overrides.skip_region_blocks;
        }
        if overrides.markdown_ignore_labels.is_some() {
            self.markdown_ignore_labels = overrides.markdown_ignore_labels.clone();
        }
        if overrides.markdown_pretty.is_some() {
            self.markdown_pretty = overrides.markdown_pretty;
        }
        if overrides.merge_labels.is_some() {
            self.merge_labels = overrides.merge_labels.clone();
        }
        if overrides.merge_blocker_labels.is_some() {
            self.merge_blocker_labels = overrides.merge_blocker_labels.clone();
        }
        if overrides.auxiliary_labels.is_some() {
            self.auxiliary_labels = overrides.auxiliary_labels.clone();
        }
        if overrides.formula_numbers.is_some() {
            self.formula_numbers = overrides.formula_numbers;
        }
        if overrides.empty_layout.is_some() {
            self.empty_layout = overrides.empty_layout;
        }
        if overrides.region_errors.is_some() {
            self.region_errors = overrides.region_errors;
        }
        if overrides.crop.is_some() {
            self.crop = overrides.crop;
        }
        if overrides.layout.is_some() {
            self.layout = overrides.layout.clone();
        }
        if overrides.render_style.is_some() {
            self.render_style = overrides.render_style.clone();
        }
        self
    }
}
