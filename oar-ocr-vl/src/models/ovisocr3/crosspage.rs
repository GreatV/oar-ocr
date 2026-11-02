//! The model-native second stage of adjacent-page table reconstruction.

use super::{OvisOcr3, prompts};
use crate::Error;
use image::RgbImage;
use regex::{Captures, Regex};
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

const NEGATIVE: &str = "No cross-page table exists between these pages.";

/// One reconstructed table; page numbers are local to this two-page request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CrossPageTable {
    /// One-based input-local page numbers contributing to the reconstructed table.
    pub source_pages: Vec<usize>,
    /// Reconstructed HTML table.
    pub html: String,
}

impl OvisOcr3 {
    /// Reconstruct tables from two adjacent images and their current Markdown.
    /// Image references use input-local page numbers 1 and 2. Returns an empty
    /// vector when either page has no table or the model returns its negative
    /// sentinel. Use [`Self::reconstruct_tables_with_page_numbers`] when the
    /// Markdown already contains document-level `page_N/` image references.
    pub fn reconstruct_tables(
        &self,
        images: &[RgbImage],
        markdown: &[&str],
        max_new_tokens: usize,
    ) -> Result<Vec<CrossPageTable>, Error> {
        self.reconstruct_tables_with_page_numbers(images, markdown, [1, 2], max_new_tokens)
    }

    /// Reconstruct tables while remapping image references for two document pages.
    ///
    /// `page_numbers` contains the positive, distinct document page numbers in
    /// input order. Qualified references must belong to this pair; unqualified
    /// crops belong to their containing input page. Returned HTML restores
    /// document page numbers, while `source_pages` stays input-local `[1, 2]`.
    /// Before another reconstruction step, re-render the affected page so that
    /// crop aliases match the current image, as in the reference reverse flow.
    pub fn reconstruct_tables_with_page_numbers(
        &self,
        images: &[RgbImage],
        markdown: &[&str],
        page_numbers: [usize; 2],
        max_new_tokens: usize,
    ) -> Result<Vec<CrossPageTable>, Error> {
        if images.len() != 2 || markdown.len() != 2 {
            return Err(Error::invalid_input(
                "OvisOCR3 table reconstruction needs two images and two Markdown pages",
            ));
        }
        if page_numbers.contains(&0) || page_numbers[0] == page_numbers[1] {
            return Err(Error::invalid_input(
                "OvisOCR3 reconstruction page numbers must be positive and distinct",
            ));
        }
        let Some(prompt) = reconstruction_prompt(markdown, page_numbers)? else {
            return Ok(Vec::new());
        };
        let raw = self.generate_joint_with_prompt(images, &prompt, max_new_tokens)?;
        parse_tables(&raw, page_numbers)
    }
}

// Adapted from OvisOCR3's Apache-2.0 document_crosspage.py: remap/parse_tables.
fn remap_images(
    text: &str,
    source_pages: [usize; 2],
    target_pages: [usize; 2],
    unqualified_page: Option<usize>,
) -> Result<String, Error> {
    static QUALIFIED: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(^|[^\w/])page_(\d+)/images/").unwrap());
    static CROP: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(^|[^\w/])(images/bbox_\d+_\d+_\d+_\d+\.jpg)").unwrap());
    for caps in QUALIFIED.captures_iter(text) {
        if !caps[2]
            .parse::<usize>()
            .is_ok_and(|page| source_pages.contains(&page))
        {
            return Err(Error::invalid_input(
                "OvisOCR3 image reference belongs to a page outside the reconstruction pair",
            ));
        }
    }
    let text = QUALIFIED.replace_all(text, |caps: &Captures<'_>| {
        let index = usize::from(caps[2].parse::<usize>().unwrap() == source_pages[1]);
        format!("{}page_{}/images/", &caps[1], target_pages[index])
    });
    if CROP.is_match(&text) && unqualified_page.is_none() {
        return Err(Error::invalid_input(
            "OvisOCR3 cross-page response contains an unqualified crop reference",
        ));
    }
    Ok(CROP
        .replace_all(&text, |caps: &Captures<'_>| {
            format!(
                "{}page_{}/{}",
                &caps[1],
                unqualified_page.unwrap(),
                &caps[2]
            )
        })
        .into_owned())
}

fn parse_tables(raw: &str, page_numbers: [usize; 2]) -> Result<Vec<CrossPageTable>, Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ResponseTable {
        source_pages: Vec<usize>,
        html: String,
    }
    let (tables, _): (Vec<ResponseTable>, usize) = crate::document::decoding::decode_json(
        raw,
        crate::document::decoding::JsonTrailing::Reject,
    )
    .map_err(|e| {
        Error::invalid_input(format!(
            "OvisOCR3 cross-page response is not a table array: {e}"
        ))
    })?;
    if tables.len() == 1 && tables[0].source_pages.is_empty() && tables[0].html == NEGATIVE {
        return Ok(Vec::new());
    }
    if tables.is_empty()
        || tables.iter().any(|table| {
            let html = table.html.trim();
            table.source_pages != [1, 2]
                || crate::render::table::table_span(
                    html,
                    crate::render::table::TableWrapper::Document,
                )
                .is_none()
        })
    {
        return Err(Error::invalid_input(
            "OvisOCR3 expected full HTML tables with source_pages exactly [1, 2]",
        ));
    }
    tables
        .into_iter()
        .map(|table| {
            Ok(CrossPageTable {
                source_pages: table.source_pages,
                html: remap_images(&table.html, [1, 2], page_numbers, None)?,
            })
        })
        .collect()
}

fn reconstruction_prompt(
    markdown: &[&str],
    page_numbers: [usize; 2],
) -> Result<Option<String>, Error> {
    if !markdown.iter().all(|page| {
        crate::render::table::table_span(page, crate::render::table::TableWrapper::Fragment)
            .is_some()
    }) {
        return Ok(None);
    }
    let context = markdown
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let text = remap_images(text, page_numbers, [1, 2], Some(index + 1))?;
            Ok(format!(
                "<INPUT_PAGE index=\"{}\">\n{text}\n</INPUT_PAGE>",
                index + 1
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?
        .join("\n\n");
    Ok(Some(format!(
        "{}\n\nInput Markdown:\n\n{context}",
        prompts::CROSSPAGE_TABLE.trim_end_matches('\n')
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_context_qualifies_figure_provenance() {
        let prompt = reconstruction_prompt(
            &[
                "<table><img src=\"images/bbox_1_2_3_4.jpg\"><img src=\"page_9/images/bbox_5_6_7_8.jpg\"></table>",
                "<table><img src=\"page_9/images/bbox_5_6_7_8.jpg\"></table>",
            ],
            [8, 9],
        )
        .unwrap()
        .unwrap();
        assert!(prompt.contains("page_1/images/bbox_1_2_3_4.jpg"));
        assert!(prompt.contains("page_2/images/bbox_5_6_7_8.jpg"));
        assert_eq!(prompt.matches("page_2/images/bbox_5_6_7_8.jpg").count(), 2);
        assert!(
            reconstruction_prompt(&["plain text", "<table></table>"], [1, 2])
                .unwrap()
                .is_none()
        );
        assert!(remap_images("page_8/images/bbox_1_2_3_4.jpg", [1, 2], [1, 2], Some(1)).is_err());

        let tables = parse_tables(r#"[{"source_pages":[1,2],"html":"<table><img src='page_2/images/bbox_1_2_3_4.jpg'></table>"}]"#, [8, 9]).unwrap();
        assert!(tables[0].html.contains("page_9/images/"));
        assert!(
            parse_tables(
                &format!(r#"[{{"source_pages":[],"html":"{NEGATIVE}"}}]"#),
                [1, 2]
            )
            .unwrap()
            .is_empty()
        );
        for raw in [
            r#"[]"#,
            r#"[{"source_pages":[1],"html":"<table></table>"}]"#,
            r#"[{"source_pages":[1,2],"html":"<table>"}]"#,
            r#"[{"source_pages":[1,2],"html":"<table></table>","extra":0}]"#,
            r#"[{"source_pages":[],"html":"wrong sentinel"}]"#,
            r#"[{"source_pages":[1,2],"html":"<table><img src='images/bbox_1_2_3_4.jpg'></table>"}]"#,
        ] {
            assert!(parse_tables(raw, [1, 2]).is_err(), "{raw}");
        }
    }
}
