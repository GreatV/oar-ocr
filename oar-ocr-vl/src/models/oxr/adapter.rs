//! Adapt OXR generation to the crate's region-recognition contract.

use super::{Oxr, OxrTask};
use crate::{BackendCapabilities, Error, RecognitionBackend, RecognitionTask};
use image::RgbImage;

impl RecognitionBackend for Oxr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        let task = match task {
            RecognitionTask::Ocr => OxrTask::Text,
            RecognitionTask::Formula => OxrTask::Formula,
            RecognitionTask::Table => OxrTask::Table,
            RecognitionTask::Chart => {
                return Err(Error::invalid_input("OXR has no chart-recognition prompt"));
            }
        };
        let options = self
            .generation_options
            .clone()
            .with_max_new_tokens(max_tokens);
        let trace = self.generate_one(&image, task.prompt(), &options, false)?;
        // Task formatting is owned by DocParser and the shared render helpers.
        self.decode_tokens(&trace.tokens)
            .map(|text| text.trim().to_string())
    }

    fn recognition_task_for_label(
        &self,
        label: &str,
        suggested: Option<RecognitionTask>,
    ) -> Option<RecognitionTask> {
        match label.to_ascii_lowercase().as_str() {
            "seal" | "stamp" | "code" | "code_block" | "code block" | "chemical_structure"
            | "chemical structure" => Some(RecognitionTask::Ocr),
            "picture" => None,
            _ => suggested,
        }
    }

    fn recognize_labeled_batch_with_options(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        labels: &[&str],
        options: &crate::GenerationOptions,
    ) -> crate::BatchResult<String> {
        if images.len() != tasks.len() || labels.len() != tasks.len() {
            return Err(Error::invalid_input(
                "OXR image, label and task counts differ",
            ));
        }
        let generation = self
            .generation_options
            .clone()
            .with_max_new_tokens(options.max_new_tokens);
        Ok(images
            .into_iter()
            .zip(tasks)
            .zip(labels)
            .map(|((image, &task), &label)| {
                let task = task_for_label(label, task)?;
                let trace = self.generate_one(&image, task.prompt(), &generation, false)?;
                self.decode_tokens(&trace.tokens)
            })
            .collect())
    }

    fn predict_reading_order(
        &self,
        image: &RgbImage,
        regions: &[crate::document::structure::LayoutElement],
        max_tokens: usize,
    ) -> Result<Option<Vec<usize>>, Error> {
        if self.use_reading_order {
            self.reading_order(image, regions, max_tokens).map(Some)
        } else {
            Ok(None)
        }
    }

    fn merge_adjacent_text_regions(&self) -> bool {
        false
    }

    fn postprocess_region_output(
        &self,
        label: &str,
        task: RecognitionTask,
        raw: &str,
    ) -> Option<String> {
        Some(format_region(label, task, raw))
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_chart: false,
            // Repeated source rows/lines are valid; never delete them with
            // a generic transcript repetition heuristic.
            truncate_repetitive_output: false,
            ..BackendCapabilities::default()
        }
    }
}

fn task_for_label(label: &str, task: RecognitionTask) -> Result<OxrTask, Error> {
    match label.to_ascii_lowercase().as_str() {
        "code" | "code_block" | "code block" | "algorithm" => Ok(OxrTask::Code),
        "chemical_structure" | "chemical structure" => Ok(OxrTask::ChemicalStructure),
        _ => match task {
            RecognitionTask::Ocr => Ok(OxrTask::Text),
            RecognitionTask::Formula => Ok(OxrTask::Formula),
            RecognitionTask::Table => Ok(OxrTask::Table),
            RecognitionTask::Chart => {
                Err(Error::invalid_input("OXR has no chart-recognition prompt"))
            }
        },
    }
}

fn strip_code_fence(raw: &str) -> &str {
    let raw = raw.trim_matches(['\r', '\n']).trim_end();
    if let Some((_, rest)) = raw.split_once("```") {
        let body = if let Some((content, _)) = rest.split_once("```")
            && !content.contains('\n')
        {
            content
        } else if let Some((_, body)) = rest.split_once('\n') {
            body.split_once("\n```")
                .map_or(body, |(content, _)| content)
        } else {
            ""
        };
        return body.trim_matches(['\r', '\n']).trim_end();
    }
    raw
}

fn strip_math(raw: &str) -> &str {
    let mut text = raw.trim();
    for (left, right) in [("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")] {
        if let Some(inner) = text.strip_prefix(left).and_then(|s| s.strip_suffix(right)) {
            text = inner.trim();
        }
    }
    text
}

fn format_region(label: &str, task: RecognitionTask, raw: &str) -> String {
    match task_for_label(label, task) {
        Ok(OxrTask::Code) => {
            let body = strip_code_fence(raw);
            if body.is_empty() {
                String::new()
            } else {
                format!("```\n{body}\n```")
            }
        }
        Ok(OxrTask::ChemicalStructure) => {
            let body = strip_math(strip_code_fence(raw));
            let body = body
                .strip_prefix(r"\smiles{")
                .and_then(|s| s.strip_suffix('}'))
                .unwrap_or(body)
                .trim();
            if body.is_empty() {
                String::new()
            } else {
                format!("$\\smiles{{{body}}}$")
            }
        }
        Ok(OxrTask::Formula) => {
            let body = strip_math(strip_code_fence(raw));
            if body.is_empty() {
                String::new()
            } else {
                format!("$$\n{body}\n$$")
            }
        }
        Ok(OxrTask::Table) => raw.trim().to_string(),
        _ => {
            let mut text = String::with_capacity(raw.len());
            for ch in raw.trim().chars() {
                if ch != '\n' || !text.ends_with('\n') {
                    text.push(ch);
                }
            }
            text
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_tasks_preserve_code_indentation_and_chemical_notation() {
        assert_eq!(
            task_for_label("Code Block", RecognitionTask::Ocr).unwrap(),
            OxrTask::Code
        );
        assert_eq!(
            task_for_label("chemical_structure", RecognitionTask::Ocr).unwrap(),
            OxrTask::ChemicalStructure
        );
        let code = "```python\ndef f():\n    return a_____b....c\n```";
        assert_eq!(
            format_region("code", RecognitionTask::Ocr, code),
            "```\ndef f():\n    return a_____b....c\n```"
        );
        assert_eq!(
            format_region("code", RecognitionTask::Ocr, "```\n    return x\n```"),
            "```\n    return x\n```"
        );
        assert_eq!(
            format_region("code", RecognitionTask::Ocr, "```code```"),
            "```\ncode\n```"
        );
        assert_eq!(
            format_region(
                "code",
                RecognitionTask::Ocr,
                "```python\ndef f():\n    return 1"
            ),
            "```\ndef f():\n    return 1\n```"
        );
        assert!(format_region("code", RecognitionTask::Ocr, "```python").is_empty());
        assert_eq!(
            format_region(
                "Chemical Structure",
                RecognitionTask::Ocr,
                r"$$\smiles{C1=CC=CC=C1}$$"
            ),
            r"$\smiles{C1=CC=CC=C1}$"
        );
        assert_eq!(
            format_region("formula", RecognitionTask::Formula, "\\[a\n+b\\]"),
            "$$\na\n+b\n$$"
        );
        assert_eq!(
            format_region("text", RecognitionTask::Ocr, "A\n\nB  C"),
            "A\nB  C"
        );
    }
}
