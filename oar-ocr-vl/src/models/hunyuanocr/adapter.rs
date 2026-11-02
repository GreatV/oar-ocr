//! Region-recognition adapter for HunyuanOCR.

use super::{HunyuanOcr, HunyuanOcrVersion, parser::clean_output};
use crate::api::error::{BatchResult, Error};
use crate::api::recognition::{RecognitionBackend, RecognitionTask};
use image::RgbImage;

fn prompt(task: RecognitionTask, version: HunyuanOcrVersion) -> &'static str {
    if version == HunyuanOcrVersion::V1_5 {
        // Official inference/utils/tasks.py fixed task wording.
        return match task {
            RecognitionTask::Ocr => "提取图中的文字。",
            RecognitionTask::Table => "把图中的表格解析为HTML。",
            RecognitionTask::Formula => "识别图片中的公式，用LaTeX格式表示。",
            RecognitionTask::Chart => {
                "解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。"
            }
        };
    }
    match task {
        // V1 has no published plain-region recipe; avoid its spotting prompt,
        // which requests coordinates in otherwise plain recognized text.
        RecognitionTask::Ocr => "Extract all text from the image.",
        RecognitionTask::Table => "Parse the table in the image into HTML.",
        RecognitionTask::Formula => {
            "Identify the formula in the image and represent it using LaTeX format."
        }
        RecognitionTask::Chart => {
            "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts."
        }
    }
}

impl RecognitionBackend for HunyuanOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        self.generate(&[image], &[prompt(task, self.version())], max_tokens)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_input("HunyuanOCR: no result returned"))?
            .map(|text| clean_output(&text, self.version()))
    }

    fn recognize_batch(
        &self,
        images: Vec<RgbImage>,
        tasks: &[RecognitionTask],
        max_tokens: usize,
    ) -> BatchResult<String> {
        let prompts: Vec<_> = tasks
            .iter()
            .map(|&task| prompt(task, self.version()))
            .collect();
        Ok(self
            .generate(&images, &prompts, max_tokens)?
            .into_iter()
            .map(|result| result.map(|text| clean_output(&text, self.version())))
            .collect())
    }

    fn recognition_batch_key(&self, image: &RgbImage, _task: RecognitionTask) -> u64 {
        HunyuanOcr::recognition_batch_key(self, image).unwrap_or(0)
    }
}
