# Page JSON Format

The page JSON is the shared, versioned interchange format for one parsed document page. Both pipelines emit it — the classic ONNX pipeline through `StructureResult::to_json` and the vision-language pipeline through `PageDocument::to_json` — so downstream consumers (CLIs, evaluation tools, post-processors) get one shape regardless of which models produced the page. It sits next to the existing `to_markdown` and `to_html` exporters; the classic `StructureResult::to_json_value` is deprecated in its favor (the raw internal result remains available through `serde_json::to_value`).

## Schema

```json
{
  "version": 1,
  "pipeline": "classic",
  "page": {
    "index": 0,
    "width": 720,
    "height": 1150,
    "angle": null
  },
  "blocks": [
    {
      "type": "doc_title",
      "label": "Title",
      "bbox": [72.0, 64.0, 360.0, 96.0],
      "order": 1,
      "confidence": 0.98,
      "angle": null,
      "content": "A Document Title"
    }
  ],
  "markdown": "# …",
  "raw_output": null,
  "diagnostics": []
}
```

- `version` is the schema version; consumers should reject versions they do not know. It is `1`.
- `pipeline` is `"classic"` for the ONNX structure pipeline and `"vl"` for the vision-language parsers.
- `page.width` and `page.height` are the pixel dimensions of the image the bounding boxes are expressed in, as passed by the caller. For the classic pipeline that is the input image (orientation correction maps boxes back to it), or the rectified image when document rectification (UVDoc) is enabled, because rectification cannot be inverted. `page.index` is the page's position in a batch when the pipeline knows it, else `null`; `page.angle` is the detected document orientation in degrees when the pipeline corrected one, else `null`.
- `blocks` lists the page in reading order. Each block has:
  - `type`: the canonical label from the PP-StructureV3 vocabulary (`doc_title`, `paragraph_title`, `text`, `content`, `abstract`, `image`, `table`, `chart`, `formula`, `figure_title`, `table_title`, `chart_title`, `figure_table_chart_title`, `header`, `header_image`, `footer`, `footer_image`, `footnote`, `seal`, `number`, `reference`, `reference_content`, `algorithm`, `formula_number`, `aside_text`, `list`, `region`, `other`). Built-in model labels with a canonical equivalent are mapped (for example MinerU's `page_number` → `number`, `equation_block` → `formula`, `table_caption` → `table_title`); a label the vocabulary does not know is passed through verbatim by both pipelines.
  - `label`: the original model label, present only when it differs from `type` (after the canonical mapping). It is omitted when the two agree.
  - `bbox`: `[x1, y1, x2, y2]` in image pixels.
  - `order`: the 1-based reading-order position when the pipeline assigns explicit order indices (headers, footers, and other auxiliary elements may have `null`); the classic pipeline reports its explicit indices, the VL pipeline numbers blocks sequentially since it stores them in reading order.
  - `confidence`: the detection confidence when the pipeline reports one, else `null`.
  - `angle`: a per-block rotation in degrees applied before recognition when the pipeline reports one, else `null`.
  - `content`: the recognized content — plain text for text blocks, HTML for tables, LaTeX for formulas — or `null` when the block was not recognized. The classic pipeline takes table content from the paired table result's HTML in plain form (no border attribute or centering wrapper), because table stitching leaves the element text empty, and formula content from the paired formula result, because inline formulas have their element text cleared.
- `markdown` is the pipeline-rendered Markdown when one was produced, else `null`: the classic pipeline fills it with its `to_markdown` rendering and the VL pipeline with the document's Markdown. Markdown rendering stays pipeline-specific; the two renderers may differ in details (for example title-level inference), and the page JSON does not normalize that.
- `raw_output` carries the parser's raw transcript only when a VL parser produced neither blocks nor Markdown (for example MonkeyOCRv2 and HPD-Parsing), so such pages are not exported empty; otherwise it is `null`, and it is always `null` for the classic pipeline.
- `diagnostics` lists non-fatal parse issues (`block_index`, `stage`, `message`); the classic pipeline currently reports none and emits an empty array.

Structured table and formula detail that the classic pipeline recognizes (cells, per-cell text, structure confidences) is not duplicated into blocks; it remains available by serializing the `StructureResult` itself (`serde_json::to_value`). Block content for tables carries the table HTML, and for formulas the LaTeX.

## Worked example

The example below is the golden fixture both crates test against: the same page emitted by either pipeline must produce these fields (except `pipeline`, `page.index`, `confidence`, and `order`, which depend on what each pipeline knows). Given a 1000×2000 image with one title block (`"Title"` label, pixel box `[100, 200, 900, 400]`, confidence `0.875`, reading order `1`) containing `"Hello"`, the classic golden additionally covers a table block — pairing the layout element with its table result to carry the plain HTML — which the VL fixture does not need:

```json
{
  "version": 1,
  "pipeline": "classic",
  "page": { "index": 0, "width": 1000, "height": 2000, "angle": null },
  "blocks": [
    {
      "type": "doc_title",
      "label": "Title",
      "bbox": [100.0, 200.0, 900.0, 400.0],
      "order": 1,
      "confidence": 0.875,
      "angle": null,
      "content": "Hello"
    },
    {
      "type": "table",
      "bbox": [100.0, 500.0, 900.0, 900.0],
      "order": 2,
      "confidence": 0.75,
      "angle": null,
      "content": "<table><tr><td>cell</td></tr></table>"
    }
  ],
  "markdown": "# Hello\n\n<div style=\"text-align: center;\"><table border=\"1\"><tr><td>cell</td></tr></table></div>",
  "raw_output": null,
  "diagnostics": []
}
```
