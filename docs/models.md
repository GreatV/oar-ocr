# Pre-trained Models

OAROCR provides pre-trained models for OCR and document understanding tasks. Download them manually from the [GitHub Releases](https://github.com/GreatV/oar-ocr/releases) page (linked in the tables below), or have the library fetch them on demand from ModelScope. See [Auto-download](#auto-download) at the bottom.

## Text Detection Models

Choose between mobile and server variants based on your needs:

- **Mobile**: Smaller, faster models suitable for real-time applications
- **Server**: Larger, more accurate models for high-precision requirements

| Version  | Category | Model File | Size | Description |
|----------|----------|------------|------|-------------|
| PP-OCRv4 | Mobile | [`pp-ocrv4_mobile_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_mobile_det.onnx) | 4.6 MiB | Mobile variant for real-time applications |
| PP-OCRv4 | Server | [`pp-ocrv4_server_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_det.onnx) | 108.2 MiB | Server variant for high-precision |
| PP-OCRv5 | Mobile | [`pp-ocrv5_mobile_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_det.onnx) | 4.6 MiB | Mobile variant for real-time applications |
| PP-OCRv5 | Server | [`pp-ocrv5_server_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_server_det.onnx) | 84.0 MiB | Server variant for high-precision |

## Text Recognition Models

### Chinese/General Models

| Version  | Category | Model File | Size | Description |
|----------|----------|------------|------|-------------|
| PP-OCRv3 | Mobile | [`pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv3_mobile_rec.onnx) | 10.2 MiB | Legacy mobile variant |
| PP-OCRv4 | Mobile | [`pp-ocrv4_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_mobile_rec.onnx) | 10.4 MiB | Mobile variant |
| PP-OCRv4 | Server | [`pp-ocrv4_server_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_rec.onnx) | 86.3 MiB | Server variant |
| PP-OCRv4 | Document | [`pp-ocrv4_server_rec_doc.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_rec_doc.onnx) | 90.5 MiB | Optimized for documents |
| PP-OCRv5 | Mobile | [`pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_rec.onnx) | 15.8 MiB | Mobile variant |
| PP-OCRv5 | Server | [`pp-ocrv5_server_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_server_rec.onnx) | 80.6 MiB | Server variant |
| SVTRv2 | Server | [`ch_svtrv2_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ch_svtrv2_rec.onnx) | 80.3 MiB | High accuracy variant |
| RepSVTR | Server | [`ch_repsvtr_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ch_repsvtr_rec.onnx) | 24.2 MiB | Balanced accuracy/speed |

### Language-Specific Models

| Version  | Language | Model File | Size | Description |
|----------|----------|------------|------|-------------|
| PP-OCRv3 | Arabic | [`arabic_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/arabic_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Arabic text recognition |
| PP-OCRv5 | Arabic | [`arabic_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/arabic_pp-ocrv5_mobile_rec.onnx) | 7.7 MiB | Arabic text recognition |
| PP-OCRv3 | Chinese Traditional | [`chinese_cht_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/chinese_cht_pp-ocrv3_mobile_rec.onnx) | 10.6 MiB | Traditional Chinese text recognition |
| PP-OCRv3 | Cyrillic | [`cyrillic_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/cyrillic_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Cyrillic script recognition |
| PP-OCRv5 | Cyrillic | [`cyrillic_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/cyrillic_pp-ocrv5_mobile_rec.onnx) | 7.7 MiB | Cyrillic script recognition |
| PP-OCRv3 | Devanagari | [`devanagari_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/devanagari_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Devanagari script recognition |
| PP-OCRv5 | Devanagari | [`devanagari_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/devanagari_pp-ocrv5_mobile_rec.onnx) | 7.6 MiB | Devanagari script recognition |
| PP-OCRv5 | Greek | [`el_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/el_pp-ocrv5_mobile_rec.onnx) | 7.5 MiB | Greek text recognition |
| PP-OCRv3 | English | [`en_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/en_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | English text recognition |
| PP-OCRv4 | English | [`en_pp-ocrv4_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/en_pp-ocrv4_mobile_rec.onnx) | 7.4 MiB | English text recognition |
| PP-OCRv5 | English | [`en_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/en_pp-ocrv5_mobile_rec.onnx) | 7.5 MiB | English text recognition |
| PP-OCRv5 | Eastern Slavic | [`eslav_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/eslav_pp-ocrv5_mobile_rec.onnx) | 7.5 MiB | Eastern Slavic languages |
| PP-OCRv3 | Japanese | [`japan_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/japan_pp-ocrv3_mobile_rec.onnx) | 9.6 MiB | Japanese text recognition |
| PP-OCRv3 | Georgian | [`ka_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ka_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Georgian text recognition |
| PP-OCRv3 | Korean | [`korean_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/korean_pp-ocrv3_mobile_rec.onnx) | 9.5 MiB | Korean text recognition |
| PP-OCRv5 | Korean | [`korean_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/korean_pp-ocrv5_mobile_rec.onnx) | 12.8 MiB | Korean text recognition |
| PP-OCRv3 | Latin | [`latin_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/latin_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Latin script recognition |
| PP-OCRv5 | Latin | [`latin_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/latin_pp-ocrv5_mobile_rec.onnx) | 7.7 MiB | Latin script recognition |
| PP-OCRv3 | Tamil | [`ta_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ta_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Tamil text recognition |
| PP-OCRv5 | Tamil | [`ta_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ta_pp-ocrv5_mobile_rec.onnx) | 7.5 MiB | Tamil text recognition |
| PP-OCRv3 | Telugu | [`te_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/te_pp-ocrv3_mobile_rec.onnx) | 8.6 MiB | Telugu text recognition |
| PP-OCRv5 | Telugu | [`te_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/te_pp-ocrv5_mobile_rec.onnx) | 7.6 MiB | Telugu text recognition |
| PP-OCRv5 | Thai | [`th_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/th_pp-ocrv5_mobile_rec.onnx) | 7.6 MiB | Thai text recognition |

## PP-OCRv6

PP-OCRv6 is the newest PP-OCR generation. The flat ONNX files and dictionaries are published in this project's [`v0.7.0` release](https://github.com/GreatV/oar-ocr/releases/tag/v0.7.0) and on ModelScope. The tables also link the original PaddlePaddle inference bundles.

> **Source / attribution.** Published by PaddlePaddle under the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) project (Apache-2.0).

### Detection

| Size | Auto-download ONNX | File size | Official bundle |
|------|--------------------|-----------|-----------------|
| tiny | [`pp-ocrv6_tiny_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_tiny_det.onnx) | 1.7 MiB | [`PP-OCRv6_tiny_det_onnx_infer.tar`](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_onnx_infer.tar) |
| small | [`pp-ocrv6_small_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_small_det.onnx) | 9.4 MiB | [`PP-OCRv6_small_det_onnx_infer.tar`](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_det_onnx_infer.tar) |
| medium | [`pp-ocrv6_medium_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_medium_det.onnx) | 59.2 MiB | [`PP-OCRv6_medium_det_onnx_infer.tar`](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_det_onnx_infer.tar) |

### Recognition

| Size | Auto-download ONNX | Dictionary | File size | Official bundle |
|------|--------------------|------------|-----------|-----------------|
| tiny | [`pp-ocrv6_tiny_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_tiny_rec.onnx) | [`ppocrv6_tiny_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/ppocrv6_tiny_dict.txt) (6904 characters) | 4.3 MiB | [`PP-OCRv6_tiny_rec_onnx_infer.tar`](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_tiny_rec_onnx_infer.tar) |
| small | [`pp-ocrv6_small_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_small_rec.onnx) | [`ppocrv6_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/ppocrv6_dict.txt) (18708 characters) | 20.2 MiB | [`PP-OCRv6_small_rec_onnx_infer.tar`](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_small_rec_onnx_infer.tar) |
| medium | [`pp-ocrv6_medium_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/pp-ocrv6_medium_rec.onnx) | [`ppocrv6_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/ppocrv6_dict.txt) (18708 characters) | 73.0 MiB | [`PP-OCRv6_medium_rec_onnx_infer.tar`](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_onnx_infer.tar) |

## Character Dictionaries

Character dictionaries are required for text recognition. Choose the appropriate dictionary for your model:

### General Dictionaries

| Version | File | Description |
|---------|------|-------------|
| PP-OCRv4 Document | [`ppocrv4_doc_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv4_doc_dict.txt) | For PP-OCRv4 document models |
| PP-OCRv5 | [`ppocrv5_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_dict.txt) | For PP-OCRv5 models |
| PP-OCRv6 Tiny | [`ppocrv6_tiny_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/ppocrv6_tiny_dict.txt) | For PP-OCRv6 tiny recognition |
| PP-OCRv6 Small/Medium | [`ppocrv6_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.7.0/ppocrv6_dict.txt) | For PP-OCRv6 small and medium recognition |
| PP-OCR Keys v1 | [`ppocr_keys_v1.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocr_keys_v1.txt) | For PP-OCRv3 general and PP-OCRv4 general models |

### PP-OCRv5 Language-Specific Dictionaries

| Language | File | Model Compatibility |
|----------|------|---------------------|
| Arabic | [`ppocrv5_arabic_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_arabic_dict.txt) | PP-OCRv5 Arabic |
| Cyrillic | [`ppocrv5_cyrillic_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_cyrillic_dict.txt) | PP-OCRv5 Cyrillic |
| Devanagari | [`ppocrv5_devanagari_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_devanagari_dict.txt) | PP-OCRv5 Devanagari |
| Greek | [`ppocrv5_el_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_el_dict.txt) | PP-OCRv5 Greek |
| English | [`ppocrv5_en_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_en_dict.txt) | PP-OCRv5 English |
| Eastern Slavic | [`ppocrv5_eslav_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_eslav_dict.txt) | PP-OCRv5 Eastern Slavic |
| Korean | [`ppocrv5_korean_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_korean_dict.txt) | PP-OCRv5 Korean |
| Latin | [`ppocrv5_latin_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_latin_dict.txt) | PP-OCRv5 Latin script |
| Tamil | [`ppocrv5_ta_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_ta_dict.txt) | PP-OCRv5 Tamil |
| Telugu | [`ppocrv5_te_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_te_dict.txt) | PP-OCRv5 Telugu |
| Thai | [`ppocrv5_th_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_th_dict.txt) | PP-OCRv5 Thai |

### PP-OCRv3 and PP-OCRv4 Language-Specific Dictionaries

The language-specific PP-OCRv3 dictionaries are different from the PP-OCRv5 dictionaries and are not part of the auto-download registry. Download them from PaddleOCR and pass the local path to the recognition builder. The PP-OCRv4 English checkpoint uses the same `en_dict.txt` file as PP-OCRv3 English.

| Language | Official dictionary | Model compatibility |
|----------|---------------------|---------------------|
| Arabic | [`arabic_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/arabic_dict.txt) | PP-OCRv3 Arabic |
| Traditional Chinese | [`chinese_cht_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/chinese_cht_dict.txt) | PP-OCRv3 Traditional Chinese |
| Cyrillic | [`cyrillic_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/cyrillic_dict.txt) | PP-OCRv3 Cyrillic |
| Devanagari | [`devanagari_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/devanagari_dict.txt) | PP-OCRv3 Devanagari |
| English | [`en_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/en_dict.txt) | PP-OCRv3 and PP-OCRv4 English |
| Japanese | [`japan_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/japan_dict.txt) | PP-OCRv3 Japanese |
| Georgian | [`ka_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ka_dict.txt) | PP-OCRv3 Georgian |
| Korean | [`korean_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/korean_dict.txt) | PP-OCRv3 Korean |
| Latin | [`latin_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/latin_dict.txt) | PP-OCRv3 Latin script |
| Tamil | [`ta_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ta_dict.txt) | PP-OCRv3 Tamil |
| Telugu | [`te_dict.txt`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/te_dict.txt) | PP-OCRv3 Telugu |

## Preprocessing Models

Models for document preprocessing and orientation detection:

| Type | Model File | Size | Description |
|------|------------|------|-------------|
| Document Orientation | [`pp-lcnet_x1_0_doc_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_doc_ori.onnx) | 6.5 MiB | Detect document rotation |
| Text Line Orientation (Light) | [`pp-lcnet_x0_25_textline_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x0_25_textline_ori.onnx) | 995 KiB | Fast text line orientation |
| Text Line Orientation | [`pp-lcnet_x1_0_textline_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_textline_ori.onnx) | 6.5 MiB | Accurate text line orientation |
| Document Rectification | [`uvdoc.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/uvdoc.onnx) | 30.2 MiB | Fix perspective distortion |

## Document Structure Models

Models for document structure analysis with `OARStructureBuilder`:

### Layout Detection

| Model | Model File | Size | Description |
|-------|------------|------|-------------|
| PicoDet-L 17cls | [`picodet-l_layout_17cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-l_layout_17cls.onnx) | 22.4 MiB | 17-class layout detection |
| PicoDet-L 3cls | [`picodet-l_layout_3cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-l_layout_3cls.onnx) | 22.4 MiB | 3-class layout detection |
| PicoDet-S 17cls | [`picodet-s_layout_17cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-s_layout_17cls.onnx) | 4.7 MiB | Fast 17-class layout |
| PicoDet-S 3cls | [`picodet-s_layout_3cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-s_layout_3cls.onnx) | 4.7 MiB | Fast 3-class layout |
| PicoDet 1x | [`picodet_layout_1x.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet_layout_1x.onnx) | 7.2 MiB | Legacy layout model |
| PicoDet 1x Table | [`picodet_layout_1x_table.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet_layout_1x_table.onnx) | 7.2 MiB | Table-focused layout |
| PP-DocLayout-S | [`pp-doclayout-s.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-s.onnx) | 4.7 MiB | Small variant |
| PP-DocLayout-M | [`pp-doclayout-m.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-m.onnx) | 22.4 MiB | Medium variant |
| PP-DocLayout-L | [`pp-doclayout-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-l.onnx) | 123.4 MiB | Large variant |
| PP-DocLayout_plus-L | [`pp-doclayout_plus-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout_plus-l.onnx) | 123.7 MiB | Enhanced large variant |
| PP-DocLayoutV2 | [`pp-doclayoutv2.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayoutv2.onnx) | 204.1 MiB | V2 with reading order (col, row) |
| PP-DocLayoutV3 | [`pp-doclayoutv3.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.6.0/pp-doclayoutv3.onnx) | 123.9 MiB | V3 with single order key |
| PP-DocBlockLayout | [`pp-docblocklayout.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-docblocklayout.onnx) | 123.3 MiB | Hierarchical ordering |
| RT-DETR-H 17cls | [`rt-detr-h_layout_17cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-h_layout_17cls.onnx) | 469.3 MiB | High accuracy 17-class |
| RT-DETR-H 3cls | [`rt-detr-h_layout_3cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-h_layout_3cls.onnx) | 469.2 MiB | High accuracy 3-class |

### Table Recognition

| Component | Model File | Size | Description |
|-----------|------------|------|-------------|
| Table Classification | [`pp-lcnet_x1_0_table_cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_table_cls.onnx) | 6.5 MiB | Wired vs wireless |
| Cell Detection (Wired) | [`rt-detr-l_wired_table_cell_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-l_wired_table_cell_det.onnx) | 123.3 MiB | RT-DETR for wired tables |
| Cell Detection (Wireless) | [`rt-detr-l_wireless_table_cell_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-l_wireless_table_cell_det.onnx) | 123.3 MiB | RT-DETR for wireless tables |
| Structure (SLANet) | [`slanet.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanet.onnx) | 7.4 MiB | Basic structure recognition |
| Structure (SLANet+) | [`slanet_plus.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanet_plus.onnx) | 7.4 MiB | Wireless table structure |
| Structure (SLANeXt Wired) | [`slanext_wired.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanext_wired.onnx) | 350.7 MiB | High accuracy wired structure |
| Structure (SLANeXt Wireless) | [`slanext_wireless.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanext_wireless.onnx) | 350.7 MiB | High accuracy wireless structure |
| Structure Dictionary | [`table_structure_dict_ch.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/table_structure_dict_ch.txt) | - | Required for structure recognition |

### Formula Recognition

| Model | Model File | Size | Description |
|-------|------------|------|-------------|
| PP-FormulaNet-S | [`pp-formulanet-s.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet-s.onnx) | 221.1 MiB | Small variant |
| PP-FormulaNet-L | [`pp-formulanet-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet-l.onnx) | 696.5 MiB | Large variant |
| PP-FormulaNet_plus-S | [`pp-formulanet_plus-s.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-s.onnx) | 221.1 MiB | Enhanced small variant |
| PP-FormulaNet_plus-M | [`pp-formulanet_plus-m.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-m.onnx) | 564.9 MiB | Enhanced medium variant |
| PP-FormulaNet_plus-L | [`pp-formulanet_plus-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-l.onnx) | 699.5 MiB | Enhanced large variant |
| PP-FormulaNet Tokenizer | [`pp-formulanet-tokenizer.json`](https://www.modelscope.cn/models/greatv/oar-ocr/file/view/master/pp-formulanet-tokenizer.json) | 2.0 MiB | Required for PP-FormulaNet variants |
| UniMERNet | [`unimernet.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet.onnx) | 1.7 GiB | Unified Math Expression Recognition |
| UniMERNet Tokenizer | [`unimernet_tokenizer.json`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet_tokenizer.json) | 2.0 MiB | Required for UniMERNet |

### Seal Text Detection

| Model | Model File | Size | Description |
|-------|------------|------|-------------|
| Seal Detection (Mobile) | [`pp-ocrv4_mobile_seal_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_mobile_seal_det.onnx) | 4.6 MiB | Fast seal detection |
| Seal Detection (Server) | [`pp-ocrv4_server_seal_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_seal_det.onnx) | 108.2 MiB | Accurate seal detection |

## Auto-download

```bash
cargo add oar-ocr --features auto-download
```

```rust,no_run
use oar_ocr::prelude::*;
let ocr = OAROCRBuilder::new(
    "pp-ocrv5_mobile_det.onnx",   // bare name resolved through the registry
    "pp-ocrv5_mobile_rec.onnx",
    "ppocrv5_dict.txt",
).build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

When the feature is enabled, registered file names are fetched from [`greatv/oar-ocr` on ModelScope](https://www.modelscope.cn/models/greatv/oar-ocr) into `$OAR_HOME` (default `~/.oar`) and verified against the expected SHA-256 before use. Subsequent runs reuse the cached copy. The bundled registry lives at [`oar_ocr::download::REGISTRY`](../oar-ocr-core/src/core/download/registry.rs).

### Path resolution rules

These rules apply only to *path* sources. Models passed as in-memory bytes (see [Loading Models from Memory](usage.md#loading-models-from-memory)) bypass path resolution entirely.

For each model path argument the builder applies these checks in order:

1. **Existing file wins.** If the path refers to a real file on disk it is used as-is — no registry lookup, no hash check, no network. A `./pp-ocrv5_mobile_det.onnx` next to the binary always shadows the registry.
2. **Only bare names or `$OAR_HOME`-rooted paths are eligible for auto-download.** A path is considered for registry resolution only when it has no parent component (e.g. `"pp-ocrv5_mobile_det.onnx"`) or when its parent equals the cache directory. Explicit paths like `./models/foo.onnx` or `/data/foo.onnx` are returned verbatim even if their file name is registered — the library never silently overrides an explicit path.
3. **Registry hit: cache or download.** If the file name appears in `REGISTRY`:
   - The cached copy is used without network access when `$OAR_HOME/<name>` exists with the expected size and SHA-256.
   - Otherwise, the file is downloaded from ModelScope, verified with SHA-256, and atomically replaced.
4. **Unregistered and missing.** The path is returned verbatim so the builder produces its normal "model not found" error.

| Input | On disk | Behaviour |
|---|---|---|
| `"pp-ocrv5_mobile_det.onnx"` | `./pp-ocrv5_mobile_det.onnx` exists | Use the local CWD file |
| `"pp-ocrv5_mobile_det.onnx"` | `$OAR_HOME/...` exists, hash OK | Use cached copy, no network |
| `"pp-ocrv5_mobile_det.onnx"` | absent or hash mismatch | Download to `$OAR_HOME`, verify, use |
| `"./models/det.onnx"` | absent | Returned as-is, resulting in "model not found" |
| `"$OAR_HOME/pp-ocrv5_mobile_det.onnx"` (absolute) | (any) | Handled like a bare name because its parent is the cache directory |

Note: the resolver compares paths verbatim — `~` is not expanded. Pass a bare filename, an absolute path under `$OAR_HOME`, or let your shell expand `~` for you.

### Cache layout

- Override the cache root with the `OAR_HOME` environment variable. Defaults to `~/.oar` (resolved via the platform home directory, while the literal `~` is not expanded by the library).
- Files land at `$OAR_HOME/<name>`, flat (no per-revision subdirectories).
- Downloads stream into a unique `$OAR_HOME/.<name>.<pid>.<n>.part` and are renamed atomically once the SHA-256 matches, so a crash mid-download won't poison the cache and concurrent processes don't clobber each other.
- After verification a `$OAR_HOME/.<name>.sha256` sidecar records the verified hash. Future loads with a matching cache file + sidecar skip the multi-second rehash. Deleting the sidecar forces a fresh hash check.
