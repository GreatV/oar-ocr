# Pre-trained Models

OAROCR provides pre-trained models for OCR and document understanding tasks. Download them from the [GitHub Releases](https://github.com/GreatV/oar-ocr/releases) page.

## Text Detection Models

Choose between mobile and server variants based on your needs:

- **Mobile**: Smaller, faster models suitable for real-time applications
- **Server**: Larger, more accurate models for high-precision requirements

| Version  | Category | Model File | Size | Description |
|----------|----------|------------|------|-------------|
| PP-OCRv4 | Mobile | [`pp-ocrv4_mobile_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_mobile_det.onnx) | 4.6MB | Mobile variant for real-time applications |
| PP-OCRv4 | Server | [`pp-ocrv4_server_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_det.onnx) | 108.2MB | Server variant for high-precision |
| PP-OCRv5 | Mobile | [`pp-ocrv5_mobile_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_det.onnx) | 4.6MB | Mobile variant for real-time applications |
| PP-OCRv5 | Server | [`pp-ocrv5_server_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_server_det.onnx) | 84.0MB | Server variant for high-precision |

## Text Recognition Models

### Chinese/General Models

| Version  | Category | Model File | Size | Description |
|----------|----------|------------|------|-------------|
| PP-OCRv3 | Mobile | [`pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv3_mobile_rec.onnx) | 10.2MB | Legacy mobile variant |
| PP-OCRv4 | Mobile | [`pp-ocrv4_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_mobile_rec.onnx) | 10.4MB | Mobile variant |
| PP-OCRv4 | Server | [`pp-ocrv4_server_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_rec.onnx) | 86.3MB | Server variant |
| PP-OCRv4 | Document | [`pp-ocrv4_server_rec_doc.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_rec_doc.onnx) | 90.5MB | Optimized for documents |
| PP-OCRv5 | Mobile | [`pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_mobile_rec.onnx) | 15.8MB | Mobile variant |
| PP-OCRv5 | Server | [`pp-ocrv5_server_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv5_server_rec.onnx) | 80.6MB | Server variant |
| SVTRv2 | Server | [`ch_svtrv2_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ch_svtrv2_rec.onnx) | 80.3MB | High accuracy variant |
| RepSVTR | Server | [`ch_repsvtr_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ch_repsvtr_rec.onnx) | 24.2MB | Balanced accuracy/speed |

### Language-Specific Models

| Version  | Language | Model File | Size | Description |
|----------|----------|------------|------|-------------|
| PP-OCRv3 | Arabic | [`arabic_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/arabic_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Arabic text recognition |
| PP-OCRv5 | Arabic | [`arabic_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/arabic_pp-ocrv5_mobile_rec.onnx) | 7.7MB | Arabic text recognition |
| PP-OCRv3 | Chinese Traditional | [`chinese_cht_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/chinese_cht_pp-ocrv3_mobile_rec.onnx) | 10.6MB | Traditional Chinese text recognition |
| PP-OCRv3 | Cyrillic | [`cyrillic_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/cyrillic_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Cyrillic script recognition |
| PP-OCRv5 | Cyrillic | [`cyrillic_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/cyrillic_pp-ocrv5_mobile_rec.onnx) | 7.7MB | Cyrillic script recognition |
| PP-OCRv3 | Devanagari | [`devanagari_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/devanagari_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Devanagari script recognition |
| PP-OCRv5 | Devanagari | [`devanagari_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/devanagari_pp-ocrv5_mobile_rec.onnx) | 7.6MB | Devanagari script recognition |
| PP-OCRv5 | Greek | [`el_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/el_pp-ocrv5_mobile_rec.onnx) | 7.5MB | Greek text recognition |
| PP-OCRv3 | English | [`en_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/en_pp-ocrv3_mobile_rec.onnx) | 8.6MB | English text recognition |
| PP-OCRv4 | English | [`en_pp-ocrv4_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/en_pp-ocrv4_mobile_rec.onnx) | 7.4MB | English text recognition |
| PP-OCRv5 | English | [`en_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/en_pp-ocrv5_mobile_rec.onnx) | 7.5MB | English text recognition |
| PP-OCRv5 | Eastern Slavic | [`eslav_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/eslav_pp-ocrv5_mobile_rec.onnx) | 7.5MB | Eastern Slavic languages |
| PP-OCRv3 | Japanese | [`japan_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/japan_pp-ocrv3_mobile_rec.onnx) | 9.6MB | Japanese text recognition |
| PP-OCRv3 | Georgian | [`ka_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ka_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Georgian text recognition |
| PP-OCRv3 | Korean | [`korean_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/korean_pp-ocrv3_mobile_rec.onnx) | 9.5MB | Korean text recognition |
| PP-OCRv5 | Korean | [`korean_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/korean_pp-ocrv5_mobile_rec.onnx) | 12.8MB | Korean text recognition |
| PP-OCRv3 | Latin | [`latin_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/latin_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Latin script recognition |
| PP-OCRv5 | Latin | [`latin_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/latin_pp-ocrv5_mobile_rec.onnx) | 7.7MB | Latin script recognition |
| PP-OCRv3 | Tamil | [`ta_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ta_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Tamil text recognition |
| PP-OCRv5 | Tamil | [`ta_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ta_pp-ocrv5_mobile_rec.onnx) | 7.5MB | Tamil text recognition |
| PP-OCRv3 | Telugu | [`te_pp-ocrv3_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/te_pp-ocrv3_mobile_rec.onnx) | 8.6MB | Telugu text recognition |
| PP-OCRv5 | Telugu | [`te_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/te_pp-ocrv5_mobile_rec.onnx) | 7.6MB | Telugu text recognition |
| PP-OCRv5 | Thai | [`th_pp-ocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/th_pp-ocrv5_mobile_rec.onnx) | 7.6MB | Thai text recognition |

## Character Dictionaries

Character dictionaries are required for text recognition. Choose the appropriate dictionary for your model:

### General Dictionaries

| Version | File | Description |
|---------|------|-------------|
| PP-OCRv4 Document | [`ppocrv4_doc_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv4_doc_dict.txt) | For PP-OCRv4 document models |
| PP-OCRv5 | [`ppocrv5_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocrv5_dict.txt) | For PP-OCRv5 models |
| PP-OCR Keys v1 | [`ppocr_keys_v1.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/ppocr_keys_v1.txt) | For older PP-OCR models |

### Language-Specific Dictionaries

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

## Preprocessing Models

Models for document preprocessing and orientation detection:

| Type | Model File | Size | Description |
|------|------------|------|-------------|
| Document Orientation | [`pp-lcnet_x1_0_doc_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_doc_ori.onnx) | 6.5MB | Detect document rotation |
| Text Line Orientation (Light) | [`pp-lcnet_x0_25_textline_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x0_25_textline_ori.onnx) | 995KB | Fast text line orientation |
| Text Line Orientation | [`pp-lcnet_x1_0_textline_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_textline_ori.onnx) | 6.5MB | Accurate text line orientation |
| Document Rectification | [`uvdoc.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/uvdoc.onnx) | 30.2MB | Fix perspective distortion |

## Document Structure Models

Models for document structure analysis with `OARStructureBuilder`:

### Layout Detection

| Model | Model File | Size | Description |
|-------|------------|------|-------------|
| PicoDet-L 17cls | [`picodet-l_layout_17cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-l_layout_17cls.onnx) | 22.4MB | 17-class layout detection |
| PicoDet-L 3cls | [`picodet-l_layout_3cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-l_layout_3cls.onnx) | 22.4MB | 3-class layout detection |
| PicoDet-S 17cls | [`picodet-s_layout_17cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-s_layout_17cls.onnx) | 4.7MB | Fast 17-class layout |
| PicoDet-S 3cls | [`picodet-s_layout_3cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet-s_layout_3cls.onnx) | 4.7MB | Fast 3-class layout |
| PicoDet 1x | [`picodet_layout_1x.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet_layout_1x.onnx) | 7.2MB | Legacy layout model |
| PicoDet 1x Table | [`picodet_layout_1x_table.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/picodet_layout_1x_table.onnx) | 7.2MB | Table-focused layout |
| PP-DocLayout-S | [`pp-doclayout-s.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-s.onnx) | 4.7MB | Small variant |
| PP-DocLayout-M | [`pp-doclayout-m.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-m.onnx) | 22.4MB | Medium variant |
| PP-DocLayout-L | [`pp-doclayout-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout-l.onnx) | 123.4MB | Large variant |
| PP-DocLayout_plus-L | [`pp-doclayout_plus-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayout_plus-l.onnx) | 123.7MB | Enhanced large variant |
| PP-DocLayoutV2 | [`pp-doclayoutv2.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-doclayoutv2.onnx) | 204.0MB | Latest version |
| PP-DocBlockLayout | [`pp-docblocklayout.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-docblocklayout.onnx) | 123.4MB | Hierarchical ordering |
| RT-DETR-H 17cls | [`rt-detr-h_layout_17cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-h_layout_17cls.onnx) | 469.2MB | High accuracy 17-class |
| RT-DETR-H 3cls | [`rt-detr-h_layout_3cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-h_layout_3cls.onnx) | 469.2MB | High accuracy 3-class |

### Table Recognition

| Component | Model File | Size | Description |
|-----------|------------|------|-------------|
| Table Classification | [`pp-lcnet_x1_0_table_cls.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_table_cls.onnx) | 6.5MB | Wired vs wireless |
| Cell Detection (Wired) | [`rt-detr-l_wired_table_cell_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-l_wired_table_cell_det.onnx) | 123.4MB | RT-DETR for wired tables |
| Cell Detection (Wireless) | [`rt-detr-l_wireless_table_cell_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/rt-detr-l_wireless_table_cell_det.onnx) | 123.4MB | RT-DETR for wireless tables |
| Structure (SLANet) | [`slanet.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanet.onnx) | 7.4MB | Basic structure recognition |
| Structure (SLANet+) | [`slanet_plus.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanet_plus.onnx) | 7.4MB | Wireless table structure |
| Structure (SLANeXt Wired) | [`slanext_wired.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanext_wired.onnx) | 350.7MB | High accuracy wired structure |
| Structure (SLANeXt Wireless) | [`slanext_wireless.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanext_wireless.onnx) | 350.7MB | High accuracy wireless structure |
| Structure Dictionary | [`table_structure_dict_ch.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/table_structure_dict_ch.txt) | - | Required for structure recognition |

### Formula Recognition

| Model | Model File | Size | Description |
|-------|------------|------|-------------|
| PP-FormulaNet-S | [`pp-formulanet-s.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet-s.onnx) | 221.1MB | Small variant |
| PP-FormulaNet-L | [`pp-formulanet-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet-l.onnx) | 696.7MB | Large variant |
| PP-FormulaNet_plus-S | [`pp-formulanet_plus-s.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-s.onnx) | 221.1MB | Enhanced small variant |
| PP-FormulaNet_plus-M | [`pp-formulanet_plus-m.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-m.onnx) | 565.0MB | Enhanced medium variant |
| PP-FormulaNet_plus-L | [`pp-formulanet_plus-l.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-l.onnx) | 699.7MB | Enhanced large variant |
| UniMERNet | [`unimernet.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet.onnx) | 1.7GB | Unified Math Expression Recognition |
| UniMERNet Tokenizer | [`unimernet_tokenizer.json`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet_tokenizer.json) | 2.0MB | Required for UniMERNet |
| LaTeX OCR | [`latex_ocr_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/latex_ocr_rec.onnx) | 97.8MB | LaTeX formula recognition |

### Seal Text Detection

| Model | Model File | Size | Description |
|-------|------------|------|-------------|
| Seal Detection (Mobile) | [`pp-ocrv4_mobile_seal_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_mobile_seal_det.onnx) | 4.6MB | Fast seal detection |
| Seal Detection (Server) | [`pp-ocrv4_server_seal_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-ocrv4_server_seal_det.onnx) | 108.2MB | Accurate seal detection |

## Recommended Configurations

### Fast Processing (Real-time)

```
Detection: pp-ocrv5_mobile_det.onnx
Recognition: pp-ocrv5_mobile_rec.onnx
Dictionary: ppocrv5_dict.txt
```

### High Accuracy

```
Detection: pp-ocrv5_server_det.onnx
Recognition: pp-ocrv5_server_rec.onnx
Dictionary: ppocrv5_dict.txt
```

### Document Processing

```
Detection: pp-ocrv4_server_det.onnx
Recognition: pp-ocrv4_server_rec_doc.onnx
Dictionary: ppocrv4_doc_dict.txt
Orientation: pp-lcnet_x1_0_doc_ori.onnx
Rectification: uvdoc.onnx
```

### Document Structure Analysis

```
Layout: pp-doclayout_plus-l.onnx
Table Classification: pp-lcnet_x1_0_table_cls.onnx
Table Structure (Wired): slanext_wired.onnx
Table Structure (Wireless): slanet_plus.onnx
Table Structure Dict: table_structure_dict_ch.txt
Formula: pp-formulanet_plus-l.onnx
```
