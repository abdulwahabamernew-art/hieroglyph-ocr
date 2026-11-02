---
title: Hieroglyph OCR
emoji: 🏺
colorFrom: green
colorTo: pink
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# Hieroglyph OCR

End-to-end Egyptian hieroglyph detection and Gardiner-code recognition using YOLO and TrOCR, with a Gradio demo and a path toward offline/mobile inference.

مشروع متكامل لاكتشاف الرموز الهيروغليفية المصرية والتعرف على أكواد قائمة جاردينر باستخدام YOLO وTrOCR، مع واجهة Gradio ودعم مسار التشغيل على الهاتف.

[![CI](https://github.com/abdulwahabamernew-art/hieroglyph-ocr/actions/workflows/ci.yml/badge.svg)](https://github.com/abdulwahabamernew-art/hieroglyph-ocr/actions/workflows/ci.yml)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-model-yellow)](https://huggingface.co/Bin-Amer/hieroglyph-ocr-models)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Gradio-orange)](https://huggingface.co/spaces/Bin-Amer/hieroglyph-ocr-demo)

## Try it now | جرّبه الآن

- Live demo: [Hieroglyph OCR on Hugging Face Spaces](https://huggingface.co/spaces/Bin-Amer/hieroglyph-ocr-demo)
- Released model weights: [Bin-Amer/hieroglyph-ocr-models](https://huggingface.co/Bin-Amer/hieroglyph-ocr-models)

## What it does | ماذا يفعل؟

The pipeline detects glyph locations in a photograph or scan, crops each glyph, reads a Gardiner code, draws the result on the input image, and exports a ZIP containing the image, crops, and JSON metadata.

يقوم النظام باكتشاف مواقع الرموز، وقص كل رمز، وقراءة كود Gardiner، ورسم النتائج على الصورة، ثم تصدير حزمة ZIP تحتوي على الصورة والقصاصات وبيانات JSON.

This is symbol/code recognition, not a semantic translation of Ancient Egyptian text.

## Architecture

```text
Input image → YOLO detector → spatial ordering → crop/padding → TrOCR recognizer → annotated image + JSON/ZIP
```

The large model artifacts are intentionally hosted in the Hugging Face model repository instead of GitHub.

## Reported results

The current model card reports the following evaluation results:

| Component | Result |
|---|---:|
| YOLO mAP@50 | 78.21% |
| TrOCR locked-test accuracy on matched YOLO boxes | 92.12% |
| MobileNetV3 Top-3 locked-test accuracy | 92.64% |
| End-to-end locked-test accuracy on known classes | 64.70% |

See the [full model card](https://huggingface.co/Bin-Amer/hieroglyph-ocr-models) for methodology and limitations.

## Repository layout

```text
app.py                         # Gradio entry point
src/hieroglyph_ocr/
  config.py                    # Model IDs and supported classes
  models.py                    # Lazy model loading
  processing.py                # Pure image-processing helpers
  pipeline.py                  # Detection and recognition pipeline
  ui.py                        # Gradio interface
tests/                         # Lightweight unit tests
.github/                       # CI and issue templates
```

## Local setup

```bash
git clone https://github.com/abdulwahabamernew-art/hieroglyph-ocr.git
cd hieroglyph-ocr
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
python app.py
```

The first inference downloads the detector and TrOCR weights from Hugging Face. Set `HF_TOKEN` only when using a private model repository; do not commit tokens or `.env` files.

## Development

```bash
ruff check .
pytest -q
```

## Limitations and responsible use

The system recognizes a released set of Gardiner codes. Rare classes and images that differ significantly from the training distribution may produce low-confidence or incorrect predictions. Results should be reviewed by a qualified Egyptologist or domain expert before scholarly or archival use. Dataset and image licenses must be checked before redistribution.

## License

The code in this repository is released under the MIT License. Model weights and datasets may have their own terms; consult the [Hugging Face model card](https://huggingface.co/Bin-Amer/hieroglyph-ocr-models).

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.
