# Architecture

The project is split into four layers so that model loading, image processing, inference, and UI code can evolve independently.

`config.py` contains repository and class configuration. `models.py` lazily downloads the YOLO detector and TrOCR recognizer from Hugging Face. `processing.py` contains deterministic helpers that are easy to unit test. `pipeline.py` coordinates detection, crop preparation, recognition, visualization, and ZIP/JSON export. `ui.py` exposes the pipeline through Gradio and keeps model initialization lazy for Spaces.

The repository intentionally stores code and configuration only. Large weights remain in `Bin-Amer/hieroglyph-ocr-models` so GitHub clones stay small and versioned model artifacts have a clear home.
