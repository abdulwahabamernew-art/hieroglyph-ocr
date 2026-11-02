"""Lazy loading for the released YOLO and TrOCR models."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

from .config import MODELS_REPO, TROCR_SUBFOLDER, YOLO_FILE_IN_REPO


@dataclass
class ModelBundle:
    detector: YOLO
    processor: TrOCRProcessor
    recognizer: VisionEncoderDecoderModel
    device: str


def load_models() -> ModelBundle:
    """Download and initialize models from the Hugging Face model repository."""
    token = os.getenv("HF_TOKEN") or None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo_weights = hf_hub_download(
        repo_id=MODELS_REPO,
        filename=YOLO_FILE_IN_REPO,
        token=token,
    )
    detector = YOLO(yolo_weights)
    processor = TrOCRProcessor.from_pretrained(
        MODELS_REPO, subfolder=TROCR_SUBFOLDER, token=token
    )
    recognizer = VisionEncoderDecoderModel.from_pretrained(
        MODELS_REPO, subfolder=TROCR_SUBFOLDER, token=token
    ).to(device)
    recognizer.config.use_cache = True
    recognizer.eval()
    return ModelBundle(detector, processor, recognizer, device)
