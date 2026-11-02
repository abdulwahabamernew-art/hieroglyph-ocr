"""End-to-end detection and Gardiner-code recognition pipeline."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .config import KNOWN_GARDINER_CODES, MODELS_REPO
from .models import ModelBundle
from .processing import mild_clahe_bgr, pad_to_square, safe_crop, spatial_order


def recognize_crop(bundle: ModelBundle, crop_bgr: np.ndarray) -> str:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = pad_to_square(Image.fromarray(crop_rgb))
    pixel_values = bundle.processor(images=image, return_tensors="pt").pixel_values.to(bundle.device)
    with torch.no_grad():
        generated = bundle.recognizer.generate(pixel_values, max_length=12, num_beams=2)
    return bundle.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run_pipeline(bundle: ModelBundle, input_image: Image.Image, confidence: float, image_size: int, apply_enhancement: bool):
    if input_image is None:
        return None, "يرجى رفع صورة أو التقاط صورة بالكاميرا أولًا.", [], "", None

    original_rgb = np.array(input_image.convert("RGB"))
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    inference_bgr = mild_clahe_bgr(original_bgr) if apply_enhancement else original_bgr.copy()
    yolo_device = 0 if bundle.device == "cuda" else "cpu"
    result = bundle.detector.predict(
        source=inference_bgr, imgsz=int(image_size), conf=float(confidence), iou=0.50,
        max_det=1000, device=yolo_device, verbose=False,
    )[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,))
    order = spatial_order(boxes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(tempfile.gettempdir()) / "hieroglyph_ocr_runs" / f"run_{timestamp}"
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rendered = original_bgr.copy()
    gallery, records, codes = [], [], []

    for output_index, box_index in enumerate(order, start=1):
        box, score = boxes[box_index], float(confidences[box_index])
        crop = safe_crop(original_bgr, box)
        if crop.size == 0:
            continue
        code_text = recognize_crop(bundle, crop)
        valid = code_text in KNOWN_GARDINER_CODES
        codes.append(code_text)
        safe_code = re.sub(r"[^A-Za-z0-9_-]+", "_", code_text).strip("_")[:24] or "EMPTY"
        crop_name = f"glyph_{output_index:04d}_{safe_code}_{score:.3f}.jpg"
        cv2.imwrite(str(crops_dir / crop_name), crop)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        gallery.append((Image.fromarray(crop_rgb), f"{output_index}: {code_text} | {score:.2f}"))
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 180, 0) if valid else (0, 140, 255)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        cv2.putText(rendered, code_text, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        records.append({
            "index": output_index, "gardiner_code": code_text, "valid_known_code": valid,
            "detection_confidence": round(score, 6),
            "xyxy": [round(float(value), 2) for value in box], "crop": f"crops/{crop_name}",
        })

    cv2.imwrite(str(run_dir / "input_original.jpg"), original_bgr)
    if apply_enhancement:
        cv2.imwrite(str(run_dir / "input_enhanced_for_inference.jpg"), inference_bgr)
    cv2.imwrite(str(run_dir / "result.jpg"), rendered)
    summary = {
        "models_repo": MODELS_REPO, "confidence": float(confidence), "image_size": int(image_size),
        "opencv_clahe": bool(apply_enhancement), "detections": len(records), "spatial_codes": codes,
        "reading_order_note": "Spatial order only; this is not a semantic translation.", "glyphs": records,
    }
    (run_dir / "result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = Path(shutil.make_archive(str(run_dir.parent / f"hieroglyph_ocr_{timestamp}"), "zip", root_dir=run_dir))
    mean_conf = float(np.mean(confidences)) if len(confidences) else 0.0
    valid_count = sum(record["valid_known_code"] for record in records)
    stats = (
        "### Analysis Summary\n\n"
        f"- Detected glyphs: `{len(records)}`\n"
        f"- Known Gardiner codes: `{valid_count}/{len(records)}`\n"
        f"- Mean YOLO confidence: `{mean_conf:.2f}`\n"
        f"- CLAHE enhancement: `{'on' if apply_enhancement else 'off'}`\n\n"
        "The displayed order is approximate spatial order, not a semantic translation."
    )
    return Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)), stats, gallery, " ".join(codes), str(zip_path)
