"""
Hieroglyph OCR — End-to-End Web Demo (YOLO26n + TrOCR)
تطبيق الواجهة التفاعلية لمنصة Hugging Face Spaces
مرتبط مباشرة بمستودع نماذجك: Bin-Amer/hieroglyph-ocr-models
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import cv2
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

# ============================================================================
# 1. إعدادات المستودع وتحميل النماذج
# ============================================================================
MODELS_REPO = "Bin-Amer/hieroglyph-ocr-models"
YOLO_FILE_IN_REPO = "yolo/hieroglyph_yolo26n_synthetic_real_v2.pt"

# قائمة الـ 171 كود جاردينر المعتمدة في المشروع
KNOWN_GARDINER_CODES = {
    "A55", "AA15", "AA26", "AA27", "AA28", "D1", "D2", "D4", "D10", "D19",
    "D21", "D28", "D34", "D35", "D36", "D39", "D46", "D52", "D53", "D54",
    "D56", "D58", "D60", "D62", "D156", "E1", "E9", "E17", "E23", "E34",
    "F4", "F9", "F12", "F13", "F16", "F18", "F21", "F22", "F23", "F26",
    "F29", "F30", "F31", "F32", "F34", "F35", "F40", "G1", "G4", "G5",
    "G7", "G10", "G14", "G17", "G21", "G25", "G26", "G29", "G35", "G36",
    "G37", "G39", "G40", "G43", "G50", "H6", "I5", "I9", "I10", "L1",
    "M1", "M3", "M4", "M8", "M12", "M16", "M17", "M18", "M20", "M23",
    "M26", "M29", "M40", "M41", "M42", "M44", "M195", "N1", "N2", "N5",
    "N14", "N16", "N17", "N18", "N19", "N24", "N25", "N26", "N29", "N30",
    "N31", "N35", "N36", "N37", "N41", "O1", "O4", "O11", "O28", "O29",
    "O31", "O34", "O49", "O50", "O51", "P1", "P3", "P6", "P8", "P98",
    "Q1", "Q3", "Q7", "R4", "R8", "S24", "S28", "S29", "S34", "S42",
    "T14", "T20", "T21", "T22", "T28", "T30", "U1", "U7", "U15", "U28",
    "U33", "U35", "V4", "V6", "V7", "V13", "V16", "V22", "V24", "V25",
    "V28", "V30", "V31", "W11", "W14", "W15", "W18", "W19", "W22", "W24",
    "W25", "X1", "X6", "X8", "Y1", "Y2", "Y3", "Y5", "Z1", "Z7", "Z11",
}

DEVICE = 0 if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """تحميل نموذج YOLO26n ونموذج TrOCR تلقائياً من المستودع."""
    print("⏳ جاري تنزيل وتحميل النماذج من Hugging Face Hub...")
    token = os.getenv("HF_TOKEN", None)

    # تحميل كاشف YOLO
    yolo_weights = hf_hub_download(
        repo_id=MODELS_REPO,
        filename=YOLO_FILE_IN_REPO,
        token=token,
    )
    yolo = YOLO(yolo_weights)

    # تحميل قارئ TrOCR
    proc = TrOCRProcessor.from_pretrained(
        MODELS_REPO, subfolder="trocr_gardiner_v2", token=token
    )
    trocr = VisionEncoderDecoderModel.from_pretrained(
        MODELS_REPO, subfolder="trocr_gardiner_v2", token=token
    ).to(TORCH_DEVICE)
    trocr.config.use_cache = True
    trocr.eval()

    print("✅ تم تحميل النموذجين بنجاح!")
    return yolo, proc, trocr


YOLO_MODEL = None
PROCESSOR = None
TROCR_MODEL = None


def ensure_models_loaded():
    global YOLO_MODEL, PROCESSOR, TROCR_MODEL
    if YOLO_MODEL is None or PROCESSOR is None or TROCR_MODEL is None:
        YOLO_MODEL, PROCESSOR, TROCR_MODEL = load_models()


# ============================================================================
# 2. دوال المعالجة البصرية والقص والترتيب المكاني والقراءة
# ============================================================================
def mild_clahe_bgr(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    return cv2.addWeighted(image_bgr, 0.65, enhanced, 0.35, 0)


def safe_crop(
    image_bgr: np.ndarray, box: np.ndarray, padding_ratio: float = 0.06
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    pad_x = max(1.0, x2 - x1) * padding_ratio
    pad_y = max(1.0, y2 - y1) * padding_ratio
    x1_i = max(0, int(round(x1 - pad_x)))
    y1_i = max(0, int(round(y1 - pad_y)))
    x2_i = min(width, int(round(x2 + pad_x)))
    y2_i = min(height, int(round(y2 + pad_y)))
    return image_bgr[y1_i:y2_i, x1_i:x2_i].copy()


def pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(
        image, ((side - image.width) // 2, (side - image.height) // 2)
    )
    return canvas


def recognize_crop(crop_bgr: np.ndarray) -> str:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = pad_to_square(Image.fromarray(crop_rgb))
    pixel_values = PROCESSOR(
        images=image, return_tensors="pt"
    ).pixel_values.to(TORCH_DEVICE)
    with torch.no_grad():
        generated = TROCR_MODEL.generate(
            pixel_values, max_length=12, num_beams=2
        )
    return PROCESSOR.batch_decode(generated, skip_special_tokens=True)[
        0
    ].strip()


def spatial_order(boxes: np.ndarray) -> list[int]:
    if len(boxes) == 0:
        return []
    heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    row_unit = max(1.0, float(np.median(heights)) * 0.75)
    return sorted(
        range(len(boxes)),
        key=lambda index: (
            round(((boxes[index, 1] + boxes[index, 3]) / 2.0) / row_unit),
            (boxes[index, 0] + boxes[index, 2]) / 2.0,
        ),
    )


# ============================================================================
# 3. خط الأنابيب المتكامل
# ============================================================================
def run_pipeline(
    input_image: Image.Image,
    confidence: float,
    image_size: int,
    apply_enhancement: bool,
):
    if input_image is None:
        return None, "⚠️ يرجى رفع صورة أو التقاط صورة بالكاميرا أولاً.", [], "", None

    ensure_models_loaded()

    original_rgb = np.array(input_image.convert("RGB"))
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    inference_bgr = (
        mild_clahe_bgr(original_bgr)
        if apply_enhancement
        else original_bgr.copy()
    )

    result = YOLO_MODEL.predict(
        source=inference_bgr,
        imgsz=int(image_size),
        conf=float(confidence),
        iou=0.50,
        max_det=1000,
        device=DEVICE,
        verbose=False,
    )[0]

    boxes = (
        result.boxes.xyxy.cpu().numpy()
        if result.boxes is not None
        else np.empty((0, 4))
    )
    confidences = (
        result.boxes.conf.cpu().numpy()
        if result.boxes is not None
        else np.empty((0,))
    )
    order = spatial_order(boxes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_root = Path(tempfile.gettempdir()) / "hieroglyph_ocr_runs"
    run_dir = temp_root / f"run_{timestamp}"
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rendered = original_bgr.copy()
    gallery, records, codes = [], [], []

    for output_index, box_index in enumerate(order, start=1):
        box = boxes[box_index]
        score = float(confidences[box_index])
        crop = safe_crop(original_bgr, box)
        if crop.size == 0:
            continue

        code_text = recognize_crop(crop)
        valid = code_text in KNOWN_GARDINER_CODES
        codes.append(code_text)

        safe_code = (
            re.sub(r"[^A-Za-z0-9_-]+", "_", code_text).strip("_")[:24]
            or "EMPTY"
        )
        crop_name = f"glyph_{output_index:04d}_{safe_code}_{score:.3f}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        gallery.append(
            (
                Image.fromarray(crop_rgb),
                f"{output_index}: {code_text} | {score:.2f}",
            )
        )

        x1, y1, x2, y2 = box.astype(int)
        color = (0, 180, 0) if valid else (0, 140, 255)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            rendered,
            code_text,
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        records.append(
            {
                "index": output_index,
                "gardiner_code": code_text,
                "valid_known_code": valid,
                "detection_confidence": round(score, 6),
                "xyxy": [round(float(value), 2) for value in box],
                "crop": f"crops/{crop_name}",
            }
        )

    cv2.imwrite(str(run_dir / "input_original.jpg"), original_bgr)
    if apply_enhancement:
        cv2.imwrite(
            str(run_dir / "input_enhanced_for_inference.jpg"), inference_bgr
        )
    cv2.imwrite(str(run_dir / "result.jpg"), rendered)

    summary = {
        "models_repo": MODELS_REPO,
        "confidence": float(confidence),
        "image_size": int(image_size),
        "opencv_clahe": bool(apply_enhancement),
        "detections": len(records),
        "spatial_codes": codes,
        "reading_order_note": (
            "ترتيب مكاني تقريبي (أعلى إلى أسفل ثم يسار إلى يمين) وليس ترتيباً دلالياً."
        ),
        "glyphs": records,
    }
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = Path(
        shutil.make_archive(
            str(temp_root / f"hieroglyph_ocr_{timestamp}"),
            "zip",
            root_dir=run_dir,
        )
    )

    mean_conf = float(np.mean(confidences)) if len(confidences) else 0.0
    valid_count = sum(record["valid_known_code"] for record in records)
    stats = f"""
### نتيجة التحليل | Analysis Summary

- **الرموز المكتشفة (Detected Glyphs):** `{len(records)}`
- **الأكواد المطابقة لقائمة Gardiner (171 فئة):** `{valid_count}/{len(records)}`
- **متوسط ثقة الكاشف (Mean YOLO Confidence):** `{mean_conf:.2f}`
- **تحسين التباين (OpenCV CLAHE):** `{"مفعّل" if apply_enhancement else "غير مفعّل"}`
- **ملاحظة علمية:** الترتيب المعروض مكاني تقريبي، والنتيجة تحدد أكواد جاردينر القياسية وليست ترجمة دلالية.
"""
    rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
    return (
        Image.fromarray(rendered_rgb),
        stats,
        gallery,
        " ".join(codes),
        str(zip_path),
    )


# ============================================================================
# 4. بناء واجهة Gradio
# ============================================================================
with gr.Blocks(title="Hieroglyph OCR — YOLO26 + TrOCR") as demo:
    gr.Markdown(
        """
        # 🏺 Hieroglyph OCR — YOLO26n + TrOCR (Gardiner Code Recognition)
        ارفع صورة لوحة حجرية أو بردية هيروغليفية (أو التقط صورة بالكاميرا).
        يقوم **YOLO26n** باكتشاف مواقع الرموز (`glyph`)، ثم يقوم **TrOCR** بقراءة كود **Gardiner** المقابل لكل رمز.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="صورة النقش أو البردية | Input Image",
                sources=["upload", "webcam", "clipboard"],
                type="pil",
                height=430,
            )
            confidence = gr.Slider(
                minimum=0.10,
                maximum=0.80,
                value=0.35,
                step=0.05,
                label="عتبة ثقة الكاشف | YOLO Confidence Threshold",
            )
            image_size = gr.Dropdown(
                choices=[640, 1024, 1280],
                value=1024,
                label="حجم الاستدلال | Inference Resolution",
            )
            enhancement = gr.Checkbox(
                value=False,
                label="تفعيل تحسين التباين الخفيف (OpenCV CLAHE) قبل الكشف",
            )

            with gr.Row():
                run_button = gr.Button("🔍 اكتشاف وقراءة الرموز", variant="primary")
                clear_button = gr.Button("🧹 مسح")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="الصناديق المكتشفة وأكواد Gardiner",
                height=520,
            )
            stats = gr.Markdown("النتائج الإحصائية ستظهر هنا بعد التحليل.")
            codes = gr.Textbox(
                label="تسلسل أكواد Gardiner (بالترتيب المكاني التقريبي)"
            )
            download = gr.File(
                label="تنزيل الحزمة الكاملة (الصورة + القصاصات + ملف JSON) بصيغة ZIP"
            )

    gallery = gr.Gallery(
        label="قصاصات الرموز المكتشفة | Cropped Glyphs",
        columns=6,
        rows=2,
        height=360,
        object_fit="contain",
    )

    run_button.click(
        fn=run_pipeline,
        inputs=[input_image, confidence, image_size, enhancement],
        outputs=[output_image, stats, gallery, codes, download],
        show_progress="full",
    )

    clear_button.click(
        fn=lambda: (
            None,
            None,
            "النتائج الإحصائية ستظهر هنا بعد التحليل.",
            [],
            "",
            None,
        ),
        inputs=[],
        outputs=[input_image, output_image, stats, gallery, codes, download],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), show_error=True)