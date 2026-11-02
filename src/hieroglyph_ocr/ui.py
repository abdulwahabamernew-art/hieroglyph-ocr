"""Gradio user interface for the OCR pipeline."""

from __future__ import annotations

import gradio as gr

from .models import load_models
from .pipeline import run_pipeline

try:
    import spaces
except ImportError:
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(function=None, **kwargs):
            if function is None:
                return lambda wrapped: wrapped
            return function


_BUNDLE = None


def _get_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_models()
    return _BUNDLE


@spaces.GPU
def infer(input_image, confidence, image_size, enhancement):
    if input_image is None:
        return None, "يرجى رفع صورة أو التقاط صورة بالكاميرا أولًا.", [], "", None
    return run_pipeline(_get_bundle(), input_image, confidence, int(image_size), enhancement)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hieroglyph OCR — YOLO26 + TrOCR") as demo:
        gr.Markdown(
            """# Hieroglyph OCR — YOLO26n + TrOCR

            ارفع صورة لوحة أو بردية هيروغليفية. يكتشف YOLO مواقع الرموز، ثم يقرأ TrOCR كود Gardiner المقابل.
            هذا المشروع يتعرف على الرموز والأكواد القياسية، ولا ينفذ ترجمة دلالية للغة المصرية القديمة.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input image", sources=["upload", "webcam", "clipboard"], type="pil", height=430)
                confidence = gr.Slider(0.10, 0.80, value=0.35, step=0.05, label="YOLO confidence threshold")
                image_size = gr.Dropdown([640, 1024, 1280], value=1024, label="Inference resolution")
                enhancement = gr.Checkbox(value=False, label="Apply mild CLAHE contrast enhancement")
                with gr.Row():
                    run_button = gr.Button("Detect and read glyphs", variant="primary")
                    clear_button = gr.Button("Clear")
            with gr.Column(scale=1):
                output_image = gr.Image(label="Detected boxes and Gardiner codes", height=520)
                stats = gr.Markdown("Results will appear here after inference.")
                codes = gr.Textbox(label="Gardiner codes in approximate spatial order")
                download = gr.File(label="Download ZIP results")
        gallery = gr.Gallery(label="Cropped glyphs", columns=6, rows=2, height=360, object_fit="contain")
        run_button.click(infer, [input_image, confidence, image_size, enhancement], [output_image, stats, gallery, codes, download], show_progress="full")
        clear_button.click(lambda: (None, None, "Results will appear here after inference.", [], "", None), [], [input_image, output_image, stats, gallery, codes, download])
    demo.queue(default_concurrency_limit=1)
    return demo
