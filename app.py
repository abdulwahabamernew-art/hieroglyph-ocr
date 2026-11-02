"""Launch the Hieroglyph OCR Gradio demo."""

from hieroglyph_ocr.ui import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(theme="soft", show_error=True)
