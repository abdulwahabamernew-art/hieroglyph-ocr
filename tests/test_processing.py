import numpy as np
from PIL import Image

from hieroglyph_ocr.processing import pad_to_square, safe_crop, spatial_order


def test_spatial_order_reads_top_to_bottom_then_left_to_right():
    boxes = np.array([
        [100, 100, 140, 140],
        [10, 10, 50, 50],
        [70, 10, 110, 50],
    ], dtype=float)
    assert spatial_order(boxes) == [1, 2, 0]


def test_safe_crop_stays_inside_image():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = safe_crop(image, np.array([-10, -5, 10, 10]))
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
    assert crop.shape[0] <= 20
    assert crop.shape[1] <= 30


def test_pad_to_square_preserves_image_content_size():
    result = pad_to_square(Image.new("RGB", (20, 10), "white"))
    assert result.size == (20, 20)
