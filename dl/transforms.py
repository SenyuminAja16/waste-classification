"""Resize, padding, dan normalisasi gambar untuk training serta demo."""

import cv2

from core.config import DL_INPUT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb224(img_bgr):
    """Pertahankan rasio aspek dengan padding hitam setelah resize ke RGB."""
    height, width = img_bgr.shape[:2]
    scale = DL_INPUT / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.cvtColor(cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    pad_x, pad_y = DL_INPUT - size[0], DL_INPUT - size[1]
    left, top = pad_x // 2, pad_y // 2
    return cv2.copyMakeBorder(rgb, top, pad_y - top, left, pad_x - left, cv2.BORDER_CONSTANT, value=0)


def to_tensor(rgb):
    from torchvision.transforms.functional import normalize, to_tensor

    return normalize(to_tensor(rgb), IMAGENET_MEAN, IMAGENET_STD)
