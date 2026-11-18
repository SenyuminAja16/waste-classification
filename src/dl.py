"""Model Deep Learning.   >>> TITIK GANTI untuk DL (Jefri) <<<

Kontrak (dipakai oleh scripts/train_dl.py DAN src/predict.py, jadi harus konsisten):

    build_model(pretrained)        -> torch.nn.Module, keluaran = len(CLASSES) logit
    set_trainable(model, mode)     -> atur bagian model yang dilatih ("head" atau "finetune")
    to_rgb224(img_bgr)             -> uint8 RGB (DL_INPUT x DL_INPUT x 3)
    to_tensor(rgb)                 -> torch.Tensor float32 (3, H, W), sudah dinormalisasi

Default: MobileNetV2 (ImageNet), classifier diganti jadi 3 kelas, input 224x224.
Model disimpan sebagai state_dict (torch.save(model.state_dict(), path)).
"""
import cv2
import numpy as np

from config import CLASSES, DL_INPUT

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(pretrained: bool = True):
    import torch
    import torchvision
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    m = torchvision.models.mobilenet_v2(weights=weights)
    m.classifier[1] = torch.nn.Linear(m.last_channel, len(CLASSES))
    return m


def set_trainable(model, mode: str, last_blocks: int = 3):
    """mode 'head'    : hanya classifier yang dilatih.
       mode 'finetune': classifier + `last_blocks` blok terakhir backbone."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if mode == "finetune":
        for blk in model.features[-last_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True


def to_rgb224(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(img_bgr, (DL_INPUT, DL_INPUT), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


def to_tensor(rgb: np.ndarray):
    import torch
    x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
