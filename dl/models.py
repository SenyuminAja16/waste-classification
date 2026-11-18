"""Model Deep Learning.

>>> BAGIAN YANG DIISI SENDIRI (DL) <<<

Kontrak (dipakai scripts/train_dl.py DAN src/predict.py, jadi harus konsisten):

    build_model(pretrained: bool) -> torch.nn.Module
        keluaran: len(config.CLASSES) logit

    set_trainable(model, mode: str) -> None
        mode "head"     : hanya lapisan klasifikasi yang dilatih
        mode "finetune" : lapisan klasifikasi + beberapa blok terakhir backbone

    to_rgb224(img_bgr) -> np.ndarray uint8 RGB berukuran (config.DL_INPUT, config.DL_INPUT, 3)

    to_tensor(rgb) -> torch.Tensor float32 (3, H, W), sudah dinormalisasi

Model disimpan sebagai state_dict:  torch.save(model.state_dict(), path)

Rencana di proposal: MobileNetV2 (ImageNet) dengan transfer learning, input 224x224.
Butuh: pip install torch torchvision
"""


def build_model(pretrained: bool = True):
    raise NotImplementedError("Isi build_model() di src/dl.py")


def set_trainable(model, mode: str):
    raise NotImplementedError("Isi set_trainable() di src/dl.py")


def to_rgb224(img_bgr):
    raise NotImplementedError("Isi to_rgb224() di src/dl.py")


def to_tensor(rgb):
    raise NotImplementedError("Isi to_tensor() di src/dl.py")
