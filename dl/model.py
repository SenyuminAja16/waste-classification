"""MobileNetV2 dan pengaturan parameter untuk dua fase training."""

from core.config import CLASSES

FINETUNE_BLOCKS = 4


def build_model(pretrained: bool = True):
    from torch import nn
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def set_trainable(model, mode: str) -> None:
    """Latih classifier saja atau tambahkan beberapa blok terakhir backbone."""
    if mode not in {"head", "finetune"}:
        raise ValueError(f"Fase training tidak dikenal: {mode}")
    model.features.requires_grad_(False)
    model.classifier.requires_grad_(True)
    if mode == "finetune":
        model.features[-FINETUNE_BLOCKS:].requires_grad_(True)
