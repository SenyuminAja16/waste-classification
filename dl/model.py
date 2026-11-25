"""Model pretrained dan pengaturan parameter untuk dua fase training."""

from core.config import CLASSES

DEFAULT_MODEL = "mobilenet_v2"
DEFAULT_DROPOUT = 0.2
MODEL_NAMES = (DEFAULT_MODEL, "efficientnet_b0", "convnextv2_femto")


def build_model(name: str = DEFAULT_MODEL, *, pretrained: bool = True, dropout: float = DEFAULT_DROPOUT):
    if name == "convnextv2_femto":
        import timm

        return timm.create_model(
            "convnextv2_femto.fcmae_ft_in1k", pretrained=pretrained, num_classes=len(CLASSES), drop_rate=dropout,
        )

    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights, efficientnet_b0, mobilenet_v2

    if name == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"Model DL tidak dikenal: {name}")
    model.classifier[0] = nn.Dropout(dropout)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def set_trainable(model, name: str, finetune_blocks: int = 0) -> None:
    """Buka head dan sejumlah blok/stage terakhir pada backbone."""
    if name == "convnextv2_femto":
        head, blocks = model.head, model.stages
    else:
        head, blocks = model.classifier, model.features
    if not 0 <= finetune_blocks <= len(blocks):
        raise ValueError(f"train.finetune_blocks untuk {name} harus antara 0 dan {len(blocks)}.")

    model.requires_grad_(False)
    head.requires_grad_(True)
    if finetune_blocks:
        for block in blocks[-finetune_blocks:]:
            block.requires_grad_(True)
