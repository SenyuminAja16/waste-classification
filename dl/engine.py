"""Forward, backward, dan evaluasi untuk satu epoch."""

import torch

from core.evaluate import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for block in model.features:
        if not any(param.requires_grad for param in block.parameters()):
            block.eval()  # Pertahankan statistik BatchNorm di blok yang dibekukan.
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def score_model(model, loader, criterion, device):
    model.eval()
    labels, predictions = [], []
    total_loss = 0.0
    for images, target in loader:
        logits = model(images.to(device))
        total_loss += criterion(logits, target.to(device)).item() * target.size(0)
        labels.extend(target.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    return evaluate(labels, predictions), total_loss / len(loader.dataset)
