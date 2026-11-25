"""Forward, backward, dan evaluasi untuk satu epoch."""

import torch
from tqdm.auto import tqdm

from core.evaluate import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device, *, description=None):
    model.train()
    for module in model.modules():
        params = tuple(module.parameters())
        if params and not any(param.requires_grad for param in params):
            module.eval()  # Pertahankan statistik BatchNorm dan mode eval pada blok beku.
    total_loss, seen = 0.0, 0
    with tqdm(loader, desc=description, unit="batch", leave=False, disable=None) as progress:
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if not progress.disable:
                progress.set_postfix(loss=f"{total_loss / seen:.3f}", refresh=False)
    return total_loss / seen


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
