from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


@dataclass
class EvalMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    latency_s: float


def model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / (1024**2)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> EvalMetrics:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds.extend(torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy().tolist())
        labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return EvalMetrics(
        accuracy=float(accuracy_score(labels, preds)),
        precision_macro=float(precision_score(labels, preds, average="macro", zero_division=0)),
        recall_macro=float(recall_score(labels, preds, average="macro", zero_division=0)),
        f1_macro=float(f1_score(labels, preds, average="macro", zero_division=0)),
        latency_s=float(elapsed / max(1, len(loader.dataset))),
    )


def energy_joules(latency_s: float, power_watts: float | None) -> float | None:
    if power_watts is None:
        return None
    return float(latency_s * power_watts)


def as_percent(value: float) -> float:
    return float(np.round(value * 100.0, 4))
