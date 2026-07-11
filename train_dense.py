from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from cofi_distilbert.data import load_task_config, load_tokenized_dataset
from cofi_distilbert.metrics import evaluate_model, model_size_mb
from cofi_distilbert.utils import default_device, ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune dense DistilBERT baseline.")
    parser.add_argument("--task", choices=["sst2", "qnli", "ag_news"], required=True)
    parser.add_argument("--config", default="configs/tasks.yaml")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config, args.task)
    set_seed(cfg.seed)
    device = default_device()

    datasets, collator = load_tokenized_dataset(cfg, args.model_name)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.train_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        datasets["eval"],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=cfg.num_labels)
    model.to(device)

    epochs = args.epochs or cfg.num_train_epochs
    lr = args.learning_rate or cfg.learning_rate
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{args.task} dense epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")

    out_dir = ensure_dir(Path(args.output_dir) / args.task / "dense")
    model.save_pretrained(out_dir)
    metrics = evaluate_model(model, eval_loader, device)
    save_json(
        {
            "task": args.task,
            "model_name": args.model_name,
            "accuracy": metrics.accuracy,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
            "f1_macro": metrics.f1_macro,
            "latency_s_per_example": metrics.latency_s,
            "parameter_size_mb": model_size_mb(model),
            "epochs": epochs,
            "learning_rate": lr,
            "device": str(device),
        },
        out_dir / "metrics.json",
    )
    print(f"Saved dense baseline to {out_dir}")


if __name__ == "__main__":
    main()
