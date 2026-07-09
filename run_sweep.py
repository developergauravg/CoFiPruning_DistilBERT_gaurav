from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from cofi_distilbert.cofi import DistilBertGateHooks, DistilBertL0Module, distillation_loss
from cofi_distilbert.data import load_task_config, load_tokenized_dataset
from cofi_distilbert.metrics import as_percent, energy_joules, evaluate_model, model_size_mb
from cofi_distilbert.utils import default_device, ensure_dir, save_json, set_seed


def parse_ratios(value: str) -> list[float]:
    return [float(x.strip()) / 100.0 if float(x.strip()) > 1 else float(x.strip()) for x in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoFi-style DistilBERT pruning sweep.")
    parser.add_argument("--task", choices=["sst2", "qnli", "ag_news"], required=True)
    parser.add_argument("--config", default="configs/tasks.yaml")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--dense-checkpoint", default=None, help="Path to fine-tuned dense teacher. Trains from model-name if omitted.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--ratios", default="10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--reg-learning-rate", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lagrangian-warmup-ratio", type=float, default=0.4)
    parser.add_argument("--droprate-init", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0, help="KL distillation weight. Not taken from FNN notebooks.")
    parser.add_argument("--beta", type=float, default=0.1, help="Hidden-state distillation weight. Not taken from FNN notebooks.")
    parser.add_argument("--gamma", type=float, default=5.0, help="Target sparsity regularization weight. Not taken from FNN notebooks.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--power-watts", type=float, default=None, help="Optional device power estimate for Joules column.")
    parser.add_argument("--save-pruned-models", action="store_true")
    return parser.parse_args()


def train_dense_teacher(args: argparse.Namespace, cfg, datasets, collator, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=cfg.num_labels)
    model.to(device)
    train_loader = DataLoader(datasets["train"], batch_size=cfg.train_batch_size, shuffle=False, collate_fn=collator)
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
        for batch in tqdm(train_loader, desc=f"{cfg.name} dense teacher epoch {epoch + 1}/{epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


def spreadsheet_row(
    task: str,
    ratio: float,
    metrics,
    effective_size_mb: float,
    parameter_size_mb: float,
    baseline_latency: float,
    baseline_size_mb: float,
    power_watts: float | None,
) -> dict[str, float | str | None]:
    speedup = baseline_latency / metrics.latency_s if metrics.latency_s > 0 else None
    compression_ratio = baseline_size_mb / effective_size_mb if effective_size_mb > 0 else None
    return {
        "Method": "CoFi",
        "Variant / Config": "Baseline/No Pruning" if ratio == 0 else f"alpha/beta/gamma CoFi, {int(ratio * 100)}% target",
        "Pruning Ratio": as_percent(ratio),
        "Dataset": {"sst2": "SST-2", "qnli": "QNLI", "ag_news": "AG News"}[task],
        "Accuracy": metrics.accuracy,
        "F1 Score (Macro)": metrics.f1_macro,
        "Precision (Macro)": metrics.precision_macro,
        "Recall (Macro)": metrics.recall_macro,
        "Effective Model Size (MB)": effective_size_mb,
        "Parameter Size (MB)": parameter_size_mb,
        "Compression Ratio": compression_ratio,
        "Sparsity (%)": as_percent(ratio),
        "Inference Latency (s)": metrics.latency_s,
        "Speedup (x)": speedup,
        "Energy (Joules)": energy_joules(metrics.latency_s, power_watts),
    }


def effective_model_size_mb(baseline_model, stats) -> float:
    total_params = sum(p.numel() for p in baseline_model.parameters())
    effective_params = total_params - stats.full_prunable_params + stats.effective_params
    return float(effective_params * 4 / (1024**2))


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config, args.task)
    set_seed(cfg.seed)
    device = default_device()
    output_root = ensure_dir(Path(args.output_dir) / args.task)

    datasets, collator = load_tokenized_dataset(cfg, args.model_name)
    train_loader = DataLoader(datasets["train"], batch_size=cfg.train_batch_size, shuffle=False, collate_fn=collator)
    eval_loader = DataLoader(datasets["eval"], batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collator)

    if args.dense_checkpoint:
        teacher = AutoModelForSequenceClassification.from_pretrained(args.dense_checkpoint, num_labels=cfg.num_labels)
        teacher.to(device)
    else:
        teacher = train_dense_teacher(args, cfg, datasets, collator, device)
        teacher.save_pretrained(ensure_dir(output_root / "dense"))

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    baseline_metrics = evaluate_model(teacher, eval_loader, device)
    baseline_size = model_size_mb(teacher)
    rows = [
        spreadsheet_row(
            args.task,
            0.0,
            baseline_metrics,
            baseline_size,
            baseline_size,
            baseline_metrics.latency_s,
            baseline_size,
            args.power_watts,
        )
    ]

    ratios = parse_ratios(args.ratios)
    epochs = args.epochs or cfg.num_train_epochs
    lr = args.learning_rate or cfg.learning_rate

    for ratio in ratios:
        student = copy.deepcopy(teacher).to(device)
        l0 = DistilBertL0Module(
            student.config,
            droprate_init=args.droprate_init,
            target_sparsity=ratio,
        ).to(device)
        optimizer = AdamW(student.parameters(), lr=lr)
        l0_optimizer = AdamW(l0.parameters(), lr=args.reg_learning_rate)
        total_steps = epochs * len(train_loader)
        warmup_steps = int(total_steps * args.lagrangian_warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )

        global_step = 0
        with DistilBertGateHooks(student, l0):
            for epoch in range(epochs):
                student.train()
                progress = tqdm(train_loader, desc=f"{args.task} CoFi {int(ratio * 100)}% epoch {epoch + 1}/{epochs}")
                for batch in progress:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    optimizer.zero_grad(set_to_none=True)
                    l0_optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            **batch,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    student_outputs = student(
                        **batch,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    loss = distillation_loss(
                        student_outputs,
                        teacher_outputs,
                        batch["labels"],
                        alpha=args.alpha,
                        beta=args.beta,
                        temperature=args.temperature,
                    )
                    loss = loss + l0.lagrangian_loss(global_step, warmup_steps, args.gamma)
                    loss.backward()
                    optimizer.step()
                    l0_optimizer.step()
                    scheduler.step()
                    l0.clamp_parameters()
                    global_step += 1
                    progress.set_postfix(loss=f"{loss.item():.4f}", sparsity=f"{l0.expected_sparsity().item():.3f}")

        stats = l0.hard_stats()
        with DistilBertGateHooks(student, l0, hard=True):
            pruned_metrics = evaluate_model(student, eval_loader, device)
        effective_size_mb = effective_model_size_mb(student, stats)
        row = spreadsheet_row(
            args.task,
            stats.total_sparsity,
            pruned_metrics,
            effective_size_mb,
            baseline_size,
            baseline_metrics.latency_s,
            baseline_size,
            args.power_watts,
        )
        row["Target Sparsity (%)"] = as_percent(ratio)
        row["Head Sparsity (%)"] = as_percent(stats.head_sparsity)
        row["FFN Sparsity (%)"] = as_percent(stats.ffn_sparsity)
        rows.append(row)

        ratio_dir = ensure_dir(output_root / f"cofi_{int(ratio * 100):02d}")
        torch.save(l0.state_dict(), ratio_dir / "l0_gates.pt")
        save_json(row, ratio_dir / "metrics.json")
        if args.save_pruned_models:
            student.save_pretrained(ratio_dir / "student_masked")

    csv_path = output_root / "cofi_distilbert_spreadsheet_results.csv"
    json_path = output_root / "cofi_distilbert_spreadsheet_results.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved spreadsheet CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
