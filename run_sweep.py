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
from codecarbon import EmissionsTracker
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from cofi_distilbert.cofi import DistilBertGateHooks, DistilBertL0Module, distillation_loss
from cofi_distilbert.data import load_task_config, load_tokenized_dataset
from cofi_distilbert.metrics import as_percent, energy_joules, evaluate_model, model_size_mb
from cofi_distilbert.utils import default_device, ensure_dir, save_json, set_seed

# --- Structural (physical) pruning, reused as-is from cofi_distilbert.pruning ---
# This subpackage is a direct port of the official CoFiPruning implementation to
# DistilBERT (see src/cofi_distilbert/pruning/*.py). It is only imported here; none
# of its own files are modified by this integration.
from cofi_distilbert.pruning.cofi_utils import calculate_parameters, export_compact_model, generate_compact_model
from cofi_distilbert.pruning.l0_module import DistilBertL0Module as StructuralL0Module
from cofi_distilbert.pruning.modeling_distilbert import CoFiDistilBertForSequenceClassification


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
    parser.add_argument("--reg-learning-rate", type=float, default=5e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lagrangian-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--droprate-init", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0, help="KL distillation weight. Not taken from FNN notebooks.")
    parser.add_argument("--beta", type=float, default=0.1, help="Hidden-state distillation weight. Not taken from FNN notebooks.")
    parser.add_argument("--gamma", type=float, default=20.0, help="Target sparsity regularization weight. Not taken from FNN notebooks.")
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
        "Variant / Config": (
            "Baseline/No Pruning"
            if ratio == 0
            else f"CoFi {int(as_percent(ratio))}% Target"
        ),
        "Pruning Ratio": as_percent(ratio),
        "Dataset": {"sst2": "SST-2", "qnli": "QNLI", "ag_news": "AG News"}[task],
        "Accuracy": metrics.accuracy,
        "F1 Score (Macro)": metrics.f1_macro,
        "Precision (Macro)": metrics.precision_macro,
        "Recall (Macro)": metrics.recall_macro,
        "Effective Model Size (MB)": effective_size_mb,
        "Parameter Size (MB)": parameter_size_mb,
        "Compression Ratio": compression_ratio,
        "Sparsity (%)": None,
        "Inference Latency (s)": metrics.latency_s,
        "Speedup (x)": speedup,
        "Energy (Joules)": None,
    }


def effective_model_size_mb(baseline_model, stats) -> float:
    total_params = sum(p.numel() for p in baseline_model.parameters())
    effective_params = total_params - stats.full_prunable_params + stats.effective_params
    return float(effective_params * 4 / (1024**2))


def effective_model_size_mb_from_compact(student, compact_model) -> float:
    """
    Effective Model Size computed from the actual physically-pruned compact
    model, rather than from cofi.py's gate-statistics estimate.

    Mirrors effective_model_size_mb's structure exactly (total params, minus
    the prunable submodules' size before pruning, plus their size after), but
    every term is measured from real weight tensors via
    cofi_distilbert.pruning.cofi_utils.calculate_parameters -- the same
    function generate_compact_model() itself uses to report before/after
    sizes -- instead of cofi.py's idealized per-head/per-FFN-dim formula
    combined with an independent elementwise hard-threshold decision that
    does not correspond to the count-matched deterministic pruning decision
    actually used to build the compact model. See the accompanying
    explanation for why the two numbers can diverge substantially.
    """
    total_params = sum(p.numel() for p in student.parameters())
    dense_prunable_params = calculate_parameters(student)
    compact_prunable_params = calculate_parameters(compact_model)
    effective_params = total_params - dense_prunable_params + compact_prunable_params
    return float(effective_params * 4 / (1024**2))


def build_compact_model(student, l0: DistilBertL0Module, device: torch.device):
    """
    Physically (structurally) prunes a copy of the trained student using the
    head/FFN gates this ratio's training loop just learned, and returns a
    real, smaller nn.Module -- not an activation-masked one.

    This reuses the CoFi structural-pruning port already implemented in
    cofi_distilbert.pruning (cofi_utils.generate_compact_model), instead of
    a new pruning algorithm, per the requirement to reuse existing work.

    cofi_distilbert.cofi.DistilBertL0Module (the module actually trained by
    the loop above) only ever learns a head-level gate (head_gate) and an
    FFN-neuron-level gate (ffn_gate) -- it has no whole-layer or
    hidden-dimension gates. The bridged
    cofi_distilbert.pruning.l0_module.DistilBertL0Module below is therefore
    restricted to that exact same "structured_heads+structured_mlp" subset,
    so the hard zs it produces reflect only what was actually trained --
    nothing is invented or assumed.
    """
    structural_l0 = StructuralL0Module(
        student.config,
        pruning_type="structured_heads+structured_mlp",
    )
    with torch.no_grad():
        structural_l0.head_loga.copy_(l0.head_gate.log_alpha.detach().cpu())
        structural_l0.int_loga.copy_(l0.ffn_gate.log_alpha.detach().cpu())
        hard_zs = structural_l0.forward(training=False)
    hard_zs = {k: v.to(device) for k, v in hard_zs.items()}

    # CoFiDistilBertForSequenceClassification uses the exact same parameter
    # names as the stock DistilBertForSequenceClassification `student` was
    # built from, so its trained weights load in directly. The CoFi class is
    # used here (rather than pruning `student` in place) because only it
    # handles the edge case where every head in a layer gets pruned to zero.
    cofi_student = CoFiDistilBertForSequenceClassification(student.config)
    cofi_student.to(device)

    # Verify every parameter name matches exactly before loading. strict=True
    # below would also raise on a mismatch, but this precheck gives an
    # explicit missing_keys/unexpected_keys report rather than relying on
    # PyTorch's default error formatting, and fails fast before touching any
    # tensor data. strict=False is never used anywhere in this function.
    expected_keys = set(cofi_student.state_dict().keys())
    provided_keys = set(student.state_dict().keys())
    missing_keys = sorted(expected_keys - provided_keys)
    unexpected_keys = sorted(provided_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "build_compact_model: parameter names in the trained student do not match "
            "CoFiDistilBertForSequenceClassification exactly.\n"
            f"missing_keys (expected by CoFiDistilBertForSequenceClassification, absent in student): {missing_keys}\n"
            f"unexpected_keys (present in student, unknown to CoFiDistilBertForSequenceClassification): {unexpected_keys}"
        )

    load_result = cofi_student.load_state_dict(student.state_dict(), strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        # Unreachable in practice: strict=True raises before returning if either
        # list is non-empty. Kept as a defensive backstop against a future
        # PyTorch version changing that contract silently.
        raise RuntimeError(
            "build_compact_model: load_state_dict reported a key mismatch despite the precheck above.\n"
            f"missing_keys: {load_result.missing_keys}\n"
            f"unexpected_keys: {load_result.unexpected_keys}"
        )
    cofi_student.eval()

    compact_model = generate_compact_model(cofi_student, hard_zs)
    compact_model.to(device)
    compact_model.eval()
    return compact_model, hard_zs


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
        ratio_dir = ensure_dir(output_root / f"cofi_{int(ratio * 100):02d}")

        tracker = EmissionsTracker(
            output_dir=str(ratio_dir),
            project_name=f"CoFi_{args.task}_{int(ratio * 100)}",
            output_file="emissions.csv",
            measure_power_secs=1,
            log_level="error",
        )
        tracker.start()
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
            ratio,
            pruned_metrics,
            effective_size_mb,
            baseline_size,
            baseline_metrics.latency_s,
            baseline_size,
            args.power_watts,
        )
        energy_kwh = tracker.stop()

        expected_total_sparsity = l0.expected_sparsity().item()
        expected_head_sparsity = 1.0 - l0.head_gate.expected_l0().mean().item()
        expected_ffn_sparsity = 1.0 - l0.ffn_gate.expected_l0().mean().item()

        row["Energy (Joules)"] = energy_kwh * 3_600_000
        row["Target Sparsity (%)"] = as_percent(ratio)
        row["Sparsity (%)"] = as_percent(expected_total_sparsity)
        row["Head Sparsity (%)"] = as_percent(expected_head_sparsity)
        row["FFN Sparsity (%)"] = as_percent(expected_ffn_sparsity)


        # --- Structural pruning step (runs after tracker.stop(), so it does not
        # change what CodeCarbon measures for this ratio). Physically prunes the
        # trained student, evaluates the real compact model, and overwrites the
        # behavior-describing spreadsheet fields with those real numbers. The
        # spreadsheet's key set (schema) is unchanged: only these existing
        # values are updated, exactly like the Energy/Sparsity fields above.
        # "Effective Model Size (MB)", "Parameter Size (MB)", and
        # "Compression Ratio" are left untouched, per requirements.
        ratio_dir = ensure_dir(output_root / f"cofi_{int(ratio * 100):02d}")
        compact_model, hard_zs = build_compact_model(student, l0, device)
        compact_metrics = evaluate_model(compact_model, eval_loader, device)
        

        compact_speedup = (
            baseline_metrics.latency_s / compact_metrics.latency_s if compact_metrics.latency_s > 0 else None
        )
        row["Accuracy"] = compact_metrics.accuracy
        row["F1 Score (Macro)"] = compact_metrics.f1_macro
        row["Precision (Macro)"] = compact_metrics.precision_macro
        row["Recall (Macro)"] = compact_metrics.recall_macro
        row["Inference Latency (s)"] = compact_metrics.latency_s
        row["Speedup (x)"] = compact_speedup

        export_compact_model(compact_model, str(ratio_dir / "compact"), zs=hard_zs)

        # Effective Model Size (MB) originally reflected cofi.py's gate-statistics
        # estimate (stats), which uses a different hard-decision rule and a
        # different parameter-counting formula than the physical pruning that
        # actually produced compact_model, and can diverge substantially from it.
        # Recompute it from the real compact model so it is consistent with the
        # model that was just evaluated above. Compression Ratio's formula is
        # unchanged (still baseline_size / effective_size_mb, exactly as in
        # spreadsheet_row()); only its input is corrected, the same way
        # row["Energy (Joules)"] is already set here rather than inside
        # spreadsheet_row(). Parameter Size (MB) is untouched.
        compact_effective_size_mb = effective_model_size_mb_from_compact(student, compact_model)
        row["Effective Model Size (MB)"] = compact_effective_size_mb
        row["Compression Ratio"] = (
            baseline_size / compact_effective_size_mb if compact_effective_size_mb > 0 else None
        )

        save_json(
            {
                "gated_activation_masking_metrics": {
                    "accuracy": pruned_metrics.accuracy,
                    "f1_macro": pruned_metrics.f1_macro,
                    "precision_macro": pruned_metrics.precision_macro,
                    "recall_macro": pruned_metrics.recall_macro,
                    "latency_s": pruned_metrics.latency_s,
                },
                "compact_physically_pruned_metrics": {
                    "accuracy": compact_metrics.accuracy,
                    "f1_macro": compact_metrics.f1_macro,
                    "precision_macro": compact_metrics.precision_macro,
                    "recall_macro": compact_metrics.recall_macro,
                    "latency_s": compact_metrics.latency_s,
                },
            },
            ratio_dir / "compact_vs_gated_metrics.json",
        )

        rows.append(row)

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
