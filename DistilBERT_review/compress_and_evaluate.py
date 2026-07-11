"""
Turn a soft-masked, CoFi-trained DistilBERT checkpoint (produced by
``run_sweep.py``) into a *physically* pruned, smaller-on-disk compact
model, then evaluate that compact model for real.

This script does not modify ``run_sweep.py``, ``cofi.py``, ``data.py`` or
``metrics.py`` in any way -- the existing activation-masking training
pipeline keeps working exactly as it does today. This script only *reads*
what that pipeline already writes to disk when it's run with
``--save-pruned-models``:

    outputs/<task>/cofi_<ratio>/l0_gates.pt        (trained HardConcreteGate state)
    outputs/<task>/cofi_<ratio>/student_masked/    (trained student weights, dense HF format)

and produces new artifacts alongside them:

    outputs/<task>/cofi_<ratio>/compact/                  (physically pruned checkpoint)
    outputs/<task>/cofi_<ratio>/compact_eval_metrics.json (real metrics measured on that checkpoint)

Bridging note: ``cofi.py``'s ``DistilBertL0Module`` only ever trains a
head-level gate and an FFN-neuron-level gate (no whole-layer or
hidden-dimension gates). This script therefore constructs
``pruning.DistilBertL0Module`` with ``pruning_type="structured_heads+structured_mlp"``
-- the exact subset of structural pruning that was actually trained -- and
copies the trained ``log_alpha`` values across. No sparsity numbers are
invented: everything reported by this script comes from a real forward
pass through a real, physically-resized model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cofi_distilbert.data import load_task_config, load_tokenized_dataset
from cofi_distilbert.metrics import evaluate_model, model_size_mb
from cofi_distilbert.utils import default_device, ensure_dir, save_json

from cofi_distilbert.pruning.cofi_utils import export_compact_model, generate_compact_model
from cofi_distilbert.pruning.l0_module import DistilBertL0Module
from cofi_distilbert.pruning.modeling_distilbert import CoFiDistilBertForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physically prune and evaluate a CoFi-trained DistilBERT checkpoint from run_sweep.py."
    )
    parser.add_argument("--task", choices=["sst2", "qnli", "ag_news"], required=True)
    parser.add_argument("--config", default="configs/tasks.yaml")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument(
        "--student-dir",
        required=True,
        help="Directory produced by run_sweep.py for one ratio, e.g. outputs/sst2/cofi_30 "
        "(must contain l0_gates.pt and student_masked/, i.e. run_sweep.py was called with --save-pruned-models).",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <student-dir>/compact")
    return parser.parse_args()


def bridge_l0_module(student_dir: Path, config) -> DistilBertL0Module:
    """
    Load the trained ``cofi.DistilBertL0Module`` state (head_gate +
    ffn_gate only) and copy its ``log_alpha`` values into a freshly built
    ``pruning.DistilBertL0Module`` restricted to the same two gate types,
    so ``forward(training=False)`` reproduces the same hard head/FFN
    decisions ``run_sweep.py`` already evaluated under
    ``DistilBertGateHooks(..., hard=True)`` -- just as physical pruning
    instead of activation masking.
    """
    gates_path = student_dir / "l0_gates.pt"
    if not gates_path.exists():
        raise FileNotFoundError(
            f"{gates_path} not found. Re-run run_sweep.py for this ratio with --save-pruned-models."
        )
    old_state = torch.load(gates_path, map_location="cpu")

    l0 = DistilBertL0Module(
        config,
        pruning_type="structured_heads+structured_mlp",
        target_sparsity=0.0,  # unused for inference; hard zs come from the trained log_alpha directly
    )
    with torch.no_grad():
        l0.head_loga.copy_(old_state["head_gate.log_alpha"])
        l0.int_loga.copy_(old_state["ffn_gate.log_alpha"])
    return l0


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config, args.task)
    device = default_device()

    student_dir = Path(args.student_dir)
    student_weights_dir = student_dir / "student_masked"
    if not student_weights_dir.exists():
        raise FileNotFoundError(
            f"{student_weights_dir} not found. Re-run run_sweep.py for this ratio with --save-pruned-models."
        )

    output_dir = Path(args.output_dir) if args.output_dir else student_dir / "compact"

    datasets, collator = load_tokenized_dataset(cfg, args.model_name)
    eval_loader = DataLoader(datasets["eval"], batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collator)

    print(f"Loading trained student weights from {student_weights_dir} ...")
    model = CoFiDistilBertForSequenceClassification.from_pretrained(str(student_weights_dir))
    model.to(device)
    dense_size_mb = model_size_mb(model)

    print(f"Bridging trained gates from {student_dir / 'l0_gates.pt'} ...")
    l0 = bridge_l0_module(student_dir, model.config)
    with torch.no_grad():
        hard_zs = l0.forward(training=False)
    hard_zs = {k: v.to(model.device) for k, v in hard_zs.items()}

    print("Physically pruning a compact copy of the model ...")
    compact_model = generate_compact_model(model, hard_zs)
    compact_model.to(device)
    compact_model.eval()

    export_compact_model(compact_model, str(output_dir), zs=hard_zs)

    print("Evaluating the compact (physically pruned) model ...")
    compact_metrics = evaluate_model(compact_model, eval_loader, device)
    compact_size_mb = model_size_mb(compact_model)

    result = {
        "task": args.task,
        "student_dir": str(student_dir),
        "compact_output_dir": str(output_dir),
        "dense_model_size_mb": dense_size_mb,
        "compact_model_size_mb": compact_size_mb,
        "physical_compression_ratio": dense_size_mb / compact_size_mb if compact_size_mb > 0 else None,
        "accuracy": compact_metrics.accuracy,
        "precision_macro": compact_metrics.precision_macro,
        "recall_macro": compact_metrics.recall_macro,
        "f1_macro": compact_metrics.f1_macro,
        "latency_s": compact_metrics.latency_s,
    }

    metrics_path = student_dir / "compact_eval_metrics.json"
    save_json(result, metrics_path)
    print(json.dumps(result, indent=2))
    print(f"Saved compact-model metrics to {metrics_path}")


if __name__ == "__main__":
    main()
