from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


@dataclass(frozen=True)
class TaskConfig:
    name: str
    hf_dataset: str
    hf_subset: str | None
    text_fields: list[str]
    label_field: str
    num_labels: int
    train_split: str
    eval_split: str
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    num_train_epochs: int
    seed: int
    train_subset: int | None = None


def load_task_config(path: str, task: str) -> TaskConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if task not in raw:
        raise KeyError(f"Unknown task '{task}'. Available tasks: {', '.join(sorted(raw))}")
    return TaskConfig(name=task, **raw[task])


def _tokenize(example: dict[str, Any], tokenizer: AutoTokenizer, cfg: TaskConfig) -> dict[str, Any]:
    texts = [example[field] for field in cfg.text_fields]
    encoded = tokenizer(*texts, truncation=True, max_length=cfg.max_length)
    encoded["labels"] = example[cfg.label_field]
    return encoded


def load_tokenized_dataset(cfg: TaskConfig, model_name: str) -> tuple[DatasetDict, DataCollatorWithPadding]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if cfg.hf_subset:
        dataset = load_dataset(cfg.hf_dataset, cfg.hf_subset)
    else:
        dataset = load_dataset(cfg.hf_dataset)

    train = dataset[cfg.train_split]
    if cfg.train_subset:
        train = train.shuffle(seed=cfg.seed).select(range(min(cfg.train_subset, len(train))))

    eval_ds = dataset[cfg.eval_split]
    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized = DatasetDict(
        {
            "train": train.map(lambda x: _tokenize(x, tokenizer, cfg), remove_columns=train.column_names),
            "eval": eval_ds.map(lambda x: _tokenize(x, tokenizer, cfg), remove_columns=eval_ds.column_names),
        }
    )
    tokenized.set_format(type="torch", columns=keep_cols)
    return tokenized, DataCollatorWithPadding(tokenizer)
