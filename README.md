# CoFi-Style Pruning for DistilBERT

This repo adapts the CoFi pruning idea from `princeton-nlp/CoFiPruning` to `distilbert-base-uncased` for:

- SST-2
- QNLI
- AG News

It uses the hyperparameters from the provided DistilBERT FNN pruning notebooks:

| Task | Model | Max length | Batch size | LR | Epochs | Seed | Train subset |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SST-2 | `distilbert-base-uncased` | 128 | 16 | `2e-5` | 2 | 42 | full |
| QNLI | `distilbert-base-uncased` | 128 | 16 | `2e-5` | 2 | 42 | full |
| AG News | `distilbert-base-uncased` | 128 | 8 | `2e-5` | 1 | 42 | 20,000 |

`alpha`, `beta`, and `gamma` are not copied from the FNN notebooks. They are exposed as command-line arguments because they control the CoFi/distillation objective:

- `--alpha`: KL distillation loss weight
- `--beta`: hidden-state distillation loss weight
- `--gamma`: sparsity target regularization weight

## What This Implements

The original CoFi repository ships custom BERT/RoBERTa model files. DistilBERT has a different module layout, so this repo implements DistilBERT-specific hard-concrete L0 gates:

- attention-head gates on every DistilBERT layer
- FFN intermediate-channel gates on every DistilBERT layer
- dense teacher distillation
- target sparsity sweep from 10% to 90%
- hard-gated evaluation
- spreadsheet-ready CSV output

The exported model is a masked structured-pruning model. The CSV reports effective model size from the number of kept structured parameters. Physical tensor compaction is intentionally not mixed into the training script because it changes DistilBERT module shapes and is easier to audit separately.

## Install on SSH Server

```bash
unzip cofi-distilbert-sweeps.zip
cd cofi-distilbert-sweeps
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build that matches your server first, then install the rest of the requirements.

## Run One Dataset

```bash
bash run_one.sh sst2
bash run_one.sh qnli
bash run_one.sh ag_news
```

## Run All Datasets

```bash
bash run_all.sh
```

Outputs are written to:

```text
outputs/<task>/cofi_distilbert_spreadsheet_results.csv
outputs/<task>/cofi_distilbert_spreadsheet_results.json
outputs/<task>/cofi_10/metrics.json
outputs/<task>/cofi_10/l0_gates.pt
...
```

## Useful Options

Run a smaller test first:

```bash
bash run_one.sh sst2 --ratios 10 --epochs 1
```

Change CoFi weights:

```bash
bash run_one.sh qnli --alpha 1.0 --beta 0.1 --gamma 10.0
```

Use an already fine-tuned dense checkpoint:

```bash
bash run_one.sh sst2 --dense-checkpoint outputs/sst2/dense
```

Add an approximate energy column if you know the average server power draw:

```bash
bash run_one.sh ag_news --power-watts 250
```

## Spreadsheet Columns

The generated CSV contains columns aligned with your sheet:

- Method
- Variant / Config
- Pruning Ratio
- Dataset
- Accuracy
- F1 Score (Macro)
- Precision (Macro)
- Recall (Macro)
- Effective Model Size (MB)
- Parameter Size (MB)
- Compression Ratio
- Sparsity (%)
- Inference Latency (s)
- Speedup (x)
- Energy (Joules)

Extra columns are also included for auditability:

- Target Sparsity (%)
- Head Sparsity (%)
- FFN Sparsity (%)

## Notes

- SST-2 and QNLI use GLUE validation splits.
- AG News uses the test split.
- Dataloaders keep `shuffle=False` to match the uploaded notebooks.
- Default pruning ratios are `10,15,20,...,90`, matching the spreadsheet layout.
