# REGCAP

Reference implementation of REGCAP as described in the submitted ACM TOSEM manuscript *REGCAP: Region-Centric Modeling for Vulnerability Detection*.

## Overview

REGCAP detects vulnerable C/C++ functions with source-ordered AST regions. It encodes token, AST, CFG, PDG, and 31-dimensional Regional Distributional Prior (RDP) views, fuses them with five-view softmax gates, and produces function predictions and region importance scores. Diff-derived vulnerable source lines provide weak region targets: the fraction of a region's covered lines marked vulnerable.

## Repository structure

- `src/regcap/preprocessing.py` and `regions.py`: Joern graph preparation, syntax regions, weak targets, and 31-feature RDP.
- `src/regcap/rdp_schema.py`: canonical feature order and VE/SE semantic metadata.
- `src/regcap/data_loading.py`: splits, source-order region selection, padding, and batching.
- `src/regcap/model.py` and `training.py`: five-view fusion, region scoring, optimization, and metrics.
- `src/regcap/cli.py`: preprocessing, training, evaluation, prediction, and smoke commands.
- `scripts/`: shell wrappers; `docs/`: input format and reproduction details.

## Environment

The recorded stack is Python 3.8.20, PyTorch 2.4.1, PyG 2.6.1, PyTorch Lightning 2.4.0, CUDA 12.1, and Joern v2.0.86. Install the package from the repository root:

```bash
conda env create -f environment.yml
conda activate regcap
python -m pip install -e . --no-deps
python -m regcap.cli --help
```

Use a compatible CUDA/PyG build for GPU training. `requirements.txt` lists Python dependencies.

## Quick start

Prepare FFmpeg+Qemu or DiverseVul inputs in the [documented layout](docs/dataset_format.md), then run:

```bash
python -m regcap.cli preprocess --dataset ffmpeg_qemu --data-root /path/to/data --output-dir processed
python -m regcap.cli train --data processed/ffmpeg_qemu-31.pkl --output-dir runs/ffmpeg_qemu
```

For PrimeVul, preserve the provided chronological split:

```bash
python -m regcap.cli preprocess --dataset primevul_train --data-root /path/to/data --output-dir processed
python -m regcap.cli preprocess --dataset primevul_valid --data-root /path/to/data --output-dir processed
python -m regcap.cli preprocess --dataset primevul_test --data-root /path/to/data --output-dir processed
python -m regcap.cli train --train processed/primevul_train-31.pkl --valid processed/primevul_valid-31.pkl --test processed/primevul_test-31.pkl --output-dir runs/primevul
```

The commands also accept explicit dataset paths. See [reproduction steps](docs/reproduction.md) and the five `scripts/*.sh` wrappers.

## Data preparation and program representation

Preprocessing requires one C/C++ function file and matching Joern JSON per sample, a vulnerable-line annotation JSON, and dataset-specific 128-dimensional Gensim Word2Vec vectors. The repository expects pre-generated Joern exports; it does not acquire benchmarks, export Joern graphs, or train Word2Vec. The [data format](docs/dataset_format.md) specifies filenames, graph keys, line numbering, and output layout.

Regions follow top-down AST structure using `METHOD`, `BLOCK`, and `CONTROL_STRUCTURE` anchors. Boundaries use syntax alone. Token, AST, CFG, PDG, and RDP representations align to those regions. At most 17 source-ordered regions are retained by uniform index selection; shorter sequences are masked and padded.

## 31-dimensional Regional Distributional Prior

Each region has a normalized 31-value vector in the fixed order in [RDP features](docs/rdp_features.md). VE-oriented and SE-oriented labels are semantic metadata for the released implementation.

## Training and evaluation

The main defaults are hidden size 128, six CFG GGNN steps, four PDG GAT heads, three token Transformer layers, 17 maximum regions, batch size 16, 50 epochs, AdamW at `1e-4`, and region supervision weight 1.0. The checkpoint with highest validation F1 is selected. The weak region target is `|region lines ∩ vulnerable lines| / |region lines|`, derived from fixing-diff line evidence. Empty regions receive zero.

```bash
python -m regcap.cli evaluate --data processed/primevul_test-31.pkl --checkpoint /path/to/best.ckpt --output-dir results/primevul
```

Evaluation writes accuracy, precision, recall, F1, region MSE, and prediction files.

## Region importance and expected outputs

```bash
python -m regcap.cli predict --data processed/cases-31.pkl --checkpoint /path/to/best.ckpt --output-dir results/cases
```

`region_predictions.jsonl` contains function probability, binary prediction, threshold, and region importance scores in retained order. Preprocessing writes `<dataset>-31.pkl` and feature statistics; training writes a validation-selected checkpoint and predictions. Checkpoints must match the released 31-dimensional, five-view architecture. Load processed pickle files only from trusted sources.

## Datasets and reproducibility scope

Supported input layouts cover FFmpeg+Qemu, DiverseVul, and PrimeVul. No raw datasets, Joern exports, Word2Vec models, processed pickles, or trained checkpoints are bundled. `scripts/smoke_test.sh` validates the core path on one prepared sample with untrained weights. Its probability is not a benchmark result. Numerical results reported in the paper are paper-reported results unless explicitly marked as rerun.

## Citation and license

Citation details will be added after publication. This repository is released under the MIT License; see `LICENSE`.
