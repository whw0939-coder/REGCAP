# Configuration

The `regcap.cli` flags hold the practical runtime configuration. The main defaults follow the manuscript: 128-dimensional embeddings, 17 retained regions, batch size 16, 50 epochs, AdamW learning rate `1e-4`, and classification/region-loss weights of 1.0. Use `python -m regcap.cli train --help` to inspect overrides. Dataset layouts are selected with `preprocess --dataset` and can be overridden with explicit input paths.
