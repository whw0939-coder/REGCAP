# Reproducing the main pipeline

1. Create the environment with `conda env create -f environment.yml`, activate it, and install the local package with `python -m pip install -e . --no-deps`.
2. Obtain one of the three benchmarks and prepare matching C files, Joern JSON, vulnerable-line annotations, and 128-dimensional Word2Vec vectors. See [dataset format](dataset_format.md).
3. Run `python -m regcap.cli preprocess --dataset ffmpeg_qemu --data-root /path/to/data --output-dir processed`. Use `diversevul` for DiverseVul. For PrimeVul, preprocess `primevul_train`, `primevul_valid`, and `primevul_test` separately.
4. Run `python -m regcap.cli train --data processed/ffmpeg_qemu-31.pkl --output-dir runs/ffmpeg_qemu` for a single-dataset 80/10/10 split. For PrimeVul, pass `--train`, `--valid`, and `--test` files matching its official chronological split.
5. Inspect `runs/<dataset>/checkpoints/` for the highest-validation-F1 checkpoint. Training also runs a final test and writes `predictions/summary_test.json` plus per-sample JSONL/CSV.
6. Run `python -m regcap.cli evaluate --data processed/<test-file>.pkl --checkpoint /path/to/best.ckpt --output-dir results/evaluation` for a separate evaluation. `summary_evaluation.json` reports accuracy, precision, recall, and F1.
7. Run `python -m regcap.cli predict --data processed/<case-file>.pkl --checkpoint /path/to/best.ckpt --output-dir results/cases` to save function probabilities and region scores.

For a one-function wiring check, run `scripts/smoke_test.sh /path/to/sample.c /path/to/sample.json /path/to/vulnerable_lines.json /path/to/wordvectors`. The smoke test uses untrained weights and does not estimate detection performance.

Preprocessing and training can be time-consuming for complete datasets. The repository has been checked with a small local integration run; full manuscript-table replication requires the full prepared data and substantial training time. Do not interpret a smoke-test metric as a paper result.
