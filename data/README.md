# Data inputs

No benchmark files are included. Download FFmpeg+Qemu, DiverseVul, or PrimeVul from the maintainers linked in [docs/dataset_format.md](../docs/dataset_format.md), then prepare C functions, Joern JSON, vulnerable-line annotations, and dataset-specific 128-dimensional Word2Vec vectors in the documented layout. Use `--data-root` for the preset layout or pass the four paths explicitly.

```bash
python -m regcap.cli preprocess --dataset ffmpeg_qemu --data-root /path/to/data --output-dir processed
python -m regcap.cli preprocess --dataset diversevul --data-root /path/to/data --output-dir processed
python -m regcap.cli preprocess --dataset primevul_train --data-root /path/to/data --output-dir processed
python -m regcap.cli preprocess --dataset primevul_valid --data-root /path/to/data --output-dir processed
python -m regcap.cli preprocess --dataset primevul_test --data-root /path/to/data --output-dir processed
```
