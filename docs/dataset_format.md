# Dataset and graph format

REGCAP accepts prepared C/C++ function files and matching Joern graph JSON. It does not redistribute benchmark content or include a Joern exporter. The paper used Joern v2.0.86. If you start from raw benchmark records, export each function and construct a JSON record with the keys below before running REGCAP preprocessing.

## Obtain the benchmarks

| Benchmark | Maintainer source | Preparation note |
| --- | --- | --- |
| FFmpeg+Qemu | [Devign release](https://github.com/epicosy/devign) | Extract labeled functions from the FFmpeg and Qemu records. |
| DiverseVul | [DiverseVul release](https://github.com/wagner-group/diversevul) | Extract labeled functions from the released records. |
| PrimeVul | [PrimeVul release](https://github.com/DLVulDet/PrimeVul) | Preserve the official chronological train/validation/test split. |

The benchmark releases do not directly provide this repository's C-file, Joern-JSON, vulnerable-line, and Word2Vec layout. Prepare those derived inputs locally. Record the source release, split, Joern version, and embedding training settings used for a reproduction. The source code here does not include a script that converts raw benchmark JSON/JSONL into the exact prepared layout.

## Prepared layout

```text
/path/to/data/
├── devign/
│   ├── c/<sample_id>_<label>.c
│   ├── js/<sample_id>_<label>.json
│   ├── Devign_vulnerable_lines.json
│   └── W2V/Devign-128-20.wordvectors
├── DiverseVul/
│   ├── c/<sample_id>_<label>.c
│   ├── js/<sample_id>_<label>.json
│   ├── DiverseVul_vulnerable_lines.json
│   └── W2V/DiverseVul-128-20.wordvectors
└── PrimeVul/
    ├── train_c/ and train_js/
    ├── valid_c/ and valid_js/
    ├── test_c/ and test_js/
    ├── vulnerable_lines_train.json
    ├── vulnerable_lines_valid.json
    ├── vulnerable_lines_test.json
    └── w2v/primevul-128-20.wordvectors
```

`<label>` is `0` for benign or `1` for vulnerable. The C and JSON basenames must match. The annotation file is a JSON object keyed by C basename. Each value is a list of 1-based vulnerable source-line numbers; samples without known vulnerable lines may use an empty list. Word2Vec files must be readable by Gensim `KeyedVectors.load` and have 128-dimensional vectors.

Each Joern JSON file must contain `ast_nodes`, `ast_edges`, `cfg_edges`, `cdg_edges`, and `ddg_edges`. Nodes require a unique `id`, `_label`, `code`, and `lineNumber`; additional Joern properties may be present. Each edge list contains `[source_id, target_id]` pairs referring to those node IDs. CFG, CDG, and DDG edge endpoints must align with the AST node IDs. The source-line numbers and C file must refer to the same function text. Incorrect alignment changes regions, RDP values, and weak region targets.

Use `--c-dir`, `--js-dir`, `--vul-lines`, and `--w2v` to override preset paths without editing source code. Preprocessing writes `<dataset>-31.pkl` and `<dataset>-region_attr_stats.json` to `--output-dir`. The pickle contains PyG `Data` samples; each sample includes `region_attr` with 31 columns and `region_attr_names` in canonical order.
