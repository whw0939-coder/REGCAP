"""Independent reproduction helper for PrimeVul C and paired-function inputs.

This script was written after the paper experiments; it is not an author
historical preparation script. Its output has not been established as fully
equivalent to the original experiment's preparation procedure.
"""

import argparse
import json
from pathlib import Path


SPLITS = ("train", "valid", "test")


def historical_bytes(code):
    if not isinstance(code, str):
        raise ValueError("func must be a string")
    return code.replace("\n", "\r\n").encode("utf-8")


def write_checked(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing file differs: {path}")
        return
    path.write_bytes(content)


def build_split(raw_root, out_root, split):
    c_dir = out_root / f"{split}_c"
    patch_dir = out_root / f"{split}_diff"
    c_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    positives = 0
    names = set()
    with (raw_root / f"primevul_{split}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            idx, target = record["idx"], record["target"]
            if not isinstance(idx, int) or target not in (0, 1):
                raise ValueError(f"Invalid idx/target in {split} row {rows + 1}")
            name = f"{idx}_{target}.c"
            if name in names:
                raise ValueError(f"Duplicate sample name: {name}")
            names.add(name)
            write_checked(c_dir / name, historical_bytes(record["func"]))
            rows += 1
            positives += target

    pairs = 0
    pending = None
    with (raw_root / f"primevul_{split}_paired.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if pending is None:
                if record.get("target") != 1:
                    raise ValueError(f"Expected vulnerable record at pair {pairs + 1}")
                pending = record
                continue
            if record.get("target") != 0:
                raise ValueError(f"Expected fixed record at pair {pairs + 1}")
            name = f"{pending['idx']}_1.c"
            if name not in names:
                raise ValueError(f"Paired vulnerable sample absent from split: {name}")
            if (c_dir / name).read_bytes() != historical_bytes(pending["func"]):
                raise ValueError(f"Paired vulnerable code differs from split: {name}")
            write_checked(patch_dir / name, historical_bytes(record["func"]))
            pending = None
            pairs += 1
    if pending is not None:
        raise ValueError(f"Odd number of paired records in {split}")
    return {"rows": rows, "positive": positives, "negative": rows - positives, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = {split: build_split(args.raw_root, args.output_root, split) for split in SPLITS}
    (args.output_root / "conversion_counts.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
