"""Export classified PrimeVul JSONL records as function files.

Adapted from an author-provided preparation script. The split mapping,
filenames, text-mode writing and record handling are retained.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, help="Directory containing classified PrimeVul JSONL files.")
    parser.add_argument("--output-root", required=True, help="Output directory; existing C files may be overwritten.")
    args = parser.parse_args()
    input_files = {
        split: os.path.join(args.raw_root, f"primevul_{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    output_root = args.output_root

    for split, jsonl_path in input_files.items():
        out_dir = os.path.join(output_root, split)
        os.makedirs(out_dir, exist_ok=True)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[SKIP] {jsonl_path}, line {line_num}: invalid JSON: {e}")
                    continue

                idx = data.get("idx")
                target = data.get("target")
                func = data.get("func", "")

                if idx is None or target is None:
                    print(f"[SKIP] {jsonl_path}, line {line_num}: missing idx or target")
                    continue

                filename = f"{idx}_{target}.c"
                file_path = os.path.join(out_dir, filename)

                with open(file_path, "w", encoding="utf-8") as out_f:
                    out_f.write(func)

        print(f"[DONE] {jsonl_path} -> {out_dir}")

    print("Conversion complete.")


if __name__ == "__main__":
    main()
