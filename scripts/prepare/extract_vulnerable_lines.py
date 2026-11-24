import json
from pathlib import Path


def extract_vulnerable_lines(input_path: str) -> dict:
    """
    Extract the vulnerable_lines field from detailed annotations
    into a simple filename-to-line-list mapping:
        {
            "xxx.c": [1, 2, 3],
            ...
        }
    """
    input_path = Path(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

    for filename, info in data.items():
        if not isinstance(info, dict):
            raise ValueError(f"{filename} must contain a dictionary to extract vulnerable_lines")

        vulnerable_lines = info.get("vulnerable_lines", [])

        if vulnerable_lines is None:
            vulnerable_lines = []

        if not isinstance(vulnerable_lines, list):
            raise ValueError(f"{filename}: vulnerable_lines must be a list")

        result[filename] = vulnerable_lines

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert detailed split annotations to simple line lists.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing vulnerable_lines_{test,train,valid}.json.")
    parser.add_argument("--merged-output", required=True, type=Path, help="Merged output JSON; existing outputs may be overwritten.")
    args = parser.parse_args()
    input_files = [args.input_dir / f"vulnerable_lines_{split}.json" for split in ("test", "train", "valid")]
    merged_result = {}

    for input_file in input_files:
        input_path = Path(input_file)

        simple_result = extract_vulnerable_lines(input_file)

        # Save the simple annotation mapping for each split
        output_path = input_path.with_name(input_path.stem + "_simple.json")

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(simple_result, f, indent=4, ensure_ascii=False)

        print(f"Saved: {output_path}")

        # Also collect entries for a merged output
        for filename, lines in simple_result.items():
            if filename in merged_result:
                print(f"Warning: duplicated filename found: {filename}")
            merged_result[filename] = lines

    # Save the merged mapping in the original test/train/valid processing order
    merged_output_path = args.merged_output

    with merged_output_path.open("w", encoding="utf-8") as f:
        json.dump(merged_result, f, indent=4, ensure_ascii=False)

    print(f"Saved merged file: {merged_output_path}")


if __name__ == "__main__":
    main()
