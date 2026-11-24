import json
import os


def _prefix_key(filename: str):
    base = os.path.basename(filename)
    prefix = base.split('_')[0]
    if prefix.isdigit():
        return (0, int(prefix), base)
    return (1, base)


def remove_no_diff_entries(input_json_path: str, output_json_path: str):
    """
    Remove entries with fix_type == "no_diff" and status == "ok"
    from detailed line annotations and save the filtered JSON.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect keys to remove
    to_remove = []
    for fname, info in data.items():
        if isinstance(info, dict) and info.get("status") == "ok" and info.get("fix_type") == "no_diff":
            to_remove.append(fname)

    # Remove selected entries
    for fname in to_remove:
        data.pop(fname, None)

    # Sort by numeric filename prefix before writing
    sorted_data = {k: data[k] for k in sorted(data.keys(), key=_prefix_key)}

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)

    print(f"Removed no_diff entries: {len(to_remove)}")
    print(f"Filtered JSON saved to: {output_json_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filter no_diff entries from detailed line annotations.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True, help="Output JSON; an existing file may be overwritten.")
    args = parser.parse_args()
    remove_no_diff_entries(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
