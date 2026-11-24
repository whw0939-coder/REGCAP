import os
import json
import difflib
from typing import List, Tuple, Dict, Any


def _prefix_key(filename: str):
    base = os.path.basename(filename)
    prefix = base.split('_')[0]
    if prefix.isdigit():
        return (0, int(prefix), base)
    return (1, base)


def extract_effective_lines(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract effective lines using the historical blank/comment filtering rules.
    Preserve original line numbers and text, excluding line terminators.
    Returns (original_line_number, text) pairs.
    """
    effective = []
    in_comment_block = False
    current_line = 1

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            stripped_line = raw_line.rstrip('\n\r')
            line_content = stripped_line.strip()

            if not line_content:
                current_line += 1
                continue

            # Inside a block comment
            if in_comment_block:
                if '*/' in line_content:
                    in_comment_block = False
                current_line += 1
                continue

            # Line comment or start of a block comment
            if line_content.startswith(('//', '/*')):
                if line_content.startswith('/*') and '*/' not in line_content:
                    in_comment_block = True
                current_line += 1
                continue

            # Historical heuristic: skip lines starting with an asterisk
            if line_content.startswith('*'):
                current_line += 1
                continue

            effective.append((current_line, stripped_line))
            current_line += 1

    return effective


def diff_effective_lines_enhanced(
    vul_effective: List[Tuple[int, str]],
    patch_effective: List[Tuple[int, str]]
) -> Dict[str, Any]:
    """
    Compare effective-line sequences using SequenceMatcher and map back to source lines.
    Return unchanged lines, equal/replacement alignment pairs, and unmatched
    replacement lines on both sides in addition to changed lines.
    """
    vul_texts = [t for (_, t) in vul_effective]
    patch_texts = [t for (_, t) in patch_effective]

    sm = difflib.SequenceMatcher(a=vul_texts, b=patch_texts, autojunk=False)

    vul_removed_or_replaced = []
    patch_added_or_replaced = []

    equal_lines_in_vul = []
    equal_lines_in_patch = []

    aligned_pairs_equal: List[Tuple[int, int]] = []
    aligned_pairs_replace: List[Tuple[int, int]] = []
    replace_unmatched_vul_lines: List[int] = []
    replace_unmatched_patch_lines: List[int] = []

    stats = {"equal": 0, "delete": 0, "insert": 0, "replace": 0}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        stats[tag] += 1

        if tag == "equal":
            # Collect unchanged line numbers
            vul_eq = [vul_effective[k][0] for k in range(i1, i2)]
            patch_eq = [patch_effective[k][0] for k in range(j1, j2)]
            equal_lines_in_vul.extend(vul_eq)
            equal_lines_in_patch.extend(patch_eq)

            # Align equal segments line by line
            # Equal segment lengths: i2-i1 == j2-j1
            for k in range(i2 - i1):
                aligned_pairs_equal.append((vul_effective[i1 + k][0], patch_effective[j1 + k][0]))

        elif tag == "delete":
            # Deleted lines on the vulnerable side [i1, i2)
            vul_removed_or_replaced.extend([vul_effective[k][0] for k in range(i1, i2)])

        elif tag == "insert":
            # Inserted lines on the fixed side [j1, j2)
            patch_added_or_replaced.extend([patch_effective[k][0] for k in range(j1, j2)])

        elif tag == "replace":
            # Replaced vulnerable-side lines are candidate weak labels
            vul_rep = [vul_effective[k][0] for k in range(i1, i2)]
            vul_removed_or_replaced.extend(vul_rep)

            # Replacement lines on the fixed side
            patch_rep = [patch_effective[k][0] for k in range(j1, j2)]
            patch_added_or_replaced.extend(patch_rep)

            # Align up to the shorter segment; mark remaining lines as unmatched
            minlen = min(i2 - i1, j2 - j1)
            for k in range(minlen):
                aligned_pairs_replace.append((vul_effective[i1 + k][0], patch_effective[j1 + k][0]))

            if (i2 - i1) > minlen:
                replace_unmatched_vul_lines.extend([vul_effective[i1 + k][0] for k in range(minlen, i2 - i1)])

            if (j2 - j1) > minlen:
                replace_unmatched_patch_lines.extend([patch_effective[j1 + k][0] for k in range(minlen, j2 - j1)])

    # Deduplicate and sort line-number lists
    vul_removed_or_replaced = sorted(set(vul_removed_or_replaced))
    patch_added_or_replaced = sorted(set(patch_added_or_replaced))
    equal_lines_in_vul = sorted(set(equal_lines_in_vul))
    equal_lines_in_patch = sorted(set(equal_lines_in_patch))

    # Deduplicate alignment pairs without changing their order
    # Keep the first occurrence of each pair
    def _dedup_pairs_keep_order(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        seen = set()
        out = []
        for a, b in pairs:
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b))
        return out

    aligned_pairs_equal = _dedup_pairs_keep_order(aligned_pairs_equal)
    aligned_pairs_replace = _dedup_pairs_keep_order(aligned_pairs_replace)

    replace_unmatched_vul_lines = sorted(set(replace_unmatched_vul_lines))
    replace_unmatched_patch_lines = sorted(set(replace_unmatched_patch_lines))

    return {
        "vulnerable_lines": vul_removed_or_replaced,                  # Vulnerable-side deleted/replaced lines (weak evidence)
        "added_lines_in_patch": patch_added_or_replaced,              # Fixed-side inserted/replaced lines
        "equal_lines_in_vul": equal_lines_in_vul,                     # Unchanged vulnerable-side lines
        "equal_lines_in_patch": equal_lines_in_patch,                 # Unchanged fixed-side lines
        "aligned_pairs_equal": aligned_pairs_equal,                   # Equal alignment: (vul_line, patch_line)
        "aligned_pairs_replace": aligned_pairs_replace,               # Approximate replacement alignment pairs
        "replace_unmatched_vul_lines": replace_unmatched_vul_lines,   # Unmatched vulnerable-side replacement lines
        "replace_unmatched_patch_lines": replace_unmatched_patch_lines,# Unmatched fixed-side replacement lines
        "stats": stats
    }


def classify_fix(vul_lines: List[int], patch_lines: List[int], stats: Dict[str, int]) -> str:
    """
    Classify the textual change by its affected sides.
    """
    has_vul_change = len(vul_lines) > 0
    has_patch_add = len(patch_lines) > 0

    if not has_vul_change and not has_patch_add:
        return "no_diff"
    if not has_vul_change and has_patch_add:
        return "additive"
    if has_vul_change and not has_patch_add:
        return "modifying"
    return "mixed"


def find_vulnerable_lines_enhanced(patch_folder: str, all_folder: str, output_file: str):
    """
    Match each fixed function in patch_folder to its same-named vulnerable file
    in all_folder. Output changed and unchanged line lists, alignment pairs,
    unmatched replacement lines, fix_type, and statistics.
    """
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith('.c')]
    patch_files = sorted(patch_files, key=_prefix_key)

    result: Dict[str, Any] = {}

    for patch_file in patch_files:
        patch_path = os.path.join(patch_folder, patch_file)
        vul_path = os.path.join(all_folder, patch_file)

        if not os.path.exists(vul_path):
            result[patch_file] = {
                "status": "missing_pair",
                "vulnerable_lines": [],
                "added_lines_in_patch": [],
                "equal_lines_in_vul": [],
                "equal_lines_in_patch": [],
                "aligned_pairs_equal": [],
                "aligned_pairs_replace": [],
                "replace_unmatched_vul_lines": [],
                "replace_unmatched_patch_lines": [],
                "fix_type": "missing_pair",
                "stats": {}
            }
            print(f"[WARN] No matching vulnerable file in all_folder: {patch_file}; recorded missing_pair")
            continue

        vul_eff = extract_effective_lines(vul_path)
        patch_eff = extract_effective_lines(patch_path)

        diff_info = diff_effective_lines_enhanced(vul_eff, patch_eff)

        vul_lines = diff_info["vulnerable_lines"]
        patch_added_lines = diff_info["added_lines_in_patch"]
        stats = diff_info["stats"]
        fix_type = classify_fix(vul_lines, patch_added_lines, stats)

        result[patch_file] = {
            "status": "ok",
            "vulnerable_lines": vul_lines,
            "added_lines_in_patch": patch_added_lines,

            # Additional textual alignment metadata
            "equal_lines_in_vul": diff_info["equal_lines_in_vul"],
            "equal_lines_in_patch": diff_info["equal_lines_in_patch"],
            "aligned_pairs_equal": diff_info["aligned_pairs_equal"],
            "aligned_pairs_replace": diff_info["aligned_pairs_replace"],
            "replace_unmatched_vul_lines": diff_info["replace_unmatched_vul_lines"],
            "replace_unmatched_patch_lines": diff_info["replace_unmatched_patch_lines"],

            "fix_type": fix_type,
            "stats": {
                **stats,
                "vul_effective_lines_count": len(vul_eff),
                "patch_effective_lines_count": len(patch_eff)
            }
        }

    sorted_result = {k: result[k] for k in sorted(result, key=_prefix_key)}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_result, f, indent=4, ensure_ascii=False)

    print(f"Detailed line annotations saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Identify diff-derived vulnerable source lines from matched C files.")
    parser.add_argument("--patch-folder", required=True, help="Directory of fixed functions named after their vulnerable counterparts.")
    parser.add_argument("--all-folder", required=True, help="Directory containing the matching vulnerable functions.")
    parser.add_argument("--output-file", required=True, help="Detailed output JSON; an existing file may be overwritten.")
    args = parser.parse_args()
    find_vulnerable_lines_enhanced(args.patch_folder, args.all_folder, args.output_file)


if __name__ == "__main__":
    main()
