import argparse
import sys
from typing import Dict

# Standard bracket sets
OPEN_SET = {"("}
CLOSE_TO_OPEN = {")": "("}


def parse_all_pairs(db: str) -> Dict[int, int]:
    """Return base-pair positions for the structure string."""
    stacks = {op: [] for op in OPEN_SET}
    pairs = {}
    for i, ch in enumerate(db):
        if ch in OPEN_SET:
            stacks[ch].append(i)
        elif ch in CLOSE_TO_OPEN:
            opener = CLOSE_TO_OPEN[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs[j], pairs[i] = i, j
    return pairs


def detect_rna_structures(db: str, mode: str = "strict"):
    """Assign structure-element labels to a dot-bracket string."""
    db = db.strip()
    n = len(db)
    all_pairs = parse_all_pairs(db)

    primary_pairs = []
    stack = []
    for i, ch in enumerate(db):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                primary_pairs.append((j, i))

    labels = ["."] * n
    for i in all_pairs:
        labels[i] = "S"

    parent_of = [None] * n
    for u, v in sorted(primary_pairs, key=lambda x: x[1] - x[0], reverse=True):
        for i in range(u + 1, v):
            parent_of[i] = (u, v)

    children_of = {p: [] for p in primary_pairs}
    for c_start, c_end in sorted(primary_pairs):
        p = parent_of[c_start]
        if p:
            children_of[p].append((c_start, c_end))

    has_s_left = [False] * n
    last_s = False
    for i in range(n):
        has_s_left[i] = last_s
        if labels[i] == "S":
            last_s = True

    has_s_right = [False] * n
    next_s = False
    for i in range(n - 1, -1, -1):
        has_s_right[i] = next_s
        if labels[i] == "S":
            next_s = True

    for i in range(n):
        if labels[i] == "S":
            continue

        p = parent_of[i]

        if p is None:
            if has_s_left[i] and has_s_right[i]:
                labels[i] = "M" if mode == "pragmatic" else "X"
            else:
                labels[i] = "E"
        else:
            kids = children_of[p]
            if not kids:
                labels[i] = "H"
            elif len(kids) == 1:
                c_start, c_end = kids[0]
                u, v = p
                l_gap = c_start - u - 1
                r_gap = v - c_end - 1
                labels[i] = "B" if (l_gap == 0 or r_gap == 0) else "I"
            else:
                labels[i] = "M"

    return "".join(labels)


def main():
    """Command-line interface for structure labeling."""
    parser = argparse.ArgumentParser(
        description="RNA Structure Labeler CLI: Converts dot-bracket notation into structural labels."
    )

    parser.add_argument(
        "structure",
        type=str,
        help="The RNA secondary structure in dot-bracket notation (e.g., '((...))').",
    )

    parser.add_argument(
        "--mode",
        choices=["strict", "pragmatic", "both"],
        default="both",
        help="Select which labeling mode to display (default: both).",
    )

    args = parser.parse_args()

    try:
        strict_label = None
        pragmatic_label = None

        if args.mode in ["strict", "both"]:
            strict_label = detect_rna_structures(args.structure, mode="strict")

        if args.mode in ["pragmatic", "both"]:
            pragmatic_label = detect_rna_structures(args.structure, mode="pragmatic")

        print(f"\nLabels for: {args.structure}")
        print("-" * (len(args.structure) + 14))
        if strict_label is not None:
            print(f"Strict Mode:    {strict_label}")
        if pragmatic_label is not None:
            print(f"Pragmatic Mode: {pragmatic_label}")
        print("-" * (len(args.structure) + 14))
        print(f"Total Length: {len(args.structure)} bases")

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
