#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path


STANDARD = set("GUAC")
IUPAC_EXT = set("RYSWKMBDHVNIT")


def print_group(title: str, chars: set[str], seq_counts: dict[str, int], total_counts: dict[str, int]) -> None:
    print(f"\n[{title}]")
    rows = [
        (ch, seq_counts.get(ch, 0), total_counts.get(ch, 0))
        for ch in chars
        if ch in total_counts
    ]
    if not rows:
        print("  (none found)")
        return

    print(f"  {'char':<8} {'seqs':>10} {'total':>12}")
    print(f"  {'-' * 32}")
    for ch, seqs, total in sorted(rows, key=lambda x: (-x[2], x[0])):
        print(f"  {repr(ch):<8} {seqs:>10} {total:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sequence and structure characters")
    parser.add_argument("input", type=Path, help="Path to merged/filtered structs pickle")
    args = parser.parse_args()

    with args.input.open("rb") as f:
        data: dict[str, dict[str, int]] = pickle.load(f)

    seq_counts: dict[str, int] = {}
    seq_total: dict[str, int] = {}

    for seq in data:
        seen = set()
        for ch in seq.upper():
            seq_total[ch] = seq_total.get(ch, 0) + 1
            if ch not in seen:
                seq_counts[ch] = seq_counts.get(ch, 0) + 1
                seen.add(ch)

    found_seq_chars = set(seq_total)
    unknown_seq = found_seq_chars - STANDARD - IUPAC_EXT

    print_group("STANDARD — GUAC", STANDARD, seq_counts, seq_total)
    print_group("IUPAC extended — RYSWKMBDHVNIT", IUPAC_EXT, seq_counts, seq_total)
    print_group("UNKNOWN", unknown_seq, seq_counts, seq_total)

    struct_seq_counts: dict[str, int] = {}
    struct_total: dict[str, int] = {}

    for counts in data.values():
        seen = set()
        for struct in counts:
            for ch in struct:
                struct_total[ch] = struct_total.get(ch, 0) + 1
                if ch not in seen:
                    struct_seq_counts[ch] = struct_seq_counts.get(ch, 0) + 1
                    seen.add(ch)

    print("\n[STRUCT CHARS]")
    if not struct_total:
        print("  (none found)")
        return

    print(f"  {'char':<8} {'seqs':>10} {'total':>12}")
    print(f"  {'-' * 32}")
    for ch, total in sorted(struct_total.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {repr(ch):<8} {struct_seq_counts.get(ch, 0):>10} {total:>12}")


if __name__ == "__main__":
    main()