#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

from normalize_dbn import parse_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect adjacent base pairs in structures")
    parser.add_argument("input", type=Path, help="Path to merged/filtered structs pickle")
    args = parser.parse_args()

    with args.input.open("rb") as f:
        data: dict[str, dict[str, int]] = pickle.load(f)

    total_structs = 0
    affected_structs = 0
    affected_seqs = 0

    for counts in data.values():
        seq_affected = False
        for struct in counts:
            total_structs += 1
            pairs = parse_pairs(struct)
            if any(abs(i - j) <= 1 for i, j in pairs):
                affected_structs += 1
                seq_affected = True
        if seq_affected:
            affected_seqs += 1

    print(f"Total structs:          {total_structs}")
    print(f"Structs with adj pairs: {affected_structs}")
    print(f"Seqs affected:          {affected_seqs}")


if __name__ == "__main__":
    main()