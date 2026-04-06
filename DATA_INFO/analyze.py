#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path


def load_data(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def analyze_struct_counts(data: dict[str, dict[str, int]]) -> None:
    total_seqs = len(data)
    total_unique_structs = sum(len(counts) for counts in data.values())
    total_structs_with_dups = sum(sum(counts.values()) for counts in data.values())
    multi_struct_seqs = sum(1 for counts in data.values() if len(counts) > 1)

    print(f"Sequences:                  {total_seqs}")
    print(f"Unique (seq, struct) pairs: {total_unique_structs}")
    print(f"Struct count incl. dups:    {total_structs_with_dups}")
    print(f"Seqs with >1 struct:        {multi_struct_seqs}")


def analyze_tagging(data: dict) -> None:
    items = data["data"]
    meta = data.get("meta", {})

    print(f"Items:                      {len(items)}")
    print(f"Max length:                 {meta.get('max_length', 'n/a')}")
    print(f"Base mode:                  {meta.get('base_mode', 'n/a')}")
    print(f"Seq alphabet:               {meta.get('seq_alphabet', [])}")
    print(f"Struct alphabet:            {meta.get('struct_alphabet', [])}")
    print(f"Bracket pairs:              {meta.get('bracket_pairs', {})}")
    print(f"Bracket type order:         {meta.get('bracket_type_order', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a pickle file")
    parser.add_argument("input", type=Path, help="Path to pickle file")
    args = parser.parse_args()

    data = load_data(args.input)

    if isinstance(data, dict) and "meta" in data and "data" in data:
        analyze_tagging(data)
        return

    if isinstance(data, dict):
        analyze_struct_counts(data)
        return

    raise SystemExit("Unsupported pickle format")


if __name__ == "__main__":
    main()