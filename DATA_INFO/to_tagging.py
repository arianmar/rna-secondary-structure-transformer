#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

from normalize_dbn import BRACKET_ORDER, BRACKET_PAIRS


def build_struct_alphabet(used_openers: set[str]) -> list[str]:
    alphabet: list[str] = []
    for op, cl in BRACKET_ORDER:
        if op in used_openers:
            alphabet.extend([op, cl])
    alphabet.append(".")
    return alphabet


def build_bracket_type_counts(single: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for struct in single.values():
        for ch in struct:
            if ch in BRACKET_PAIRS:
                counts[ch + BRACKET_PAIRS[ch]] += 1
    return dict(counts)


def build_bracket_type_order(type_counts: dict[str, int]) -> list[str]:
    order_index = {pair: i for i, pair in enumerate(BRACKET_ORDER)}
    return sorted(
        type_counts,
        key=lambda pair: (-type_counts[pair], order_index[pair]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert filtered pickle to tagging format")
    parser.add_argument("input", type=Path, help="Path to filtered pickle")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-mode", type=int, required=True, choices=[0, 1, 2])
    args = parser.parse_args()

    with args.input.open("rb") as f:
        data: dict[str, dict[str, int]] = pickle.load(f)

    single = {seq: next(iter(counts)) for seq, counts in data.items() if len(counts) == 1}
    dropped_multi = len(data) - len(single)

    if not single:
        raise SystemExit("No single-structure sequences left")

    seq_alphabet = sorted({ch for seq in single for ch in seq})
    max_length = max(len(seq) for seq in single)

    used_openers = {ch for struct in single.values() for ch in struct if ch in BRACKET_PAIRS}
    struct_alphabet = build_struct_alphabet(used_openers)

    bracket_pairs = {op: cl for op, cl in BRACKET_ORDER if op in used_openers}
    bracket_type_counts = build_bracket_type_counts(single)
    bracket_type_order = build_bracket_type_order(bracket_type_counts)

    meta = {
        "base_mode": int(args.base_mode),
        "seq_alphabet": seq_alphabet,
        "struct_alphabet": struct_alphabet,
        "bracket_pairs": bracket_pairs,
        "bracket_order": BRACKET_ORDER,
        "bracket_type_order": bracket_type_order,
        "bracket_type_counts": bracket_type_counts,
        "max_length": int(max_length),
        "nitems": int(len(single)),
    }

    out = {
        "meta": meta,
        "data": list(single.items()),
    }

    out_path = args.output or args.input.parent / f"{args.input.stem}_tag.pkl"
    with out_path.open("wb") as f:
        pickle.dump(out, f, protocol=4)

    print(f"Total seqs:        {len(data)}")
    print(f"Single-struct:     {len(single)}")
    print(f"Dropped (multi):   {dropped_multi}")
    print(f"Saved: {out_path}")
    print(f"  base_mode:           {meta['base_mode']}")
    print(f"  seq_alphabet:        {meta['seq_alphabet']}")
    print(f"  struct_alphabet:     {meta['struct_alphabet']}")
    print(f"  max_length:          {meta['max_length']}")
    print(f"  bracket_pairs:       {meta['bracket_pairs']}")
    print(f"  bracket_type_order:  {meta['bracket_type_order']}")


if __name__ == "__main__":
    main()