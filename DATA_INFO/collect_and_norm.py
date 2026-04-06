#!/usr/bin/env python3
import argparse
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

from normalize_dbn import normalize


def read_non_comment_lines(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> list[str]:
    lines: list[str] = []
    with zf.open(info) as f:
        for raw in f:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def collect_from_zip(zip_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = [info for info in zf.infolist() if not info.is_dir()]
        total = len(files)

        for idx, info in enumerate(files, start=1):
            lines = read_non_comment_lines(zf, info)
            if len(lines) < 2:
                print(f"skip {info.filename}: less than 2 non-comment lines")
                continue

            seq = lines[0].upper()
            struct = lines[1]

            if len(seq) != len(struct):
                print(f"skip {info.filename}: len mismatch ({len(seq)} vs {len(struct)})")
                continue

            try:
                norm = normalize(struct)
            except ValueError as e:
                print(f"skip {info.filename}: {e}")
                continue

            out[seq].append(norm)

            if idx % 10_000 == 0 or idx == total:
                print(f"... {idx} / {total} files read")

    return dict(out)


def to_counts(structs: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for struct in structs:
        counts[struct] += 1
    return dict(counts)


def print_stats(label: str, raw: dict[str, list[str]], counted: dict[str, dict[str, int]]) -> None:
    total_raw = sum(len(v) for v in raw.values())
    total_with_dups = sum(sum(v.values()) for v in counted.values())
    total_unique_pairs = sum(len(v) for v in counted.values())
    deduped_same_only = sum(
        1 for seq, structs in raw.items() if len(structs) > 1 and len(set(structs)) == 1
    )
    single_struct = sum(1 for counts in counted.values() if len(counts) == 1)
    multi_struct = sum(1 for counts in counted.values() if len(counts) > 1)

    print(f"\n[{label}]")
    print(f"  unique sequences:          {len(counted)}")
    print(f"  structs before dedup:      {total_raw}")
    print(f"  structs incl. dups:        {total_with_dups}")
    print(f"  unique (seq, struct):      {total_unique_pairs}")
    print(f"  seqs deduped to one struct:{deduped_same_only}")
    print(f"  seqs with 1 struct:        {single_struct}")
    print(f"  seqs with >1 struct:       {multi_struct}")


def merge_counts(
    left: dict[str, dict[str, int]],
    right: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {}

    for seq in set(left) | set(right):
        combined: dict[str, int] = defaultdict(int)
        for struct, cnt in left.get(seq, {}).items():
            combined[struct] += cnt
        for struct, cnt in right.get(seq, {}).items():
            combined[struct] += cnt
        merged[seq] = dict(combined)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and normalize DBN/STA files")
    parser.add_argument("--dbn-zip", type=Path, default=Path("./dataRaw/dbnFiles.zip"))
    parser.add_argument("--sta-zip", type=Path, default=Path("./dataRaw/staFiles.zip"))
    parser.add_argument("--output", type=Path, default=Path("./data/merged_structs.pkl"))
    args = parser.parse_args()

    print("Reading DBN...")
    dbn_raw = collect_from_zip(args.dbn_zip)
    dbn_data = {seq: to_counts(structs) for seq, structs in dbn_raw.items()}
    print_stats("DBN", dbn_raw, dbn_data)

    print("\nReading STA...")
    sta_raw = collect_from_zip(args.sta_zip)
    sta_data = {seq: to_counts(structs) for seq, structs in sta_raw.items()}
    print_stats("STA", sta_raw, sta_data)

    dbn_pairs = {(seq, struct) for seq, counts in dbn_data.items() for struct in counts}
    sta_pairs = {(seq, struct) for seq, counts in sta_data.items() for struct in counts}

    only_dbn = dbn_pairs - sta_pairs
    only_sta = sta_pairs - dbn_pairs
    both = dbn_pairs & sta_pairs

    print("\n[diff]")
    print(f"  (seq, struct) only in DBN: {len(only_dbn)}")
    print(f"  (seq, struct) only in STA: {len(only_sta)}")
    print(f"  (seq, struct) in both:     {len(both)}")

    merged = merge_counts(dbn_data, sta_data)

    total_with_dups = sum(sum(v.values()) for v in merged.values())
    total_unique_pairs = sum(len(v) for v in merged.values())
    single_struct = sum(1 for v in merged.values() if len(v) == 1)
    multi_struct = sum(1 for v in merged.values() if len(v) > 1)

    print("\n[merged]")
    print(f"  total unique sequences:    {len(merged)}")
    print(f"  total structs incl. dups:  {total_with_dups}")
    print(f"  total unique (seq, struct):{total_unique_pairs}")
    print(f"  seqs with 1 struct:        {single_struct}")
    print(f"  seqs with >1 struct:       {multi_struct}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(merged, f, protocol=4)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()