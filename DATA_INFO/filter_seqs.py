#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path


STANDARD = set("GUAC")
IUPAC = set("RYSWKMBDHVN")
I_OR_T = {"I", "T"}
KNOWN_NON_GUAC = STANDARD | IUPAC | I_OR_T


def is_unknown(ch: str) -> bool:
    return ch.upper() not in KNOWN_NON_GUAC


def process_seq(seq: str, args) -> str | None:
    seq = seq.upper()

    if args.t2u:
        seq = seq.replace("T", "U")

    if args.i2g:
        seq = seq.replace("I", "G")

    if args.unk2n:
        seq = "".join(ch if not is_unknown(ch) else "N" for ch in seq)

    if args.guacn:
        seq = "".join("N" if ch in IUPAC else ch for ch in seq)

    if args.maxlen is not None and len(seq) > args.maxlen:
        return None

    if args.guac:
        if not all(ch in STANDARD for ch in seq):
            return None
    elif args.guac_plus or args.guacn:
        if not all(ch in (STANDARD | IUPAC) for ch in seq):
            return None

    return seq


def build_output_name(args) -> str:
    parts = ["filtered"]

    if args.maxlen is not None:
        parts.append(str(args.maxlen))
        
    if args.guac:
        parts.append("guac")
    elif args.guacn:
        parts.append("guacn")
    elif args.guac_plus:
        parts.append("guac+")

    if args.t2u:
        parts.append("t2u")
    if args.i2g:
        parts.append("i2g")
    if args.unk2n:
        parts.append("unk2n")
    if args.single_struct:
        parts.append("single")

    return "_".join(parts) + ".pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter and transform merged_structs.pkl")
    parser.add_argument("--input", type=Path, default=Path("./data/merged_structs.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--maxlen", type=int, default=None)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--guac", action="store_true", help="Keep only GUAC sequences")
    mode.add_argument("--guac+", dest="guac_plus", action="store_true", help="Keep GUAC + IUPAC")
    mode.add_argument("--guacn", action="store_true", help="Replace IUPAC codes with N")

    parser.add_argument("--t2u", action="store_true", help="Replace T with U")
    parser.add_argument("--i2g", action="store_true", help="Replace I with G")
    parser.add_argument("--unk2n", action="store_true", help="Replace unknown chars with N")
    parser.add_argument("--single-struct", action="store_true", help="Keep only sequences with exactly 1 struct")
    args = parser.parse_args()

    if args.guac and args.unk2n:
        raise SystemExit("--guac and --unk2n are incompatible")

    with args.input.open("rb") as f:
        data: dict[str, dict[str, int]] = pickle.load(f)

    out: dict[str, dict[str, int]] = {}
    dropped_seqs = 0

    for seq, counts in data.items():
        new_seq = process_seq(seq, args)
        if new_seq is None:
            dropped_seqs += 1
            continue

        dst = out.setdefault(new_seq, {})
        for struct, cnt in counts.items():
            dst[struct] = dst.get(struct, 0) + cnt

    dropped_multi = 0
    if args.single_struct:
        before = len(out)
        out = {seq: counts for seq, counts in out.items() if len(counts) == 1}
        dropped_multi = before - len(out)

    output_name = args.output if args.output is not None else build_output_name(args)
    output_path = args.output_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(out, f, protocol=4)

    total_structs = sum(sum(counts.values()) for counts in out.values())

    print(f"Loaded sequences:         {len(data)}")
    print(f"Dropped by seq filter:    {dropped_seqs}")
    if args.single_struct:
        print(f"Dropped by single-struct: {dropped_multi}")
    print(f"Kept sequences:           {len(out)}")
    print(f"Total structs:            {total_structs}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()