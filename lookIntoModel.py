#!/usr/bin/env python3
import argparse
import pprint
from contextlib import redirect_stdout
from pathlib import Path

import torch


def shape_of_state_dict(state_dict: dict) -> dict[str, tuple[int, ...] | str]:
    out: dict[str, tuple[int, ...] | str] = {}
    for key, value in state_dict.items():
        shape = getattr(value, "shape", None)
        out[key] = tuple(shape) if shape is not None else type(value).__name__
    return out


def print_section(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def dump_checkpoint(ckpt_path: Path, show_raw: bool) -> None:
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")

    print_section("CHECKPOINT PATH")
    print(str(ckpt_path))

    print()
    print_section("TOP-LEVEL KEYS")
    if not isinstance(obj, dict):
        print(type(obj).__name__)
        return
    print(list(obj.keys()))

    print()
    print_section("TOP-LEVEL VALUES")
    for key, value in obj.items():
        if isinstance(value, dict):
            print(f"{key}: dict(len={len(value)})")
        elif isinstance(value, list):
            print(f"{key}: list(len={len(value)})")
        else:
            print(f"{key}: {type(value).__name__} = {value}")

    print()
    print_section("META")
    meta = obj.get("meta")
    if isinstance(meta, dict):
        pprint.pprint(meta, sort_dicts=False)
    else:
        print("no 'meta' present")

    print()
    print_section("MODEL STATE_DICT KEYS")
    model_sd = obj.get("model")
    if isinstance(model_sd, dict):
        for key in model_sd:
            print(key)
    else:
        print("no 'model' present")

    print()
    print_section("MODEL STATE_DICT SHAPES")
    if isinstance(model_sd, dict):
        pprint.pprint(shape_of_state_dict(model_sd), sort_dicts=False)
    else:
        print("no 'model' present")

    print()
    print_section("OPTIMIZER")
    opt = obj.get("opt")
    if isinstance(opt, dict):
        print("keys:", list(opt.keys()))
        if "param_groups" in opt:
            print("\nparam_groups:")
            pprint.pprint(opt["param_groups"], sort_dicts=False)
        if "state" in opt and isinstance(opt["state"], dict):
            print(f"\nstate entries: {len(opt['state'])}")
    else:
        print("no 'opt' present")

    if show_raw:
        print()
        print_section("RAW FULL OBJECT")
        pprint.pprint(obj, sort_dicts=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump a checkpoint to stdout or a text file")
    parser.add_argument("ckpt", type=Path, help="Path to checkpoint file, e.g. best.pt")
    parser.add_argument("--output", type=Path, default=None, help="Optional output text file path")
    parser.add_argument("--raw", action="store_true", help="Also print the raw full checkpoint object")
    args = parser.parse_args()

    ckpt_path = args.ckpt.expanduser().resolve()

    if args.output is None:
        dump_checkpoint(ckpt_path, show_raw=args.raw)
        return

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f, redirect_stdout(f):
        dump_checkpoint(ckpt_path, show_raw=args.raw)

    print(f"written to: {out_path}")


if __name__ == "__main__":
    main()