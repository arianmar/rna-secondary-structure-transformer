#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x.strip())
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


@dataclass
class EpochSeries:
    epoch: list[int]
    gs: list[int]
    train_loss: list[float]
    val_loss: list[float]
    val_acc: list[float]
    val_seq: list[float]
    val_f1: list[float]
    val_inv: list[float]


WORK_DIRNAME = "work"


def resolve_run_dir(path_arg: str) -> Path:
    path = Path(path_arg).expanduser().resolve()

    if path.is_file():
        if path.name != "best.pt":
            raise SystemExit("If a file path is given, it must point to best.pt")
        parent = path.parent
        if parent.name == WORK_DIRNAME:
            return parent
        return parent

    if path.is_dir():
        w = path / WORK_DIRNAME
        return w if w.is_dir() else path

    raise SystemExit(f"path not found: {path}")


def read_metrics(metrics_path: Path) -> EpochSeries:
    epochs = EpochSeries([], [], [], [], [], [], [], [])
    bad_lines = 0

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            if obj.get("kind") != "epoch":
                continue

            ep = obj.get("epoch")
            gs = obj.get("global_step")
            tr = to_float(obj.get("train_loss"))
            vl = to_float(obj.get("val_loss"))

            if ep is None or gs is None or tr is None or vl is None:
                continue

            val_acc = to_float(obj.get("val_acc"))
            val_seq = to_float(obj.get("val_seq"))
            val_f1 = to_float(obj.get("val_f1"))
            val_inv = to_float(obj.get("val_inv"))

            epochs.epoch.append(int(ep))
            epochs.gs.append(int(gs))
            epochs.train_loss.append(tr)
            epochs.val_loss.append(vl)
            epochs.val_acc.append(val_acc if val_acc is not None else float("nan"))
            epochs.val_seq.append(val_seq if val_seq is not None else float("nan"))
            epochs.val_f1.append(val_f1 if val_f1 is not None else float("nan"))
            epochs.val_inv.append(val_inv if val_inv is not None else float("nan"))

    if bad_lines:
        print(f"[WARN] skipped {bad_lines} bad lines")

    print(f"[INFO] epoch points: {len(epochs.epoch)}")
    return epochs


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[OK] wrote {path}")


def plot_line(x, ys: list[tuple[list[float], str]], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    for values, label in ys:
        plt.plot(x, values, linewidth=2, label=label)
    plt.xlabel("epoch")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    save_fig(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training and validation metrics from a run directory or best.pt path"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to run dir or best.pt",
    )
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.path)
    metrics_path = run_dir / "metrics.jsonl"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise SystemExit(f"metrics file not found: {metrics_path}")

    ep = read_metrics(metrics_path)

    if ep.epoch:
        plot_line(
            ep.epoch,
            [(ep.train_loss, "train_loss"), (ep.val_loss, "val_loss")],
            "Train vs val loss",
            "loss",
            out_dir / "epoch_losses.png",
        )

        plot_line(
            ep.epoch,
            [(ep.val_acc, "val_token_acc"), (ep.val_seq, "val_seq_exact")],
            "Validation accuracy",
            "score",
            out_dir / "val_accuracy.png",
        )

        plot_line(
            ep.epoch,
            [(ep.val_f1, "val_paired_f1"), (ep.val_inv, "val_invalid_rate")],
            "Validation F1 + invalid rate",
            "score",
            out_dir / "val_f1_invalid.png",
        )

    print(f"\nDone. Plots are in: {out_dir}\n")


if __name__ == "__main__":
    main()