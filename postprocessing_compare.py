#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
import time

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader

from utils.checkpointing import load_model_from_checkpoint
from core.data import load_dataset
from core.dataset import RNADataset, make_collate
from core.postproc_dpgs import build_type_id_pairs, repair_dpgs
from core.postproc_ktd import repair_kill_to_dot
from core.structure import dotbracket_is_valid_ids


OUT_DIR_NAME = "postprocessing_compare"

MODE_ORDER = ("raw", "ktd", "dpgs")
MODE_LABELS = {"raw": "raw", "ktd": "KTD", "dpgs": "DPGS"}
MODE_STYLES = {"raw": "-", "ktd": "--", "dpgs": ":"}
METRICS = ("token_acc", "seq_exact", "paired_f1", "invalid", "pp_ms_per_seq")


def parse_snapshot_epoch(name: str) -> int | None:
    m = re.fullmatch(r"epoch_(\d+)", name)
    return None if m is None else int(m.group(1))


def list_snapshot_dirs(model_dir: str) -> list[tuple[int, str]]:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model dir not found: {model_dir}")

    items: list[tuple[int, str]] = []

    for entry in os.scandir(model_dir):
        if not entry.is_dir():
            continue

        epoch = parse_snapshot_epoch(entry.name)
        if epoch is None:
            continue

        ckpt_path = os.path.join(entry.path, "best.pt")
        split_path = os.path.join(entry.path, "split.pkl")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"missing checkpoint file: {ckpt_path}")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"missing split file: {split_path}")

        items.append((epoch, entry.path))

    items.sort(key=lambda x: x[0])

    if not items:
        raise FileNotFoundError(
            f"no epoch_* snapshot dirs with best.pt + split.pkl found in: {model_dir}"
        )

    return items


def build_val_loader(ds, split_path: str, batch_eval: int):
    with open(split_path, "rb") as f:
        split = pickle.load(f)

    val_items = [ds.items[i] for i in split["val"]]
    collate_fn = make_collate(ds.vocab.pad_x, ds.vocab.pad_y)

    return DataLoader(
        RNADataset(val_items, ds.vocab.base2id, ds.vocab.struct2id),
        batch_size=batch_eval,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def init_metric_store() -> dict[str, dict[str, list[float]]]:
    return {mode: {metric: [] for metric in METRICS} for mode in MODE_ORDER}


def append_mode_results(
    results: dict[str, dict[str, list[float]]],
    by_mode: dict[str, dict[str, float]],
) -> None:
    for mode in MODE_ORDER:
        for metric in METRICS:
            results[mode][metric].append(by_mode[mode][metric])


def print_mode_results(by_mode: dict[str, dict[str, float]]) -> None:
    for mode in MODE_ORDER:
        vals = by_mode[mode]
        print(
            f"  {MODE_LABELS[mode]:>4} | "
            f"acc={vals['token_acc']:.4f} | "
            f"seq={vals['seq_exact']:.4f} | "
            f"f1={vals['paired_f1']:.4f} | "
            f"inv={vals['invalid']:.4f} | "
            f"pp_ms={vals['pp_ms_per_seq']:.4f}"
        )


@torch.no_grad()
def evaluate_postprocessing_modes(
    model,
    loader,
    device: str,
    ds,
    type_id_pairs: list[tuple[int, int]],
    dot_id: int,
    progress_label: str = "",
) -> dict[str, dict[str, float]]:
    model.eval()

    paired_mask = ds.structure.paired_id_mask.cpu()

    correct = {mode: 0 for mode in MODE_ORDER}
    seq_exact = {mode: 0 for mode in MODE_ORDER}
    tp = {mode: 0 for mode in MODE_ORDER}
    fp = {mode: 0 for mode in MODE_ORDER}
    fn = {mode: 0 for mode in MODE_ORDER}
    invalid_seq = {mode: 0 for mode in MODE_ORDER}
    pp_time_s = {mode: 0.0 for mode in MODE_ORDER}

    token_count = 0
    seq_total = 0
    total_batches = len(loader)
    progress_every = max(1, total_batches // 5)

    for batch_idx, (x, y, mask) in enumerate(loader, start=1):
        if progress_label and (
            batch_idx == 1
            or batch_idx == total_batches
            or batch_idx % progress_every == 0
        ):
            print(
                f"[PROGRESS] {progress_label} batch {batch_idx}/{total_batches}",
                flush=True,
            )

        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        logits = model(x)
        valid = (y != ds.vocab.pad_y) & mask

        token_count += int(valid.sum().item())
        seq_total += int(x.size(0))

        y_cpu = y.cpu()
        valid_cpu = valid.cpu()
        logits_cpu = logits.cpu()

        raw_pred = logits_cpu.argmax(dim=-1)
        preds: dict[str, torch.Tensor] = {"raw": raw_pred}

        t0 = time.perf_counter()
        ktd_pred = raw_pred.clone()
        for b in range(ktd_pred.size(0)):
            ktd_pred[b] = repair_kill_to_dot(
                pred_ids=ktd_pred[b],
                valid_mask=valid_cpu[b],
                structure=ds.structure,
                dot_id=dot_id,
            )
        pp_time_s["ktd"] += time.perf_counter() - t0
        preds["ktd"] = ktd_pred

        t0 = time.perf_counter()
        dpgs_pred = raw_pred.clone()
        for b in range(dpgs_pred.size(0)):
            dpgs_pred[b] = repair_dpgs(
                pred_ids=dpgs_pred[b],
                logits=logits_cpu[b],
                valid_mask=valid_cpu[b],
                type_id_pairs=type_id_pairs,
                dot_id=dot_id,
            )
        pp_time_s["dpgs"] += time.perf_counter() - t0
        preds["dpgs"] = dpgs_pred

        y_paired = paired_mask[y_cpu] & valid_cpu

        for mode in MODE_ORDER:
            pred = preds[mode]

            correct[mode] += int(((pred == y_cpu) & valid_cpu).sum().item())
            seq_exact[mode] += int(
                (((pred == y_cpu) | (~valid_cpu)).all(dim=1)).sum().item()
            )

            p_paired = paired_mask[pred] & valid_cpu
            tp[mode] += int((p_paired & y_paired).sum().item())
            fp[mode] += int((p_paired & (~y_paired)).sum().item())
            fn[mode] += int(((~p_paired) & y_paired).sum().item())

            for b in range(pred.size(0)):
                if not dotbracket_is_valid_ids(pred[b], valid_cpu[b], ds.structure):
                    invalid_seq[mode] += 1

    out: dict[str, dict[str, float]] = {}

    for mode in MODE_ORDER:
        precision = tp[mode] / max(tp[mode] + fp[mode], 1)
        recall = tp[mode] / max(tp[mode] + fn[mode], 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

        out[mode] = {
            "token_acc": float(correct[mode] / max(token_count, 1)),
            "seq_exact": float(seq_exact[mode] / max(seq_total, 1)),
            "paired_f1": float(f1),
            "invalid": float(invalid_seq[mode] / max(seq_total, 1)),
            "pp_ms_per_seq": (
                0.0
                if mode == "raw"
                else float((pp_time_s[mode] / max(seq_total, 1)) * 1000.0)
            ),
        }

    return out


def save_fig(path: str) -> None:
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[OK] wrote {path}")


def plot_modes(
    epochs: list[int],
    results: dict[str, dict[str, list[float]]],
    metric: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    plt.figure()

    for mode in MODE_ORDER:
        plt.plot(
            epochs,
            results[mode][metric],
            linestyle=MODE_STYLES[mode],
            linewidth=2,
            label=MODE_LABELS[mode],
        )

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    save_fig(out_path)


def write_results(
    out_dir: str,
    epochs: list[int],
    results: dict[str, dict[str, list[float]]],
) -> None:
    compare_json_path = os.path.join(out_dir, "compare_results.json")
    with open(compare_json_path, "w", encoding="utf-8") as f:
        json.dump({"epochs": epochs, "results": results}, f, ensure_ascii=False, indent=2)

    for mode in MODE_ORDER:
        mode_json_path = os.path.join(out_dir, f"{mode}_results.json")
        with open(mode_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epochs": epochs,
                    "mode": mode,
                    "metrics": results[mode],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print(f"\n[OK] wrote compare json: {compare_json_path}")
    print(f"[OK] wrote per-mode json files in: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare transformer postprocessing modes"
    )
    parser.add_argument("model_dir", type=str)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--batch-eval", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    snapshot_items = list_snapshot_dirs(args.model_dir)
    out_dir = os.path.join(args.model_dir, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    first_ckpt_path = os.path.join(snapshot_items[0][1], "best.pt")
    first_split_path = os.path.join(snapshot_items[0][1], "split.pkl")

    _, first_bundle, first_ckpt = load_model_from_checkpoint(first_ckpt_path, "cpu")
    first_meta = first_ckpt.get("meta") or {}
    ckpt_data_path = first_meta.get("data_path")
    
    if args.data_path is not None:
        ds_path = args.data_path
    elif ckpt_data_path is not None:
        ds_path = ckpt_data_path
    else:
        raise SystemExit(
            "No dataset path available. Pass --data-path explicitly or use a checkpoint "
            "that contains meta['data_path']."
        )

    ds = load_dataset(ds_path)

    if ds.base_mode != first_bundle["base_mode"]:
        raise SystemExit(
            f"Dataset/checkpoint mismatch: dataset base_mode={ds.base_mode}, "
            f"checkpoint base_mode={first_bundle['base_mode']}"
        )

    if ds.vocab.max_len != first_bundle["max_len"]:
        raise SystemExit(
            f"Dataset/checkpoint mismatch: dataset max_len={ds.vocab.max_len}, "
            f"checkpoint max_len={first_bundle['max_len']}"
        )

    if ds.vocab.seq_alphabet != first_bundle["seq_alphabet"]:
        raise SystemExit("Dataset/checkpoint mismatch: seq_alphabet differs")

    if ds.vocab.struct_alphabet != first_bundle["struct_alphabet"]:
        raise SystemExit("Dataset/checkpoint mismatch: struct_alphabet differs")

    if ds.structure.bracket_pairs != first_bundle["bracket_pairs"]:
        raise SystemExit("Dataset/checkpoint mismatch: bracket_pairs differs")
    
    val_loader = build_val_loader(ds, first_split_path, args.batch_eval)

    device = "cpu" if args.cpu else get_device()
    print(f"[INFO] device={device}")
    print(f"[INFO] model_dir={args.model_dir}")
    print(f"[INFO] data_path={ds_path}")
    print(f"[INFO] base_mode={ds.base_mode}")
    print(f"[INFO] checkpoints={[epoch for epoch, _ in snapshot_items]}")

    dot_id = ds.vocab.struct2id["."]
    type_id_pairs = build_type_id_pairs(
        first_bundle["struct2id"],
        first_bundle["structure"].bracket_pairs,
        first_bundle["structure"].bracket_type_order,
    )

    results = init_metric_store()
    epochs: list[int] = []
    total_models = len(snapshot_items)

    for model_idx, (snapshot_epoch, snapshot_dir) in enumerate(snapshot_items, start=1):
        ckpt_path = os.path.join(snapshot_dir, "best.pt")
        model, _, ckpt = load_model_from_checkpoint(ckpt_path, device)

        epochs.append(snapshot_epoch)

        print(
            f"\n[INFO] model {model_idx}/{total_models} | "
            f"snapshot_epoch={snapshot_epoch} | "
            f"best_epoch={ckpt.get('epoch', -1)}",
            flush=True,
        )

        t0 = time.time()
        by_mode = evaluate_postprocessing_modes(
            model=model,
            loader=val_loader,
            device=device,
            ds=ds,
            type_id_pairs=type_id_pairs,
            dot_id=dot_id,
            progress_label=f"snapshot_epoch={snapshot_epoch} VAL",
        )
        print(f"[INFO] total={time.time() - t0:.1f}s")

        print_mode_results(by_mode)
        append_mode_results(results, by_mode)

    plot_modes(
        epochs,
        results,
        "token_acc",
        "Validation token accuracy",
        "token_acc",
        os.path.join(out_dir, "val_token_acc_compare.png"),
    )
    plot_modes(
        epochs,
        results,
        "seq_exact",
        "Validation sequence exact",
        "seq_exact",
        os.path.join(out_dir, "val_seq_exact_compare.png"),
    )
    plot_modes(
        epochs,
        results,
        "paired_f1",
        "Validation paired F1",
        "paired_f1",
        os.path.join(out_dir, "val_paired_f1_compare.png"),
    )
    plot_modes(
        epochs,
        results,
        "invalid",
        "Validation invalid rate",
        "invalid",
        os.path.join(out_dir, "val_invalid_compare.png"),
    )
    plot_modes(
        epochs,
        results,
        "pp_ms_per_seq",
        "Validation postprocessing time per sequence",
        "pp_ms_per_seq",
        os.path.join(out_dir, "val_postproc_ms_compare.png"),
    )

    write_results(out_dir, epochs, results)
    print(f"[OK] wrote plots in: {out_dir}\n")


if __name__ == "__main__":
    main()