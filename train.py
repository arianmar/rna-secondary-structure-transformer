# train.py
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from core.data import load_dataset, make_split
from core.dataset import make_collate
from core.engine import build_eval_loaders, evaluate, run_test_eval, train_epoch
from core.model import TransformerEncoderTagger
from utils.checkpointing import (
    append_hist_if_changed,
    build_ckpt_meta,
    build_model_dir,
    load_checkpoint,
    meta_last,
    save_checkpoint_dir_snapshot,
    save_ckpt,
    validate_model_dir_for_resume,
    work_dir,
)
from utils.config import parse_args
from utils.logger import end_logging, log, log_startup, metrics_write, setup_logging

_SNAPSHOT_DIR_RE = re.compile(r"epoch_(\d+)$")
_SNAPSHOT_FILES = ("last.pt", "best.pt", "split.pkl", "train.log", "metrics.jsonl")


def _resolve_snapshot_dir(path_arg: str) -> str | None:
    p = Path(path_arg).expanduser().resolve()
    if not p.is_dir() or _SNAPSHOT_DIR_RE.fullmatch(p.name) is None:
        return None
    return str(p)


def _ensure_full_snapshot(snapshot_dir: str) -> None:
    missing = [n for n in _SNAPSHOT_FILES if not (Path(snapshot_dir) / n).is_file()]
    if missing:
        raise SystemExit(
            "[ERR] snapshot is incomplete (expected: last.pt, best.pt, split.pkl, train.log, metrics.jsonl). "
            f"Missing in {snapshot_dir}: {', '.join(missing)}"
        )


def _import_snapshot_to_parent_work(snapshot_dir: str) -> str:
    _ensure_full_snapshot(snapshot_dir)

    snap = Path(snapshot_dir)
    model_dir = snap.parent
    wdir = model_dir / "work"
    wdir.mkdir(parents=True, exist_ok=True)

    for name in _SNAPSHOT_FILES:
        shutil.copy2(snap / name, wdir / name)

    return str(model_dir)


def _resolve_current_model_dir(path_arg: str) -> str:
    p = Path(path_arg).expanduser().resolve()

    if not p.exists() or not p.is_dir():
        raise SystemExit(f"[ERR] mode='current': path not found or not a directory: {path_arg}")

    snap_dir = _resolve_snapshot_dir(str(p))
    if snap_dir is not None:
        return _import_snapshot_to_parent_work(snap_dir)

    if p.name == "work":
        p = p.parent

    wdir = p / "work"
    if not wdir.is_dir():
        raise SystemExit(
            "[ERR] mode='current': model directory must contain a 'work/' directory "
            f"with last.pt, best.pt, split.pkl, train.log, metrics.jsonl. Missing work/ in: {p}"
        )

    return str(p)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def paths_in(model_dir: str) -> tuple[str, str, str, str, str]:
    w = work_dir(model_dir)
    return (
        os.path.join(w, "train.log"),
        os.path.join(w, "metrics.jsonl"),
        os.path.join(w, "last.pt"),
        os.path.join(w, "best.pt"),
        os.path.join(w, "split.pkl"),
    )


def flag_was_set(name: str) -> bool:
    p = f"--{name}"
    return any(a == p or a.startswith(p + "=") for a in sys.argv)


def _update_meta_immutables(meta: dict, cfg, ds, data_path: str) -> None:
    meta.update(
        {
            "data_path": os.path.abspath(data_path),
            "max_length": int(ds.vocab.max_len),
            "base_mode": int(ds.base_mode),
            "d_model": int(cfg.d_model),
            "n_heads": int(cfg.n_heads),
            "n_layers": int(cfg.n_layers),
            "d_head": int(cfg.d_head),
            "d_ff": int(cfg.d_ff),
            "dropout": float(cfg.dropout),
            "vocab_in": int(ds.vocab.vocab_in),
            "num_classes": int(ds.vocab.num_classes),
            "pad_x": int(ds.vocab.pad_x),
            "pad_y": int(ds.vocab.pad_y),
            "seq_alphabet": ds.vocab.seq_alphabet,
            "struct_alphabet": ds.vocab.struct_alphabet,
            "bracket_pairs": ds.structure.bracket_pairs,
            "bracket_order": ds.structure.bracket_order,
            "bracket_type_order": ds.structure.bracket_type_order,
            "bracket_type_counts": ds.structure.bracket_type_counts,
        }
    )


def cosine_lr_for_epoch(epoch: int, start_epoch: int, run_epochs: int, lr: float, eta_min: float) -> float:
    if run_epochs <= 1:
        return float(eta_min)
    end_epoch = start_epoch + run_epochs - 1
    if epoch <= start_epoch:
        return float(lr)
    if epoch >= end_epoch:
        return float(eta_min)
    t = (epoch - start_epoch) / float(run_epochs - 1)
    c = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(eta_min + (lr - eta_min) * c)


def set_optimizer_lr(opt, lr: float) -> None:
    lr = float(lr)
    for pg in opt.param_groups:
        pg["lr"] = lr


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)

    cfg = parse_args()
    device = get_device()

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0

    if cfg.mode == "new":
        if cfg.data_path is None:
            raise SystemExit("--data-path is required in mode 'new'")

        data_path = cfg.data_path
        ds = load_dataset(data_path)

        model_dir = build_model_dir(cfg, ds)
        log_path, metrics_path, last_ckpt_path, best_ckpt_path, split_path = paths_in(model_dir)
        setup_logging(log_path, metrics_path, "new")

        ckpt_meta = build_ckpt_meta(cfg, ds, data_path, base_meta=None, start_epoch=1)
        ckpt = None

    else:
        if cfg.data_path is None:
            raise SystemExit("--data-path is required in mode 'current'")

        model_dir = _resolve_current_model_dir(cfg.data_path)
        validate_model_dir_for_resume(model_dir)

        log_path, metrics_path, last_ckpt_path, best_ckpt_path, split_path = paths_in(model_dir)
        setup_logging(log_path, metrics_path, "current")

        ckpt = load_checkpoint(last_ckpt_path, "cpu")
        base_meta = ckpt.get("meta") or {}
        data_path = base_meta.get("data_path")
        if not data_path:
            raise SystemExit(f"mode='current' but {last_ckpt_path} missing meta['data_path']")

        ds = load_dataset(data_path)

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf"))))
        bad_epochs = int(ckpt.get("bad_epochs", 0))

        immut = {
            "d-model": ("d_model", int(cfg.d_model)),
            "n-heads": ("n_heads", int(cfg.n_heads)),
            "n-layers": ("n_layers", int(cfg.n_layers)),
            "d-ff": ("d_ff", int(cfg.d_ff)),
            "dropout": ("dropout", float(cfg.dropout)),
        }
        for flag, (k, new_v) in immut.items():
            if flag_was_set(flag):
                old_v = base_meta.get(k)
                if old_v is not None and new_v != old_v:
                    raise SystemExit(f"mode='current': {k} mismatch (ckpt={old_v} vs arg={new_v})")

        if int(base_meta.get("base_mode", ds.base_mode)) != int(ds.base_mode):
            raise SystemExit("mode='current': base_mode mismatch")
        if int(base_meta.get("max_length", ds.vocab.max_len)) != int(ds.vocab.max_len):
            raise SystemExit("mode='current': max_length mismatch")
        if list(base_meta.get("seq_alphabet", ds.vocab.seq_alphabet)) != list(ds.vocab.seq_alphabet):
            raise SystemExit("mode='current': seq_alphabet mismatch")
        if list(base_meta.get("struct_alphabet", ds.vocab.struct_alphabet)) != list(ds.vocab.struct_alphabet):
            raise SystemExit("mode='current': struct_alphabet mismatch")
        if dict(base_meta.get("bracket_pairs", ds.structure.bracket_pairs)) != dict(ds.structure.bracket_pairs):
            raise SystemExit("mode='current': bracket_pairs mismatch")
        if int(base_meta.get("pad_x", ds.vocab.pad_x)) != int(ds.vocab.pad_x) or int(base_meta.get("pad_y", ds.vocab.pad_y)) != int(ds.vocab.pad_y):
            raise SystemExit("mode='current': pad_x/pad_y mismatch")
        if int(base_meta.get("num_classes", ds.vocab.num_classes)) != int(ds.vocab.num_classes):
            raise SystemExit("mode='current': num_classes mismatch")

        ckpt_meta = dict(base_meta)

        def upd(flag: str, key: str, cast, value, se: int):
            if flag_was_set(flag):
                append_hist_if_changed(ckpt_meta, key, cast(value), se)

        upd("batch-train", "batch_train", int, cfg.batch_train, start_epoch)
        upd("batch-val", "batch_val", int, cfg.batch_val, start_epoch)
        upd("epochs", "epochs_per_run", int, cfg.epochs, start_epoch)
        upd("patience", "patience", int, cfg.patience, start_epoch)
        upd("min-delta", "min_delta", float, cfg.min_delta, start_epoch)
        upd("lr", "lr", float, cfg.lr, start_epoch)
        upd("weight-decay", "weight_decay", float, cfg.weight_decay, start_epoch)
        upd("eta-min", "eta_min", float, cfg.eta_min, start_epoch)
        upd("log-every", "log_every", int, cfg.log_every, start_epoch)

        if flag_was_set("max-steps-per-epoch"):
            append_hist_if_changed(
                ckpt_meta,
                "max_steps_per_epoch",
                None if cfg.max_steps_per_epoch is None else int(cfg.max_steps_per_epoch),
                start_epoch,
            )

        upd("split-seed", "split_seed", int, cfg.split_seed, start_epoch)
        upd("snap-every", "snap_every", int, cfg.snap_every, start_epoch)

        _update_meta_immutables(ckpt_meta, cfg, ds, data_path)

    batch_train = int(meta_last(ckpt_meta, "batch_train", cfg.batch_train))
    batch_val = int(meta_last(ckpt_meta, "batch_val", cfg.batch_val))
    run_epochs = int(meta_last(ckpt_meta, "epochs_per_run", cfg.epochs))
    patience = int(meta_last(ckpt_meta, "patience", cfg.patience))
    min_delta = float(meta_last(ckpt_meta, "min_delta", cfg.min_delta))
    lr = float(meta_last(ckpt_meta, "lr", cfg.lr))
    weight_decay = float(meta_last(ckpt_meta, "weight_decay", cfg.weight_decay))
    eta_min = float(meta_last(ckpt_meta, "eta_min", cfg.eta_min))
    log_every = int(meta_last(ckpt_meta, "log_every", cfg.log_every))
    msteps = meta_last(ckpt_meta, "max_steps_per_epoch", cfg.max_steps_per_epoch)
    max_steps_per_epoch = None if msteps is None else int(msteps)
    split_seed = int(meta_last(ckpt_meta, "split_seed", cfg.split_seed))
    snap_every = int(meta_last(ckpt_meta, "snap_every", cfg.snap_every))

    if run_epochs <= 0:
        raise SystemExit("--epochs must be >= 1")
    if snap_every <= 0:
        raise SystemExit("--snap-every must be >= 1")
    if lr <= 0 or eta_min <= 0:
        raise SystemExit("lr and eta_min must be > 0")
    if eta_min > lr:
        raise SystemExit(f"eta_min ({eta_min}) must be <= lr ({lr})")

    collate_fn = make_collate(ds.vocab.pad_x, ds.vocab.pad_y)
    train_items, val_items, test_items, split_info = make_split(ds.items, split_path, split_seed, cfg.mode)
    val_loader, test_loader = build_eval_loaders(val_items, test_items, ds, batch_val, collate_fn)

    log_startup(f"device: {device}")
    if device == "cuda":
        log_startup(f"cuda device: {torch.cuda.get_device_name(0)}")
    log_startup(f"model dir: {model_dir}")
    log_startup(f"work dir: {work_dir(model_dir)}")
    log_startup(f"data: {data_path}")
    log_startup(f"split: train={split_info['train_size']} val={split_info['val_size']} test={split_info['test_size']}")

    model = TransformerEncoderTagger(
        vocab_in=ds.vocab.vocab_in,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        pad_x=ds.vocab.pad_x,
        num_classes=ds.vocab.num_classes,
        max_len=ds.vocab.max_len,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if cfg.mode == "current":
        if ckpt is None:
            ckpt = load_checkpoint(last_ckpt_path, device)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt and isinstance(ckpt["opt"], dict):
            try:
                opt.load_state_dict(ckpt["opt"])
            except Exception:
                pass

    log(
        "[INFO] params: "
        f"MAX_LEN={ds.vocab.max_len} | BASE_MODE={ds.base_mode} | "
        f"BATCH_TRAIN={batch_train} | BATCH_VAL={batch_val} | "
        f"EPOCHS={run_epochs} | PATIENCE={patience} | MIN_DELTA={min_delta} | "
        f"LR={lr} | ETA_MIN={eta_min} | WEIGHT_DECAY={weight_decay} | "
        f"D_MODEL={cfg.d_model} | N_HEADS={cfg.n_heads} | N_LAYERS={cfg.n_layers} | D_HEAD={cfg.d_head} | "
        f"D_FF={cfg.d_ff} | DROPOUT={cfg.dropout} | SNAP_EVERY={snap_every}"
    )

    header_logged = False

    def metric_header():
        nonlocal header_logged
        if header_logged:
            return
        log(
            f"{'ep':>2} | {'train_loss':>10} | {'val_loss':>8} | {'val_acc':>7} | {'val_seq':>7} | "
            f"{'val_f1':>6} | {'val_inv':>7} | {'val_s':>5} | {'lr':>8}"
        )
        header_logged = True

    end_epoch = start_epoch + run_epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        current_lr = cosine_lr_for_epoch(epoch, start_epoch, run_epochs, lr, eta_min)
        set_optimizer_lr(opt, current_lr)

        train_loss, train_steps = train_epoch(
            model=model,
            train_items=train_items,
            ds=ds,
            batch_size=batch_train,
            split_seed=split_seed,
            epoch=epoch,
            opt=opt,
            device=device,
            log_every=log_every,
            max_steps=max_steps_per_epoch,
            collate_fn=collate_fn,
            bad_epochs=bad_epochs,
        )
        global_step += train_steps

        t0 = time.time()
        val_acc, val_loss, val_seq_acc, val_f1_paired, val_invalid = evaluate(model, val_loader, device, ds)
        val_time = time.time() - t0

        metric_header()
        log(
            f"{epoch:>2d} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | "
            f"{val_seq_acc:>7.4f} | {val_f1_paired:>6.4f} | {val_invalid:>7.4f} | {val_time:>5.1f} | "
            f"{current_lr:>8.2e}"
        )

        metrics_write(
            {
                "kind": "epoch",
                "epoch": int(epoch),
                "lr": float(current_lr),
                "global_step": int(global_step),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_seq": float(val_seq_acc),
                "val_f1": float(val_f1_paired),
                "val_inv": float(val_invalid),
                "val_s": float(val_time),
                "best_val_loss_before": float(best_val_loss),
                "bad_epochs_before": int(bad_epochs),
            },
            cfg.mode,
        )

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = float(val_loss)
            bad_epochs = 0
            save_ckpt(
                best_ckpt_path,
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "meta": ckpt_meta,
                },
            )
            metrics_write(
                {
                    "kind": "best",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                },
                cfg.mode,
            )
        else:
            bad_epochs += 1

        save_ckpt(
            last_ckpt_path,
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_val_loss": float(best_val_loss),
                "bad_epochs": int(bad_epochs),
                "meta": ckpt_meta,
            },
        )

        should_snap = (epoch % snap_every == 0) or (epoch == end_epoch)
        snapped = False
        if should_snap:
            dst = save_checkpoint_dir_snapshot(model_dir, epoch, "epoch")
            log(f"[INFO] saved snapshot: {dst}")
            snapped = True

        if bad_epochs >= patience:
            if not snapped:
                dst = save_checkpoint_dir_snapshot(model_dir, epoch, "early-stop")
                log(f"[INFO] saved snapshot: {dst}")
            log(f"[INFO] Early stopping (val_loss not improved for {patience} epochs)")
            break

    run_test_eval(model=model, test_loader=test_loader, best_ckpt_path=best_ckpt_path, device=device, ds=ds)
    end_logging()


if __name__ == "__main__":
    main()