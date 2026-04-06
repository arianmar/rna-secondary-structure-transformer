import os
import shutil

import torch

from core.data import DatasetBundle, StructureInfo
from core.model import TransformerEncoderTagger
from utils.config import TrainConfig

_WORK_DIRNAME = "work"
_WORK_FILES = ("last.pt", "best.pt", "split.pkl", "train.log", "metrics.jsonl")


def save_ckpt(path: str, state: dict) -> None:
    torch.save(state, path)


def next_available_dir(base: str) -> str:
    if not os.path.exists(base):
        return base
    i = 2
    while True:
        cand = f"{base}_{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


def work_dir(model_dir: str) -> str:
    return os.path.join(model_dir, _WORK_DIRNAME)


def build_model_dir(cfg: TrainConfig, ds: DatasetBundle) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    base = os.path.join(
        cfg.output_dir,
        f"model_{ds.base_mode}_{ds.vocab.max_len}_{cfg.d_model}_{cfg.n_heads}_{cfg.n_layers}_{cfg.d_ff}",
    )
    model_dir = next_available_dir(base)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(work_dir(model_dir), exist_ok=True)
    return model_dir


def is_hist(v) -> bool:
    return isinstance(v, list) and (not v or (isinstance(v[-1], (list, tuple)) and len(v[-1]) == 2))


def meta_last(meta: dict, key: str, fallback):
    v = meta.get(key)
    if is_hist(v) and v:
        return v[-1][0]
    return fallback if v is None else v


def ensure_hist_value(meta: dict, key: str, value, start_epoch: int) -> None:
    v = meta.get(key)
    if is_hist(v):
        if not v:
            meta[key] = [(value, start_epoch)]
        return
    meta[key] = [(v, 1)] if v is not None else [(value, start_epoch)]


def append_hist_if_changed(meta: dict, key: str, value, start_epoch: int) -> None:
    ensure_hist_value(meta, key, value, start_epoch)
    cur = meta.get(key, [])
    last = cur[-1][0] if cur else None
    if value != last:
        cur.append((value, start_epoch))
        meta[key] = cur


def build_ckpt_meta(
    cfg: TrainConfig,
    ds: DatasetBundle,
    data_path: str,
    *,
    base_meta: dict | None = None,
    start_epoch: int = 1,
) -> dict:
    meta = dict(base_meta) if isinstance(base_meta, dict) else {}

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

    append_hist_if_changed(meta, "batch_train", int(cfg.batch_train), start_epoch)
    append_hist_if_changed(meta, "batch_val", int(cfg.batch_val), start_epoch)
    append_hist_if_changed(meta, "epochs_per_run", int(cfg.epochs), start_epoch)
    append_hist_if_changed(meta, "patience", int(cfg.patience), start_epoch)
    append_hist_if_changed(meta, "min_delta", float(cfg.min_delta), start_epoch)
    append_hist_if_changed(meta, "lr", float(cfg.lr), start_epoch)
    append_hist_if_changed(meta, "weight_decay", float(cfg.weight_decay), start_epoch)
    append_hist_if_changed(meta, "eta_min", float(cfg.eta_min), start_epoch)
    append_hist_if_changed(meta, "log_every", int(cfg.log_every), start_epoch)
    append_hist_if_changed(
        meta,
        "max_steps_per_epoch",
        None if cfg.max_steps_per_epoch is None else int(cfg.max_steps_per_epoch),
        start_epoch,
    )
    append_hist_if_changed(meta, "split_seed", int(cfg.split_seed), start_epoch)
    append_hist_if_changed(meta, "snap_every", int(cfg.snap_every), start_epoch)

    return meta


def load_checkpoint(ckpt_path: str, device: str) -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    return torch.load(ckpt_path, map_location=device)


def resolve_model_dir(path_arg: str) -> str:
    path = os.path.abspath(path_arg)
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        name = os.path.basename(path)
        if name not in {"last.pt", "best.pt"}:
            raise FileNotFoundError(f"checkpoint file must be last.pt or best.pt, got: {path}")
        parent = os.path.dirname(path)
        if os.path.basename(parent) == _WORK_DIRNAME:
            return os.path.dirname(parent)
        return parent
    raise FileNotFoundError(f"path not found: {path}")


def validate_model_dir_for_resume(model_dir: str) -> None:
    wdir = work_dir(model_dir)
    missing = [name for name in _WORK_FILES if not os.path.exists(os.path.join(wdir, name))]
    if missing:
        raise FileNotFoundError(
            f"missing required resume files in {wdir}: {', '.join(missing)}"
        )


def save_checkpoint_dir_snapshot(model_dir: str, epoch: int, tag: str) -> str:
    src_dir = work_dir(model_dir)
    dst_dir = os.path.join(model_dir, f"epoch_{epoch}")
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for name in _WORK_FILES:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))

    return dst_dir


def build_runtime_bundle_from_meta(meta: dict):
    required = (
        "max_length",
        "d_model",
        "n_heads",
        "n_layers",
        "d_ff",
        "dropout",
        "num_classes",
        "seq_alphabet",
        "struct_alphabet",
        "bracket_pairs",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"checkpoint metadata missing: {', '.join(missing)}")

    seq_alphabet = list(meta["seq_alphabet"])
    struct_alphabet = list(meta["struct_alphabet"])
    bracket_pairs = dict(meta["bracket_pairs"])
    bracket_order = list(meta.get("bracket_order", []))
    bracket_type_order = list(meta.get("bracket_type_order", []))
    bracket_type_counts = dict(meta.get("bracket_type_counts", {}))

    base2id = {ch: i for i, ch in enumerate(seq_alphabet)}
    struct2id = {ch: i for i, ch in enumerate(struct_alphabet)}
    inv_struct = {i: ch for ch, i in struct2id.items()}

    pad_x = len(base2id)
    pad_y = len(struct2id)
    vocab_in = pad_x + 1
    num_classes = int(meta["num_classes"])

    paired_chars = set(bracket_pairs.keys()) | set(bracket_pairs.values())
    paired_id_mask = torch.zeros(pad_y + 1, dtype=torch.bool)
    for ch, idx in struct2id.items():
        if ch in paired_chars:
            paired_id_mask[idx] = True

    open_ids = tuple(struct2id[ch] for ch in bracket_pairs.keys())
    close_ids = tuple(struct2id[ch] for ch in bracket_pairs.values())

    structure = StructureInfo(
        bracket_pairs=bracket_pairs,
        bracket_order=bracket_order,
        bracket_type_order=bracket_type_order,
        bracket_type_counts=bracket_type_counts,
        paired_id_mask=paired_id_mask,
        open_ids=open_ids,
        close_ids=close_ids,
        open_id_set=frozenset(open_ids),
        close_id_set=frozenset(close_ids),
        closeid_to_openid={struct2id[cl]: struct2id[op] for op, cl in bracket_pairs.items()},
    )

    return {
        "max_len": int(meta["max_length"]),
        "base_mode": int(meta.get("base_mode", -1)),
        "d_model": int(meta["d_model"]),
        "n_heads": int(meta["n_heads"]),
        "n_layers": int(meta["n_layers"]),
        "d_ff": int(meta["d_ff"]),
        "dropout": float(meta["dropout"]),
        "num_classes": num_classes,
        "seq_alphabet": seq_alphabet,
        "struct_alphabet": struct_alphabet,
        "bracket_pairs": bracket_pairs,
        "bracket_order": bracket_order,
        "bracket_type_order": bracket_type_order,
        "bracket_type_counts": bracket_type_counts,
        "base2id": base2id,
        "struct2id": struct2id,
        "inv_struct": inv_struct,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "vocab_in": vocab_in,
        "structure": structure,
    }


def load_model_from_checkpoint(ckpt_path: str, device: str):
    ckpt = load_checkpoint(ckpt_path, device)
    meta = ckpt.get("meta") or {}
    bundle = build_runtime_bundle_from_meta(meta)

    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    model = TransformerEncoderTagger(
        vocab_in=bundle["vocab_in"],
        d_model=bundle["d_model"],
        n_heads=bundle["n_heads"],
        n_layers=bundle["n_layers"],
        d_ff=bundle["d_ff"],
        dropout=bundle["dropout"],
        pad_x=bundle["pad_x"],
        num_classes=bundle["num_classes"],
        max_len=bundle["max_len"],
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, bundle, ckpt