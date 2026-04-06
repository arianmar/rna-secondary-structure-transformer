#!/usr/bin/env python3
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch

EPOCH_DIR_RE = re.compile(r"epoch_(\d+)$")
WORK_DIRNAME = "work"


def _as_work_dir_if_present(p: Path) -> Path:
    w = p / WORK_DIRNAME
    return w if w.is_dir() else p


def _resolve_model_root_from_run_dir(run_dir: Path) -> Path:
    if EPOCH_DIR_RE.match(run_dir.name) or run_dir.name == WORK_DIRNAME:
        return run_dir.parent
    return run_dir


def next_available_dir(base: Path) -> Path:
    if not base.exists():
        return base
    i = 2
    while True:
        cand = base.parent / f"{base.name}{i}"
        if not cand.exists():
            return cand
        i += 1


def safe_read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def iter_jsonl(path: Path) -> Iterable[dict]:
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                yield obj
    if bad:
        print(f"[WARN] {path}: skipped {bad} bad json lines")


def to_float(x: Any) -> Optional[float]:
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


def find_latest_epoch_dir(model_dir: Path) -> Optional[Path]:
    if not model_dir.is_dir():
        return None
    best_ep: int | None = None
    best_dir: Path | None = None
    for p in model_dir.iterdir():
        if not p.is_dir():
            continue
        m = EPOCH_DIR_RE.match(p.name)
        if not m:
            continue
        ep = int(m.group(1))
        if best_ep is None or ep > best_ep:
            best_ep, best_dir = ep, p
    return best_dir


def resolve_run_dir(path_arg: str) -> Path:
    p = Path(path_arg).expanduser().resolve()

    if p.is_dir():
        if EPOCH_DIR_RE.match(p.name):
            return p

        latest = find_latest_epoch_dir(p)
        if latest is not None:
            return latest

        run = _as_work_dir_if_present(p)
        if (run / "metrics.jsonl").exists():
            return run

    raise SystemExit(f"[ERR] cannot resolve run dir from: {path_arg}")


def discover_model_roots(models_dir: Path) -> list[Path]:
    if not models_dir.is_dir():
        raise SystemExit(f"[ERR] --models-dir must be a directory: {models_dir}")

    out: list[Path] = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        if find_latest_epoch_dir(p) is not None:
            out.append(p.resolve())
            continue
        w = p / WORK_DIRNAME
        if (w / "metrics.jsonl").exists():
            out.append(p.resolve())

    out.sort(key=lambda x: x.name.lower())
    if not out:
        raise SystemExit(f"[ERR] no model workspaces found in: {models_dir}")
    return out


@dataclass
class RawSeries:
    epochs: list[int]
    val_loss: list[float]
    val_acc: list[float]
    val_seq: list[float]
    val_f1: list[float]
    val_inv: list[float]


def extract_raw_series(run_dir: Path) -> Optional[RawSeries]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    ep: list[int] = []
    vl: list[float] = []
    va: list[float] = []
    vs: list[float] = []
    vf: list[float] = []
    vi: list[float] = []

    for obj in iter_jsonl(metrics_path):
        if obj.get("kind") != "epoch":
            continue
        e = obj.get("epoch")
        vloss = to_float(obj.get("val_loss"))
        if e is None or vloss is None:
            continue

        ep.append(int(e))
        vl.append(float(vloss))

        vacc = to_float(obj.get("val_acc"))
        vseq = to_float(obj.get("val_seq"))
        vf1 = to_float(obj.get("val_f1"))
        vinv = to_float(obj.get("val_inv"))

        va.append(float(vacc) if vacc is not None else float("nan"))
        vs.append(float(vseq) if vseq is not None else float("nan"))
        vf.append(float(vf1) if vf1 is not None else float("nan"))
        vi.append(float(vinv) if vinv is not None else float("nan"))

    if not ep:
        return None
    return RawSeries(ep, vl, va, vs, vf, vi)


@dataclass
class PostprocSeries:
    epochs: list[int]
    token_acc: list[float]
    seq_exact: list[float]
    paired_f1: list[float]
    invalid: list[float]
    pp_ms_per_seq: list[float]


def extract_postproc_series(model_dir_or_run_dir: Path, mode: str) -> Optional[PostprocSeries]:
    base = _resolve_model_root_from_run_dir(model_dir_or_run_dir)

    pp_dir = base / "postprocessing_compare"
    obj = safe_read_json(pp_dir / f"{mode}_results.json")
    if obj is None:
        return None

    epochs = obj.get("epochs")
    metrics = obj.get("metrics")
    if not isinstance(epochs, list) or not isinstance(metrics, dict):
        return None

    def get_list(key: str) -> Optional[list[float]]:
        v = metrics.get(key)
        if not isinstance(v, list):
            return None
        out: list[float] = []
        for x in v:
            fx = to_float(x)
            out.append(float("nan") if fx is None else float(fx))
        return out

    token_acc = get_list("token_acc")
    seq_exact = get_list("seq_exact")
    paired_f1 = get_list("paired_f1")
    invalid = get_list("invalid")
    pp_ms = get_list("pp_ms_per_seq")

    if token_acc is None or seq_exact is None or paired_f1 is None or invalid is None or pp_ms is None:
        return None

    return PostprocSeries(
        epochs=[int(e) for e in epochs],
        token_acc=token_acc,
        seq_exact=seq_exact,
        paired_f1=paired_f1,
        invalid=invalid,
        pp_ms_per_seq=pp_ms,
    )


def read_ckpt_meta(run_dir: Path) -> dict:
    base = run_dir if EPOCH_DIR_RE.match(run_dir.name) else _as_work_dir_if_present(run_dir)
    for name in ("best.pt", "last.pt"):
        p = base / name
        if not p.exists():
            continue
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception:
            return {}
        if isinstance(obj, dict):
            meta = obj.get("meta")
            return meta if isinstance(meta, dict) else {}
        return {}
    return {}


def read_split_bytes(run_dir: Path) -> Optional[bytes]:
    base = run_dir if EPOCH_DIR_RE.match(run_dir.name) else _as_work_dir_if_present(run_dir)
    p = base / "split.pkl"
    if not p.exists():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def fairness_key(run_dir: Path) -> tuple[str, tuple]:
    meta = read_ckpt_meta(run_dir)
    split_bytes = read_split_bytes(run_dir)

    max_len = meta.get("max_length")
    base_mode = meta.get("base_mode")
    seq_alphabet = tuple(meta.get("seq_alphabet") or [])
    struct_alphabet = tuple(meta.get("struct_alphabet") or [])
    bracket_pairs = tuple(sorted((meta.get("bracket_pairs") or {}).items()))

    ok = (
        isinstance(max_len, int)
        and isinstance(base_mode, int)
        and bool(seq_alphabet)
        and bool(struct_alphabet)
        and bool(bracket_pairs)
        and split_bytes is not None
    )

    if ok:
        return ("ok", (max_len, base_mode, seq_alphabet, struct_alphabet, bracket_pairs, split_bytes))

    missing_bits = (
        "ml" if not isinstance(max_len, int) else "",
        "bm" if not isinstance(base_mode, int) else "",
        "sa" if not seq_alphabet else "",
        "sta" if not struct_alphabet else "",
        "bp" if not bracket_pairs else "",
        "split" if split_bytes is None else "",
    )
    tag = "missing_" + "_".join([x for x in missing_bits if x])
    if not tag:
        tag = "missing"
    return (tag, (max_len, base_mode, seq_alphabet, struct_alphabet, bracket_pairs, split_bytes))


def group_name(key: tuple) -> str:
    max_len = key[0] if len(key) > 0 else None
    base_mode = key[1] if len(key) > 1 else None
    ml = str(max_len) if isinstance(max_len, int) else "unknown"
    bm = str(base_mode) if isinstance(base_mode, int) else "unknown"
    return f"group_{ml}_bm{bm}"


def save_fig(path: Path) -> None:
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"[OK] wrote {path}")


def _last_finite_idx(y: list[float]) -> int:
    j = len(y) - 1
    while j >= 0:
        try:
            if math.isfinite(float(y[j])):
                return j
        except Exception:
            pass
        j -= 1
    return -1


def plot_overlay(
    out_path: Path,
    title: str,
    ylabel: str,
    series: list[tuple[list[int], list[float], str, str]],
) -> bool:
    if len(series) <= 1:
        return False

    plt.figure()
    for x, y, sid, color in series:
        line = plt.plot(x, y, linewidth=1.0, color=color)[0]
        if x:
            j = _last_finite_idx(y)
            if 0 <= j < len(x):
                plt.text(x[j], y[j], f" {sid}", fontsize=8, va="center", color=line.get_color())

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(out_path)
    return True


def best_so_far(y: list[float], mode: str) -> list[float]:
    out: list[float] = []
    best: Optional[float] = None
    for v in y:
        if not math.isfinite(v):
            out.append(best if best is not None else float("nan"))
            continue
        best = v if best is None else (min(best, v) if mode == "min" else max(best, v))
        out.append(best)
    return out


def plot_best_so_far(
    out_path: Path,
    title: str,
    ylabel: str,
    mode: str,
    series: list[tuple[list[int], list[float], str, str]],
) -> bool:
    if len(series) <= 1:
        return False

    plt.figure()
    for x, y, sid, color in series:
        by = best_so_far(y, mode)
        line = plt.plot(x, by, linewidth=1.0, color=color)[0]
        if x:
            j = _last_finite_idx(by)
            if 0 <= j < len(x):
                plt.text(x[j], by[j], f" {sid}", fontsize=8, va="center", color=line.get_color())

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    save_fig(out_path)
    return True


def plot_rank_bar(out_path: Path, title: str, ylabel: str, items: list[tuple[float, str, str]]) -> bool:
    if len(items) <= 1:
        return False

    xs = list(range(len(items)))
    ys = [v for v, _, _ in items]
    labels = [sid for _, sid, _ in items]
    colors = [c for _, _, c in items]

    plt.figure(figsize=(max(6, 0.35 * len(items)), 4))
    plt.bar(xs, ys, color=colors)
    plt.xticks(xs, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] wrote {out_path}")
    return True


def _fmt_seq_alphabet(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "".join(str(x) for x in v)
    if isinstance(v, dict):
        return "".join(str(k) for k in v.keys())
    return str(v)


def _fmt_bracket_pairs(v: Any) -> str:
    if isinstance(v, dict):
        items = list(v.items())
        if not items:
            return ""
        return "".join(f"{op}{cl}" for op, cl in items)

    if isinstance(v, (list, tuple)):
        pairs = []
        for it in v:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                pairs.append((it[0], it[1]))
        pairs.sort(key=lambda kv: (str(kv[0]), str(kv[1])))
        return "".join(f"{op}{cl}" for op, cl in pairs)

    return str(v)


def _meta_get_table(meta: dict, key: str, default: str = "n/a") -> str:
    v = meta.get(key)
    if isinstance(v, list) and v and isinstance(v[-1], (list, tuple)) and len(v[-1]) == 2:
        v = v[-1][0]
    if v is None:
        return default

    if key in {"seq_alphabet", "struct_alphabet"}:
        return _fmt_seq_alphabet(v)
    if key == "bracket_pairs":
        return _fmt_bracket_pairs(v)

    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


_TABLE_META_KEYS = [
    "max_length",
    "base_mode",
    "seq_alphabet",
    "struct_alphabet",
    "bracket_pairs",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "dropout",
]


def write_models_table(out_path: Path, items: list["ModelItem"]) -> bool:
    if not items:
        return False

    cols = ["", "id"] + _TABLE_META_KEYS
    rows: list[list[str]] = []
    for it in items:
        rows.append(["", it.sid] + [_meta_get_table(it.meta, k) for k in _TABLE_META_KEYS])

    ncols = len(cols)
    col_max = [len(str(c)) for c in cols]
    for r in rows:
        for j in range(min(len(r), ncols)):
            col_max[j] = max(col_max[j], len(str(r[j])))

    col_max[0] = 1

    pad = 2
    weights = [max(1, w + pad) for w in col_max]
    weights[0] = 1

    total_w = float(sum(weights))
    col_widths = [w / total_w for w in weights]

    width_in = max(6.0, 0.10 * total_w)
    height_in = max(2.8, 0.35 * (len(rows) + 1))

    plt.figure(figsize=(width_in, height_in))
    plt.axis("off")

    tbl = plt.table(
        cellText=rows,
        colLabels=cols,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)

    try:
        tbl[(0, 0)].set_facecolor("white")
    except Exception:
        pass

    for i, it in enumerate(items, start=1):
        try:
            cell = tbl[(i, 0)]
            cell.set_facecolor(it.color)
            cell.get_text().set_text("")
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[OK] wrote {out_path}")
    return True


@dataclass
class ModelItem:
    sid: str
    run_dir: Path
    model_root: Path
    meta: dict
    group_kind: str
    group_key: tuple
    color: str
    raw: Optional[RawSeries]
    pp_raw: Optional[PostprocSeries]
    pp_ktd: Optional[PostprocSeries]
    pp_dpgs: Optional[PostprocSeries]


def build_model_item(path_arg: str, sid: str, color: str) -> ModelItem:
    run_dir = resolve_run_dir(path_arg)
    model_root = _resolve_model_root_from_run_dir(run_dir)

    meta = read_ckpt_meta(run_dir)
    gkind, gkey = fairness_key(run_dir)

    return ModelItem(
        sid=sid,
        run_dir=run_dir,
        model_root=model_root,
        meta=meta,
        group_kind=gkind,
        group_key=gkey,
        color=color,
        raw=extract_raw_series(run_dir),
        pp_raw=extract_postproc_series(model_root, "raw"),
        pp_ktd=extract_postproc_series(model_root, "ktd"),
        pp_dpgs=extract_postproc_series(model_root, "dpgs"),
    )


def build_groups(items: list[ModelItem], enable_group: bool) -> dict[str, list[ModelItem]]:
    if not enable_group:
        return {"all": items}

    buckets: dict[tuple[str, tuple], list[ModelItem]] = {}
    for it in items:
        buckets.setdefault((it.group_kind, it.group_key), []).append(it)

    out: dict[str, list[ModelItem]] = {}
    used: dict[str, int] = {}

    for (_kind, key), lst in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        name = group_name(key)
        n = used.get(name, 0) + 1
        used[name] = n
        if n > 1:
            name = f"{name}_{n}"
        out[name] = lst

    return out


def plot_group(out_root: Path, group_name_str: str, items: list[ModelItem]) -> bool:
    group_dir = out_root if group_name_str == "all" else out_root / group_name_str

    raw_items = [(it.sid, it.color, it.raw) for it in items if it.raw is not None]
    pp_raw_items = [(it.sid, it.color, it.pp_raw) for it in items if it.pp_raw is not None]
    pp_ktd_items = [(it.sid, it.color, it.pp_ktd) for it in items if it.pp_ktd is not None]
    pp_dpgs_items = [(it.sid, it.color, it.pp_dpgs) for it in items if it.pp_dpgs is not None]

    has_any_plot = (
        len(raw_items) > 1
        or len(pp_raw_items) > 1
        or len(pp_ktd_items) > 1
        or len(pp_dpgs_items) > 1
    )
    if not has_any_plot:
        return False

    group_dir.mkdir(parents=True, exist_ok=True)
    write_models_table(group_dir / "models_table.png", items)

    created: dict[str, Path] = {}

    def subdir(*parts: str) -> Path:
        key = "/".join(parts)
        p = created.get(key)
        if p is not None:
            return p
        p = group_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        created[key] = p
        return p

    if len(raw_items) > 1:
        raw_dir = subdir("raw")
        series_loss = [(rs.epochs, rs.val_loss, sid, color) for sid, color, rs in raw_items]
        series_acc = [(rs.epochs, rs.val_acc, sid, color) for sid, color, rs in raw_items]
        series_seq = [(rs.epochs, rs.val_seq, sid, color) for sid, color, rs in raw_items]
        series_f1 = [(rs.epochs, rs.val_f1, sid, color) for sid, color, rs in raw_items]
        series_inv = [(rs.epochs, rs.val_inv, sid, color) for sid, color, rs in raw_items]

        plot_overlay(raw_dir / "val_loss_overlay.png", "val_loss (training raw)", "val_loss", series_loss)
        plot_overlay(raw_dir / "val_acc_overlay.png", "val_acc / token_acc (training raw)", "score", series_acc)
        plot_overlay(raw_dir / "val_seq_overlay.png", "val_seq_exact (training raw)", "score", series_seq)
        plot_overlay(raw_dir / "val_f1_overlay.png", "val_paired_f1 (training raw)", "score", series_f1)
        plot_overlay(raw_dir / "val_inv_overlay.png", "val_invalid_rate (training raw)", "rate", series_inv)

    def plot_pp(mode: str, items_pp: list[tuple[str, str, Optional[PostprocSeries]]], include_inv: bool) -> None:
        if len(items_pp) <= 1:
            return

        folder = subdir("postprocessing", mode)
        label_mode = "postproc raw" if mode == "raw" else mode

        series_acc = [(pp.epochs, pp.token_acc, sid, color) for sid, color, pp in items_pp if pp is not None]
        series_seq = [(pp.epochs, pp.seq_exact, sid, color) for sid, color, pp in items_pp if pp is not None]
        series_f1 = [(pp.epochs, pp.paired_f1, sid, color) for sid, color, pp in items_pp if pp is not None]

        plot_overlay(folder / "val_acc_overlay.png", f"val_token_acc ({label_mode})", "score", series_acc)
        plot_overlay(folder / "val_seq_overlay.png", f"val_seq_exact ({label_mode})", "score", series_seq)
        plot_overlay(folder / "val_f1_overlay.png", f"val_paired_f1 ({label_mode})", "score", series_f1)

        if include_inv:
            series_inv = [(pp.epochs, pp.invalid, sid, color) for sid, color, pp in items_pp if pp is not None]
            plot_overlay(folder / "val_inv_overlay.png", f"val_invalid_rate ({label_mode})", "rate", series_inv)

    plot_pp("raw", pp_raw_items, include_inv=True)
    plot_pp("ktd", pp_ktd_items, include_inv=False)
    plot_pp("dpgs", pp_dpgs_items, include_inv=False)

    extra_dir: Optional[Path] = None

    def ensure_extra() -> Path:
        nonlocal extra_dir
        if extra_dir is None:
            extra_dir = subdir("extra")
        return extra_dir

    if len(raw_items) > 1:
        series_loss = [(rs.epochs, rs.val_loss, sid, color) for sid, color, rs in raw_items]
        series_f1 = [(rs.epochs, rs.val_f1, sid, color) for sid, color, rs in raw_items]

        plot_best_so_far(
            ensure_extra() / "best_so_far_val_loss_training_raw.png",
            "best-so-far val_loss (training raw)",
            "val_loss",
            "min",
            series_loss,
        )
        plot_best_so_far(
            ensure_extra() / "best_so_far_val_f1_training_raw.png",
            "best-so-far val_f1 (training raw)",
            "score",
            "max",
            series_f1,
        )

        ranked_loss: list[tuple[float, str, str]] = []
        for it in items:
            if it.raw is None:
                continue
            vals = [v for v in it.raw.val_loss if math.isfinite(v)]
            if vals:
                ranked_loss.append((min(vals), it.sid, it.color))
        ranked_loss.sort(key=lambda x: x[0])
        plot_rank_bar(
            ensure_extra() / "rank_best_val_loss_training_raw.png",
            "best val_loss per model (training raw)",
            "best val_loss",
            ranked_loss,
        )

    def best_pp_f1(
        items_pp: list[tuple[str, str, Optional[PostprocSeries]]],
        out_best: str,
        title_best: str,
        out_rank: str,
        title_rank: str,
    ) -> None:
        if len(items_pp) <= 1:
            return

        series_f1 = [(pp.epochs, pp.paired_f1, sid, color) for sid, color, pp in items_pp if pp is not None]
        if len(series_f1) <= 1:
            return

        plot_best_so_far(ensure_extra() / out_best, title_best, "score", "max", series_f1)

        ranked: list[tuple[float, str, str]] = []
        for sid, color, pp in items_pp:
            if pp is None:
                continue
            vals = [v for v in pp.paired_f1 if math.isfinite(v)]
            if vals:
                ranked.append((max(vals), sid, color))
        ranked.sort(key=lambda x: -x[0])
        plot_rank_bar(ensure_extra() / out_rank, title_rank, "best paired_f1", ranked)

    best_pp_f1(
        pp_raw_items,
        "best_so_far_postproc_raw_f1.png",
        "best-so-far paired_f1 (postproc raw)",
        "rank_best_f1_postproc_raw.png",
        "best paired_f1 per model (postproc raw)",
    )
    best_pp_f1(
        pp_ktd_items,
        "best_so_far_ktd_f1.png",
        "best-so-far paired_f1 (ktd)",
        "rank_best_f1_ktd.png",
        "best paired_f1 per model (ktd)",
    )
    best_pp_f1(
        pp_dpgs_items,
        "best_so_far_dpgs_f1.png",
        "best-so-far paired_f1 (dpgs)",
        "rank_best_f1_dpgs.png",
        "best paired_f1 per model (dpgs)",
    )

    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare model runs with overlay plots (training raw + postprocessing)"
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--models", nargs="+", help="Paths to model workspaces to compare")
    src.add_argument("--models-dir", type=str, help="Directory that contains multiple model workspaces; all are compared")

    ap.add_argument("--group", action="store_true")
    ap.add_argument("--out", type=str, default="compare")
    args = ap.parse_args()

    if args.models_dir is not None:
        roots = discover_model_roots(Path(args.models_dir).expanduser().resolve())
        paths = [str(p) for p in roots]
    else:
        paths = list(args.models or [])

    out_root = next_available_dir(Path(args.out).expanduser().resolve())

    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    items: list[ModelItem] = []
    for i, p in enumerate(paths, start=1):
        items.append(build_model_item(p, f"M{i}", cycle[(i - 1) % len(cycle)]))

    groups = build_groups(items, args.group)

    print("\n[INFO] grouping summary", flush=True)
    for gname, glist in groups.items():
        if not glist:
            continue
        saved = plot_group(out_root, gname, glist)
        if not saved:
            print(f"[INFO] skipped (no plots): {gname}")

    print(f"\n[OK] done. Output: {out_root}\n")


if __name__ == "__main__":
    main()