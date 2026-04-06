#!/usr/bin/env python3
import argparse
import math
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from utils.checkpointing import load_model_from_checkpoint
from core.postproc_dpgs import repair_dpgs
from core.postproc_ktd import repair_kill_to_dot
from core.structure import dotbracket_is_valid_ids
from label import detect_rna_structures


_WS = set(b" \t\r\n")


def read_sequence(
    seq_arg: str | None,
    path: str | None,
    base2id: dict[str, int],
    max_len: int,
) -> str:
    if (seq_arg is None) == (path is None):
        raise SystemExit("Provide exactly one of --seq or --path")

    if path is not None:
        with open(path, "rb") as f:
            data = f.read()

        seq_bytes = b""
        for line in data.splitlines():
            line = line.strip()
            if line:
                seq_bytes = line
                break

        if not seq_bytes:
            raise SystemExit("File is empty")
    else:
        seq_bytes = seq_arg.encode("utf-8", errors="ignore")

    if any(b in _WS for b in seq_bytes):
        seq_bytes = bytes(b for b in seq_bytes if b not in _WS)

    try:
        seq = seq_bytes.decode("ascii").upper()
    except UnicodeDecodeError as e:
        raise SystemExit(f"Sequence contains non-ASCII byte at position {e.start}")

    if not seq:
        raise SystemExit("Sequence is empty")

    for i, ch in enumerate(seq):
        if ch not in base2id:
            raise SystemExit(f"Invalid character at pos {i}: {ch!r}")

    if len(seq) > max_len:
        raise SystemExit(f"Sequence too long: {len(seq)} > MAX_LEN={max_len}")

    return seq


def encode_sequence(seq: str, base2id: dict[str, int]) -> torch.Tensor:
    return torch.tensor([base2id[ch] for ch in seq], dtype=torch.long).unsqueeze(0)


def fmt_f(x: float, nd: int = 2) -> str:
    if math.isnan(x) or math.isinf(x):
        return "nan"
    return f"{x:.{nd}f}"


def print_score_table(
    delta: float | None,
    seq_logprob: float,
    avg_logprob: float,
    mean_confidence: float,
    mean_entropy: float,
) -> None:
    delta_str = "nan" if delta is None else fmt_f(delta, 4)

    headers = [
        "delta",
        "seq_logprob",
        "avg_logprob",
        "mean_confidence",
        "mean_entropy",
    ]
    values = [
        delta_str,
        fmt_f(seq_logprob, 4),
        fmt_f(avg_logprob, 4),
        fmt_f(mean_confidence, 4),
        fmt_f(mean_entropy, 4),
    ]

    widths = [max(len(h), len(v)) for h, v in zip(headers, values)]

    def hline() -> str:
        return "|-" + "-|-".join("-" * w for w in widths) + "-|"

    def row(items: list[str]) -> str:
        return "| " + " | ".join(s.ljust(w) for s, w in zip(items, widths)) + " |"

    print(row(headers))
    print(row(values))


def print_token_table(
    seq: str,
    pred: str,
    topk_chars: list[list[str]],
    topk_probs: list[list[float]],
    confidence: list[float],
    entropy: list[float],
) -> None:
    print("sequence:")
    print(seq)
    print("prediction:")
    print(pred)
    print()

    idx_w = max(3, len(str(len(seq) - 1)))
    seq_w = max(3, max((len(ch) for ch in seq), default=1))
    pred_w = max(4, max((len(ch) for ch in pred), default=1))
    top_w = max(
        8,
        max(
            len(f"{topk_chars[i][j]}{fmt_f(topk_probs[i][j])}")
            for i in range(len(seq))
            for j in range(3)
        ) if seq else 8,
    )
    conf_w = max(10, max((len(fmt_f(x)) for x in confidence), default=4))
    ent_w = max(7, max((len(fmt_f(x)) for x in entropy), default=4))

    headers = [
        ("idx", idx_w),
        ("seq", seq_w),
        ("pred", pred_w),
        ("top1", top_w),
        ("top2", top_w),
        ("top3", top_w),
        ("confidence", conf_w),
        ("entropy", ent_w),
    ]

    def hline() -> str:
        return "+" + "+".join("-" * (w + 2) for _, w in headers) + "+"

    def make_row(values: list[str]) -> str:
        cells = []
        for value, (_, width) in zip(values, headers):
            cells.append(" " + value[:width].ljust(width) + " ")
        return "|" + "|".join(cells) + "|"

    print(hline())
    print(make_row([name for name, _ in headers]))
    print(hline())

    for i in range(len(seq)):
        row = [
            str(i),
            seq[i],
            pred[i],
            f"{topk_chars[i][0]}{fmt_f(topk_probs[i][0])}",
            f"{topk_chars[i][1]}{fmt_f(topk_probs[i][1])}",
            f"{topk_chars[i][2]}{fmt_f(topk_probs[i][2])}",
            fmt_f(confidence[i]),
            fmt_f(entropy[i]),
        ]
        print(make_row(row))

    print(hline())


def print_structure_labels(pred: str, mode: str) -> None:
    if mode in {"strict", "both"}:
        strict_label = detect_rna_structures(pred, mode="strict")
        print("structure_labels_strict:")
        print(strict_label)

    if mode in {"pragmatic", "both"}:
        pragmatic_label = detect_rna_structures(pred, mode="pragmatic")
        print("structure_labels_pragmatic:")
        print(pragmatic_label)


def apply_postprocessing(
    pred_ids: torch.Tensor,
    logits_cpu: torch.Tensor,
    valid_mask: torch.Tensor,
    bundle: dict,
    postproc: str | None,
    type_id_pairs: list[tuple[int, int]] | None,
) -> torch.Tensor:
    if postproc is None:
        return pred_ids

    dot_id = bundle["struct2id"]["."]

    if postproc == "ktd":
        return repair_kill_to_dot(
            pred_ids=pred_ids,
            valid_mask=valid_mask,
            structure=bundle["structure"],
            dot_id=dot_id,
        )

    if postproc == "dpgs":
        if type_id_pairs is None:
            raise ValueError("type_id_pairs missing for dpgs")
        return repair_dpgs(
            pred_ids=pred_ids,
            logits=logits_cpu,
            valid_mask=valid_mask,
            type_id_pairs=type_id_pairs,
            dot_id=dot_id,
        )

    raise ValueError(f"Unknown postproc mode: {postproc}")


def compute_score_summary(
    probs: torch.Tensor,
    entropy: torch.Tensor,
    pred_ids: torch.Tensor,
    raw_pred_ids: torch.Tensor,
) -> dict[str, float | None]:
    T = int(pred_ids.numel())

    chosen_probs = probs[torch.arange(T), pred_ids].clamp_min(1e-12)
    raw_chosen_probs = probs[torch.arange(T), raw_pred_ids].clamp_min(1e-12)

    seq_logprob = float(chosen_probs.log().sum().item())
    raw_seq_logprob = float(raw_chosen_probs.log().sum().item())
    avg_logprob = seq_logprob / max(T, 1)
    mean_confidence = float(probs.max(dim=-1).values.mean().item())
    mean_entropy = float(entropy.mean().item())
    delta = seq_logprob - raw_seq_logprob

    return {
        "delta": delta,
        "seq_logprob": seq_logprob,
        "avg_logprob": avg_logprob,
        "mean_confidence": mean_confidence,
        "mean_entropy": mean_entropy,
    }


@torch.no_grad()
def predict(
    model,
    device: str,
    seq: str,
    bundle: dict,
    postproc: str | None,
    type_id_pairs: list[tuple[int, int]] | None,
    need_token_table: bool,
    need_scores: bool,
):
    x = encode_sequence(seq, bundle["base2id"]).to(device)
    logits = model(x)[0]
    logits_cpu = logits.cpu()

    raw_pred_ids = logits_cpu.argmax(dim=-1)
    valid_mask = torch.ones(raw_pred_ids.numel(), dtype=torch.bool)

    pred_ids = apply_postprocessing(
        pred_ids=raw_pred_ids,
        logits_cpu=logits_cpu,
        valid_mask=valid_mask,
        bundle=bundle,
        postproc=postproc,
        type_id_pairs=type_id_pairs,
    )
    pred = "".join(bundle["inv_struct"][int(i)] for i in pred_ids.tolist())

    need_probs = need_token_table or need_scores
    topk_chars = None
    topk_probs = None
    confidence = None
    entropy_vals = None
    score_summary = None

    if need_probs:
        probs = torch.softmax(logits_cpu, dim=-1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)

        if need_token_table:
            topk_p, topk_i = probs.topk(k=3, dim=-1)

            topk_chars = [
                [bundle["inv_struct"][int(topk_i[t, j])] for j in range(3)]
                for t in range(topk_i.size(0))
            ]
            topk_probs = [
                [float(topk_p[t, j]) for j in range(3)]
                for t in range(topk_p.size(0))
            ]
            confidence = [row[0] for row in topk_probs]
            entropy_vals = [float(entropy[t]) for t in range(entropy.numel())]

        if need_scores:
            score_summary = compute_score_summary(
                probs=probs,
                entropy=entropy,
                pred_ids=pred_ids,
                raw_pred_ids=raw_pred_ids,
            )

    return (
        pred,
        pred_ids,
        topk_chars,
        topk_probs,
        confidence,
        entropy_vals,
        score_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer RNA structure with transformer checkpoint"
    )
    parser.add_argument("ckpt", type=str, help="Path to checkpoint file")
    parser.add_argument("--seq", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--token-table", action="store_true")
    parser.add_argument("--show-validity", action="store_true")
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument(
        "--postproc",
        type=str,
        choices=["ktd", "dpgs"],
        default=None,
    )
    parser.add_argument(
        "--label-mode",
        choices=["strict", "pragmatic", "both"],
        default=None,
        help="Show structure-element labels for the predicted dot-bracket output",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")

    device = "cpu"
    if not args.cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    model, bundle, _ = load_model_from_checkpoint(args.ckpt, device)

    type_id_pairs = None
    if args.postproc == "dpgs":
        type_order = bundle["structure"].bracket_type_order
        if type_order:
            type_id_pairs = [
                (bundle["struct2id"][pair[0]], bundle["struct2id"][pair[1]])
                for pair in type_order
            ]
        else:
            type_id_pairs = [
                (bundle["struct2id"][op], bundle["struct2id"][cl])
                for op, cl in bundle["structure"].bracket_pairs.items()
            ]

    seq = read_sequence(
        args.seq,
        args.path,
        bundle["base2id"],
        bundle["max_len"],
    )

    (
        pred,
        pred_ids,
        topk_chars,
        topk_probs,
        confidence,
        entropy_vals,
        score_summary,
    ) = predict(
        model=model,
        device=device,
        seq=seq,
        bundle=bundle,
        postproc=args.postproc,
        type_id_pairs=type_id_pairs,
        need_token_table=args.token_table,
        need_scores=args.show_scores,
    )

    info_parts = []
    if args.postproc is not None:
        info_parts.append(f"postproc={args.postproc}")
    if args.show_validity:
        valid_mask = torch.ones(len(seq), dtype=torch.bool)
        is_valid = dotbracket_is_valid_ids(pred_ids, valid_mask, bundle["structure"])
        info_parts.append(f"valid={is_valid}")

    if info_parts:
        print("[INFO] " + " | ".join(info_parts))

    if args.show_scores:
        print()
        print_score_table(
            delta=score_summary["delta"],
            seq_logprob=score_summary["seq_logprob"],
            avg_logprob=score_summary["avg_logprob"],
            mean_confidence=score_summary["mean_confidence"],
            mean_entropy=score_summary["mean_entropy"],
        )
        print()

    if args.token_table:
        print_token_table(seq, pred, topk_chars, topk_probs, confidence, entropy_vals)
    else:
        print(seq)
        print(pred)

    if args.label_mode is not None:
        print()
        print_structure_labels(pred, args.label_mode)


if __name__ == "__main__":
    main()