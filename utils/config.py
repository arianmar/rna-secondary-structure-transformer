# utils/config.py
import argparse
import os
from dataclasses import dataclass
from typing import Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, "models")


@dataclass(frozen=True)
class TrainConfig:
    base: str
    mode: str
    data_path: str | None
    output_dir: str

    batch_train: Optional[int]
    batch_val: Optional[int]
    epochs: Optional[int]
    patience: Optional[int]
    min_delta: Optional[float]
    lr: Optional[float]
    weight_decay: Optional[float]
    eta_min: Optional[float]

    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float

    log_every: Optional[int]
    max_steps_per_epoch: Optional[int]
    split_seed: Optional[int]
    snap_every: Optional[int]

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="RNA structure transformer training")

    p.add_argument("--mode", type=str, default="new", choices=["new", "current"])
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "mode='new': path to dataset pickle | "
            "mode='current': path to a model directory or an epoch_* snapshot directory"
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=MODELS_DIR,
        help="Root directory where model workspaces are created (default: models/)",
    )

    pre, _ = p.parse_known_args()
    is_current = pre.mode == "current"

    def dflt(v):
        return None if is_current else v

    p.add_argument("--batch-train", type=int, default=dflt(64))
    p.add_argument("--batch-val", type=int, default=dflt(64))
    p.add_argument("--epochs", type=int, default=dflt(20))
    p.add_argument("--patience", type=int, default=dflt(3))
    p.add_argument("--min-delta", type=float, default=dflt(0.0))
    p.add_argument("--lr", type=float, default=dflt(5e-4))
    p.add_argument("--weight-decay", type=float, default=dflt(1e-2))
    p.add_argument("--eta-min", type=float, default=dflt(1e-5))

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--log-every", type=int, default=dflt(250))
    p.add_argument("--max-steps-per-epoch", type=int, default=dflt(None))
    p.add_argument("--split-seed", type=int, default=dflt(1337))
    p.add_argument("--snap-every", type=int, default=dflt(5))

    args = p.parse_args()

    if args.d_model % args.n_heads != 0:
        p.error(f"--d-model ({args.d_model}) must be divisible by --n-heads ({args.n_heads})")
    if args.snap_every is not None and args.snap_every <= 0:
        p.error("--snap-every must be >= 1")

    return TrainConfig(
        base=BASE,
        mode=args.mode,
        data_path=args.data_path,
        output_dir=os.path.abspath(args.output),
        batch_train=args.batch_train,
        batch_val=args.batch_val,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eta_min=args.eta_min,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        log_every=args.log_every,
        max_steps_per_epoch=args.max_steps_per_epoch,
        split_seed=args.split_seed,
        snap_every=args.snap_every,
    )