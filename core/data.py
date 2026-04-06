import os
import pickle
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VocabInfo:
    seq_alphabet: list[str]
    struct_alphabet: list[str]
    base2id: dict[str, int]
    struct2id: dict[str, int]
    pad_x: int
    pad_y: int
    vocab_in: int
    num_classes: int
    max_len: int


@dataclass(frozen=True)
class StructureInfo:
    bracket_pairs: dict[str, str]
    bracket_order: list[str]
    bracket_type_order: list[str]
    bracket_type_counts: dict[str, int]
    paired_id_mask: torch.Tensor
    open_ids: tuple[int, ...]
    close_ids: tuple[int, ...]
    open_id_set: frozenset[int]
    close_id_set: frozenset[int]
    closeid_to_openid: dict[int, int]


@dataclass(frozen=True)
class DatasetBundle:
    items: list[tuple[str, str]]
    vocab: VocabInfo
    structure: StructureInfo
    base_mode: int


def load_dataset(data_path: str) -> DatasetBundle:
    pkl_path = data_path
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"dataset not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data_obj = pickle.load(f)

    meta = data_obj["meta"]
    base_mode = int(meta["base_mode"])
    seq_alphabet = list(meta["seq_alphabet"])
    struct_alphabet = list(meta["struct_alphabet"])
    bracket_pairs = dict(meta["bracket_pairs"])
    bracket_order = list(meta.get("bracket_order", []))
    bracket_type_order = list(meta.get("bracket_type_order", []))
    bracket_type_counts = dict(meta.get("bracket_type_counts", {}))
    max_len = int(meta["max_length"])

    base2id = {ch: i for i, ch in enumerate(seq_alphabet)}
    pad_x = len(base2id)
    vocab_in = pad_x + 1

    struct2id = {ch: i for i, ch in enumerate(struct_alphabet)}
    pad_y = len(struct2id)
    num_classes = pad_y

    paired_chars = set(bracket_pairs.keys()) | set(bracket_pairs.values())
    paired_id_mask = torch.zeros(pad_y + 1, dtype=torch.bool)
    for ch, idx in struct2id.items():
        if ch in paired_chars:
            paired_id_mask[idx] = True

    open_ids = tuple(struct2id[ch] for ch in bracket_pairs.keys())
    close_ids = tuple(struct2id[ch] for ch in bracket_pairs.values())
    open_id_set = frozenset(open_ids)
    close_id_set = frozenset(close_ids)
    closeid_to_openid = {
        struct2id[cl]: struct2id[op] for op, cl in bracket_pairs.items()
    }

    vocab = VocabInfo(
        seq_alphabet=seq_alphabet,
        struct_alphabet=struct_alphabet,
        base2id=base2id,
        struct2id=struct2id,
        pad_x=pad_x,
        pad_y=pad_y,
        vocab_in=vocab_in,
        num_classes=num_classes,
        max_len=max_len,
    )

    structure = StructureInfo(
        bracket_pairs=bracket_pairs,
        bracket_order=bracket_order,
        bracket_type_order=bracket_type_order,
        bracket_type_counts=bracket_type_counts,
        paired_id_mask=paired_id_mask,
        open_ids=open_ids,
        close_ids=close_ids,
        open_id_set=open_id_set,
        close_id_set=close_id_set,
        closeid_to_openid=closeid_to_openid,
    )

    return DatasetBundle(
        items=data_obj["data"],
        vocab=vocab,
        structure=structure,
        base_mode=base_mode,
    )


def make_split(items, split_path: str, split_seed: int, mode: str):
    n = len(items)

    created_new_split = False
    split_missing_in_current = False

    if mode == "new":
        rng = random.Random(split_seed)
        idxs = list(range(n))
        rng.shuffle(idxs)

        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        train_idx = idxs[:n_train]
        val_idx = idxs[n_train : n_train + n_val]
        test_idx = idxs[n_train + n_val :]

        with open(split_path, "wb") as f:
            pickle.dump(
                {
                    "seed": split_seed,
                    "train": train_idx,
                    "val": val_idx,
                    "test": test_idx,
                },
                f,
            )

        created_new_split = True

    elif mode == "current":
        if not os.path.exists(split_path):
            split_missing_in_current = True
            raise FileNotFoundError(
                f"MODE='current' but split file missing: {split_path}"
            )

        with open(split_path, "rb") as f:
            split = pickle.load(f)

        train_idx = split["train"]
        val_idx = split["val"]
        test_idx = split["test"]

    else:
        raise ValueError(f"unknown mode: {mode}")

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]

    split_info = {
        "created_new_split": created_new_split,
        "split_missing_in_current": split_missing_in_current,
        "train_size": len(train_items),
        "val_size": len(val_items),
        "test_size": len(test_items),
    }

    return train_items, val_items, test_items, split_info