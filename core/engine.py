import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.dataset import RNADataset, PermutedRNADataset
from core.structure import dotbracket_is_valid_ids
from utils.logger import log


def make_train_loader(train_items, ds, batch_size, split_seed, epoch, collate_fn):
    perm = list(range(len(train_items)))
    random.Random(split_seed + epoch).shuffle(perm)
    return DataLoader(
        PermutedRNADataset(train_items, perm, ds.vocab.base2id, ds.vocab.struct2id),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def build_eval_loaders(val_items, test_items, ds, batch_size, collate_fn):
    mk = lambda items: DataLoader(
        RNADataset(items, ds.vocab.base2id, ds.vocab.struct2id),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return mk(val_items), mk(test_items)


@torch.no_grad()
def evaluate(model, loader, device, ds):
    model.eval()
    paired_mask = ds.structure.paired_id_mask.to(device)

    correct = total = seq_total = seq_exact = tp = fp = fn = invalid_seq = 0
    loss_sum = 0.0
    token_count = 0.0

    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x)

        valid = (y != ds.vocab.pad_y) & mask
        loss_sum += F.cross_entropy(
            logits.reshape(-1, ds.vocab.num_classes),
            y.reshape(-1),
            ignore_index=ds.vocab.pad_y,
            reduction="sum",
        ).item()
        token_count += valid.sum().item()

        pred = logits.argmax(dim=-1)
        correct += ((pred == y) & valid).sum().item()
        total += valid.sum().item()

        seq_total += x.size(0)
        seq_exact += ((pred == y) | (~valid)).all(dim=1).sum().item()

        y_paired = paired_mask[y] & valid
        p_paired = paired_mask[pred] & valid
        tp += (p_paired & y_paired).sum().item()
        fp += (p_paired & (~y_paired)).sum().item()
        fn += ((~p_paired) & y_paired).sum().item()

        pred_cpu = pred.detach().cpu()
        valid_cpu = valid.detach().cpu()
        for b in range(pred_cpu.size(0)):
            if not dotbracket_is_valid_ids(pred_cpu[b], valid_cpu[b], ds.structure):
                invalid_seq += 1

    acc = correct / max(total, 1)
    loss = loss_sum / max(token_count, 1)
    seq_exact_acc = seq_exact / max(seq_total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_paired = (2 * precision * recall) / max(precision + recall, 1e-12)
    invalid_rate = invalid_seq / max(seq_total, 1)
    return acc, loss, seq_exact_acc, f1_paired, invalid_rate


def train_epoch(model, train_items, ds, batch_size, split_seed, epoch, opt, device, log_every, max_steps, collate_fn, bad_epochs):
    model.train()
    loader = make_train_loader(train_items, ds, batch_size, split_seed, epoch, collate_fn)

    running = 0.0
    steps = 0
    header = False

    for step, (x, y, _mask) in enumerate(loader, start=1):
        if step == 1:
            log(f"[INFO] epoch {epoch}: first batch B={x.size(0)} T={x.size(1)} bad_epochs={bad_epochs}")

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, ds.vocab.num_classes), y.reshape(-1), ignore_index=ds.vocab.pad_y)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running += loss.item()
        steps += 1

        if step <= 10 or step % log_every == 0:
            if not header:
                log(f"{'ep':>2} | {'step':>4} | {'loss':>7} | {'avg_loss':>8}")
                header = True
            log(f"{epoch:>2d} | {step:>4d} | {loss.item():>7.4f} | {running / steps:>8.4f}")

        if max_steps is not None and step >= max_steps:
            break

    return running / max(steps, 1), steps


def run_test_eval(model, test_loader, best_ckpt_path, device, ds):
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        log(f"[INFO] loaded best checkpoint from {best_ckpt_path} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})")

    t0 = time.time()
    test_acc, test_loss, test_seq_acc, test_f1_paired, test_invalid = evaluate(model, test_loader, device, ds)
    dt = time.time() - t0

    log(f"{'test_loss':>9} | {'test_acc':>8} | {'test_seq':>8} | {'test_f1':>8} | {'test_inv':>8} | {'test_s':>6}")
    log(f"{test_loss:>9.4f} | {test_acc:>8.4f} | {test_seq_acc:>8.4f} | {test_f1_paired:>8.4f} | {test_invalid:>8.4f} | {dt:>6.1f}")