import torch


def repair_kill_to_dot(
    pred_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    structure,
    dot_id: int,
) -> torch.Tensor:
    out = pred_ids.clone()
    stacks = {op_id: [] for op_id in structure.open_ids}

    for t in range(int(out.numel())):
        if not bool(valid_mask[t]):
            continue

        tok = int(out[t])

        if tok in structure.open_id_set:
            stacks[tok].append(t)
        elif tok in structure.close_id_set:
            op = structure.closeid_to_openid[tok]
            if stacks[op]:
                stacks[op].pop()
            else:
                out[t] = dot_id

    for positions in stacks.values():
        for pos in positions:
            out[pos] = dot_id

    return out