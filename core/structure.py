import torch

from core.data import StructureInfo


def dotbracket_is_valid_ids(
    pred_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    structure: StructureInfo,
) -> bool:
    stacks = {op_id: 0 for op_id in structure.open_ids}
    T = int(pred_ids.numel())

    for t in range(T):
        if not bool(valid_mask[t]):
            continue

        tok = int(pred_ids[t])

        if tok in structure.open_id_set:
            stacks[tok] += 1
        elif tok in structure.close_id_set:
            op = structure.closeid_to_openid.get(tok)
            if op is None:
                continue
            if stacks[op] <= 0:
                return False
            stacks[op] -= 1

    return all(v == 0 for v in stacks.values())