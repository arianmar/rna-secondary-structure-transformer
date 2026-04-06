import torch
import torch.nn.functional as F


def build_type_id_pairs(
    struct2id: dict[str, int],
    bracket_pairs: dict[str, str],
    bracket_type_order: list[str] | None = None,
) -> list[tuple[int, int]]:
    if bracket_type_order:
        return [
            (struct2id[pair[0]], struct2id[pair[1]])
            for pair in bracket_type_order
        ]
    return [(struct2id[op], struct2id[cl]) for op, cl in bracket_pairs.items()]


def repair_dpgs(
    pred_ids: torch.Tensor,
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    type_id_pairs: list[tuple[int, int]],
    dot_id: int,
) -> torch.Tensor:
    out = pred_ids.clone()
    valid_idx = valid_mask.nonzero(as_tuple=False).flatten().tolist()
    if not valid_idx:
        return out

    log_probs = F.log_softmax(logits, dim=-1)
    neg_inf = float("-inf")

    for open_id, close_id in type_id_pairs:
        rel_pos: list[int] = []
        score_dot: list[float] = []
        score_open: list[float] = []
        score_close: list[float] = []

        for pos in valid_idx:
            tok = int(out[pos])
            if tok == dot_id or tok == open_id or tok == close_id:
                rel_pos.append(pos)
                row = log_probs[pos]
                score_dot.append(float(row[dot_id]))
                score_open.append(float(row[open_id]))
                score_close.append(float(row[close_id]))

        m = len(rel_pos)
        if m == 0:
            continue

        dp = [neg_inf] * (m + 1)
        dp[0] = 0.0
        choices = [bytearray(m + 1) for _ in range(m)]

        for i in range(m):
            remaining = m - i - 1
            nxt = [neg_inf] * (m + 1)

            s_dot = score_dot[i]
            s_open = score_open[i]
            s_close = score_close[i]

            choice_row = choices[i]
            max_depth = min(i, remaining + 1)

            for depth in range(max_depth + 1):
                cur = dp[depth]
                if cur == neg_inf:
                    continue

                cand = cur + s_dot
                if cand > nxt[depth]:
                    nxt[depth] = cand
                    choice_row[depth] = 0

                if depth + 1 <= remaining:
                    cand = cur + s_open
                    if cand > nxt[depth + 1]:
                        nxt[depth + 1] = cand
                        choice_row[depth + 1] = 1

                if depth > 0:
                    cand = cur + s_close
                    if cand > nxt[depth - 1]:
                        nxt[depth - 1] = cand
                        choice_row[depth - 1] = 2

            dp = nxt

        if dp[0] == neg_inf:
            continue

        depth = 0
        for i in range(m - 1, -1, -1):
            action = choices[i][depth]
            pos = rel_pos[i]

            if action == 0:
                out[pos] = dot_id
            elif action == 1:
                out[pos] = open_id
                depth -= 1
            else:
                out[pos] = close_id
                depth += 1

    return out