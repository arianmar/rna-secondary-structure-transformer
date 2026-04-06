#!/usr/bin/env python3

BRACKET_ORDER = [
    "()",
    "[]",
    "{}",
    "<>",
    "Aa",
    "Bb",
    "Cc",
    "Dd",
    "Ee",
    "Ff",
    "Gg",
    "Hh",
    "Ii",
    "Jj",
    "Kk",
]

BRACKET_PAIRS = {op: cl for op, cl in BRACKET_ORDER}
OPEN_SET = frozenset(BRACKET_PAIRS)
CLOSE_TO_OPEN = {cl: op for op, cl in BRACKET_ORDER}
VALID_CHARS = OPEN_SET | frozenset(CLOSE_TO_OPEN) | {"."}


def parse_pairs(dbn: str) -> list[tuple[int, int]]:
    stacks = {op: [] for op in OPEN_SET}
    pairs: list[tuple[int, int]] = []

    for idx, ch in enumerate(dbn):
        if ch not in VALID_CHARS:
            raise ValueError(f"invalid character {ch!r} at pos {idx}")

        if ch == ".":
            continue

        if ch in OPEN_SET:
            stacks[ch].append(idx)
            continue

        op = CLOSE_TO_OPEN[ch]
        if not stacks[op]:
            raise ValueError(f"unmatched closing bracket {ch!r} at pos {idx}")

        start = stacks[op].pop()
        pairs.append((start, idx))

    for op, rest in stacks.items():
        if rest:
            raise ValueError(f"unmatched opening bracket {op!r} at pos {rest[-1]}")

    return sorted(pairs)


def assign_bracket_types(pairs: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    n = len(pairs)
    conflicts = [set() for _ in range(n)]

    for a in range(n):
        ia, ja = pairs[a]
        for b in range(a + 1, n):
            ib, jb = pairs[b]
            if (ia < ib < ja < jb) or (ib < ia < jb < ja):
                conflicts[a].add(b)
                conflicts[b].add(a)

    colors: list[int] = [-1] * n
    for i in range(n):
        used = {colors[j] for j in conflicts[i] if colors[j] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[i] = color

    return {pairs[i]: colors[i] for i in range(n)}


def normalize(dbn: str) -> str:
    pairs = parse_pairs(dbn)
    if not pairs:
        if all(ch == "." for ch in dbn):
            return dbn
        raise ValueError("invalid dot-bracket string")

    type_map = assign_bracket_types(pairs)
    result = ["."] * len(dbn)

    for (i, j), t in type_map.items():
        if t >= len(BRACKET_ORDER):
            raise ValueError(f"too many pseudoknot levels: need bracket type {t}")
        op, cl = BRACKET_ORDER[t]
        result[i] = op
        result[j] = cl

    return "".join(result)