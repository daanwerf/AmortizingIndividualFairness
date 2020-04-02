def attention_geometric(j: int, k: int, p: float) -> float:
    return p * (1 - p) ** (j - 1)


def attention_singular(j: int) -> int:
    return 1 if (j == 1) else 0
