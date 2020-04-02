import numpy as np


def attention_geometric(k: int, p: float):
    return lambda j: p * (1 - p) ** (j - 1) if j < k else 0


def attention_model_singular():
    return lambda j: 1 if (j == 1) else 0


def dcg(k, r):
    return (np.exp2(np.asarray(r)[0:k]) / np.log2(np.arange(2, k + 2))).sum()


def ndcg(k, r_1, r_2):
    return dcg(k, r_1) / dcg(k, r_2)
