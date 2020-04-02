import logging

import numpy as np


def attention_geometric(k: int, p: float):
    return lambda j: p * (1 - p) ** (j - 1) if j < k else 0


def attention_model_singular():
    return lambda j: 1 if (j == 0) else 0


def dcg(k, r):
    return ((np.exp2((np.asarray(r))[0:k]) - 1) / np.log2(np.arange(2, k + 2))).sum()


def ndcg(k, r_1, r_2):
    return dcg(k, r_1) / dcg(k, r_2)


def get_simple_logger(filename="out.log"):
    # create logger
    logger = logging.getLogger('project logger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    ch = logging.FileHandler(filename, mode="w")
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
