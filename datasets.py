import os

import numpy as np
import pandas as pd
from numpy.random import exponential, triangular

SYNTHETIC_DATASET_TYPES = ["uniform", "linear", "exponential"]


def generate_uniform(n):
    return np.ones(n)


def generate_exponential(n, scale=4):
    return exponential(scale, n)


def generate_linear(n):
    return np.abs(triangular(-1, 0, 1, 2 * n)[:n])


def generate_data(method: str, n):
    if method == "uniform":
        return generate_uniform(n)
    elif method == "linear":
        return generate_linear(n)
    elif method == "exponential":
        return generate_exponential(n)


class SingleQueryDataset(object):
    def __init__(self, relevance):
        self.relevance = np.asarray(relevance)
        self.relevance.sort()
        self.relevance = relevance[::-1]
        self.relevance /= self.relevance.sum()

        # generate baseline ranking p
        self.p = list(range(0, len(self.relevance)))


class Synthetic(SingleQueryDataset):
    def __init__(self, method, n=100):
        self.method = method
        self.n = n
        super().__init__(generate_data(method, n))

    def __str__(self):
        return f"syn-{self.method}_n={self.n}"


class AirBNBSingleQuery(SingleQueryDataset):
    def __init__(self, filename):
        self.name = os.path.basename(filename)
        df = pd.read_csv(filename, usecols=['review_scores_rating'])
        df.dropna(inplace=True)
        relevance = df['review_scores_rating'].values

        super().__init__(relevance)

    def __str__(self):
        return f"AirBNB-{self.name}"
