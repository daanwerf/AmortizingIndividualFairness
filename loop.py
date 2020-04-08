import math
import os
from sys import stdout as out, maxsize

import pandas as pd
from mip import Model, xsum, minimize, BINARY

from core import *
from datasets import Synthetic, AirBNBSingleQuery

np.set_printoptions(precision=2)


def build_model(A, R, r, w, k, theta, idcg):
    # Index variables
    I = [i for i in range(0, len(R))]

    m = Model('ILP Fair ranking', )

    # Create the variables to be optimized
    x = [[m.add_var(var_type=BINARY) for j in I] for i in I]

    # Create optimization function
    m.objective = minimize(xsum((A[i] * w(j) - (R[i] + r[i])) * x[i][j] for j in I for i in I))

    # Constraints

    # Only assign 1 ranking per items
    for i in I:
        m += xsum(x[i][j] for j in I) == 1

    # Only one per position
    for j in I:
        m += xsum(x[i][j] for i in I) == 1

    m += xsum(((2 ** r[i] - 1) / math.log(j + 2, 2)) * x[i][j] for i in I for j in range(0, k)) >= theta * idcg

    return m, x


def convert_solution_to_ranking(x, verbose=False):
    ranking = [-1 for i in x]
    I = range(0, len(x))
    for i in I:
        if verbose: out.write("[[ ")
        for j in I:
            if verbose: out.write(f"{x[i][j].x} ")
            if x[i][j].x == 1:
                ranking[j] = i
        if verbose: out.write("]]\n")
    return ranking


def compute_unfairness(A, R):
    return np.abs(np.asarray(A) - np.asarray(R))


def prefilter_selection(A, R, r, D, k):
    k_top_relevant = np.argsort(r)[::-1]
    unfairness = np.asarray(A) - (np.asarray(R) + r)
    unfairness[k_top_relevant[:k]] = -maxsize
    selection = np.argsort(unfairness)[:D]
    return selection


def run_model(r, w, k, theta, D=20, iterations=350):
    # Attention so far
    A = list(np.zeros(len(r)))
    # Relevance so far
    R = list(np.zeros(len(r)))

    logger = get_simple_logger(level=logging.INFO)
    # logger = get_simple_logger(level=logging.DEBUG)
    results = []

    for iteration in range(1, iterations):

        ideal_ranking = np.argsort(r)[::-1]
        idcg = dcg(k, np.asarray(r)[ideal_ranking])

        selection = prefilter_selection(A, R, r, D, k)
        _A = np.asarray(A)[selection]
        _R = np.asarray(R)[selection]
        _r = np.asarray(r)[selection]

        m, x = build_model(_A, _R, _r, w, k, theta, idcg)
        m.verbose = 0
        m.optimize()

        print(f"IDCG@{k} is {idcg:0.3f}")

        if m.num_solutions:
            print(f"{iteration} Ranking with a total cost of {m.objective_value}")
            print("Number of solutions: %d" % m.num_solutions)

            # array when the value on the i_th position indicates the subject at rank i
            new_ranking = convert_solution_to_ranking(x)
            new_ranking = selection[new_ranking]

            # Add attention each subject receives
            for rank, subject in enumerate(new_ranking):
                A[subject] += w(rank)

            # For single query this should be just factor (needs to be checked)
            R = list(np.asarray(R) + np.asarray(r))

            new_ranking_dcg = dcg(k, np.asarray(r)[new_ranking])
            new_ranking_ndcg = new_ranking_dcg / idcg
            unfairness = compute_unfairness(A, R).sum()

            logger.info(f"---- (theta:{theta}, k:{k}) ITERATION: {iteration} ----")
            logger.info(f"Unfairness: \t\t\t\t\t\t{unfairness}")
            logger.info(
                f"New ranking DCG@{k},IDCG@{k}, NDCG@{k}: \t{new_ranking_dcg:0.3f}, {idcg:0.3f}, {new_ranking_ndcg:0.3f}")
            logger.debug(f"Relevance r_i: \t\t\t\t\t\t{r}")
            logger.debug(f"Optimal ranking: \t\t\t\t\t{ideal_ranking}")
            logger.debug(f"New ranking after iteration \t\t{np.asarray(new_ranking)}")
            logger.debug(f"Attention accumulated: \t\t\t\t{np.asarray(A)}")
            logger.debug(f"Relevance accumulated: \t\t\t\t{np.asarray(R)}")

            results.append([iteration, idcg, new_ranking_dcg, new_ranking_ndcg, unfairness, k, theta])
        else:
            Exception("This should never happen")

    return pd.DataFrame(results, columns=["it", "idcg", "dcg", "ndcg", "unfairness", "k", "theta"])


def store_results(results):
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/results.csv", float_format='%.3f', index=False)


def toy_model():
    THETA = 0.8
    k = 3
    # Current relevance
    r = np.asarray([4, 3, 2, 1, 1, 6])
    r = r / r.sum()

    # Attention model
    w = attention_geometric(k, 0.5)

    results = run_model(r, w, k, THETA)


def run_on_datasets():
    THETA = 0.6
    k = 1

    ds = Synthetic("exponential", n=30000)
    ds = Synthetic("uniform", n=25)
    ds = AirBNBSingleQuery("data/boston.csv")
    r = ds.relevance

    # Attention model
    # w = attention_geometric(25, 0.5)
    w = attention_model_singular()

    results = run_model(r, w, k, THETA, iterations=10000)
    store_results(results)


if __name__ == '__main__':
    run_on_datasets()
