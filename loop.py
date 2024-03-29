import math
import os
from collections import namedtuple
from sys import stdout as out, maxsize

import pandas as pd
from mip import Model, xsum, minimize, BINARY

from core import *

np.set_printoptions(precision=2)
logger = get_simple_logger(level=logging.INFO)

Experiment = namedtuple("Experiment", ['dataset', 'k', 'w', 'thetas', 'iterations', 'D'])


def build_model(A, R, r, w, k, theta, idcg):
    # Index variables
    I = [i for i in range(0, len(R))]

    m = Model('ILP Fair ranking', )

    # Create the variables to be optimized
    x = [[m.add_var(var_type=BINARY) for j in I] for i in I]

    # Create optimization function
    m.objective = minimize(xsum(abs(A[i] + w(j) - (R[i] + r[i])) * x[i][j] for j in I for i in I))

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


def relevance_model(r, w, iterations=350):
    results = []
    ideal_ranking = np.argsort(r)[::-1]
    ranks = np.arange(0, len(r))

    for it in range(0, iterations):
        wvec = np.vectorize(lambda x: w(x) * it)
        A = wvec(ranks)
        R = r * it
        unfairness = np.abs((R - A[ideal_ranking])).sum()
        results.append([it, 0, 0, -1, unfairness, 0, "relevance"])

    return pd.DataFrame(results, columns=["it", "idcg", "dcg", "ndcg", "unfairness", "k", "model"])


def run_model(r, w, k, theta, D=20, iterations=350):
    # Attention so far
    A = np.zeros(len(r))
    # Relevance so far
    R = np.zeros(len(r))
    r = np.asarray(r)

    results = []
    total_relevance = 0

    for iteration in range(1, iterations):

        ideal_ranking = np.argsort(r)[::-1]
        idcg = dcg(k, r[ideal_ranking])

        selection = prefilter_selection(A, R, r, D, k)
        _A = A[selection]
        _R = R[selection]
        _r = r[selection]

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
                if rank > k + 1:
                    break

            # Add gained relevance
            R += r

            new_ranking_dcg = dcg(k, r[new_ranking])
            new_ranking_ndcg = new_ranking_dcg / idcg
            unfairness = compute_unfairness(A, R).sum()
            total_relevance += r.sum()  # just a sanity check this should always be equation to it*1

            logger.info(f"---- (theta:{theta}, k:{k}, D:{D}) ITERATION: {iteration} ----")
            logger.info(f"Unfairness/total_relevance: \t\t{unfairness:0.2f}/{total_relevance:.2f}")
            logger.info(
                f"New ranking DCG@{k},IDCG@{k}, NDCG@{k}: \t{new_ranking_dcg:0.3f}, {idcg:0.3f}, {new_ranking_ndcg:0.3f}")
            logger.debug(f"Relevance r_i: \t\t\t\t\t\t{r}")
            logger.debug(f"Optimal ranking: \t\t\t\t\t{ideal_ranking}")
            logger.debug(f"New ranking after iteration \t\t{np.asarray(new_ranking)}")
            logger.debug(f"Attention accumulated: \t\t\t\t{A}")
            logger.debug(f"Relevance accumulated: \t\t\t\t{R}")

            results.append([iteration, idcg, new_ranking_dcg, new_ranking_ndcg, unfairness, k, f'theta={theta}'])
        else:
            Exception("This should never happen")
    return pd.DataFrame(results, columns=["it", "idcg", "dcg", "ndcg", "unfairness", "k", "model"])


def run_model_prob(r, k, w, iterations=350, D=50, swaps=1, rate=0.7):
    # Attention so far
    A = np.zeros(len(r))
    # Relevance so far
    R = np.zeros(len(r))
    r = np.asarray(r)

    # Compute the ideal ranking
    ideal_ranking = np.argsort(r)[::-1]
    idcg = dcg(k, r[ideal_ranking])
    results = []

    total_relevance = 0

    for iteration in range(0, iterations):
        new_ranking = np.copy(ideal_ranking)

        # Compute unfairness
        U = A - (R + r)

        # Most unfair objects
        max_unfairness = np.argsort(U)[:D]

        # Swap with prob
        u = U[max_unfairness]
        u[u >= 0] = 0
        u = np.abs(u)
        if u.sum() > 0:
            u /= u.sum()

        cdf_u = np.cumsum(u)

        swappend = set()
        swap = 0
        while swap < swaps:
            sample_u = np.random.rand()
            sample = np.argmax(cdf_u > sample_u)
            swap_candidate = max_unfairness[sample]  # index of most unfair cand sampled
            swap_idx = np.where(new_ranking == swap_candidate)  # position in ranking

            # swap_pos = np.random.poisson(rate)
            swap_pos = np.random.geometric(rate) - 1
            swap_pos = min(swap_pos, len(new_ranking) - 1)
            if swap_candidate in swappend or swap_pos in swappend:
                continue
            else:
                swappend.add(swap_pos)
                swappend.add(swap_candidate)

            new_ranking[swap_pos], new_ranking[swap_idx] = new_ranking[swap_idx], new_ranking[swap_pos]
            swap += 1

        # Add gained relevance
        R += r

        # Add attention
        # Add attention each subject receives
        for rank, subject in enumerate(new_ranking):
            A[subject] += w(rank)
            if rank > k + 1:
                break

        new_ranking_dcg = dcg(k, r[new_ranking])
        new_ranking_ndcg = new_ranking_dcg / idcg
        unfairness = compute_unfairness(A, R).sum()
        total_relevance += r.sum()  # just a sanity check this should always be equation to it*1

        if iteration % 100 == 0:
            logger.info(f"---- (k:{k}, D:) ITERATION: {iteration} ----")
            logger.info(f"Unfairness/total_relevance: \t\t{unfairness:0.2f}/{total_relevance:.2f}")
            logger.info(
                f"New ranking DCG@{k},IDCG@{k}, NDCG@{k}: \t{new_ranking_dcg:0.3f}, {idcg:0.3f}, {new_ranking_ndcg:0.3f}")
        logger.debug(f"Relevance r_i: \t\t\t\t\t\t{r}")
        logger.debug(f"Optimal ranking: \t\t\t\t\t{ideal_ranking}")
        logger.debug(f"New ranking after iteration \t\t{np.asarray(new_ranking)}")
        logger.debug(f"Attention accumulated: \t\t\t\t{A}")
        logger.debug(f"Relevance accumulated: \t\t\t\t{R}")

        results.append([iteration, idcg, new_ranking_dcg, new_ranking_ndcg, unfairness, k, 'prob'])

    return pd.DataFrame(results, columns=["it", "idcg", "dcg", "ndcg", "unfairness", "k", "model"])


def store_results(results, filename="results.csv"):
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/" + filename, float_format='%.3f', index=False)


def get_experiment_filename(exp: Experiment, attention_model):
    return f"results_{exp.dataset}_k={exp.k}_{attention_model}.csv"


def run_experiment(exp: Experiment, include_baseline=True):
    results = []
    for theta in exp.thetas:
        results.append(run_model(exp.dataset.relevance, exp.w, exp.k, theta, exp.D, exp.iterations))

    if include_baseline:
        results.append(relevance_model(exp.dataset.relevance, exp.w, exp.iterations))

    return pd.concat(results)
