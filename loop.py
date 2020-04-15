import math
import os
from collections import namedtuple
from sys import stdout as out, maxsize

import pandas as pd
from mip import Model, xsum, minimize, BINARY

from core import *
from datasets import Synthetic

np.set_printoptions(precision=2)
logger = get_simple_logger(level=logging.INFO)

Experiment = namedtuple("Experiment", ['dataset', 'k', 'w', 'thetas', 'iterations', 'D'])


def build_model(A, R, r, w, k, theta, idcg, extended):
    # Index variables
    I = [i for i in range(0, len(R))]

    m = Model('ILP Fair ranking', )

    # Create the variables to be optimized
    x = [[m.add_var(var_type=BINARY) for j in I] for i in I]

    if(extended):
        # Create extended optimization function
        m.objective = minimize(xsum((A[i] + w(j) - (R[i] + r[i]))**2 * x[i][j] for j in I for i in I))
    else:
        # Create original optimization function
        m.objective = minimize(xsum(abs(A[i] + w(j) - (R[i] + r[i])) * x[i][j] for j in I for i in I))

    # Try with KL divergence
    # Try with Jensen - Shannon divergence


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
        results.append([it, 0, 0, 1, unfairness, 0, "relevance"])

    return pd.DataFrame(results, columns=["it", "idcg", "dcg", "ndcg", "unfairness", "k", "theta"])


def run_model(r, w, k, theta, D=20, iterations=350):
    # Attention so far
    A = np.zeros(len(r))
    # Relevance so far
    R = np.zeros(len(r))
    r = np.asarray(r)

    results = []
    unfairness = []
    total_relevance = 0

    #Set this to true if you want to use the quadratic function
    extended = True

    for iteration in range(1, iterations):

        ideal_ranking = np.argsort(r)[::-1]
        idcg = dcg(k, r[ideal_ranking])

        selection = prefilter_selection(A, R, r, D, k)
        _A = A[selection]
        _R = R[selection]
        _r = r[selection]

        m, x = build_model(_A, _R, _r, w, k, theta, idcg, extended)
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

            computed_unfairness = compute_unfairness(A,R)
            unfairness.append([iteration, computed_unfairness])
            unfairness_sum = computed_unfairness.sum()
            total_relevance += r.sum()  # just a sanity check this should always be equation to it*1

            logger.info(f"---- (theta:{theta}, k:{k}, D:{D}) ITERATION: {iteration} ----")
            logger.info(f"Unfairness/total_relevance: \t\t{unfairness_sum:0.2f}/{total_relevance:.2f}")
            logger.info(
                f"New ranking DCG@{k},IDCG@{k}, NDCG@{k}: \t{new_ranking_dcg:0.3f}, {idcg:0.3f}, {new_ranking_ndcg:0.3f}")
            logger.debug(f"Relevance r_i: \t\t\t\t\t\t{r}")
            logger.debug(f"Optimal ranking: \t\t\t\t\t{ideal_ranking}")
            logger.debug(f"New ranking after iteration \t\t{np.asarray(new_ranking)}")
            logger.debug(f"Attention accumulated: \t\t\t\t{A}")
            logger.debug(f"Relevance accumulated: \t\t\t\t{R}")

            results.append([iteration, idcg, new_ranking_dcg, new_ranking_ndcg, unfairness_sum, k, f'{theta}'])
        else:
            Exception("This should never happen")
    return pd.DataFrame(results, columns=["it", "idcg", "dcg", "ndcg", "unfairness", "k", "theta"]) , pd.DataFrame(unfairness, columns=["it", "unfairness"])



def store_results(results, filename="results.csv"):
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/" + filename, float_format='%.3f', index=False)

def store_unfairness_array(unfairness_array, filename ="resultss.csv"):
    os.makedirs("resultss_unfar", exist_ok=True)
    unfairness_array.to_csv("resultss_unfar/" + filename, float_format='%.3f', index=False)

def get_experiment_filename(exp: Experiment, attention_model):
    return f"results_{exp.dataset}_k={exp.k}_{attention_model}.csv"


def run_experiment(exp: Experiment, include_baseline=True):
    results = []

    for theta in exp.thetas:
        res, unfairness_arr = run_model(exp.dataset.relevance, exp.w, exp.k, theta, exp.D, exp.iterations)
        results.append(res)

        #results.append(run_model(exp.dataset.relevance, exp.w, exp.k, theta, exp.D, exp.iterations))

    if include_baseline:
        results.append(relevance_model(exp.dataset.relevance, exp.w, exp.iterations))

    return pd.concat(results), unfairness_arr


if __name__ == '__main__':

    # Executing from this file is for debugging
    exp = Experiment(Synthetic("uniform", n=300), 1, attention_model_singular(), [0.6, 0.8], 200, 35)
    df = run_experiment(exp)
    # plot_results(df)
