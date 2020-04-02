import math
from sys import stdout as out

from mip import Model, xsum, minimize, BINARY

# original ranking
from core import *


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


def convert_solution_to_ranking(x):
    ranking = [-1 for i in x]
    I = range(0, len(x))
    for i in I:
        out.write("[[ ")
        for j in I:
            out.write(f"{x[i][j].x} ")
            if x[i][j].x == 1:
                ranking[j] = i
        out.write("]]\n")
    return ranking


def run_model(r, w, theta):
    # Attention so far
    A = list(np.zeros(len(r)))
    # Relevance so far
    R = list(np.zeros(len(r)))

    logger = get_simple_logger()

    for iteration in range(1, 5):

        ideal_ranking = np.argsort(r)[::-1]
        idcg = dcg(k, np.asarray(r)[ideal_ranking])

        m, x = build_model(A, R, r, w, k, theta, idcg)
        m.optimize()

        print(f"IDCG@{k} is {idcg:0.3f}")

        if m.num_solutions:
            print(f"Ranking with a total cost of {m.objective_value}")
            print("Number of solutions: %d" % m.num_solutions)

            # array when the value on the i_th position indicates the subject at rank i
            new_ranking = convert_solution_to_ranking(x)

            # Add attention each subject receives
            for rank, subject in enumerate(new_ranking):
                A[subject] += w(rank)

            # For single query this should be just factor (needs to be checked)
            for subject, relevance in enumerate(r):
                R[subject] += relevance

            new_ranking_dcg = dcg(k, np.asarray(r)[new_ranking])
            new_ranking_ndcg = new_ranking_dcg / idcg
            logger.info(f"---- (theta:{theta}) ITERATION: {iteration} ----")
            logger.info(f"Relevance r_i: \t\t\t\t\t\t{r}")
            logger.info(f"Optimal ranking: \t\t\t\t\t{list(ideal_ranking)}")
            logger.info(f"New ranking after iteration \t\t{new_ranking}")
            logger.info(
                f"New ranking DCG@{k},IDCG@{k}, NDCG@{k}: \t{new_ranking_dcg:0.3f}, {idcg:0.3f}, {new_ranking_ndcg:0.3f}")
            logger.info(f"Attention accumulated: \t\t\t\t{A}")
            logger.info(f"Relevance accumulated: \t\t\t\t{R}")


THETA = 0.8
k = 3
# Current relevance
r = [4, 3, 2, 1, 1, 6]
# Attention model
w = attention_geometric(k, 0.5)

run_model(r, w, THETA)
