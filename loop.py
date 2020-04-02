import math
from sys import stdout as out

import numpy as np
from mip import Model, xsum, minimize, BINARY

# original ranking
from core import dcg, attention_model_singular

# p = [1, 2, 2, 3, 4, 5]

# Attention so far
A = [0, 0, 0, 0, 0, 0]
# Relevance so far
R = [1, 1, 1, 1, 1, 1]
# Current relevance
r = [4, 3, 2, 1, 1, 6]
# Attention model
w = attention_model_singular()

k = 3

# Get optimal ordering using relevance:
optimal = np.argsort(r)[::-1]
idcg = dcg(k, np.asarray(r)[optimal])
theta = 1

# Index variables
I = [i for i in range(0, len(R))]

m = Model('ILP Fair ranking')

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

m.optimize()

print(f"IDCG is {idcg:0.3f}")

if m.num_solutions:
    print(f"Ranking with a total cost of {m.objective_value}")
    print("Number of solutions: %d" % m.num_solutions)

    new_ranking = [-1 for i in I]
    for i in I:
        out.write("[[ ")
        for j in I:
            out.write(f"{x[i][j].x} ")
            if x[i][j].x == 1:
                new_ranking[j] = i
        out.write("]]\n")
    print("")
    print("Optimal ranking: \t", list(optimal))
    print("New ranking: \t\t", new_ranking)
