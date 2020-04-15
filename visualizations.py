import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from core import attention_geometric
from datasets import Synthetic
from loop import Experiment, attention_model_singular, run_experiment, get_experiment_filename, store_results, store_unfairness_array


def load_or_run(exp, filename, overwrite=False):
    if os.path.exists("results/" + filename) and not overwrite:
        return pd.read_csv("results/" + filename)
    else:
        results_df, unf_arr_df = run_experiment(exp, True)
        store_results(results_df, filename)
        store_unfairness_array(unf_arr_df, filename)
        return results_df,  unf_arr_df


def plot_results(df):
    fig = plt.figure()
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    plt.show()


def plot1_synthetic_singular():
    scale = 0.7
    fig = plt.figure(figsize=(21 * scale, 6 * scale))

    iterations = 350
    D = 25
    N = 300
    thetas = [0.0, 0.6, 0.8, 1.0]
    overwrite = False

    # ------
    plt.subplot(1, 3, 1)
    exp = Experiment(Synthetic("uniform", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df, unf_arr = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 2)
    exp = Experiment(Synthetic("linear", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df, unf_arr = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 3)
    exp = Experiment(Synthetic("exponential", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df, unf_arr = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    plt.tight_layout()
    plt.savefig("plots/plot1.png")

    plt.show()


def plot2_synthetic_geometric():
    scale = 0.7
    fig = plt.figure(figsize=(21 * scale, 6 * scale))

    iterations = 350
    D = 50
    N = 300
    thetas = [0.0, 0.6, 0.8, 1.0]
    overwrite = False
    k = 5
    p = 0.5

    # ------
    plt.subplot(1, 3, 1)
    exp = Experiment(Synthetic("uniform", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df, unf_arr = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 2)
    exp = Experiment(Synthetic("linear", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df, unf_arr = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 3)
    exp = Experiment(Synthetic("exponential", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df, unf_arr = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    plt.tight_layout()
    plt.savefig("plots/plot2.png")

    plt.show()


def create_distribution_over_subjects(pad=False):
    att_models = ["syn-exponential", "syn-linear", "syn-uniform"]
    functions = ["L1", "L2"]
    filenames = ["results" + "_" + str(t) + "_n=300_k=1_singular_" + str(f) for f in functions for t in att_models]

    header = ""
    if pad:
        header = "theta\tmean\t\tvar\t\t\t\t\std"
    else:
        header = "theta\tmean\t\tvar\t\tstd"

    for i, file_name in enumerate(filenames):

        #quadratic function L2
        df_arr_L2 = pd.read_csv("resultss_unfar/" + file_name + "_L2.csv")
        U_L2 = abs(df_arr_L2[:, 1] - df_arr_L2[:, 2])

        #original function L1
        df_arr_L1 = pd.read_csv("resultss_unfar/" + file_name + "_L1.csv")
        U_L1 = abs(df_arr_L1[:, 1] - df_arr_L2[:, 2])

        if i == 0:
            print(header)
        if not pad:
            print('{:.1f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'.format(np.mean(U_L1), np.mean(U_L2),
                                                                                  np.var(U_L1), np.var(U_L2),
                                                                                  np.std(U_L1), np.std(U_L2)))
        else:
            print('{:.1f}\t{:.3f}\t{:.3f}\t{:8.3f}\t{:8.3f}\t{:8.3f}\t{:8.3f}'.format(np.mean(U_L1), np.mean(U_L2),
                                                                                      np.var(U_L1), np.var(U_L2),
                                                                                      np.std(U_L1), np.std(U_L2)))










if __name__ == '__main__':
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    # Plot 0
    plot1_synthetic_singular()

    # Plot 1
    plot2_synthetic_geometric()

    # Create a table for new distribution of unfairness over m subjects
    #create_distribution_over_subjects()

    # Create a plot for new distribution of unfairness over m subjects

