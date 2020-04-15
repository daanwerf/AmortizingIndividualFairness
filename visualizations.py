import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core import attention_geometric
from datasets import Synthetic
from loop import Experiment, attention_model_singular, run_experiment, get_experiment_filename, store_results


def load_or_run(exp, filename, overwrite=True):
    if os.path.exists("results/" + filename) and not overwrite:
        return pd.read_csv("results/" + filename)
    else:
        df = run_experiment(exp, True)
        store_results(df, filename)
        return df


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
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 2)
    exp = Experiment(Synthetic("linear", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 3)
    exp = Experiment(Synthetic("exponential", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df = load_or_run(exp, filename, overwrite=overwrite)
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
    overwrite = True
    k = 5
    p = 0.5

    # ------
    plt.subplot(1, 3, 1)
    exp = Experiment(Synthetic("uniform", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 2)
    exp = Experiment(Synthetic("linear", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 3)
    exp = Experiment(Synthetic("exponential", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    plt.tight_layout()
    plt.savefig("plots/plot2.png")

    plt.show()


if __name__ == '__main__':
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Plot 0
    plot1_synthetic_singular()

    # Plot 1
    plot2_synthetic_geometric()
