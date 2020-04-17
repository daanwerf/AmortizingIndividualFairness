import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import loop
from core import attention_geometric, attention_model_singular
from datasets import Synthetic, AirBNBSingleQuery
from loop import Experiment, run_experiment, get_experiment_filename, store_results


def load_or_run(exp, filename, overwrite=False):
    if os.path.exists("results/" + filename) and not overwrite:
        return pd.read_csv("results/" + filename)
    else:
        df = run_experiment(exp)
        store_results(df, filename)
        return df


def plot_results(df):
    fig = plt.figure()
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='theta', legend="full")
    g.set(xlabel="iterations")

    plt.show()


def plot1_synthetic_singular(overwrite=True):
    scale = 0.7
    fig = plt.figure(figsize=(21 * scale, 6 * scale))

    iterations = 350
    D = 50
    N = 300
    thetas = [0.0, 0.6, 0.8, 1.0]

    # ------
    plt.subplot(1, 3, 1)
    exp = Experiment(Synthetic("uniform", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 2)
    exp = Experiment(Synthetic("linear", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 3)
    exp = Experiment(Synthetic("exponential", n=N), 1, attention_model_singular(), thetas, iterations, D)

    filename = get_experiment_filename(exp, "singular")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
    g.set(xlabel="iterations")

    plt.tight_layout()
    plt.savefig("plots/plot1.png")

    plt.show()


def plot2_synthetic_geometric(overwrite=True):
    scale = 0.7
    fig = plt.figure(figsize=(21 * scale, 6 * scale))

    iterations = 350
    D = 50
    N = 300
    thetas = [0.0, 0.6, 0.8, 1.0]
    k = 5
    p = 0.5

    # ------
    plt.subplot(1, 3, 1)
    exp = Experiment(Synthetic("uniform", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 2)
    exp = Experiment(Synthetic("linear", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
    g.set(xlabel="iterations")

    # ------
    plt.subplot(1, 3, 3)
    exp = Experiment(Synthetic("exponential", n=N), k, attention_geometric(k, p), thetas, iterations, D)

    filename = get_experiment_filename(exp, "geometric")
    df = load_or_run(exp, filename, overwrite=overwrite)
    g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
    g.set(xlabel="iterations")

    plt.tight_layout()
    plt.savefig("plots/plot2.png")

    plt.show()


def plot3_prob(overwrite=True):
    scale = 0.7
    fig = plt.figure(figsize=(21 * scale, 12 * scale))

    iterations = 350
    N = 100
    k = 5

    attention_models = {
        'attention-singular': (attention_model_singular(), 1, 0.8, 1),  # model, k, rate, swaps
        f'attention-geometric@{k}': (attention_geometric(k, 0.5), k, 0.6, 3)
    }

    # ------
    for i, ds in enumerate(["uniform", "linear", "exponential"]):
        for j, w_name in enumerate(attention_models.keys()):
            w, k, rate, swaps = attention_models[w_name]
            plt.subplot(2, 3, j * 3 + i + 1)
            filename = f"results/results_{ds}_prob_k={k}_geometric_{w_name}.csv"
            data = Synthetic(ds, n=N)
            if os.path.exists(filename) and not overwrite:
                df = pd.read_csv(filename)
            else:
                df = loop.run_model_prob(data.relevance, k, w, iterations=iterations, swaps=swaps, rate=rate, D=25)
                df = pd.concat([df, loop.relevance_model(data.relevance, w, iterations)], ignore_index=True)
                df.to_csv(filename)

            mean_ndcg = df[df['model'] == 'prob']['ndcg'].mean()
            std_ndcg = df[df['model'] == 'prob']['ndcg'].std()

            g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
            g.set(xlabel="Iterations", ylabel="Unfairness |A-R|",
                  title=f"{ds}(n={N}) + {w_name}\n ndcg@{k}(μ, σ)=({mean_ndcg:.2f},{std_ndcg:.2f})")

    plt.tight_layout()
    plt.savefig("plots/plot_prob_extension.png")
    plt.show()


def plot3_prob_airbnb(overwrite=True):
    scale = 0.7
    fig = plt.figure(figsize=(21 * scale, 12 * scale))

    iterations = 10000
    k = 10
    p = 0.5

    attention_models = {
        'attention-singular': (attention_model_singular(), 1, 0.8, 1),  # model, k, rate, swaps
        f'attention-geometric@{k}': (attention_geometric(k, p), k, 0.6, 4)
    }

    # ------
    for i, ds in enumerate(["data/Boston.csv", "data/Geneva.csv", "data/HongKong.csv"]):
        for j, w_name in enumerate(attention_models.keys()):
            ds_name = ds.replace('data/', 'AirBnB ').replace(".csv", "")
            w, k, rate, swaps = attention_models[w_name]
            plt.subplot(2, 3, j * 3 + i + 1)
            filename = f"results/results_{ds_name}_prob_k={k}_geometric_{w_name}.csv"
            data = AirBNBSingleQuery(ds)
            if os.path.exists(filename) and not overwrite:
                df = pd.read_csv(filename)
            else:
                df = loop.run_model_prob(data.relevance, k, w, iterations=iterations, swaps=swaps, rate=rate, D=25)
                df = pd.concat([df, loop.relevance_model(data.relevance, w, iterations)], ignore_index=True)
                df.to_csv(filename)

            mean_ndcg = df[df['model'] == 'prob']['ndcg'].mean()
            std_ndcg = df[df['model'] == 'prob']['ndcg'].std()

            g = sns.lineplot(data=df, x='it', y='unfairness', hue='model', legend="full")
            g.set(xlabel="Iterations", ylabel="Unfairness |A-R|",
                  title=f"{ds_name}(n={len(data.relevance)}) + {w_name}\n ndcg@{k}(μ, σ)=({mean_ndcg:.2f},{std_ndcg:.2f})")

    plt.tight_layout()
    plt.savefig("plots/plot_prob_extension_airnbnb.png")
    plt.show()


if __name__ == '__main__':
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # # Plot 1
    plot1_synthetic_singular(overwrite=True)

    # Plot 2
    plot2_synthetic_geometric(overwrite=True)

    # # Plot 3
    plot3_prob(overwrite=True)

    # plot 4
    plot3_prob_airbnb(overwrite=True)
