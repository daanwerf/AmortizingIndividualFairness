import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from datasets import SingleQueryDataset, Synthetic


def plot_single_query_dataset(dataset: SingleQueryDataset):
    fig = plt.figure()
    g = sns.distplot(dataset.relevance, norm_hist=True, bins=50)
    g.set(title=f"n={dataset.relevance.size}")
    plt.show()


def plot_results(filename):
    data = pd.read_csv(filename)
    g = sns.lineplot(data=data, x="it", y="unfairness")
    plt.show()


if __name__ == '__main__':
    os.makedirs("plots", exist_ok=True)
    # dataset = AirBNBSingleQuery('data/Geneva.csv')
    dataset = Synthetic('exponential', n=1000000)
    plot_single_query_dataset(dataset)
    # plot_results("results/results.csv")
