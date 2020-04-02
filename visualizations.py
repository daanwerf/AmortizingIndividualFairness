import matplotlib.pyplot as plt
import seaborn as sns

from synthetic import SingleQueryDataset, AirBNBSingleQuery


def plot_single_query_dataset(dataset: SingleQueryDataset):
    fig = plt.figure()
    g = sns.distplot(dataset.relevance, norm_hist=True)
    g.set(title=f"n={dataset.relevance.size}")
    plt.show()


if __name__ == '__main__':
    dataset = AirBNBSingleQuery('datasets/Geneva.csv')
    plot_single_query_dataset(dataset)
