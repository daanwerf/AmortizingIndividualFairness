import pandas as pd
import numpy as np

def createSyntheticData(distributionType, n):
    if distributionType == "uniform":
        print("handle distribution type uniform")
    elif distributionType == "linear":
        print("handle distribution type linear")
    elif distributionType == "exponential":
        print("handle distribution type exponential")
    else:
        raise ValueError("not the distribution type we support")

    return -1


def createAirbnbData(file, query):
    # do some preprocessing out of the csv files

    # handle the queries
    if query == "single":
        print("handle single query")
    elif query == "multi":
        print("handle multi query")
    else:
        raise ValueError("not valid query")

    # create array

    return -1


def createAttentionDistribution(shape, n):

    if shape == "singular":
        print("handle singular")
    elif shape == "geometric":
        print("handle geometric ")
    else:
        raise ValueError("not valid shape")

    return -1

