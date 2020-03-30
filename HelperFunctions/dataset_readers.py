import pandas as pd


def get_dataset_reader(city_name):
    if city_name != "Boston" and city_name != "Geneva" and city_name != "HongKong":
        print("Invalid city name")
    return pd.read_csv("../Datasets/" + city_name + ".csv")


def get_review_scores_rating(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_rating']

    return res


def get_review_scores_accuracy(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_accuracy']

    return res


def get_review_scores_cleanliness(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_cleanliness']

    return res


def get_review_scores_checkin(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_checkin']

    return res


def get_review_scores_communication(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_communication']

    return res


def get_review_scores_location(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_location']

    return res


def get_review_scores_value(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = row['review_scores_value']

    return res


def get_review_all(city_name):
    df = get_dataset_reader(city_name)
    res = {}

    for i, row in df.iterrows():
        res[int(row["id"])] = [row['review_scores_rating'], row['review_scores_accuracy'],
                               row['review_scores_cleanliness'], row['review_scores_checkin'],
                               row['review_scores_communication'], row['review_scores_location'],
                               row['review_scores_value']]

    return res


