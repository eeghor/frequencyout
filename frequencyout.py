import pandas as pd

categorical_columns = [
    "entry_key",
    "user_key",
    "location_id",
    "department_id",
    "account_name",
]
d = pd.read_csv("data/d_.csv.gz")


class SPAD:
    """
    Simple Probabilistic Method (SPAD)
    """

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def predict(self, X):
        return X
