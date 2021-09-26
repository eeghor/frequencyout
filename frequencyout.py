import pandas as pd
import numpy as np
import copy
from collections import Counter
from itertools import combinations
import time


class CategoricalHistogramBasedDetector:
    """
    Histogram-Based Outlier/Anomaly Detector
    """

    def __init__(self, score_type: str = None, combination_size: int = 2) -> None:

        self.score_type = score_type
        self.combination_size = combination_size

    def fit(self, X: pd.DataFrame = None):

        self.feature_names = sorted(X.columns)

        X = X.assign(
            **{
                "+".join(combination): X[combination[0]].astype(str)
                + "|"
                + X[combination[1]].astype(str)
                for combination in combinations(self.feature_names, self.combination_size)
            }
        )

        print(X.columns)
        print(X.head())

        return self


class SPAD:
    """
    Simple Probabilistic Anomaly Detector
    """

    @classmethod
    def get_name(cls):
        return cls.__name__

    def fit(self, X: pd.DataFrame = None, plus: bool = False):

        X_ = copy.deepcopy(X)

        for pair in combinations(X.columns, 2):
            X_["+".join(pair)] = X[pair[0]].astype(str) + "|" + X[pair[1]].astype(str)

        print(X_.head())

        self.counts = {c: Counter(X_[c]) for c in X_.columns}
        self.n = len(X_)

        return self

    def predict(self, X):

        scores = np.zeros(shape=(len(X),))

        for c in X.columns:
            scores += np.log(
                X[c]
                .apply(
                    lambda x: (self.counts[c].get(x, 0) + 1)
                    / (self.n + len(self.counts[c]))
                )
                .values
            )

        return pd.Series(
            data=scores, name=f"{self.get_name().lower()}_scores", index=X.index
        )


class HBOS:
    """
    Histogram-Based Outlier Score
    performs well on global anomaly detection problems but cannot detect local outliers.

    returns a series of scores; the larger the score the more an outlier
    """

    @classmethod
    def get_name(cls):
        return cls.__name__

    def fit(self, X: pd.DataFrame = None):

        self.counts = {c: Counter(X[c]) for c in X.columns}
        self.n = len(X)

        return self

    def predict(self, X):

        scores = np.zeros(shape=(len(X),))

        for c in X.columns:

            max_count_in_c = max(self.counts[c].values())

            scores += np.log(
                X[c]
                .apply(lambda x: max_count_in_c / self.counts[c].get(x, 1e-10))
                .values
            )

        return pd.Series(
            data=scores, name=f"{self.get_name().lower()}_scores", index=X.index
        )


if __name__ == "__main__":

    categorical_columns = [
        "entry_key",
        "user_key",
        "location_id",
        "department_id",
        "account_name",
    ]

    data = pd.read_csv("data/d_.csv.gz", usecols=categorical_columns, dtype=str)

    n_train = 1000000
    n_test = 100000

    X_train = data.sample(n_train)
    X_test = data.sample(n_test)

    t_start = time.time()

    spad = SPAD()
    spad.fit(X_train)
    y_pred = spad.predict(X_test)
    print(y_pred)

    print("SPAD")
    print(f"trained on {n_train:,} samples")
    print(f"predictions on {n_test:,} samples")
    print(f"elapsed time: {time.time() - t_start: .4f} sec")

    t_start = time.time()

    hbos = HBOS()
    hbos.fit(X_train)
    y_pred = hbos.predict(X_test)
    print(y_pred)

    print("HBOS")
    print(f"trained on {n_train:,} samples")
    print(f"predictions on {n_test:,} samples")
    print(f"elapsed time: {time.time() - t_start: .4f} sec")

    chbd = CategoricalHistogramBasedDetector()
    print(chbd.fit(X_train))
