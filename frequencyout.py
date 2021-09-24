import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import time


class SPAD:
    """
    Simple Probabilistic Anomaly Detector
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
