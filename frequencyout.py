import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import time


class SPAD:
    """
    Simple Probabilistic Method (SPAD)
    """

    def fit(self, X: pd.DataFrame = None):

        self.counts = {c: Counter(X[c]) for c in X.columns}
        self.n = len(X)

        return self

    def predict(self, X):

        spad_scores = np.zeros(shape=(len(X),))

        for c in X.columns:
            spad_scores += np.log(
                X[c]
                .apply(
                    lambda x: (self.counts[c].get(x, 0) + 1)
                    / (self.n + len(self.counts[c]))
                )
                .values
            )

        return pd.Series(data=spad_scores, name="spad_scores", index=X.index)


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

    print(f"trained on {n_train:,} samples")
    print(f"predictions on {n_test:,} samples")
    print(f"elapsed time: {time.time() - t_start: .4f} sec")
