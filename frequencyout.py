import pandas as pd
import numpy as np
from collections import defaultdict, Counter


class SPAD:
    """
    Simple Probabilistic Method (SPAD)
    """

    def fit(self, X: pd.DataFrame = None):

        self.count_table = {c: Counter(X[c]) for c in X.columns}
        self.n = len(X)

        return self

    def predict(self, X):

        spad_scores = np.zeros(shape=(1,))

        for c in X.columns:
            spad_scores = spad_scores + np.log(
                X[c]
                .apply(
                    lambda x: (self.count_table[c].get(x, 0) + 1)
                    / (self.n + len(self.count_table[c]))
                )
                .values
            )

        return X.assign(spad_score=spad_scores).sort_values("spad_score")


if __name__ == "__main__":

    categorical_columns = [
        "entry_key",
        "user_key",
        "location_id",
        "department_id",
        "account_name",
    ]
    d = pd.read_csv("data/d_.csv.gz")[categorical_columns]

    spad = SPAD()

    spad.fit(X=d.iloc[:1000])

    print(spad.predict(d.sample(400)))
