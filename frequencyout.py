import pandas as pd  # type:ignore
import numpy as np  # type:ignore
from collections import Counter
from itertools import combinations
import time
from typing import Literal, Dict


class CategoricalHistogramBasedDetector:
    def __init__(
        self, score_type: Literal["spad", "hbos"] = None, combination_size: int = 2
    ) -> None:

        self.score_type = score_type
        self.combination_size = combination_size

    def __add_combined_features(self, X: pd.DataFrame) -> pd.DataFrame:

        """
        Parameters
        ----------
        X : data frame where every column is a categorical feature

        Returns
        -------
        original data frame with new columns with combinations
        of categorical features
        """

        return X.assign(
            **{
                "+".join(combination): X[combination[0]].astype(str)
                + "|"
                + X[combination[1]].astype(str)
                for combination in combinations(
                    self.feature_names, self.combination_size
                )
            }
        )

    def __get_spad_scores(self, X: pd.Series) -> pd.Series:

        """
        Parameters
        ----------
        X : series corresponding to a categorical feature

        Returns
        -------
        series where every element is a SPAD score
        """

        return np.log(
            X.apply(
                lambda _: (self.counts[X.name].get(_, 0) + 1)
                / (len(X.index) + len(self.counts[X.name]))
            )
        )

    def __get_hbos_scores(self, X: pd.Series) -> pd.Series:

        """
        Parameters
        ----------
        X : series corresponding to a categorical feature

        Returns
        -------
        series where every element is a HBOS score
        """

        max_count = max(self.counts[X.name].values())

        return np.log(X.apply(lambda _: max_count / self.counts[X.name].get(_, 1e-10)))

    def fit(self, X: pd.DataFrame) -> "CategoricalHistogramBasedDetector":

        """
        Parameters
        ----------
        X : data frame where every column is a categorical feature

        Returns
        -------
        self
        """

        self.feature_names = sorted(X.columns)

        # create and attach new combined features if required
        if self.combination_size > 1:
            X = self.__add_combined_features(X)

        # count how many times each possible value of each feature occurs
        self.counts: Dict = {c: Counter(X[c]) for c in X.columns}

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:

        """
        Parameters
        ----------
        X : data frame where every column is a categorical feature

        Returns
        -------
        where every element is a score for the corresponding sample
        """

        if self.combination_size > 1:
            X = self.__add_combined_features(X)

        scores = np.zeros(shape=(len(X.index),))

        if self.score_type == "spad":
            for c in X.columns:
                scores += self.__get_spad_scores(X[c])

        elif self.score_type == "hbos":
            for c in X.columns:
                scores += self.__get_hbos_scores(X[c])

        return pd.Series(data=scores, name=f"{self.score_type}_scores", index=X.index)


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
    chbd = CategoricalHistogramBasedDetector(score_type="hbos", combination_size=1)
    chbd.fit(X_train)
    y_pred = chbd.predict(X_test)

    print(y_pred)
    print(f"trained on {n_train:,} samples")
    print(f"predictions on {n_test:,} samples")
    print(f"elapsed time: {time.time() - t_start: .4f} sec")

    t_start = time.time()
    chbd = CategoricalHistogramBasedDetector(score_type="spad", combination_size=1)
    chbd.fit(X_train)
    y_pred = chbd.predict(X_test)

    print(y_pred)
    print(f"trained on {n_train:,} samples")
    print(f"predictions on {n_test:,} samples")
    print(f"elapsed time: {time.time() - t_start: .4f} sec")
