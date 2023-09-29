import pandas as pd  # type:ignore
import numpy as np  # type:ignore
from collections import Counter
from itertools import combinations
from typing import Literal, Dict


class CategoricalHistogramBasedDetector:

    """
    This anomaly/outlier detector implements two anomaly/outlier methods for
    samples with categorical features.

     - SPAD: Simple Probabilistic Anomaly Detector proposed in
             Aryal et al, 2016: Revisiting Attribute Independence Assumption in
                                Probabilistic Unsupervised Anomaly Detection
             note: lower score indicate more unusual

     - HBOS: Histogram-Based Outlier Score as described in
             Goldstein and Dengel, 2012: Histogram-based Outlier Score (HBOS):
                                         A Fast Unsupervised Anomaly Detection Algorithm
             note: higher scores mean more unusual
    """

    def __init__(
        self, score_type: Literal["spad", "hbos"] = None, combination_size: int = 2
    ) -> None:
        """
        Parameters
        ----------
        score_type : type of scores to use; spad or hbos
        combination_size : number of original features to include in each combined feature;
                           default: 2, i.e. create additional combined feature
                           from original feature pairs

        """

        self.score_type = score_type
        self.combination_size = combination_size

    def __add_combined_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : data frame where every column is a categorical feature

        Returns
        -------
        original data frame with new columns created from combinations
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

        """

        self.feature_names = sorted(X.columns)

        # create and attach new combined features if required
        if self.combination_size > 1:
            X = self.__add_combined_features(X)

        # count how many times each possible value of each feature occurs
        self.counts: Dict = {c: Counter(X[c]) for c in X.columns}

        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        """
        Parameters
        ----------
        X : data frame where every column is a categorical feature

        Returns
        -------
        a series indexed like the supplied data frame and named like {spad, hbos}_scores
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
