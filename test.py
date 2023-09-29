import unittest
import pandas as pd  # type: ignore
from collections import Counter
from hbod import CategoricalHistogramBasedDetector


class testSomething(unittest.TestCase):
    def test_spad(self):
        X_train = pd.DataFrame(
            {
                "v1": ["a", "a", "b", "a", "a", "c", "b", "b", "c"],
                "v2": ["c", "a", "c", "c", "a", "b", "c", "c", "a"],
            }
        )

        X_test = pd.DataFrame(
            {
                "v1": ["a", "a", "b", "c", "a", "b", "c", "c"],
                "v2": ["b", "a", "c", "b", "c", "b", "c", "a"],
            }
        )

        chbd = CategoricalHistogramBasedDetector(score_type="spad", combination_size=2)
        chbd.fit(X_train)

        X_test_more_to_less_unusual = X_test.assign(
            scores=chbd.score(X_test)
        ).sort_values("scores")

        results = [row for row in X_test_more_to_less_unusual.itertuples(index=False)]

        self.assertListEqual(
            [
                (results[0].v1, results[0].v2),
                (results[1].v1, results[1].v2),
                (results[-1].v1, results[-1].v2),
            ],
            [("b", "b"), ("a", "b"), ("b", "c")],
        )

    def test_hbos(self):
        X_train = pd.DataFrame(
            {
                "v1": ["a", "a", "b", "a", "a", "c", "b", "b", "c"],
                "v2": ["c", "a", "c", "c", "a", "b", "c", "c", "a"],
            }
        )

        X_test = pd.DataFrame(
            {
                "v1": ["a", "a", "b", "c", "a", "b", "c", "c"],
                "v2": ["b", "a", "c", "b", "c", "b", "c", "a"],
            }
        )

        chbd = CategoricalHistogramBasedDetector(score_type="hbos", combination_size=2)
        chbd.fit(X_train)

        X_test_more_to_less_unusual = X_test.assign(
            scores=chbd.score(X_test)
        ).sort_values("scores", ascending=False)

        results = [row for row in X_test_more_to_less_unusual.itertuples(index=False)]

        self.assertListEqual(
            [
                (results[0].v1, results[0].v2),
                (results[1].v1, results[1].v2),
                (results[-1].v1, results[-1].v2),
            ],
            [("b", "b"), ("a", "b"), ("b", "c")],
        )


if __name__ == "__main__":
    unittest.main()
