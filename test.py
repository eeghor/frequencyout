import unittest

class testSomething(unittest.TestCase):
    def test_spad(self):

        chbd = CategoricalHistogramBasedDetector(score_type='spad', combination_size=2)
        chbd.fit(X_train)

        self.assertTrue(
            CatFeeder()._is_valid_id(some_id="a4d2cc9d-367a-4ec1-882a-618f34195aa1")
        )

if __name__ == "__main__":
    unittest.main()