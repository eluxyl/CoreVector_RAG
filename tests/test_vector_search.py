import unittest

import numpy as np

from corevector_rag import ADCVectorSearchEngine, ProductQuantizer


class ProductQuantizerTests(unittest.TestCase):
    def test_fit_and_encode_outputs_expected_shapes(self) -> None:
        vectors = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.1, 0.1],
                [9.8, 9.9, 10.0, 10.1],
                [10.1, 10.0, 9.9, 10.0],
            ],
            dtype=np.float32,
        )
        pq = ProductQuantizer(num_subvectors=2, codebook_size=2, random_state=7)
        pq.fit(vectors)
        codes = pq.encode(vectors)

        self.assertEqual(codes.shape, (4, 2))
        self.assertEqual(pq.codebooks.shape, (2, 2, 2))
        self.assertEqual(codes.dtype, np.uint8)


class ADCVectorSearchEngineTests(unittest.TestCase):
    def test_search_returns_nearest_cluster_first(self) -> None:
        vectors = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.2, 0.1],
                [10.0, 10.0, 10.0, 10.0],
                [9.9, 10.1, 9.8, 10.2],
            ],
            dtype=np.float32,
        )
        ids = np.array([101, 102, 201, 202], dtype=np.int64)
        engine = ADCVectorSearchEngine(num_subvectors=2, codebook_size=2, random_state=3)
        engine.fit(vectors, ids=ids)

        nearest_ids, distances = engine.search(np.array([0.05, 0.02, 0.01, 0.03], dtype=np.float32), top_k=2)

        self.assertEqual(nearest_ids.shape, (2,))
        self.assertTrue(set(nearest_ids.tolist()).issubset({101, 102}))
        self.assertTrue(np.all(np.diff(distances) >= 0))

    def test_add_indexes_new_vectors(self) -> None:
        base = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [8.0, 8.0, 8.0, 8.0],
                [9.0, 9.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        engine = ADCVectorSearchEngine(num_subvectors=2, codebook_size=2, random_state=1)
        engine.fit(base, ids=np.array([1, 2, 3]))

        engine.add(np.array([[0.05, 0.02, 0.03, 0.01]], dtype=np.float32), ids=np.array([99]))
        nearest_ids, _ = engine.search(np.array([0.04, 0.01, 0.01, 0.02], dtype=np.float32), top_k=2)

        self.assertIn(99, nearest_ids.tolist())


if __name__ == "__main__":
    unittest.main()
