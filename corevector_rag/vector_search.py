from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProductQuantizer:
    num_subvectors: int
    codebook_size: int = 256
    max_iter: int = 25
    random_state: int | None = None

    def __post_init__(self) -> None:
        if self.num_subvectors <= 0:
            raise ValueError("num_subvectors must be positive")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        self.codebooks: np.ndarray | None = None
        self._subvector_dim: int | None = None
        self._rng = np.random.default_rng(self.random_state)

    def fit(self, vectors: np.ndarray) -> "ProductQuantizer":
        vectors = self._as_2d_float32(vectors)
        n_samples, dimension = vectors.shape
        if dimension % self.num_subvectors != 0:
            raise ValueError("Vector dimension must be divisible by num_subvectors")

        subvector_dim = dimension // self.num_subvectors
        codebooks = np.empty(
            (self.num_subvectors, self.codebook_size, subvector_dim), dtype=np.float32
        )

        for subvector_index in range(self.num_subvectors):
            start = subvector_index * subvector_dim
            end = start + subvector_dim
            subvectors = vectors[:, start:end]
            codebooks[subvector_index] = self._kmeans(subvectors)

        self._subvector_dim = subvector_dim
        self.codebooks = codebooks
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        vectors = self._as_2d_float32(vectors)
        expected_dim = self.num_subvectors * self._subvector_dim
        if vectors.shape[1] != expected_dim:
            raise ValueError(f"Expected vectors with dimension {expected_dim}")

        code_dtype = np.uint8 if self.codebook_size <= 256 else np.uint16
        codes = np.empty((vectors.shape[0], self.num_subvectors), dtype=code_dtype)

        for subvector_index in range(self.num_subvectors):
            start = subvector_index * self._subvector_dim
            end = start + self._subvector_dim
            subvectors = vectors[:, start:end]
            centroids = self.codebooks[subvector_index]
            distances = ((subvectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            codes[:, subvector_index] = np.argmin(distances, axis=1)

        return codes

    def distance_table(self, query_vector: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")

        expected_dim = self.num_subvectors * self._subvector_dim
        if query_vector.shape[0] != expected_dim:
            raise ValueError(f"Expected query vector with dimension {expected_dim}")

        table = np.empty((self.num_subvectors, self.codebook_size), dtype=np.float32)
        for subvector_index in range(self.num_subvectors):
            start = subvector_index * self._subvector_dim
            end = start + self._subvector_dim
            query_subvector = query_vector[start:end]
            centroids = self.codebooks[subvector_index]
            table[subvector_index] = ((centroids - query_subvector) ** 2).sum(axis=1)

        return table

    def _kmeans(self, subvectors: np.ndarray) -> np.ndarray:
        n_samples = subvectors.shape[0]
        if n_samples == 0:
            raise ValueError("Cannot train ProductQuantizer with empty vectors")

        replace = n_samples < self.codebook_size
        initial_indices = self._rng.choice(
            n_samples, size=self.codebook_size, replace=replace
        )
        centroids = subvectors[initial_indices].copy()

        for _ in range(self.max_iter):
            distances = ((subvectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            assignments = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for centroid_index in range(self.codebook_size):
                mask = assignments == centroid_index
                if np.any(mask):
                    new_centroids[centroid_index] = subvectors[mask].mean(axis=0)
                else:
                    random_index = self._rng.integers(0, n_samples)
                    new_centroids[centroid_index] = subvectors[random_index]

            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        return centroids.astype(np.float32, copy=False)

    @staticmethod
    def _as_2d_float32(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Expected a two-dimensional array of vectors")
        return vectors

    def _ensure_fitted(self) -> None:
        if self.codebooks is None or self._subvector_dim is None:
            raise RuntimeError("ProductQuantizer must be fitted before use")


class ADCVectorSearchEngine:
    def __init__(self, num_subvectors: int, codebook_size: int = 256, random_state: int | None = None):
        self.pq = ProductQuantizer(
            num_subvectors=num_subvectors,
            codebook_size=codebook_size,
            random_state=random_state,
        )
        self._codes: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    def fit(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> "ADCVectorSearchEngine":
        vectors = np.asarray(vectors, dtype=np.float32)
        self.pq.fit(vectors)
        self._codes = self.pq.encode(vectors)

        if ids is None:
            self._ids = np.arange(vectors.shape[0], dtype=np.int64)
        else:
            ids = np.asarray(ids)
            if ids.shape[0] != vectors.shape[0]:
                raise ValueError("ids length must match number of vectors")
            self._ids = ids
        return self

    def add(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> None:
        self._ensure_fitted()
        vectors = np.asarray(vectors, dtype=np.float32)
        new_codes = self.pq.encode(vectors)

        if ids is None:
            start = int(np.max(self._ids)) + 1 if self._ids.size else 0
            ids = np.arange(start, start + vectors.shape[0], dtype=np.int64)
        else:
            ids = np.asarray(ids)
            if ids.shape[0] != vectors.shape[0]:
                raise ValueError("ids length must match number of vectors")

        self._codes = np.concatenate([self._codes, new_codes], axis=0)
        self._ids = np.concatenate([self._ids, ids], axis=0)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_fitted()
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        table = self.pq.distance_table(query_vector)
        subvector_indexes = np.arange(self.pq.num_subvectors)[:, None]
        adc_distances = table[subvector_indexes, self._codes.T].sum(axis=0)

        result_count = min(top_k, adc_distances.shape[0])
        nearest_indices = np.argpartition(adc_distances, result_count - 1)[:result_count]
        ordered = nearest_indices[np.argsort(adc_distances[nearest_indices])]
        return self._ids[ordered], adc_distances[ordered]

    def _ensure_fitted(self) -> None:
        if self._codes is None or self._ids is None:
            raise RuntimeError("ADCVectorSearchEngine must be fitted before use")
