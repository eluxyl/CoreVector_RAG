"""
Product Quantization Engine for Vector Compression and Fast Search.
"""
import logging
import numpy as np
from sklearn.cluster import KMeans
from config.settings import VectorConfig

logger = logging.getLogger("CoreVector.Quantizer")

class ProductQuantizer:
    """Compresses high-dimensional vectors and performs Asymmetric Distance Computation."""
    
    def __init__(self, config: VectorConfig):
        self.M = config.M_SUBVECTORS
        self.K = config.K_CENTROIDS
        self.sub_dim = config.EMBEDDING_DIM // self.M
        self.codebooks = np.zeros((self.M, self.K, self.sub_dim), dtype=np.float32)
        
    def fit(self, X: np.ndarray):
        """Trains K-Means codebooks for each sub-vector space."""
        logger.info(f"Training Product Quantizer (M={self.M}, K={self.K})...")
        for m in range(self.M):
            # Extract the m-th sub-vector slice across all training data
            sub_vectors = X[:, m * self.sub_dim : (m + 1) * self.sub_dim]
            
            kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=3, max_iter=100)
            kmeans.fit(sub_vectors)
            self.codebooks[m] = kmeans.cluster_centers_
            
            if m % 4 == 0 or m == self.M - 1:
                logger.info(f"  -> Trained subspace {m+1}/{self.M}")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Compresses float32 vectors into uint8 byte codes."""
        N = X.shape[0]
        codes = np.zeros((N, self.M), dtype=np.uint8)
        
        for m in range(self.M):
            sub_vectors = X[:, m * self.sub_dim : (m + 1) * self.sub_dim]
            # Compute L2 distance from sub-vectors to codebook centroids
            diff = sub_vectors[:, np.newaxis, :] - self.codebooks[m][np.newaxis, :, :]
            dist = np.sum(diff ** 2, axis=2)
            codes[:, m] = np.argmin(dist, axis=1).astype(np.uint8)
            
        return codes

    def search_adc(self, queries: np.ndarray, compressed_db: np.ndarray, top_k: int = 10):
        """
        Asymmetric Distance Computation (ADC).
        Computes distances using a lookup table instead of vector math.
        """
        N_queries = queries.shape[0]
        N_db = compressed_db.shape[0]
        results = np.zeros((N_queries, top_k), dtype=int)
        
        for q_idx in range(N_queries):
            query = queries[q_idx]
            
            # 1. Build the lookup table for this specific query
            # Shape: (M, K) - Distance from each query sub-vector to all codebook centroids
            lookup_table = np.zeros((self.M, self.K), dtype=np.float32)
            for m in range(self.M):
                q_sub = query[m * self.sub_dim : (m + 1) * self.sub_dim]
                diff = q_sub[np.newaxis, :] - self.codebooks[m]
                lookup_table[m, :] = np.sum(diff ** 2, axis=1)
            
            # 2. Compute approximate distance to ALL database vectors using pure array indexing
            # For each vector, we just sum the pre-calculated distances from the table
            m_indices = np.arange(self.M)
            approx_distances = np.sum(lookup_table[m_indices, compressed_db], axis=1)
            
            # 3. Retrieve Top-K nearest neighbors
            results[q_idx] = np.argsort(approx_distances)[:top_k]
            
        return results