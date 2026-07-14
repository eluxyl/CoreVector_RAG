"""
Configuration settings for the CoreVector Engine.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class VectorConfig:
    # Data ingestion parameters
    OPENML_ID: int = 554  # MNIST_784 dataset ID
    EMBEDDING_DIM: int = 256  # Simulate dense embeddings via PCA
    
    # Dataset splits
    N_INDEX: int = 20000  # Number of vectors in our database
    N_QUERY: int = 100    # Number of queries to test search latency/accuracy
    RANDOM_STATE: int = 42
    
    # Product Quantization (PQ) Parameters
    # We split 256 dimensions into 16 sub-vectors of 16 dimensions each.
    M_SUBVECTORS: int = 64 
    # We use 256 clusters per sub-vector so the ID fits perfectly in a uint8 byte.
    K_CENTROIDS: int = 256
