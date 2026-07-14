"""
Data Ingestion and Dense Vector Simulation Layer.
"""
import logging
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from config.settings import VectorConfig

logger = logging.getLogger("CoreVector.Ingestion")

class VectorDataLoader:
    """Fetches real open-source data and transforms it into dense embeddings."""
    
    def __init__(self, config: VectorConfig):
        self.config = config

    def load_dense_embeddings(self):
        """Fetches MNIST and applies PCA to simulate continuous dense embedding spaces."""
        logger.info(f"Downloading OpenML dataset ID {self.config.OPENML_ID}...")
        X, _ = fetch_openml(data_id=self.config.OPENML_ID, return_X_y=True, as_frame=False, parser='auto')
        
        # Normalize pixel intensities
        X = X / 255.0
        
        logger.info(f"Applying PCA to project into {self.config.EMBEDDING_DIM}-D dense space...")
        pca = PCA(n_components=self.config.EMBEDDING_DIM, random_state=self.config.RANDOM_STATE)
        dense_vectors = pca.fit_transform(X)
        
        # Normalize to unit length (standard for Cosine/L2 embedding search)
        norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
        dense_vectors = dense_vectors / np.maximum(norms, 1e-9)
        
        # Split into Index (database) and Query sets
        X_index, X_query = train_test_split(
            dense_vectors, 
            train_size=self.config.N_INDEX, 
            test_size=self.config.N_QUERY, 
            random_state=self.config.RANDOM_STATE
        )
        
        logger.info(f"Database shape: {X_index.shape}. Query shape: {X_query.shape}.")
        return X_index.astype(np.float32), X_query.astype(np.float32)