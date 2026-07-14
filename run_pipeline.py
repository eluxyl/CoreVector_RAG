"""
CoreVector Engine Execution Orchestrator.
"""
import os
import sys
import logging
import time

# System Path injection for direct script execution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import VectorConfig
from src.ingestion import VectorDataLoader
from src.quantizer import ProductQuantizer
from src.evaluation import compute_exact_knn, evaluate_system

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("CoreVector")

def main():
    config = VectorConfig()
    
    # 1. Ingestion
    loader = VectorDataLoader(config)
    X_index, X_query = loader.load_dense_embeddings()
    
    # 2. Quantization (Compression)
    pq = ProductQuantizer(config)
    pq.fit(X_index)
    
    logger.info("Compressing the entire vector database...")
    t0 = time.time()
    compressed_db = pq.encode(X_index)
    logger.info(f"Database compressed in {time.time() - t0:.2f} seconds.")
    
    # 3. Search & Benchmarking
    logger.info("Running Exact Brute-Force Search for Ground Truth...")
    exact_results = compute_exact_knn(X_query, X_index, top_k=10)
    
    logger.info("Running Accelerated Asymmetric Distance Search (ADC)...")
    approx_results = pq.search_adc(X_query, compressed_db, top_k=10)
    
    # 4. Evaluation Outputs
    evaluate_system(exact_results, approx_results, X_index, compressed_db)

if __name__ == "__main__":
    main()