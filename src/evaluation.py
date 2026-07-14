"""
System Benchmark and Recall Evaluation.
"""
import sys
import numpy as np

def compute_exact_knn(queries: np.ndarray, db: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Standard L2 brute-force search for baseline comparison."""
    N_queries = queries.shape[0]
    exact_results = np.zeros((N_queries, top_k), dtype=int)
    
    for i in range(N_queries):
        # Brute force exact Euclidean distance
        diff = db - queries[i]
        dist = np.sum(diff ** 2, axis=1)
        exact_results[i] = np.argsort(dist)[:top_k]
        
    return exact_results

def evaluate_system(exact_ids: np.ndarray, approx_ids: np.ndarray, raw_db: np.ndarray, compressed_db: np.ndarray):
    """Calculates Memory Compression Ratio and Recall@K."""
    
    # 1. Memory Calculation
    raw_bytes = raw_db.nbytes
    compressed_bytes = compressed_db.nbytes
    compression_ratio = raw_bytes / compressed_bytes
    
    # 2. Recall Calculation (Intersection of approximate results vs exact results)
    N_queries = exact_ids.shape[0]
    recall_sum = 0.0
    
    for i in range(N_queries):
        exact_set = set(exact_ids[i])
        approx_set = set(approx_ids[i])
        recall_sum += len(exact_set.intersection(approx_set)) / len(exact_set)
        
    mean_recall = recall_sum / N_queries
    
    print("\n" + "="*50)
    print("      SYSTEM INFRASTRUCTURE BENCHMARKS")
    print("="*50)
    print(f"Original Database Size : {raw_bytes / (1024**2):.2f} MB (Float32)")
    print(f"Quantized Database Size: {compressed_bytes / (1024**2):.2f} MB (UInt8)")
    print(f"Memory Compression     : {compression_ratio:.1f}x Reduction")
    print("-" * 50)
    print(f"Search Recall@10       : {mean_recall * 100:.2f}%")
    print("="*50 + "\n")