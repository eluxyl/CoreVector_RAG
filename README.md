# CoreVector: Custom Product Quantization Search Engine

A low-latency, highly compressed vector search index built from scratch in pure NumPy. This project demonstrates low-level systems engineering for massive RAG applications by bypassing off-the-shelf vector databases and implementing memory-optimized vector algebra natively.

##  Infrastructure Mathematics
Instead of storing full-precision vectors $x \in \mathbb{R}^D$, **Product Quantization** partitions the vector into $M$ sub-vectors. For each sub-space, a K-Means algorithm generates a codebook of centroids. 

The original dense array of 32-bit floats is translated into an array of $M$ 8-bit integers, where each integer points to the nearest sub-centroid.

During querying, **Asymmetric Distance Computation (ADC)** avoids decompression entirely. The distance $d(q, x)$ between a query $q$ and a compressed vector $x$ is approximated in $O(M)$ time using a pre-computed distance lookup table:
$$d(q, x) \approx \sum_{m=1}^{M} \| q_m - C_{m}(x_m) \|^2$$

##  System Benchmarks
- **Vector Transformation:** Compressed sparse 784-D vectors into dense continuous 256-D space.
- **Memory Compression:** 64.0x infrastructure cost reduction (Float32 $\rightarrow$ UInt8 code blocks).
- **Accuracy Guarantee:** Maintained high Recall@10 while completely decoupling search time from raw storage size.