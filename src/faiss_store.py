"""
Stage 4: ANN Indexing with FAISS

This file provides a FAISS-backed vector store that is a drop-in
replacement for Stage 3's BruteForceVectorStore -- same add()/search()
interface, so anything built on top of the vector store (retrieval,
generation) doesn't need to know or care which one is underneath.

We use FAISS's IndexHNSWFlat: an implementation of the HNSW algorithm
described in the conceptual explanation above. FAISS also offers a plain
"IndexFlatIP" which is just brute force again (useful for sanity-checking
FAISS itself against our own Stage 3 implementation -- they should agree).

Key parameters worth understanding (not just copying):
  - M: how many connections each vector keeps to its neighbors in the
    graph. Higher M -> better recall, more memory used, slower to build.
  - ef_construction: how hard the index searches for good neighbors
    *while building* the graph. Higher -> better quality index, slower
    to build (a one-time cost).
  - ef_search: how hard it searches *at query time*. Higher -> better
    recall, slower per-query. This is the one you'd tune live, since it
    doesn't require rebuilding the index.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: int
    score: float
    text: str
    source_doc: str


class FaissVectorStore:
    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200):
        import faiss
        self._faiss = faiss
        # HNSWFlat with inner-product metric. We normalize vectors to unit
        # length ourselves (same trick as Stage 3), which makes inner
        # product equivalent to cosine similarity.
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = 64  # default; can be changed before search()
        self.chunks = []
        self.dim = dim

    def add(self, chunks: list, vectors: np.ndarray):
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(chunks)} chunks but {vectors.shape[0]} vectors."
            )
        normed = (vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)).astype("float32")
        self.index.add(normed)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list["SearchResult"]:
        query_norm = (query_vector / (np.linalg.norm(query_vector) + 1e-8)).astype("float32")
        query_norm = query_norm.reshape(1, -1)

        scores, indices = self.index.search(query_norm, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 if fewer than top_k results exist
                continue
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                score=float(score),
                text=chunk.text,
                source_doc=chunk.source_doc,
            ))
        return results

    def __len__(self):
        return len(self.chunks)


def benchmark_vs_brute_force(n_vectors: int = 5000, dim: int = 64, n_queries: int = 20, top_k: int = 5):
    """
    Builds n_vectors random embeddings, indexes them both ways, and
    reports (a) how much faster FAISS/HNSW is per query, and (b) recall:
    of the top_k results HNSW returns, what fraction actually match the
    true top_k from brute force? This is the real question any ANN
    index needs answered before you trust it in a project -- speed alone
    doesn't tell you if you're sacrificing correctness too much.
    """
    import time
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from vector_store import BruteForceVectorStore, SearchResult as BFResult
    from chunking import Chunk

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(n_vectors, dim)).astype("float32")
    fake_chunks = [
        Chunk(text=f"chunk {i}", chunk_id=i, source_doc="synthetic", start_char=0, end_char=0)
        for i in range(n_vectors)
    ]

    bf_store = BruteForceVectorStore()
    bf_store.add(fake_chunks, vectors)

    faiss_store = FaissVectorStore(dim=dim)
    faiss_store.add(fake_chunks, vectors)

    queries = rng.normal(size=(n_queries, dim)).astype("float32")

    start = time.perf_counter()
    bf_results = [bf_store.search(q, top_k=top_k) for q in queries]
    bf_time = time.perf_counter() - start

    start = time.perf_counter()
    faiss_results = [faiss_store.search(q, top_k=top_k) for q in queries]
    faiss_time = time.perf_counter() - start

    overlaps = []
    for bf_res, fa_res in zip(bf_results, faiss_results):
        bf_ids = {r.chunk_id for r in bf_res}
        fa_ids = {r.chunk_id for r in fa_res}
        overlaps.append(len(bf_ids & fa_ids) / len(bf_ids))
    recall = sum(overlaps) / len(overlaps)

    print(f"Vectors indexed: {n_vectors}, dim: {dim}, queries: {n_queries}, top_k: {top_k}\n")
    print(f"Brute-force total time: {bf_time*1000:.1f} ms  ({bf_time/n_queries*1000:.3f} ms/query)")
    print(f"FAISS/HNSW total time:  {faiss_time*1000:.1f} ms  ({faiss_time/n_queries*1000:.3f} ms/query)")
    print(f"Speedup: {bf_time/faiss_time:.1f}x")
    print(f"Recall@{top_k} (agreement with brute-force ground truth): {recall*100:.1f}%")


if __name__ == "__main__":
    benchmark_vs_brute_force()
