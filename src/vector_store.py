"""
Stage 3: Vector Store (from scratch)

A vector store's job is simple to state and easy to get wrong to implement
fast: given a new "query vector" (the embedding of a question), find the
K stored vectors (chunk embeddings) that are most similar to it.

This file implements the simplest correct version -- brute-force linear
search -- comparing the query against every stored vector one at a time.

Why start here instead of jumping straight to FAISS (Stage 4)?
  - It's the *ground truth*: brute force always finds the actual best
    matches, because it checks literally everything. Faster methods
    (like the HNSW index FAISS uses) are approximations that trade a
    little bit of accuracy for a lot of speed -- and you can only judge
    "how much accuracy did we trade away" if you have this exact version
    to compare against.
  - At small scale (a few thousand chunks -- typical for a portfolio
    project), brute force is plenty fast. The need for a smarter index
    only shows up at hundreds of thousands+ of vectors.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: int
    score: float
    text: str
    source_doc: str


class BruteForceVectorStore:
    def __init__(self):
        # We keep the raw chunk objects (for their text/metadata) and a
        # single matrix of their vectors, kept in the same order, so
        # row i of self.vectors corresponds to self.chunks[i].
        self.chunks = []
        self.vectors = None  # shape: (n_chunks, dim), filled in on add()

    def add(self, chunks: list, vectors: np.ndarray):
        """
        chunks: list of Chunk objects (from chunking.py)
        vectors: matrix of shape (len(chunks), dim) -- one embedding per
                 chunk, in the same order as `chunks`.
        """
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(chunks)} chunks but {vectors.shape[0]} vectors "
                f"-- these must line up 1-to-1."
            )

        # Pre-normalize vectors to unit length once, at insert time, so
        # search doesn't have to redo that work on every query. This is
        # the first real "make it faster without changing the answer"
        # optimization in this project.
        normed = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        if self.vectors is None:
            self.vectors = normed
        else:
            self.vectors = np.vstack([self.vectors, normed])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        """
        Find the top_k stored chunks most similar to query_vector.

        Because both the stored vectors and (below) the query vector are
        normalized to length 1, a plain dot product IS the cosine
        similarity -- no need to divide by lengths at search time. This
        is exactly why we normalized at insert time above: it turns
        "cosine similarity" into "one matrix multiply", which numpy does
        very fast.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        # (n_chunks, dim) dot (dim,) -> (n_chunks,) : one similarity
        # score per stored chunk, computed all at once.
        scores = self.vectors @ query_norm

        # argsort gives indices that would sort ascending; we want the
        # highest scores, so take the last top_k and reverse them.
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                score=float(scores[idx]),
                text=chunk.text,
                source_doc=chunk.source_doc,
            ))
        return results

    def __len__(self):
        return len(self.chunks)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from chunking import sentence_chunk

    # --- Fake embeddings for a runnable demo without needing the real
    # model or internet access. Each "topic" gets a distinct direction in
    # a small vector space, with a little noise, so similarity search has
    # something meaningful to find. Stage 2's real EmbeddingModel is a
    # drop-in replacement for this in actual use.
    def fake_embed(texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        topic_directions = {
            "animal": np.array([1.0, 0.0, 0.0, 0.0]),
            "finance": np.array([0.0, 1.0, 0.0, 0.0]),
            "food": np.array([0.0, 0.0, 1.0, 0.0]),
        }
        vectors = []
        for t in texts:
            lower = t.lower()
            if any(w in lower for w in ["cat", "dog", "feline", "animal"]):
                base = topic_directions["animal"]
            elif any(w in lower for w in ["stock", "market", "finance", "crash"]):
                base = topic_directions["finance"]
            else:
                base = topic_directions["food"]
            noise = rng.normal(0, 0.1, size=4)
            vectors.append(base + noise)
        return np.array(vectors)

    documents = [
        "The cat sat on the mat all afternoon.",
        "A dog barked at the mailman this morning.",
        "The stock market crashed sharply yesterday.",
        "Investors panicked as the finance sector fell.",
        "I made pasta with tomato sauce for dinner.",
        "The pizza had extra cheese and pepperoni.",
    ]

    chunks = []
    for i, doc in enumerate(documents):
        chunks.extend(sentence_chunk(doc, source_doc=f"doc_{i}.txt", max_chars=200))

    vectors = fake_embed([c.text for c in chunks])

    store = BruteForceVectorStore()
    store.add(chunks, vectors)
    print(f"Indexed {len(store)} chunks.\n")

    query = "Tell me about pets and animals"
    query_vec = fake_embed([query])[0]

    print(f"Query: {query!r}\n")
    results = store.search(query_vec, top_k=3)
    for r in results:
        print(f"  score={r.score:.3f}  [{r.source_doc}]  {r.text!r}")
