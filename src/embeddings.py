"""
Stage 2: Embeddings

An embedding model turns a piece of text into a fixed-length list of numbers
(a "vector") such that texts with similar *meaning* end up as vectors that
are close together in that number-space, and texts with different meaning
end up far apart.

Concretely: "The cat sat on the mat" and "A feline rested on the rug" will
produce vectors that are close together, even though they share almost no
words -- because the model was trained on meaning, not spelling.

We use `sentence-transformers` here, a library built on top of models
specifically trained to produce good sentence-level embeddings (as opposed
to raw word embeddings, which don't capture whole-sentence meaning well).

The model we default to, "all-MiniLM-L6-v2", is:
  - small (~90MB) and fast enough to run on a laptop CPU, no GPU needed
  - produces 384-dimensional vectors (i.e. each chunk becomes a list of
    384 numbers)
  - a solid, widely-used baseline -- not the best possible model, but a
    good one to learn on before swapping in something larger later
"""

import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Imported lazily so that chunking.py (Stage 1) never needs this
        # heavy dependency installed just to run its own demo.
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{model_name}' "
              f"(first run downloads it, then it's cached locally)...")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Turn a list of strings into a matrix of shape (n_texts, dim).
        Each row is one text's embedding vector.
        """
        if not texts:
            return np.zeros((0, self.dim))
        return self.model.encode(texts, convert_to_numpy=True,
                                  show_progress_bar=False)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for embedding a single string."""
        return self.embed([text])[0]


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    The standard way to measure "how similar are these two vectors"?
    It's the cosine of the angle between them:
      - 1.0  -> pointing in exactly the same direction (very similar meaning)
      - 0.0  -> perpendicular (unrelated)
      - -1.0 -> pointing in opposite directions (rare in practice for
                sentence embeddings, but possible)

    We divide by each vector's length (norm) so that *direction* is all
    that matters, not magnitude -- two vectors pointing the same way but
    of different length should still count as maximally similar.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


if __name__ == "__main__":
    model = EmbeddingModel()

    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed yesterday.",
        "Wall Street saw a huge drop today.",
        "I love eating pizza on weekends.",
    ]

    vectors = model.embed(sentences)
    print(f"\nEmbedded {len(sentences)} sentences into vectors of "
          f"shape {vectors.shape} (i.e. {vectors.shape[1]} numbers each)\n")

    print("Similarity matrix (1.0 = identical meaning, 0.0 = unrelated):\n")
    header = "".join(f"{i:>7}" for i in range(len(sentences)))
    print(f"{'':40}{header}")
    for i, sent_i in enumerate(sentences):
        row = "".join(
            f"{cosine_similarity(vectors[i], vectors[j]):7.2f}"
            for j in range(len(sentences))
        )
        label = sent_i[:37] + "..." if len(sent_i) > 37 else sent_i
        print(f"{label:40}{row}")

    print("\nNotice: sentences 0&1 (cat/feline) and 2&3 (stock market) "
          "score high with each other, but low with sentence 4 (pizza) "
          "-- even though none of these sentence pairs share many exact "
          "words. That's the embedding model capturing *meaning*.")
