"""
Run the entire RAG-from-scratch project end to end, stage by stage,
with clear headers so you can see exactly what each part does and
what its output looks like.

Usage:
    python run_all.py

This is meant for demoing/reviewing the project (e.g. showing it to
someone, or refreshing your own memory on what you built) -- not for
production use. Each stage below is copy-pasted from that stage's own
__main__ block, so this file has no logic of its own; it's a guided
tour through the actual project files.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    # ---------------------------------------------------------------
    section("STAGE 1: CHUNKING")
    # ---------------------------------------------------------------
    from chunking import fixed_size_chunk, sentence_chunk

    sample = (
        "Retrieval-augmented generation combines a retriever with a "
        "generator. The retriever finds relevant documents from a corpus. "
        "It typically uses vector similarity search. Meanwhile, cooking "
        "pasta requires boiling salted water first."
    )
    print("Fixed-size chunking (notice it cuts words in half):")
    for c in fixed_size_chunk(sample, "demo.txt", chunk_size=100, overlap=20):
        print(f"  [{c.chunk_id}] {c.text!r}")

    print("\nSentence chunking (never cuts a sentence in half):")
    for c in sentence_chunk(sample, "demo.txt", max_chars=100):
        print(f"  [{c.chunk_id}] {c.text!r}")

    # ---------------------------------------------------------------
    section("STAGE 2: EMBEDDINGS")
    # ---------------------------------------------------------------
    from embeddings import EmbeddingModel, cosine_similarity

    embed_model = EmbeddingModel()  # downloads/caches on first run
    sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed yesterday.",
        "I love eating pizza on weekends.",
    ]
    vectors = embed_model.embed(sentences)
    print(f"Embedded {len(sentences)} sentences into {vectors.shape[1]}-dim vectors.\n")
    print("Similarity: cat-sentence vs feline-sentence (should be high):",
          f"{cosine_similarity(vectors[0], vectors[1]):.3f}")
    print("Similarity: cat-sentence vs pizza-sentence (should be low): ",
          f"{cosine_similarity(vectors[0], vectors[3]):.3f}")

    # ---------------------------------------------------------------
    section("STAGE 3 & 4: VECTOR STORE + FAISS BENCHMARK")
    # ---------------------------------------------------------------
    from faiss_store import benchmark_vs_brute_force
    try:
        benchmark_vs_brute_force(n_vectors=2000, dim=64, n_queries=10, top_k=5)
    except ImportError:
        print("(Skipped: install faiss-cpu to run this part -- pip install faiss-cpu)")

    # ---------------------------------------------------------------
    section("STAGE 5 & 6: FULL PIPELINE + EVALUATION")
    # ---------------------------------------------------------------
    from rag_pipeline import RAGPipeline
    from evaluation import EvalExample, run_full_eval

    documents = {
        "rag_basics.txt": (
            "Retrieval-augmented generation, or RAG, is a technique that "
            "combines a retriever with a language model. The retriever "
            "searches a corpus of documents to find relevant passages. "
            "The language model then uses those passages as context to "
            "generate a grounded, factual answer. This reduces "
            "hallucination compared to asking a language model a question "
            "with no supporting context."
        ),
        "vector_search.txt": (
            "Vector search works by converting text into numerical "
            "embeddings using a machine learning model. Similar pieces of "
            "text end up with similar embeddings. To find relevant "
            "documents, we convert the search query into an embedding "
            "too, then find the stored embeddings closest to it using a "
            "similarity measure like cosine similarity."
        ),
    }

    pipeline = RAGPipeline()
    pipeline.ingest(documents)

    test_set = [
        EvalExample(
            question="What problem does RAG solve?",
            expected_context_substrings=["reduces hallucination"],
            expected_answer_keywords=["hallucination"],
        ),
        EvalExample(
            question="How does vector search find relevant documents?",
            expected_context_substrings=["cosine similarity"],
            expected_answer_keywords=["embedding", "similarity"],
        ),
        EvalExample(
            question="What does the retriever do in a RAG system?",
            expected_context_substrings=["searches a corpus"],
            expected_answer_keywords=["search", "passages"],
        ),
    ]
    run_full_eval(pipeline, test_set)

    # ---------------------------------------------------------------
    section("DONE")
    # ---------------------------------------------------------------
    print("That's the full project: chunking -> embeddings -> vector store")
    print("-> FAISS indexing -> retrieval+generation -> evaluation.")


if __name__ == "__main__":
    main()
