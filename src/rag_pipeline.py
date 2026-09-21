"""
End-to-end RAG pipeline: Stages 1-5 wired together.

This is the file that turns "four separate exercises" into "one working
RAG system." It takes raw documents in, and lets you ask questions out,
by running:

    documents -> chunking -> embeddings -> vector store -> retrieval -> generation

Run it directly to see a full question answered end to end.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from chunking import sentence_chunk
from embeddings import EmbeddingModel
from vector_store import BruteForceVectorStore
from generation import build_prompt, LocalGenerator


class RAGPipeline:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2",
                 gen_model_name: str = "google/flan-t5-small"):
        self.embedder = EmbeddingModel(embed_model_name)
        self.store = BruteForceVectorStore()
        self.generator = LocalGenerator(gen_model_name)

    def ingest(self, documents: dict[str, str], max_chars: int = 300):
        """documents: {source_name: full_text}"""
        all_chunks = []
        for source_name, text in documents.items():
            all_chunks.extend(sentence_chunk(text, source_name, max_chars=max_chars))

        vectors = self.embedder.embed([c.text for c in all_chunks])
        self.store.add(all_chunks, vectors)
        print(f"Ingested {len(all_chunks)} chunks from {len(documents)} document(s).")

    def ask(self, question: str, top_k: int = 3, verbose: bool = True) -> str:
        query_vec = self.embedder.embed_one(question)
        results = self.store.search(query_vec, top_k=top_k)

        if verbose:
            print(f"\nRetrieved {len(results)} chunks:")
            for r in results:
                print(f"  score={r.score:.3f} [{r.source_doc}] {r.text[:80]!r}")

        prompt = build_prompt(question, [r.text for r in results])
        answer = self.generator.generate(prompt)
        return answer


if __name__ == "__main__":
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

    question = "What problem does RAG solve?"
    print(f"\nQuestion: {question}")
    answer = pipeline.ask(question)
    print(f"\nFinal answer: {answer}")
