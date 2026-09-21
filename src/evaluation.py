"""
Stage 6: Evaluation

The core idea: build a small "test set" of questions where you already
know which chunk(s) should be retrieved and roughly what the answer
should say. Run the pipeline on every question automatically, and score
retrieval and generation SEPARATELY -- because, as we just saw with the
"I don't know" example, a RAG system can retrieve the right context and
still generate a bad answer (or vice versa: retrieve the wrong context
and still stumble into a right-sounding answer). If you only look at the
final answer, you can't tell which half is actually broken.

This file defines:
  - EvalExample: one labeled test case
  - evaluate_retrieval(): precision/recall for the retrieval step
  - evaluate_generation(): keyword-overlap scoring for the generation step
    (a crude but genuinely useful proxy -- see the docstring below for why)
"""

from dataclasses import dataclass, field


@dataclass
class EvalExample:
    question: str
    # Substrings that should appear somewhere in a correctly-retrieved
    # chunk. We match on substrings rather than exact chunk IDs so the
    # test set survives you tweaking chunk_size/overlap later.
    expected_context_substrings: list[str]
    # Words/phrases we'd expect a correct answer to contain.
    expected_answer_keywords: list[str] = field(default_factory=list)


def evaluate_retrieval(pipeline, examples: list[EvalExample], top_k: int = 3) -> dict:
    """
    For each example, checks whether ANY retrieved chunk contains the
    expected substring. This is "retrieval hit rate" -- a simplified
    stand-in for precision/recall that's easy to reason about by hand,
    which matters when you're still building trust in your own metric.

    Precision would ask: "of what we retrieved, how much was relevant?"
    Recall would ask: "of what's relevant, how much did we retrieve?"
    Hit rate (what we compute here) answers a simpler, related question:
    "did we retrieve at least one relevant chunk at all?" -- good enough
    to catch the class of bug we just found (relevant chunk never
    reaching the generator), without needing a fully labeled corpus.
    """
    hits = 0
    details = []

    for ex in examples:
        query_vec = pipeline.embedder.embed_one(ex.question)
        results = pipeline.store.search(query_vec, top_k=top_k)
        retrieved_text = " ".join(r.text for r in results).lower()

        found = any(sub.lower() in retrieved_text for sub in ex.expected_context_substrings)
        hits += int(found)
        details.append({
            "question": ex.question,
            "hit": found,
            "retrieved": [r.text[:80] for r in results],
        })

    hit_rate = hits / len(examples) if examples else 0.0
    return {"hit_rate": hit_rate, "hits": hits, "total": len(examples), "details": details}


def evaluate_generation(pipeline, examples: list[EvalExample], top_k: int = 3) -> dict:
    """
    Keyword-overlap scoring: for each question, generate a real answer,
    then check what fraction of `expected_answer_keywords` show up in it.

    This is intentionally crude rather than using another LLM to "judge"
    the answer (an approach called LLM-as-judge, which is common in real
    RAG eval but adds its own cost, latency, and judgment-quality
    questions). Keyword overlap is transparent, free, and deterministic
    -- you can see exactly why a score is what it is, which matters most
    while you're still learning to trust (or distrust) your own metrics.
    A production system would likely layer in something like RAGAS's
    faithfulness/answer-relevance scores on top of this, not instead of it.
    """
    results = []
    total_score = 0.0

    for ex in examples:
        answer = pipeline.ask(ex.question, top_k=top_k, verbose=False)
        answer_lower = answer.lower()

        if ex.expected_answer_keywords:
            matched = [kw for kw in ex.expected_answer_keywords if kw.lower() in answer_lower]
            score = len(matched) / len(ex.expected_answer_keywords)
        else:
            score = None

        total_score += score if score is not None else 0.0
        results.append({
            "question": ex.question,
            "answer": answer,
            "keyword_score": score,
        })

    avg_score = total_score / len(examples) if examples else 0.0
    return {"avg_keyword_score": avg_score, "details": results}


def run_full_eval(pipeline, examples: list[EvalExample], top_k: int = 3):
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)
    retrieval_results = evaluate_retrieval(pipeline, examples, top_k=top_k)
    print(f"Hit rate: {retrieval_results['hits']}/{retrieval_results['total']} "
          f"({retrieval_results['hit_rate']*100:.1f}%)\n")
    for d in retrieval_results["details"]:
        status = "HIT " if d["hit"] else "MISS"
        print(f"  [{status}] {d['question']}")

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)
    generation_results = evaluate_generation(pipeline, examples, top_k=top_k)
    print(f"Average keyword score: {generation_results['avg_keyword_score']*100:.1f}%\n")
    for d in generation_results["details"]:
        score_str = f"{d['keyword_score']*100:.0f}%" if d["keyword_score"] is not None else "n/a"
        print(f"  [{score_str}] Q: {d['question']}")
        print(f"         A: {d['answer']!r}")

    return retrieval_results, generation_results


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from rag_pipeline import RAGPipeline

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

    pipeline = RAGPipeline()
    pipeline.ingest(documents)
    run_full_eval(pipeline, test_set)
