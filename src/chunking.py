"""
Stage 1: Chunking

Why chunking matters: an LLM's context window is finite, and embedding
models produce better vectors for shorter, semantically coherent text than
for a whole document at once. Chunking is the (deceptively simple-looking)
step that decides what "unit of knowledge" your retriever can ever return.

Get it wrong and:
  - chunks too big -> embeddings become "average of everything", retrieval
    gets fuzzy, and you blow your generation context budget
  - chunks too small -> you lose surrounding context needed to answer,
    and you multiply your vector count (cost + noise)
  - naive splitting (e.g. every N characters) -> you slice sentences and
    even words in half, destroying meaning at chunk boundaries

We implement three strategies here, worst-to-best, so you can literally see
the difference in the demo notebook later.
"""

from dataclasses import dataclass, field
import re


@dataclass
class Chunk:
    text: str
    chunk_id: int
    source_doc: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def fixed_size_chunk(text: str, source_doc: str, chunk_size: int = 500,
                      overlap: int = 50) -> list[Chunk]:
    """
    Naive baseline: slice by raw character count with overlap.

    Overlap exists so that information straddling a chunk boundary in the
    original document isn't lost entirely to one side. This is the
    "everyone's first RAG tutorial" approach — we build it first so later
    stages have something to beat in evaluation.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    chunk_id = 0
    step = chunk_size - overlap

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            source_doc=source_doc,
            start_char=start,
            end_char=end,
        ))
        chunk_id += 1
        if end == len(text):
            break
        start += step

    return chunks


def sentence_chunk(text: str, source_doc: str, max_chars: int = 500,
                    overlap_sentences: int = 1) -> list[Chunk]:
    """
    Better: split on sentence boundaries first, then group sentences into
    chunks up to max_chars. Never cuts a sentence in half.

    This is a simple regex-based sentence splitter. It's not perfect (it
    will trip on abbreviations like "Dr." or "e.g."), which is exactly the
    kind of real-world messiness worth knowing about before you reach for
    a library like nltk or spacy to handle it properly.
    """
    # Split after ., !, or ? followed by whitespace and a capital letter
    # or end of string -- a deliberately simple heuristic.
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    raw_sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks = []
    chunk_id = 0
    current_sentences: list[str] = []
    current_len = 0
    char_cursor = 0
    chunk_start_char = 0

    def flush():
        nonlocal chunk_id, current_sentences, current_len, chunk_start_char
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            source_doc=source_doc,
            start_char=chunk_start_char,
            end_char=chunk_start_char + len(chunk_text),
        ))
        chunk_id += 1

    for sent in sentences:
        if current_len + len(sent) > max_chars and current_sentences:
            flush()
            # carry over last N sentences for continuity
            carry = current_sentences[-overlap_sentences:] if overlap_sentences else []
            chunk_start_char = char_cursor
            current_sentences = list(carry)
            current_len = sum(len(s) for s in current_sentences)

        current_sentences.append(sent)
        current_len += len(sent)
        char_cursor += len(sent) + 1

    flush()
    return chunks


def semantic_chunk(text: str, source_doc: str, embed_fn, max_chars: int = 800,
                    similarity_threshold: float = 0.5) -> list[Chunk]:
    """
    Most sophisticated: split on sentences, embed each one, and start a new
    chunk when consecutive sentences' embeddings diverge past a similarity
    threshold -- i.e. when the topic actually shifts, not just when we hit
    a character limit.

    embed_fn: a function (list[str]) -> np.ndarray of shape (n, dim),
    injected here so this module has zero dependency on Stage 2's embedding
    model -- keeps the stages decoupled and independently testable.

    This is the most "correct" strategy conceptually, and also the most
    expensive (you're embedding at the sentence level just to decide where
    to cut, then re-embedding at the chunk level for retrieval later).
    """
    import numpy as np

    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    raw_sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if len(sentences) <= 1:
        return [Chunk(text=text, chunk_id=0, source_doc=source_doc,
                       start_char=0, end_char=len(text))]

    embeddings = embed_fn(sentences)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    chunks = []
    chunk_id = 0
    current_sentences = [sentences[0]]
    current_len = len(sentences[0])
    char_cursor = 0
    chunk_start_char = 0

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        would_exceed = current_len + len(sentences[i]) > max_chars
        topic_shift = sim < similarity_threshold

        if (topic_shift or would_exceed) and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source_doc=source_doc,
                start_char=chunk_start_char,
                end_char=chunk_start_char + len(chunk_text),
                metadata={"split_reason": "topic_shift" if topic_shift else "max_chars"},
            ))
            chunk_id += 1
            chunk_start_char = char_cursor
            current_sentences = []
            current_len = 0

        current_sentences.append(sentences[i])
        current_len += len(sentences[i])
        char_cursor += len(sentences[i]) + 1

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            chunk_id=chunk_id,
            source_doc=source_doc,
            start_char=chunk_start_char,
            end_char=chunk_start_char + len(chunk_text),
        ))

    return chunks


if __name__ == "__main__":
    sample = (
        "Retrieval-augmented generation combines a retriever with a "
        "generator. The retriever finds relevant documents from a corpus. "
        "It typically uses vector similarity search. Meanwhile, cooking "
        "pasta requires boiling salted water first. Add the pasta once "
        "the water reaches a rolling boil. Drain it a minute before the "
        "package says, since it keeps cooking in the sauce."
    )

    print("=== Fixed-size chunking ===")
    for c in fixed_size_chunk(sample, "demo.txt", chunk_size=100, overlap=20):
        print(f"[{c.chunk_id}] {c.text!r}")

    print("\n=== Sentence chunking ===")
    for c in sentence_chunk(sample, "demo.txt", max_chars=100):
        print(f"[{c.chunk_id}] {c.text!r}")
