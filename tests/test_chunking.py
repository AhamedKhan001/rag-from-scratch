import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import fixed_size_chunk, sentence_chunk


def test_fixed_size_covers_full_text():
    text = "a" * 1000
    chunks = fixed_size_chunk(text, "doc1", chunk_size=100, overlap=10)
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == 1000
    # step size should be chunk_size - overlap = 90
    assert chunks[1].start_char == 90


def test_fixed_size_rejects_bad_overlap():
    try:
        fixed_size_chunk("hello", "doc1", chunk_size=10, overlap=10)
        assert False, "should have raised"
    except ValueError:
        pass


def test_sentence_chunk_never_splits_a_sentence():
    text = "First sentence here. Second sentence here. Third one too."
    chunks = sentence_chunk(text, "doc1", max_chars=25, overlap_sentences=0)
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?"))


def test_sentence_chunk_handles_single_sentence():
    text = "Just one sentence with no split needed."
    chunks = sentence_chunk(text, "doc1", max_chars=500)
    assert len(chunks) == 1
    assert chunks[0].text == text


if __name__ == "__main__":
    test_fixed_size_covers_full_text()
    test_fixed_size_rejects_bad_overlap()
    test_sentence_chunk_never_splits_a_sentence()
    test_sentence_chunk_handles_single_sentence()
    print("All chunking tests passed.")
