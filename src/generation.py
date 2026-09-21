"""
Stage 5: Generation

Takes retrieved chunks + a question, builds a prompt that grounds the LLM
in that context, and gets an answer back.

We use flan-t5-small here: a small, instruction-tuned, open-source model
that runs on CPU with no API key. It's not going to write beautifully --
it's a ~80M parameter model, tiny by modern standards -- but it's enough
to prove the RAG mechanism actually works end to end for free. Swapping
in a bigger local model or a hosted API later is a one-function change
(see `AnthropicGenerator` below for that shape), because everything else
in this project only depends on the `generate(prompt: str) -> str`
interface, not on which model provides it.
"""


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    This is the single most important design decision in a RAG system's
    "generation" half: how you phrase the instruction to the model shapes
    whether it actually sticks to the provided context or wanders off and
    makes things up.

    Key ingredients of a good RAG prompt:
      - Explicitly say the answer must come from the context
      - Give an explicit "I don't know" escape hatch, so the model isn't
        pressured into guessing when the context doesn't cover it
      - Clearly separate "context" from "question" so the model doesn't
        confuse the two
    """
    context_block = "\n\n".join(
        f"[Chunk {i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )
    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say "
        "\"I don't know based on the given context.\"\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class LocalGenerator:
    """
    Free, local, no API key -- runs a small instruction-tuned model on CPU.

    We load the tokenizer and model directly (rather than using
    transformers' pipeline() shortcut) so this doesn't depend on a
    specific pipeline "task name" existing in your installed transformers
    version -- and so you can see the actual three steps involved:
    tokenize the text into numbers, run the model, decode the numbers
    back into text.
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print(f"Loading local generation model '{model_name}' "
              f"(first run downloads it, then it's cached)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        # Step 1: turn the prompt text into token IDs (numbers) the model
        # understands. truncation=True prevents an error if the prompt
        # is longer than the model's max input length.
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        # Step 2: the model reads those numbers and produces new numbers
        # (its answer, still as token IDs at this point).
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Step 3: turn those output numbers back into human-readable text.
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


class AnthropicGenerator:
    """
    Optional swap-in if you have an Anthropic API key and want noticeably
    better answers. Same generate(prompt) -> str interface as
    LocalGenerator, so nothing else in this project needs to change to
    use it -- that interface-matching is the whole point of building
    generation as its own small module.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set the ANTHROPIC_API_KEY environment "
                "variable, or pass api_key= explicitly."
            )

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


if __name__ == "__main__":
    context = [
        "Retrieval-augmented generation (RAG) combines a retriever with a "
        "generator model to answer questions using external documents.",
        "The retriever finds relevant chunks of text using vector "
        "similarity search over embeddings.",
    ]
    question = "What are the two main components of a RAG system?"

    prompt = build_prompt(question, context)
    print("=== Prompt sent to the model ===\n")
    print(prompt)
    print("\n=== Generating (loading model on first run)... ===\n")

    generator = LocalGenerator()
    answer = generator.generate(prompt)
    print(f"Answer: {answer}")
