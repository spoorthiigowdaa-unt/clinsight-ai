from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.rag.retriever import retrieve


MODEL_NAME = "google/flan-t5-small"


# --------------------------------------------------
# Load local model and tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# RAG answer generation
# --------------------------------------------------

def generate_answer(query: str, top_k: int = 3):

    results = retrieve(
        query=query,
        top_k=top_k
    )

    if not results:
        return {
            "answer": (
                "The available CareGuard knowledge base "
                "does not contain enough information."
            ),
            "sources": []
        }

    context_blocks = []
    sources = []

    for i, result in enumerate(results, start=1):

        context_blocks.append(
            f"""
Source {i}: {result['source']}
{result['content']}
"""
        )

        sources.append(result["source"])

    context = "\n".join(context_blocks)

    prompt = f"""
Answer the question using only the provided healthcare context.

Do not use outside knowledge.
Do not provide diagnosis or treatment advice.

If the context does not contain enough information, say:
"The available CareGuard knowledge base does not contain enough information."

Question:
{query}

Context:
{context}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return {
        "answer": answer,
        "sources": sorted(set(sources))
    }


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    test_query = (
        "Why might a patient have high healthcare utilization?"
    )

    result = generate_answer(test_query)

    print("\nQUESTION:")
    print(test_query)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:
        print("-", source)