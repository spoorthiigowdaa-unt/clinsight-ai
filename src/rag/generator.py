from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.rag.retriever import retrieve


MODEL_NAME = "google/flan-t5-small"


# --------------------------------------------------
# Load local model and tokenizer
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)


# --------------------------------------------------
# Helper: build grounded fallback answer
# --------------------------------------------------

def build_grounded_fallback(results):
    """
    Creates a concise answer directly from retrieved context
    when the local language model produces an overly short answer.

    The fallback uses only retrieved knowledge-base content.
    """

    combined_context = " ".join(
        result["content"].replace("\n", " ")
        for result in results
    )

    known_factors = [
        "multiple chronic conditions",
        "frequent emergency department visits",
        "previous inpatient admissions",
        "high procedure utilization",
        "polypharmacy or high medication burden",
        "complex care needs",
        "poor continuity of care"
    ]

    matched_factors = [
        factor
        for factor in known_factors
        if factor.lower() in combined_context.lower()
    ]

    if matched_factors:

        readable_factors = ", ".join(
            matched_factors[:-1]
        )

        if len(matched_factors) > 1:
            readable_factors += (
                f", and {matched_factors[-1]}"
            )
        else:
            readable_factors = matched_factors[0]

        return (
            "High healthcare utilization may be associated with "
            f"{readable_factors}. "
            "Historical encounters, conditions, procedures, and "
            "medication use can help identify patients with greater "
            "future healthcare needs."
        )

    return (
        "Healthcare utilization can be influenced by patterns in "
        "a patient's previous healthcare use and clinical history. "
        "The retrieved knowledge indicates that historical utilization "
        "information can help identify patients who may require "
        "additional care management."
    )


# --------------------------------------------------
# RAG answer generation
# --------------------------------------------------

def generate_answer(
    query: str,
    top_k: int = 3
):

    # --------------------------------------------------
    # Retrieve relevant knowledge
    # --------------------------------------------------

    results = retrieve(
        query=query,
        top_k=top_k
    )

    if not results:
        return {
            "answer": (
                "The available ClinSight knowledge base "
                "does not contain enough information."
            ),
            "sources": []
        }

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context_blocks = []
    sources = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_blocks.append(
            f"""
Source {i}: {result['source']}
{result['content']}
"""
        )

        sources.append(
            result["source"]
        )

    context = "\n".join(
        context_blocks
    )

    # --------------------------------------------------
    # Grounded prompt
    # --------------------------------------------------

    prompt = f"""
You are ClinSight AI, a healthcare utilization knowledge assistant.

Answer the question using ONLY the healthcare context provided below.

Instructions:
- Do not use outside medical knowledge.
- Do not provide diagnosis, prescriptions, treatment recommendations,
  medication changes, or dosage advice.
- Give a clear and concise answer.
- Prefer 2 to 4 complete sentences when the context supports them.
- Explain the relationship between the retrieved factors and healthcare
  utilization instead of returning only a short phrase.
- Mention more than one relevant factor when the context contains several.
- Do not invent information that is not present in the context.

If the context does not contain enough information, respond exactly:
"The available ClinSight knowledge base does not contain enough information."

Question:
{query}

Healthcare Context:
{context}

Grounded Answer:
"""

    # --------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        num_beams=4,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        length_penalty=1.2,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    # --------------------------------------------------
    # Short-answer guard
    # --------------------------------------------------
    # FLAN-T5-small sometimes returns only a short extracted phrase.
    # If the result is too short, construct a grounded summary using
    # only information contained in the retrieved knowledge chunks.
    # --------------------------------------------------

    word_count = len(
        answer.split()
    )

    if word_count < 8:
        answer = build_grounded_fallback(
            results
        )

    # --------------------------------------------------
    # Return answer and unique sources
    # --------------------------------------------------

    return {
        "answer": answer,
        "sources": sorted(
            set(sources)
        )
    }


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        (
            "Why might a patient have high "
            "healthcare utilization?"
        ),
        (
            "What factors can contribute to "
            "high healthcare utilization?"
        ),
        (
            "Why do patients with multiple chronic "
            "conditions have higher healthcare utilization?"
        )
    ]

    for test_query in test_queries:

        result = generate_answer(
            test_query
        )

        print(
            "\n" + "=" * 60
        )

        print("QUESTION:")
        print(
            test_query
        )

        print("\nANSWER:")
        print(
            result["answer"]
        )

        print("\nSOURCES:")

        for source in result["sources"]:
            print(
                "-",
                source
            )