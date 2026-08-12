from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.ingest import load_documents, split_documents


BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "careguard_healthcare_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# --------------------------------------------------
# Persistent Chroma client
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "CareGuard healthcare RAG knowledge base"
    }
)


# --------------------------------------------------
# Build vector database
# --------------------------------------------------

def build_vector_store():

    documents = load_documents()
    chunks = split_documents(documents)

    if not chunks:
        print("No knowledge-base chunks found.")
        return

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    ids = [
        f"chunk-{i}"
        for i in range(len(chunks))
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    # Clear old entries so repeated runs do not create duplicates
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )


# --------------------------------------------------
# Semantic retrieval
# --------------------------------------------------

def retrieve(query: str, top_k: int = 3):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_results = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved_results.append(
            {
                "content": document,
                "source": metadata.get(
                    "source",
                    "unknown"
                ),
                "distance": float(distance)
            }
        )

    return retrieved_results


# --------------------------------------------------
# Local test
# --------------------------------------------------

if __name__ == "__main__":

    build_vector_store()

    test_query = (
        "Why might a patient need a lot of medical care?"
    )

    print("\nQuery:")
    print(test_query)

    results = retrieve(
        test_query,
        top_k=3
    )

    for i, result in enumerate(
        results,
        start=1
    ):
        print(f"\n--- Result {i} ---")
        print("Source:", result["source"])
        print(
            "Distance:",
            round(result["distance"], 4)
        )
        print(result["content"])