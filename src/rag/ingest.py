from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge_base"


def load_documents():
    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name

        documents.extend(loaded_docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print("Documents loaded:", len(documents))
    print("Chunks created:", len(chunks))

    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)