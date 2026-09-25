import os
import glob
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "zepto_policies"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=ef
    )

def initialize_vector_store():
    collection = get_chroma_collection()
    if collection.count() > 0:
        return collection

    doc_files = glob.glob("docs/doc_*.txt")
    documents = []
    ids = []
    metadatas = []

    for filepath in sorted(doc_files):
        doc_id = os.path.basename(filepath).split('.')[0]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
            documents.append(text)
            ids.append(doc_id)
            metadatas.append({"source": doc_id})

    if documents:
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    return collection

if __name__ == "__main__":
    col = initialize_vector_store()
    print(f"Indexed {col.count()} documents into ChromaDB.")