import os
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "storage"
DATA_DIR = r"C:\Users\Hp Pc\Documents\HHRG-118-JU00-20240312-SD001.pdf"

def get_index(persist_dir=PERSIST_DIR, data_file=DATA_DIR):
    if persist_dir.exists():
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context)
        print("index loaded from storage...")
    else:
        reader = SimpleDirectoryReader(input_files=[str(data_file)])
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print("Index created and persisted to storage...")

    return index


def main():
    index = get_index()
    query_engine = index.as_query_engine()
    response = query_engine.query("What is this documents about?")
    print(response)


if __name__ == "__main__":
    main()


