from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

# Both LLM and embeddings run locally via Ollama
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="llama3.2")

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "ollama_storage"
DATA_DIR = r"C:\Users\Hp Pc\Documents\HHRG-118-JU00-20240312-SD001.pdf"


def get_index(persist_dir=PERSIST_DIR, data_file=DATA_DIR):
    if persist_dir.exists():
        storage_context = StorageContext.from_defaults(
            persist_dir=str(persist_dir)
        )
        index = load_index_from_storage(storage_context)
        print("### Index loaded from storage ###...")
    else:
        document = SimpleDirectoryReader(input_files=[str(data_file)]).load_data()
        index = VectorStoreIndex.from_documents(document)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print("### Index created and persisted ###...")
    return index


def main():
    index = get_index()
    query_engine = index.as_query_engine()
    response = query_engine.query("What is this document about?")
    print(response)

if __name__ == "__main__":
    main()