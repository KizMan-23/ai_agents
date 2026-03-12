from pathlib import Path
import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
load_dotenv()

google_api = os.getenv("GOOGLE_API_KEY")
Settings.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=google_api)
Settings.embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001", api_key=google_api)

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "google_storage"
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
    return index


def main():
    index = get_index()
    query_engine = index.as_query_engine()
    response = query_engine.query("what is this document about?")
    print(response)

if __name__ == "__main__":
    main()