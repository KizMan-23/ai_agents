from pathlib import Path
from llama_index.core import (
    VectoreStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
    Settings,
)

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI