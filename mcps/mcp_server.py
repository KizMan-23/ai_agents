import os
import logging
import dotenv
from mcp.types import (
    Resource,
    TextContent,
    Tool,
    ImageContent,
    EmbeddedResource,
    GetPromptResult,
    Prompt,
    PromptMessage,
    PromptArgument
)
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio as stdio
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import glob
from importlib import metadata
from langchain_community.document_loaders import PyPDFLoader


# ==============================================
# Logging
# ==============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document-search-mcp")

# ==============================================
# Initialize Server
# ==============================================

server = Server("document-search")

# ==============================================
# Initialize ChromaDB client
# ==============================================
client = None
embedding_function = None
collection = None

try:
    client = chromadb.PersistentClient(path="./chroma_db")
    logger.info("Successfully connected to ChromaDB")

    #Initialize Ollama Embedding
    embedding_function = OllamaEmbeddingFunction(
        model_name="nomic-embed-text:latest",
        url="http://localhost:11434/api/embeddings",
        timeout=120
    )
    logger.info("Successfully initialized Ollama Embedding")

    #Get collection
    collection = client.get_collection(
        name="bchpdf_collection",
        embedding_function=embedding_function
    )
    logger.info(f"Successfully connected to collection with {collection.count()} documents")
except Exception as e:
    logging.error(f"Error initializing components: {e}")

# ===========================================================
# Format search result helper functio for query_document tool
# ===========================================================

def format_search_result(document: str, distance: float, metadata: dict[str, object]= None) -> str:
    """Format a search result into a readable string"""
    result = f"Score: {1 - distance:.4f} (closer to 1 is better)\n"

    if metadata:
        page_num = metadata.get("page", "Unknown")
        result += f"Page: {page_num}\n"
    
    result += f"Content: {document}"
    return result

# ===========================================================
# List available tools in the server
# ===========================================================
server.list_tools()