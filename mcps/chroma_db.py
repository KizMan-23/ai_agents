import chromadb
import httpx
import ollama
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader as pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")

# BASE_DIR = Path(__file__).resolve().parent
# PERSIST_DIR = BASE_DIR/ "biochem_docs"
DIR_PATH = r"C:\Users\Hp Pc\Desktop\SEMI\Documents\PDFs\BCH_353\\"

def load_dir(dir_path):
    """
    load a PDF file and convert to text documents
    Args:
        pdf_pathn(str): Path to PDF file
    
    Returns: 
        list: List of document pages
    """
    loader = DirectoryLoader(
        path=dir_path,
        glob="**/*.pdf",
        recursive=True,
        loader_cls=pypdf,
        show_progress=True
    )
    pages = loader.load()
    return pages

def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into overlappiing chunks

    Args:
        document(list): List of document to split
        chunk_size(int): Size of each chunk characters
        chunk_overlap(int): size of chunk character to overlap

    Returns:
        list: list of chunked texts
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

documents = load_dir(DIR_PATH)
chunks = create_chunks(documents)


embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text:latest",
    url="http://localhost:11434/api/embeddings", #explicit url
    timeout=120
)

ollama._client.Client.timeout = httpx.Timeout(120) # Set timeout for Ollama client

try:
    response = httpx.get("http://localhost:11434", timeout=5)
    print("✅ Ollama is running...")
    print(f"Ollama Report: {response.json}")
except Exception:
    raise RuntimeError("❌ Ollama is not running. Start it with: ollama serve")

collection = client.get_or_create_collection(name="bchpdf_collection", embedding_function=embedding_function)

#Add Documents to collection
documents = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
ids = [str(i) for i in range(len(chunks))]

#Add to collection in batches to avoid memory issues
BATCH_SIZE = 50

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i + BATCH_SIZE]
    batch_meta = metadatas[i:i + BATCH_SIZE]
    batch_ids  = ids[i:i + BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        metadatas=batch_meta,
        documents=batch_docs
    )
    print(f"Added batch {i // BATCH_SIZE + 1} / {len(documents) // BATCH_SIZE + 1}")

print(f"Total documents in collection: {collection.count()}")

