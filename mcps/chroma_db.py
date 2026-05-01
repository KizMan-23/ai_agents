import chromadb
import httpx
from langchain_community.document_loaders import PyPDFLoader as pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")

file_path = r"C:\Users\Hp Pc\Desktop\SEMI\Documents\PDFs\BCH_353\BCH 353- Amino acid disorder.pdf"

def load_pdf(pdf_path):
    """
    load a PDF file and convert to text documents
    Args:
        pdf_pathn(str): Path to PDF file
    
    Returns: 
        list: List of document pages
    """
    loader = pypdf(pdf_path)
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

documents = load_pdf(file_path)
chunks = create_chunks(documents)

embedding_function = OllamaEmbeddingFunction(
    model_name="nomic-embed-text:latest",
    url="http://localhost:11434/api/embeddings", #explicit url
    timeout=120)

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

#add collection
collection.add(
    ids=ids,
    metadatas=metadatas,
    documents=documents
)

print(f"Total documents in collection: {collection.count()}")
