from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader
import os
import glob
import tqdm
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

dbName = "langchain_demo"
collectionName = "collection_of_excel_blobs"
collection = client[dbName][collectionName]

dir_path = r"C:\Users\Hp Pc\Desktop\SEMI\Documents\Excel Docs\\"

# print(f"Path exists: {os.path.exists(dir_path)}")
# files = glob.glob(os.path.join(dir_path, "*.xlsx"), recursive=True)
# print(f"Files found by glob: {len(files)}")

loader = DirectoryLoader(
    dir_path, 
    glob="*.xlsx", 
    loader_cls=UnstructuredExcelLoader, 
    loader_kwargs={"mode": "elements"},
    show_progress=True)

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,   
    chunk_overlap=200, 
    separators=["\n\n","\n",".",","],
    length_function=len 
)

split_docs = text_splitter.split_documents(data)

print(f"Original docs: {len(data)} | After splitting: {len(split_docs)}")

# Then use split_docs instead of data
embedding = OllamaEmbeddings(model="nomic-embed-text:latest")

print("Starting embedding and insert...")
vector_store = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embedding, collection=collection, batch_size=50
)


#Manual batching process by 50
# for i in tqdm(range(0, len(split_docs), 50), desc="Embedding & Inserting"):
#     batch = split_docs[i:i+50]
#     MongoDBAtlasVectorSearch.from_documents(
#         batch, embedding, collection=collection
#     )


