from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader
import os
import glob
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

embedding = OllamaEmbeddings(model="nomic-embed-text:latest", dimensions=768)

vector_store = MongoDBAtlasVectorSearch.from_documents(data, embedding, collection=collection)



