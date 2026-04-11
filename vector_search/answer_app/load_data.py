from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain.llms import Qwen3
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

client = os.getenv("MONGO_URL")
dbName = "langchain_demo"
collectionName = "collection_of_cv_blobs"
collection = client[dbName][collectionName]

dir_path = r"C:\Users\Hp Pc\Documents\CVs"

loader = DirectoryLoader(dir_path, glob="**/*.txt", show_progress=True)
data = loader.load()

embedding = OllamaEmbeddings("Qwen/Qwen3-Embedding-8B")

vector_store = MongoDBAtlasVectorSearch(data, embedding, collection=collection)
