from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorestores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain.llms import Qwen3
from langchain_core.chains import RetrievalQA
import os

