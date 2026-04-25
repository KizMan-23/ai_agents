from pymongo import MongoClient
from langchain_ollama import OllamaEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader, UnstructuredExcelLoader
from langchain_ollama import ChatOllama
from langchain_community.chains import RetrievalQA
import os
import sys
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

dbName = "langchain_demo"
collectionName = "collection_of_excel_blobs"
collection = client[dbName][collectionName]

embedding = OllamaEmbeddings(model="qwen3-embedding:8b", dimensions=1024)

vector_store = MongoDBAtlasVectorSearch(embedding, collection=collection)

def query_data(query):
    docs = vector_store.similarity_search(query, k=5)
    as_output = docs[0].page_content if docs else "No relevant documents found."

    llm = ChatOllama(model="qwen3-8b", temperature=0.7)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    retriever_output = qa_chain.run(query)

    return as_output, retriever_output

if len(sys.argv) < 2:
    print("Usage: python query.py \"Your question here\"")
    sys.exit(1)

query = sys.argv[1]
print(f"Query received: {query}\n\n")

retriever_answer, llm_answer = query_data(query)
print("====== Retriever Output ======")
print(f"\n {retriever_answer[:500]}....\n " if len(retriever_answer) > 500 else retriever_answer)
print("\n\n====== LLM Answer ======")
print(f"\n {llm_answer[:500]}....\n " if len(llm_answer) > 500 else llm_answer)