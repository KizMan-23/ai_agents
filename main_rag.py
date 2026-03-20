import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
DATA_FILE = r"C:\Users\Hp Pc\Documents\HHRG-118-JU00-20240312-SD001.pdf"

documents = PyPDFLoader(DATA_FILE).load()
print(f"Loaded {len(documents)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)    
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the questions.
If you dont know the answer, just say that you don't know. Use ten sentences maximun and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate.from_template(template)
out_parser = StrOutputParser()
llm_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | out_parser
    
)
query = "What is this documents about?"

rag_response = rag_chain.invoke(query)
print(rag_response)
