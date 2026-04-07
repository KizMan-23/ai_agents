from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utils import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

FILE_PATH  = r"C:\Users\Hp Pc\Documents\HHRG-118-JU00-20240312-SD001.pdf"

document = PyPDFLoader(FILE_PATH).load()
print(document[0].page_content)

text_splitter =  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunk = text_splitter.from_documents(document)

#Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorestore = FAISS.from_documents(text_chunk, embeddings, persist_directory="./faiss_index")
retriever = vectorestore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

template ="""You are an assistant for question answering. Use the following pieces of context to answer the question at the end.
If you don't know the answer, say you don't know. use ten sentences maximum to answer the question and keep the answers concise.
Do not make up any information. Always use the relevant information from the context to answer the question. 
Always use all the relevant information from the context to
Question: {question} 
Context: {context}
"""

prompt = ChatPromptTemplate.from_template(template, input_variables=["context", "question"])
parser = StrOutputParser()
llm = Ollama(model="nomic-qa", temperature=0.9, max_tokens=500)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs , "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)


result = rag_chain.invoke({"query": "What are the main findings?"})
print(result["result"])
print("\nSources:", [d.metadata for d in result["source_documents"]])