# imports
import os
import glob
from dotenv import load_dotenv
import gradio as gr

# imports for langchain, plotly and Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ===== 切換設定：改為 "ollama" 即可使用本地模型 =====
USE_PROVIDER = "openai"  # "openai" 或 "ollama"
OLLAMA_MODEL = "llama3"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

folder = "personal_information"
loader_kwargs = {}
loader = DirectoryLoader(folder, glob = "*.pdf", loader_cls = PyPDFLoader, loader_kwargs = loader_kwargs)
folder_doc = loader.load()

documents = folder_doc
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 100)
chunks = text_splitter.split_documents(documents)
print(f"Total number of documents: {len(documents)}")
print(f"Total number of chunks: {len(chunks)}")

db_name = "personal_information_vector_db"

if USE_PROVIDER == "openai":
    MODEL = "gpt-4o-mini"
    embeddings = OpenAIEmbeddings()
else:
    MODEL = OLLAMA_MODEL
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

if os.path.exists(db_name):
    Chroma(persist_directory = db_name, embedding_function = embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents = chunks, embedding = embeddings, persist_directory = db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit = 1, include = ["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

if USE_PROVIDER == "openai":
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
else:
    llm = ChatOllama(temperature=0.7, model=MODEL)
memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
retriever = vectorstore.as_retriever(search_kwargs = {"k":25})
conversation_chain = ConversationalRetrievalChain.from_llm(llm = llm, retriever = retriever, memory = memory)

if __name__ == "__main__":
    query = "請問碩士讀哪裡?"
    result = conversation_chain.invoke({"question": query})
    print(result["answer"])