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

# ===== 設定 =====
USE_EMBED_PROVIDER = "ollama"  # embedding 用的 provider: "openai" 或 "ollama"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# 可供使用者在 UI 切換的模型清單 (顯示名稱 -> (provider, model_id))
MODEL_CHOICES = {
    "llama3 (Ollama)": ("ollama", "llama3"),
    "gemma3 (Ollama)": ("ollama", "gemma3"),
    "gpt-4o-mini (OpenAI)": ("openai", "gpt-4o-mini"),
    "gpt-4o (OpenAI)": ("openai", "gpt-4o"),
}
DEFAULT_MODEL = "llama3 (Ollama)"

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

if USE_EMBED_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings()
else:
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

retriever = vectorstore.as_retriever(search_kwargs = {"k":25})

def build_chain(provider, model_id):
    if provider == "openai":
        llm = ChatOpenAI(temperature=0.7, model_name=model_id)
    else:
        llm = ChatOllama(temperature=0.7, model=model_id)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# 用 dict 追蹤目前的 chain 與模型名稱
current = {"model_name": DEFAULT_MODEL, "chain": build_chain(*MODEL_CHOICES[DEFAULT_MODEL])}

def chat(message, _history, model_name):
    if model_name != current["model_name"]:
        current["chain"] = build_chain(*MODEL_CHOICES[model_name])
        current["model_name"] = model_name
    result = current["chain"].invoke({"question": message})
    return result["answer"]

if __name__ == "__main__":
    model_dropdown = gr.Dropdown(
        choices=list(MODEL_CHOICES.keys()),
        value=DEFAULT_MODEL,
        label="選擇模型",
    )
    gr.ChatInterface(
        fn=chat,
        title="個人資訊問答系統",
        description="根據個人文件回答問題，請輸入你的問題。",
        additional_inputs=[model_dropdown],
    ).launch()