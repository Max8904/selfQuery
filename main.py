# imports
import os
import glob
import logging
from dotenv import load_dotenv
import gradio as gr

# imports for langchain, plotly and Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
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

# ===== Logger 設定 =====
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("log", "qa_log.log"),
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class PromptLogHandler(BaseCallbackHandler):
    """攔截每次送入 LLM 的完整 prompt 並寫入 log。"""
    def __init__(self):
        self.call_count = 0

    def on_chat_model_start(self, _serialized, messages, **_kwargs):
        self.call_count += 1
        # 第 1 次呼叫 = 濃縮問題，第 2 次 = QA 回答
        stage = "濃縮問題" if self.call_count % 2 == 1 else "QA 回答"
        logger.info("=== 送入模型的 Prompt（%s階段）===", stage)
        for msg in messages[0]:
            logger.info("[%s] %s", msg.type, msg.content)

prompt_log_handler = PromptLogHandler()

# ===== 設定 =====
USE_EMBED_PROVIDER = "ollama"  # embedding 用的 provider: "openai" 或 "ollama"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Prompt 模板：依模型的語言能力選用
PROMPTS = {
    "zh": ChatPromptTemplate.from_messages([
        ("system",
         "你是一個專業的個人資訊問答助手。\n"
         "請根據以下提供的文件內容，使用繁體中文回答使用者的問題。\n"
         "如果文件中沒有相關資訊，請誠實告知。\n\n"
         "{context}"),
        ("human", "{question}"),
    ]),
    "en": ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional personal information QA assistant. "
         "You MUST answer ONLY in Traditional Chinese (繁體中文). "
         "Never respond in English. "
         "Answer the user's question based on the following context. "
         "If the context does not contain relevant information, honestly say so in Traditional Chinese.\n\n"
         "{context}"),
        ("human", "{question}"),
    ]),
}

# 可供使用者在 UI 切換的模型清單 (顯示名稱 -> (provider, model_id, prompt_key))
MODEL_CHOICES = {
    "qwen2.5 (Ollama)": ("ollama", "qwen2.5", "zh"),
    "llama3 (Ollama)": ("ollama", "llama3", "en"),
    "gemma3 (Ollama)": ("ollama", "gemma3", "en"),
    "gpt-4o-mini (OpenAI)": ("openai", "gpt-4o-mini", "zh"),
    "gpt-4o (OpenAI)": ("openai", "gpt-4o", "zh"),
}
DEFAULT_MODEL = "qwen2.5 (Ollama)"

folder = "personal_information"
db_name = "personal_information_vector_db"

if USE_EMBED_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings()
else:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

# 若 DB 已存在且有資料，直接載入；否則才讀取文件並建立
vectorstore = None
if os.path.exists(db_name):
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    count = vectorstore._collection.count()
    if count > 0:
        print(f"Loaded existing vectorstore with {count} documents")
    else:
        print("DB exists but is empty, rebuilding...")
        vectorstore.delete_collection()
        vectorstore = None

if vectorstore is None:
    loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Total documents: {len(documents)}, chunks: {len(chunks)}")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
print(f"There are {collection.count():,} vectors with {len(sample_embedding):,} dimensions")

retriever = vectorstore.as_retriever(search_kwargs = {"k":25})

def build_chain(provider, model_id, prompt_key):
    if provider == "openai":
        llm = ChatOpenAI(temperature=0.7, model_name=model_id)
    else:
        llm = ChatOllama(temperature=0.7, model=model_id)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPTS[prompt_key]},
        return_source_documents=True,
    )

# 用 dict 追蹤目前的 chain 與模型名稱
current = {"model_name": DEFAULT_MODEL, "chain": build_chain(*MODEL_CHOICES[DEFAULT_MODEL])}

def chat(message, _history, model_name):
    if model_name != current["model_name"]:
        current["chain"] = build_chain(*MODEL_CHOICES[model_name])
        current["model_name"] = model_name
        logger.info("模型切換為: %s", model_name)

    prompt_key = MODEL_CHOICES[model_name][2]
    logger.info("使用者問題: %s", message)
    logger.info("使用模型: %s | Prompt: %s", model_name, prompt_key)

    result = current["chain"].invoke(
        {"question": message},
        config={"callbacks": [prompt_log_handler]},
    )

    # 記錄檢索到的文件片段
    for i, doc in enumerate(result.get("source_documents", []), 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        logger.info("檢索文件 [%d] (來源: %s, 頁碼: %s): %s", i, source, page, doc.page_content[:200])

    logger.info("模型回答: %s", result["answer"])
    logger.info("-" * 60)

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