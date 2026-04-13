# ===== 內建模組與環境變數載入 =====
import os
import sys
import glob
import logging
import re
from dotenv import load_dotenv
import gradio as gr

# 強制設定終端機輸出為 UTF-8 (解決 Windows 亂碼)
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure') and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ===== LangChain 生態系、圖表與向量資料庫載入 =====
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
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
from datetime import datetime

os.makedirs("log", exist_ok=True)
log_filename = f"qa_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    # 將 log 儲存至檔案，設定編碼為 utf-8 並定義格式
    filename=os.path.join("log", log_filename),
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class PromptLogHandler(BaseCallbackHandler):
    """攔截每次送入 LLM 的完整 prompt 並寫入 log。"""
    def on_chat_model_start(self, _serialized, messages, **_kwargs):
        logger.info("=== 送入模型的 Prompt ===")
        for msg in messages[0]:
            if msg.type == "system":
                # System Message 通常包含龐大的檢索文件({context})，將其截斷以保持 Log 乾淨
                content_preview = msg.content.replace('\n', ' ')[:100]
                logger.info("[%s] %s ... (已省略過長的 context 內容)", msg.type, content_preview)
            else:
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
         "回答時請在語句中或結尾處，根據參考的文件使用 [文件名, 頁碼] 格式標註來源。\n"
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
         "Cite sources using the [filename, page] format within or at the end of your sentences. "
         "If the context does not contain relevant information, honestly say so in Traditional Chinese.\n\n"
         "{context}"),
        ("human", "{question}"),
    ]),
}

# 可供使用者在 UI 介面上切換的模型清單 (格式為：顯示名稱 -> (provider, model_id, prompt_key))
MODEL_CHOICES = {
    "qwen2.5 (Ollama)": ("ollama", "qwen2.5", "zh"),
    "llama3 (Ollama)": ("ollama", "llama3", "en"),
    "gemma3 (Ollama)": ("ollama", "gemma3", "en"),
    "gpt-4o-mini (OpenAI)": ("openai", "gpt-4o-mini", "zh"),
    "gpt-4o (OpenAI)": ("openai", "gpt-4o", "zh"),
}
DEFAULT_MODEL = "qwen2.5 (Ollama)"

# 設定文件資料夾與向量資料庫儲存的資料夾名稱
folder = "personal_information"
db_name = "personal_information_vector_db"

# 根據設定初始化 Embedding 模型 (負責將文字轉化為向量)
if USE_EMBED_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings()
else:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

# ===== 向量資料庫 (ChromaDB) 初始化 =====
# 若 DB 已存在且有資料，直接載入；否則才讀取文件並建立
vectorstore = None
if os.path.exists(db_name):
    # 嘗試載入本地的 ChromaDB
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    count = vectorstore._collection.count()
    if count > 0:
        print(f"Loaded existing vectorstore with {count} chunks")
    else:
        print("DB exists but is empty, rebuilding...")
        vectorstore.delete_collection()
        vectorstore = None

# 若無現存的資料庫或為空，則開始讀取文件進行重建
if vectorstore is None:
    # 使用 DirectoryLoader 尋找資料夾底下所有 PDF 並讀取
    loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    # 設定文件切割器，每個區塊 300 個字元，保留 100 字元的重疊以防語意斷層
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Total documents: {len(documents)}, chunks: {len(chunks)}")
    # 將切割好的文件區塊轉換成向量並存入本地端 ChromaDB
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    print(f"Vectorstore created with {vectorstore._collection.count()} chunks")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
print(f"There are {collection.count():,} vectors with {len(sample_embedding):,} dimensions")

# 建立檢索器 (Retriever)，設定每次找出最相似的 5 個文件片段 (k=5)
retriever = vectorstore.as_retriever(search_kwargs = {"k":5})

# ===== 建立對話核心鏈 (Chain) =====
def build_chain(provider, model_id, prompt_key):
    """根據選擇的供應商與模型，建立 LangChain 對話檢索系統"""
    if provider == "openai":
        llm = ChatOpenAI(temperature=0.7, model_name=model_id)
    else:
        llm = ChatOllama(temperature=0.7, model=model_id)
    
    # 建立記憶體模組，確保對話時能參考過去的歷史紀錄
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    # 組合 LLM、檢索器、記憶體與 Prompt 模板
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPTS[prompt_key]},
        return_source_documents=True,
    )

# 使用 dict 追蹤目前的 chain 與模型名稱，以便在使用者切換模型時可以隨時更新
current = {"model_name": DEFAULT_MODEL, "chain": build_chain(*MODEL_CHOICES[DEFAULT_MODEL])}

def chat(message, _history, model_name):
    """Gradio 的聊天主邏輯，處理使用者輸入並回傳 LLM 回答"""
    # 若使用者選擇的模型與目前不同，則重新建立 chain
    if model_name != current["model_name"]:
        current["chain"] = build_chain(*MODEL_CHOICES[model_name])
        current["model_name"] = model_name
        logger.info("模型切換為: %s", model_name)

    prompt_key = MODEL_CHOICES[model_name][2]
    logger.info("使用者問題: %s", message)
    logger.info("使用模型: %s | Prompt: %s", model_name, prompt_key)

    try:
        # 呼叫 LangChain 進行 RAG (檢索增強生成)，並傳入自訂的 log callback
        result = current["chain"].invoke(
            {"question": message},
            config={"callbacks": [prompt_log_handler]},
        )
    except ConnectionError:
        logger.error("無法連線到 Ollama 服務")
        return "無法連線到 Ollama，請確認 Ollama 已啟動（執行 `ollama serve`）。"
    except Exception as e:
        logger.error("呼叫模型時發生錯誤: %s", e)
        return f"發生錯誤：{e}"

    # 記錄檢索到的文件片段並整理成參考清單
    logger.info("=== 檢索到的參考文件 ===")
    reference_list = []
    seen_refs = set()
    for i, doc in enumerate(result.get("source_documents", []), 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", 0) + 1  # 通常 PDF 頁碼從 0 開始，顯示給使用者時 +1
        
        # 取得一小段預覽內容 (移除換行並取前 60 字)
        snippet = doc.page_content.replace('\n', ' ').strip()[:60]
        
        # 建立唯一標識，避免重複列出完全相同的來源頁面
        ref_id = f"{source}, {page}"
        if ref_id not in seen_refs:
            # 套用與文內引用一致的樣式 **`[檔名, 頁碼]`**
            styled_ref = f"**`[{ref_id}]`**"
            reference_list.append(f"{styled_ref} {snippet}...")
            seen_refs.add(ref_id)
            
        logger.info("文件 [%d] | 來源: %s (頁: %s) | 內容: %s...", i, source, page, snippet[:40])

    # 移除模型回答結尾可能存在的多餘換行
    answer_text = result["answer"].strip()
    
    # 使用最寬鬆的正規表示法尋找任何 [...] 格式
    # 只要中括號內有文字（不論是否有逗號或數字），都強制進行標註
    answer_text = re.sub(
        r'(\[[^\]\n]+\])', 
        r'**`\1`**', 
        answer_text
    )

    if reference_list:
        # 直接使用加粗文字作為分隔，並串接加工後的 answer_text
        final_answer = answer_text + "\n\n**【參考來源】**\n" + "\n".join([f"- {ref}" for ref in reference_list])
    else:
        final_answer = answer_text

    logger.info("模型回答: %s", result["answer"])
    logger.info("-" * 60)

    return final_answer

if __name__ == "__main__":
    # 建立 Gradio 下拉選單元件，讓使用者可切換模型
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
    ).launch(height=800)