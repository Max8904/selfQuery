# ===== 程式版本與環境設定 =====
APP_VERSION = "1.1.0"

# ===== 內建模組與環境變數載入 =====
import os
import sys
import glob
import logging
import re
import hashlib
import shutil
import yaml
from dotenv import load_dotenv
import gradio as gr

# 強制設定終端機輸出為 UTF-8 (解決 Windows 亂碼)
if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure') and sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ===== LangChain 生態系、圖表與向量資料庫載入 =====
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 這個模板決定了 LLM 看到的每段參考資料長什麼樣子
DOC_TEMPLATE = PromptTemplate(
    template="---文件片段---\n檔案名稱: {source}\n頁碼: {page}\n內文: {page_content}",
    input_variables=["page_content", "source", "page"]
)

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

# ===== 讀取配置設定 (config.yaml) =====
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ===== Logger 設定 =====
from datetime import datetime

# 定義 ANSI 顏色代碼
class LogColors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    """自訂顏色格式化器，根據不同的訊息內容或等級上色"""
    FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    
    def format(self, record):
        log_fmt = self.FORMAT
        # 根據 Log 等級設定基本顏色
        if record.levelno == logging.INFO:
            color = LogColors.GREEN
        elif record.levelno == logging.WARNING:
            color = LogColors.YELLOW
        elif record.levelno == logging.ERROR:
            color = LogColors.RED
        else:
            color = LogColors.RESET

        # 特殊處理內容：如果是 Prompt 或模型回答，給予特定顏色
        msg = record.msg
        if "=== 送入模型的 Prompt ===" in msg or "使用者問題:" in msg or msg.startswith("["):
            color = LogColors.CYAN
        elif "模型回答:" in msg:
            color = LogColors.BLUE
        elif "=== 檢索到的參考文件 ===" in msg or msg.startswith("文件 ["):
            color = LogColors.YELLOW

        formatter = logging.Formatter(f"{color}{log_fmt}{LogColors.RESET}", datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

os.makedirs("log", exist_ok=True)
log_filename = f"qa_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 建立 Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 檔案 Handler (現在也包含顏色代碼)
file_handler = logging.FileHandler(os.path.join("log", log_filename), encoding="utf-8")
file_handler.setFormatter(ColorFormatter())

# 終端機 Handler (帶有顏色)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColorFormatter())

# 加入 Handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
USE_EMBED_PROVIDER = config["providers"]["embed_provider"]
OLLAMA_EMBED_MODEL = config["providers"]["ollama_embed_model"]

# Prompt 模板：依模型的語言能力選用
PROMPTS = {
    "zh": ChatPromptTemplate.from_messages([
        ("system",
         "你是一個專業的個人資訊問答助手。\n"
         "請根據以下提供的文件內容，使用繁體中文回答使用者的問題。\n"
         "【回答規則】\n"
         "1. 必須在語句中或結尾處，根據參考的文件使用 [文件名, 頁碼] 格式標註來源。\n"
         "2. 請只擷取檔案路徑中的「檔名」部分（例如：xxx.pdf）進行標註。\n"
         "3. 如果文件中沒有相關資訊，請誠實告知。\n\n"
         "以下為文件的內容：\n"
         "{context}"),
        ("human", "{question}"),
    ]),
    "en": ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional personal information QA assistant. "
         "You MUST answer ONLY in Traditional Chinese (繁體中文). "
         "Never respond in English. "
         "Answer the user's question based on the following context. "
         "【Rules】\n"
         "1. Cite sources using the [filename, page] format within or at the end of your sentences.\n"
         "2. Use only the 'filename' (e.g., xxx.pdf) from the provided source path.\n"
         "3. If the context does not contain relevant information, honestly say so in Traditional Chinese.\n\n"
         "The following is the content of the documents:\n"
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
DEFAULT_MODEL = config["chat"]["default_model"]

# 設定文件資料夾 (支援列表) 與向量資料庫儲存的資料夾名稱
folders = config["vector_db"]["folders"]
db_name = config["vector_db"]["db_name"]

def get_folder_fingerprint(folder_paths, chunk_size, chunk_overlap):
    """計算多個資料夾內容與切割設定的指紋 (檔案狀態 + 切割參數 + 版本號)"""
    files = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            continue
        for ext in ["*.pdf", "*.pptx", "*.docx", "*.xlsx", "*.xls"]:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort()
    
    state = []
    for f in files:
        stats = os.stat(f)
        state.append((os.path.basename(f), stats.st_mtime, stats.st_size))
    
    # 將所有檔案狀態、切割參數與 APP_VERSION 組合在一起進行雜湊
    combined_info = {
        "files": state,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "app_version": APP_VERSION
    }
    return hashlib.md5(str(combined_info).encode()).hexdigest()

# 根據設定初始化 Embedding 模型
if USE_EMBED_PROVIDER == "openai":
    embeddings = OpenAIEmbeddings()
else:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

# ===== 向量資料庫 (ChromaDB) 初始化 (具備自動更新功能) =====
current_fingerprint = get_folder_fingerprint(
    folders, 
    config["vector_db"]["chunk_size"], 
    config["vector_db"]["chunk_overlap"]
)
fingerprint_file = os.path.join(db_name, "fingerprint.txt")

vectorstore = None
rebuild_needed = False

if os.path.exists(db_name):
    # 檢查指紋檔案是否存在且內容是否匹配
    if os.path.exists(fingerprint_file):
        with open(fingerprint_file, "r") as f:
            last_fingerprint = f.read().strip()
        
        if last_fingerprint == current_fingerprint:
            print("Detected no changes in source files or settings. Loading existing vectorstore...")
            vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
            # 如果資料庫是空的，也需要重建
            if vectorstore._collection.count() == 0:
                rebuild_needed = True
        else:
            print("Detected changes in source files or settings. Preparing to rebuild...")
            rebuild_needed = True
    else:
        print("Missing fingerprint file. Rebuilding for safety...")
        rebuild_needed = True
else:
    rebuild_needed = True

if rebuild_needed:
    # 如果資料庫已存在但需要重建，先刪除舊目錄以確保清理乾淨
    if os.path.exists(db_name):
        shutil.rmtree(db_name)
    
    os.makedirs(db_name, exist_ok=True)
    
    # 建立多種格式的載入器配置
    file_loaders = {
        "*.pdf": PyMuPDFLoader,
        "*.pptx": UnstructuredPowerPointLoader,
        "*.docx": Docx2txtLoader,
        "*.xlsx": UnstructuredExcelLoader,
        "*.xls": UnstructuredExcelLoader,
    }

    documents = []
    for folder_path in folders:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
        print(f"Loading documents from: {folder_path}")
        for glob_pattern, loader_cls in file_loaders.items():
            try:
                loader = DirectoryLoader(folder_path, glob=glob_pattern, loader_cls=loader_cls)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                if loaded_docs:
                    print(f"  - Loaded {len(loaded_docs)} documents matching {glob_pattern}")
            except Exception as e:
                print(f"Error loading {glob_pattern} in {folder_path}: {e}")
    
    # 預處理：1. 將所有文件的頁碼 +1 (從 0-indexed 改為 1-indexed)，若無頁碼預設為 1
    #         2. 將 source 路徑清理為純檔名，避免 LLM 輸出完整路徑
    for doc in documents:
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown"))

    # 設定文件切割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["vector_db"]["chunk_size"], 
        chunk_overlap=config["vector_db"]["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total documents: {len(documents)}, chunks: {len(chunks)}")
    
    # 將切割好的文件區塊轉換成向量並存入本地端 ChromaDB
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    
    # 寫入新指紋
    with open(fingerprint_file, "w") as f:
        f.write(current_fingerprint)
    print(f"Vectorstore created/rebuilt with {vectorstore._collection.count()} chunks.")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
print(f"There are {collection.count():,} vectors with {len(sample_embedding):,} dimensions")

# 建立檢索器 (Retriever)，設定每次找出最相似的文件片段 (k 來自 config.yaml)
retriever = vectorstore.as_retriever(search_kwargs = {"k": config["retriever"]["k"]})

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
        combine_docs_chain_kwargs={
            "prompt": PROMPTS[prompt_key],
            "document_prompt": DOC_TEMPLATE
        },
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
        # 由於我們在存入資料庫前已經將頁碼 +1，這裡直接讀取即可
        page = doc.metadata.get("page", 1) 
        
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