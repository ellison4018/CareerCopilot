import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── 凭证常量，各节点按需使用 ──────────────────────────────────────────────────
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY  = os.environ["LLM_API_KEY"]

EMBEDDING_BASE_URL = os.environ["EMBEDDING_BASE_URL"]
EMBEDDING_API_KEY  = os.environ["EMBEDDING_API_KEY"]
EMBEDDING_MODEL    = os.environ["EMBEDDING_MODEL"]

RERANK_BASE_URL = os.environ["RERANK_BASE_URL"]
RERANK_API_KEY  = os.environ["RERANK_API_KEY"]
RERANK_MODEL    = os.environ["RERANK_MODEL"]

# ── Embedding 单例（模型固定，无需per-节点差异化）────────────────────────────
embedding_model = OpenAIEmbeddings(
    base_url=EMBEDDING_BASE_URL,
    api_key=EMBEDDING_API_KEY,
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False,  # DashScope 不接受 tokenized 输入
)
