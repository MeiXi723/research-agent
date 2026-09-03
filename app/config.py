import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", "./data/faiss")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/research.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
WINDOW_TURNS = int(os.getenv("WINDOW_TURNS", "5"))
TRACE_DIR = os.getenv("TRACE_DIR", "./data/traces")
MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "3"))
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
