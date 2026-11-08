import os
from dotenv import load_dotenv

# override=True：.env 是本项目唯一配置源——shell 里预置的同名环境变量不得
# 静默压过项目配置（实测：旧 DEEPSEEK_API_KEY 残留在 shell 导致切 key 后仍读旧值）
load_dotenv(override=True)

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

# ---------------------------------------------------------------------------
# v2.0 配置
# ---------------------------------------------------------------------------
#
# LLM_TIMEOUT_SECONDS: 单次 LLM 请求超时。v1 依赖 OpenAI SDK 默认值，长思考会
# 挂住整个 ReAct 循环；显式超时后走既有重试通道。
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

# AI_MOCK: 罐头响应模式（录制回放）。=1 时 chat/chat_with_tools 不发真实请求，
# 回放 data/ai_mock/ 下的录制；E2E 冒烟在 CI 无 key 环境跑全链路就靠它。
AI_MOCK = os.getenv("AI_MOCK", "0") == "1"
AI_MOCK_DIR = os.getenv("AI_MOCK_DIR", "./data/ai_mock")

# MCP_SERVERS: stdio 子进程 MCP server 清单，空格分隔的 name=command [args...]
#   例：MCP_SERVERS="research=python mcp_servers/research_server.py"
#   空值 = 不启用 MCP 层，agent 退回内置三工具（与 v1 行为一致）
MCP_SERVERS = os.getenv("MCP_SERVERS", "")
MCP_TOOL_TIMEOUT = float(os.getenv("MCP_TOOL_TIMEOUT", "30"))

# EVAL_*: L4 评测层（opt-in）
EVAL_DATASETS_DIR = os.getenv("EVAL_DATASETS_DIR", "./evals/datasets")
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "5"))

# ---------------------------------------------------------------------------
# v2.0 多模态视觉专线 —— 与文本通道（DEEPSEEK_*）完全解耦的独立提供商
# ---------------------------------------------------------------------------
#
# 文本模型（DeepSeek 官方）没有视觉能力，图片理解走这条独立专线：
# 任意 OpenAI 兼容的视觉端点都可接入（智谱 GLM-4V / 方舟 doubao-vision /
# OpenRouter qwen-vl 等），格式为 messages[].content 数组 + image_url(base64)。
#
# VISION_API_KEY 为空 = 专线未配置：图片上传优雅降级（400 + 明确提示），
# 不影响文本问答全链路。
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
VISION_MODEL = os.getenv("VISION_MODEL", "glm-4v-flash")
VISION_TIMEOUT_SECONDS = float(os.getenv("VISION_TIMEOUT_SECONDS", "60"))
# 图片入库描述的默认提问
VISION_DESCRIBE_PROMPT = os.getenv(
    "VISION_DESCRIBE_PROMPT",
    "详细描述这张图片的全部文字内容与关键信息，输出可供检索的纯文本，保留数字与专有名词。",
)
