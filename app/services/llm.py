import time

from openai import OpenAI

from app.config import LLM_RETRY_BASE_DELAY, MAX_LLM_RETRIES


_llm_service = None


class LLMService:
    def __init__(self, api_key: str, base_url: str):
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url)
        self._embed_model = None

    def _ensure_embed_model(self):
        if self._embed_model is not None:
            return
        try:
            from fastembed import TextEmbedding
            self._embed_model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
        except Exception:
            self._embed_model = "fallback"

    def embed(self, text: str) -> list[float]:
        self._ensure_embed_model()
        if self._embed_model == "fallback":
            return self._fallback_embed(text)
        try:
            result = list(self._embed_model.embed([text]))
            return result[0].tolist()
        except Exception:
            return self._fallback_embed(text)

    def _fallback_embed(self, text: str, dims: int = 512) -> list[float]:
        import hashlib
        vec = [0.0] * dims
        for i in range(len(text)):
            gram = text[i:i+3]
            h = int(hashlib.md5(gram.encode()).hexdigest()[:8], 16)
            vec[h % dims] += 1.0
        norm = (sum(v * v for v in vec)) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        # Pad or truncate to 384 dims (BGE-small output dims) for FAISS compatibility
        target_dims = 384
        if len(vec) < target_dims:
            vec += [0.0] * (target_dims - len(vec))
        return vec[:target_dims]

    def chat(self, messages: list[dict], model: str = "deepseek-chat", temperature: float = 0.7) -> str:
        for attempt in range(MAX_LLM_RETRIES):
            try:
                resp = self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))

    def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto",
                        model: str = "deepseek-chat", temperature: float = 0.0):
        for attempt in range(MAX_LLM_RETRIES):
            try:
                resp = self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                return resp.choices[0].message
            except Exception:
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))
