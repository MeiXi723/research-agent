import asyncio
import hashlib
import json
import os
import time
from types import SimpleNamespace

from openai import AsyncOpenAI, OpenAI

from app.config import (
    AI_MOCK, AI_MOCK_DIR, LLM_RETRY_BASE_DELAY, LLM_TIMEOUT_SECONDS, MAX_LLM_RETRIES,
)

_llm_service = None
_async_llm_service = None

_RECORD_MODE = os.getenv("AI_MOCK_RECORD", "0") == "1"


def _mock_key(kind: str, payload: dict) -> str:
    # 录制回放的键：对请求体做稳定哈希（同请求 → 同罐头）
    body = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def _mock_lookup(key: str):
    path = os.path.join(AI_MOCK_DIR, "recordings.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") == key:
                return rec.get("response")
    return None


def _mock_record(key: str, kind: str, payload: dict, response) -> None:
    os.makedirs(AI_MOCK_DIR, exist_ok=True)
    path = os.path.join(AI_MOCK_DIR, "recordings.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "kind": kind,
                            "request": payload, "response": response},
                           ensure_ascii=False) + "\n")


def _replay_message(raw) -> SimpleNamespace:
    """把录制的消息 dict 还原成 agent.py 消费的形状（.content/.tool_calls/.function.*）。"""
    tool_calls = None
    if raw.get("tool_calls"):
        tool_calls = [
            SimpleNamespace(function=SimpleNamespace(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ))
            for tc in raw["tool_calls"]
        ]
    return SimpleNamespace(content=raw.get("content"), tool_calls=tool_calls)


class LLMService:
    def __init__(self, api_key: str, base_url: str):
        self.chat_client = OpenAI(api_key=api_key, base_url=base_url,
                                  timeout=LLM_TIMEOUT_SECONDS)
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
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if AI_MOCK or _RECORD_MODE:
            key = _mock_key("chat", payload)
            if AI_MOCK:
                raw = _mock_lookup(key)
                # 罐头缺失必须响（fail-loud）：静默编造会让冒烟测试假绿
                if raw is None:
                    raise RuntimeError(f"AI_MOCK 回放缺失录制: {key}（先开 AI_MOCK_RECORD=1 真跑录制）")
                return raw
        for attempt in range(MAX_LLM_RETRIES):
            try:
                resp = self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                content = resp.choices[0].message.content
                if _RECORD_MODE and not AI_MOCK:
                    _mock_record(_mock_key("chat", payload), "chat", payload, content)
                return content
            except Exception:
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))

    def chat_with_tools(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto",
                        model: str = "deepseek-chat", temperature: float = 0.0):
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "tools": tools, "tool_choice": tool_choice}
        if AI_MOCK or _RECORD_MODE:
            key = _mock_key("chat_with_tools", payload)
            if AI_MOCK:
                raw = _mock_lookup(key)
                if raw is None:
                    raise RuntimeError(f"AI_MOCK 回放缺失录制: {key}（先开 AI_MOCK_RECORD=1 真跑录制）")
                return _replay_message(raw)
        for attempt in range(MAX_LLM_RETRIES):
            try:
                resp = self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                message = resp.choices[0].message
                if _RECORD_MODE and not AI_MOCK:
                    record = {
                        "content": message.content,
                        "tool_calls": [
                            {"function": {"name": tc.function.name,
                                          "arguments": tc.function.arguments}}
                            for tc in (message.tool_calls or [])
                        ],
                    }
                    _mock_record(_mock_key("chat_with_tools", payload), "chat_with_tools", payload, record)
                return message
            except Exception:
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))


class AsyncLLMService:
    """多智能体角色用：事件循环内不能跑同步 HTTP——async 改写，重试语义与 LLMService 一致。"""

    def __init__(self, api_key: str, base_url: str):
        self.chat_client = AsyncOpenAI(api_key=api_key, base_url=base_url,
                                       timeout=LLM_TIMEOUT_SECONDS)

    async def chat(self, messages: list[dict], model: str = "deepseek-chat",
                   temperature: float = 0.7) -> str:
        for attempt in range(MAX_LLM_RETRIES):
            try:
                resp = await self.chat_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                await asyncio.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))
