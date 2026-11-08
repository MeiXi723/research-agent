"""并发冒烟（G2 门禁）：5 并发 /ask + 上传，验证无串话、无状态污染、错误率 0。

串话判定：mock LLM 的最终答案携带请求问题哈希标记，响应必须含自己的标记。
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


def _marker(question: str) -> str:
    return "MK" + hashlib.md5(question.encode()).hexdigest()[:6]


@pytest.fixture
def client():
    from app.main import app
    import app.main as main_module
    import app.services.agent as agent_module
    import app.services.llm as llm_module

    with TestClient(app) as c:
        mock = MagicMock()
        mock.embed.return_value = [0.1] * 384
        mock.last_usage = None  # MagicMock getattr 会自动造属性，必须显式置 None

        def chat_side_effect(messages, **kwargs):
            return "mock 摘要"

        mock.chat.side_effect = chat_side_effect

        # 直答模式：无 tool_calls，答案携带本次问题哈希——串话立即可见
        def chat_with_tools_side_effect(messages, tools, **kwargs):
            last_user = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"][-1]
            marker = _marker(last_user["content"])
            return SimpleNamespace(
                content=f"最终答案 {marker}",
                tool_calls=None,
            )

        mock.chat_with_tools.side_effect = chat_with_tools_side_effect

        agent_module._llm_service = mock
        llm_module._llm_service = mock
        main_module._llm_service = mock
        llm_module._async_llm_service = MagicMock()
        llm_module._async_llm_service.chat = _async_fail
        yield c


async def _async_fail(*args, **kwargs):
    raise RuntimeError("并发冒烟不测多 agent 路径")


def test_5_concurrent_asks_no_cross_talk(client):
    questions = [f"第{i}个并发问题：查询主题{i}号" for i in range(5)]

    def run(q):
        resp = client.post("/ask", json={"question": q, "force_route": "tool_research"})
        assert resp.status_code == 200
        body = resp.json()
        # 串话检测：答案必须携带本请求的标记
        assert _marker(q) in body["answer"], f"串话！问[{q[:12]}] 答[{body['answer'][:30]}]"
        assert body["route"] == "tool_research"
        assert body["metrics"]["route_via"] is not None
        return body

    with ThreadPoolExecutor(max_workers=5) as pool:
        bodies = list(pool.map(run, questions))

    assert len({b["conversation_id"] for b in bodies}) == 5  # 会话各自独立
    assert all(b["metrics"]["usage_captured"] is False for b in bodies)  # mock 无 usage，口径如实


def test_concurrent_uploads_isolated(client):
    def run(i):
        resp = client.post(
            "/documents/upload",
            files={"file": (f"conc_{i}.txt", f"并发上传内容 {i} 号。".encode(), "text/plain")},
        )
        assert resp.status_code == 200
        return resp.json()["id"]

    with ThreadPoolExecutor(max_workers=5) as pool:
        ids = list(pool.map(run, range(5)))
    assert len(set(ids)) == 5
