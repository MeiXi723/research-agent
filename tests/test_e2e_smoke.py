"""E2E 全链路冒烟（marker=e2e，CI 默认排除）：真实 lifespan 启停 + 全部端点，LLM 全 mock 无网络。"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.e2e


class FakeAsyncLLM:
    """多 agent 冒烟：让每个 _structured_call 走 fallback 路径（graph 仍完整跑通）。"""

    async def chat(self, messages, **kwargs):
        raise RuntimeError("e2e 冒烟模式：LLM 不可用，验证 fallback 闭环")


@pytest.fixture
def client():
    from app.main import app
    import app.main as main_module
    import app.services.agent as agent_module
    import app.services.llm as llm_module
    from app.services.memory import SemanticMemory, set_memory

    with TestClient(app) as c:  # lifespan：db/faiss/memory/llm 单例 + /health 就绪
        mock = MagicMock()
        mock.embed.return_value = [0.1] * 384
        mock.last_usage = None  # MagicMock getattr 会自动造属性，必须显式置 None
        mock.chat.return_value = "mock 摘要"
        # MagicMock(name=...) 的 name 是 mock 身份不是属性——工具调用消息必须 SimpleNamespace
        msg_tool = SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(id="t1", function=SimpleNamespace(
                name="recall_memory", arguments=json.dumps({"query": "最大步数"}),
            ))],
        )
        msg_final = SimpleNamespace(content="最终答案：ReAct 是思考与行动结合的范式。", tool_calls=None)
        mock.chat_with_tools.side_effect = [msg_tool, msg_final]

        # 三处单例都要覆盖：main.py 端点（upload 的 embed）走 main 模块全局
        agent_module._llm_service = mock
        llm_module._llm_service = mock
        main_module._llm_service = mock
        llm_module._async_llm_service = FakeAsyncLLM()
        # 语义记忆换 mock 嵌入，避免触发 fastembed 真加载
        set_memory(SemanticMemory(main_module._db, embed_fn=lambda t: [0.1] * 384))
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mcp_enabled"] is False  # 未配 MCP_SERVERS 时退回内置工具


def test_upload_then_ask_with_memory_recall(client):
    up = client.post(
        "/documents/upload",
        files={"file": ("e2e.txt", __import__("io").BytesIO("ReAct 范式将推理与行动结合。".encode()), "text/plain")},
    )
    assert up.status_code == 200

    # 预置一条记忆，让 agent 的 recall_memory 工具真有东西可召回
    from app.services.memory import get_memory
    get_memory().remember("sys_fact", "系统最大步数为 10 步")

    ask = client.post("/ask", json={"question": "系统最大步数是多少？"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["answer"].startswith("最终答案")
    assert len(body["steps"]) >= 1
    assert body["steps"][0]["tool"] == "recall_memory"
    # 召回内容真的回到了工具结果里
    assert "10 步" in (body["steps"][0]["tool_result"] or "")


def test_research_full_fallback_loop(client):
    # LLM 全部走 fallback：graph 的 plan→research→write→review→finalize 闭环仍要完整
    resp = client.post("/api/research/full", json={"task": "总结本项目检索架构", "max_iterations": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("report")
    assert body.get("iterations", 0) >= 1
