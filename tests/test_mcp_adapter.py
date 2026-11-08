"""MCP 适配层单元测试：全 mock（不拉起真 stdio 子进程）。

mock 策略：session 用 MagicMock/AsyncMock，工具定义用真实 mcp.types.Tool 构造
（比纯 MagicMock 更贴近 wire 形状，字段名由 2.3.0 实测确认）；事件循环用真后台线程，
以验证 MCPTool.call 的跨线程 run_coroutine_threadsafe 调度真实成立。
"""

import asyncio
import json
import threading
import time

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from app.mcp_client.provider import (
    NAMESPACE_SEP,
    MCPProviderManager,
    MCPTool,
    parse_servers_spec,
)
from app.tools.base import BaseTool


def _make_tool(name: str = "text_stats", required: list[str] | None = None) -> Tool:
    required = required if required is not None else ["text"]
    return Tool(
        name=name,
        description="统计文本的字符数、词数、行数。",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "待统计文本"}},
            "required": required,
        },
    )


def _make_result(text: str, is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=is_error)


@pytest.fixture
def loop_runner():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="unit-mcp-loop", daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# parse_servers_spec
# ---------------------------------------------------------------------------


def test_parse_servers_spec_multi_server():
    raw = "research=.venv/Scripts/python.exe mcp_servers/research_server.py; calc=python calc.py --flag 1"
    spec = parse_servers_spec(raw)
    assert spec == [
        ("research", [".venv/Scripts/python.exe", "mcp_servers/research_server.py"]),
        ("calc", ["python", "calc.py", "--flag", "1"]),
    ]


def test_parse_servers_spec_quoted_windows_backslash_path():
    # 实测（py3.12 shlex posix 模式）：双引号内反斜杠仅在转义 " 或 \ 时生效，
    # 引号包裹是带空格反斜杠路径的唯一安全写法；裸反斜杠会被当转义符吃掉
    raw = r'ws="C:\Program Files\my server.py"; fwd=C:/x/y.py'
    spec = parse_servers_spec(raw)
    assert spec == [
        ("ws", [r"C:\Program Files\my server.py"]),
        ("fwd", ["C:/x/y.py"]),
    ]


def test_parse_servers_spec_skips_empty_segments():
    assert parse_servers_spec("  ;; a=python x.py; ") == [("a", ["python", "x.py"])]
    assert parse_servers_spec("") == []


@pytest.mark.parametrize(
    "raw",
    [
        "python x.py",  # 缺 name= 前缀
        "research=",  # name 后无命令
        "=python x.py",  # 空 name
    ],
)
def test_parse_servers_spec_invalid_raises(raw):
    with pytest.raises(ValueError):
        parse_servers_spec(raw)


# ---------------------------------------------------------------------------
# MCPTool
# ---------------------------------------------------------------------------


def test_mcp_tool_describe_is_openai_function_schema(loop_runner):
    tool = _make_tool("text_stats")
    adapter = MCPTool("research", tool, session=None, loop=loop_runner, timeout=5.0)
    expected = {
        "name": f"research{NAMESPACE_SEP}text_stats",
        "description": "统计文本的字符数、词数、行数。",
        "parameters": tool.input_schema,
    }
    assert adapter.describe() == expected
    # parameters 原样透传服务器侧 JSON Schema
    assert adapter.describe()["parameters"] is tool.input_schema
    assert adapter.to_schema() == {"type": "function", "function": expected}
    assert isinstance(adapter, BaseTool)


def test_mcp_tool_validate_required(loop_runner):
    tool = _make_tool(required=["text"])
    adapter = MCPTool("research", tool, session=None, loop=loop_runner, timeout=5.0)
    assert adapter.validate(text="hi") is None
    assert adapter.validate() is not None
    assert "text" in adapter.validate()
    # 必填参数显式传 None 也算缺
    assert adapter.validate(text=None) is not None


def test_mcp_tool_call_success(loop_runner):
    from unittest.mock import AsyncMock, MagicMock

    session = MagicMock()
    session.call_tool = AsyncMock(return_value=_make_result('{"characters": 23, "words": 4, "lines": 2}'))
    adapter = MCPTool("research", _make_tool(), session=session, loop=loop_runner, timeout=5.0)

    out = adapter.call(text="hello 世界\nsecond")
    assert out == '{"characters": 23, "words": 4, "lines": 2}'
    # 调用的是远程原名（不带命名空间），参数原样透传
    session.call_tool.assert_awaited_once_with("text_stats", arguments={"text": "hello 世界\nsecond"})


def test_mcp_tool_call_joins_multiple_content_blocks(loop_runner):
    from unittest.mock import AsyncMock, MagicMock

    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[
                TextContent(type="text", text="line1"),
                TextContent(type="text", text="line2"),
            ],
            is_error=False,
        )
    )
    adapter = MCPTool("research", _make_tool(), session=session, loop=loop_runner, timeout=5.0)
    assert adapter.call(text="x") == "line1\nline2"


def test_mcp_tool_call_is_error_returns_error_json(loop_runner):
    # mcp 2.x 行为：服务器工具抛异常 -> is_error=True 的结果而非 raise
    from unittest.mock import AsyncMock, MagicMock

    session = MagicMock()
    session.call_tool = AsyncMock(return_value=_make_result("Error executing tool unit_convert", is_error=True))
    adapter = MCPTool("research", _make_tool(), session=session, loop=loop_runner, timeout=5.0)

    out = adapter.call(text="x")
    payload = json.loads(out)
    assert "error" in payload
    assert "unit_convert" in payload["error"]


def test_mcp_tool_call_transport_error_returns_error_json(loop_runner):
    # 连接层异常（server 死亡、管道断裂）不应炸掉 agent 循环，转 error JSON
    from unittest.mock import AsyncMock, MagicMock

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=BrokenPipeError("pipe closed"))
    adapter = MCPTool("research", _make_tool(), session=session, loop=loop_runner, timeout=5.0)

    payload = json.loads(adapter.call(text="x"))
    assert "error" in payload
    assert "pipe closed" in payload["error"]


def test_mcp_tool_call_timeout_raises_timeout_error(loop_runner):
    # future.result 硬闸：server 假死不回包也要按时抛 TimeoutError
    from unittest.mock import AsyncMock, MagicMock

    async def hang(name, arguments=None):
        await asyncio.sleep(30)

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=hang)
    adapter = MCPTool("research", _make_tool(), session=session, loop=loop_runner, timeout=0.2)

    start = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter.call(text="x")
    elapsed = time.monotonic() - start
    assert elapsed < 5  # 按硬闸超时，而不是等满 30s 协程
    session.call_tool.assert_awaited_once()


# ---------------------------------------------------------------------------
# MCPProviderManager 聚合
# ---------------------------------------------------------------------------


def _fake_entry(name: str, tool_names: list[str]) -> object:
    from app.mcp_client.provider import _ServerEntry

    entry = _ServerEntry(name, ["python", "x.py"])
    entry.tools = [_make_tool(n, required=[]) for n in tool_names]
    entry.started = True
    entry.thread = threading.Thread(target=lambda: None, name=f"fake-{name}")  # 未启动，仅满足“已配置”检查
    entry.loop = None  # 聚合路径不执行协程，无需真 loop
    entry.session = None
    return entry


def test_provider_get_tools_aggregates_namespaces():
    manager = MCPProviderManager(
        [("research", ["python", "a.py"]), ("calc", ["python", "b.py"])], tool_timeout=9.0
    )
    manager._entries = [_fake_entry("research", ["text_stats"]), _fake_entry("calc", ["evaluate"])]
    tools = manager.get_tools()

    assert set(tools) == {
        f"research{NAMESPACE_SEP}text_stats",
        f"calc{NAMESPACE_SEP}evaluate",
    }
    assert all(isinstance(t, BaseTool) for t in tools.values())
    assert all(t.describe()["name"] == key for key, t in tools.items())
    # 同名工具跨 server 不冲突：命名空间隔离，远程调用名保持原名
    manager._entries.append(_fake_entry("other", ["text_stats"]))
    tools = manager.get_tools()
    assert tools[f"research{NAMESPACE_SEP}text_stats"].remote_tool_name == "text_stats"
    assert tools[f"other{NAMESPACE_SEP}text_stats"].remote_tool_name == "text_stats"


def test_provider_get_tools_skips_unstarted_entries():
    manager = MCPProviderManager([("research", ["python", "a.py"])], tool_timeout=9.0)
    entry = _fake_entry("research", ["text_stats"])
    entry.started = False
    manager._entries = [entry]
    assert manager.get_tools() == {}


def test_provider_get_tools_before_start_raises():
    manager = MCPProviderManager([("research", ["python", "a.py"])], tool_timeout=9.0)
    with pytest.raises(RuntimeError, match="start_all"):
        manager.get_tools()


def test_provider_duplicate_server_name_raises():
    with pytest.raises(ValueError, match="重复"):
        MCPProviderManager([("a", ["python", "x.py"]), ("a", ["python", "y.py"])])


def test_provider_timeout_defaults_from_config():
    from unittest.mock import patch

    manager = MCPProviderManager([("a", ["python", "x.py"])])
    assert manager._tool_timeout == 30.0  # config.MCP_TOOL_TIMEOUT 默认值
    with patch("app.config.MCP_TOOL_TIMEOUT", 7.5):
        assert MCPProviderManager([("a", ["python", "x.py"])])._tool_timeout == 7.5


def test_shutdown_all_idempotent_and_orphan_free():
    # 未启动线程的 manager 也要能干净 shutdown（幂等 + 无孤儿）
    manager = MCPProviderManager([("a", ["python", "x.py"])], tool_timeout=5.0)
    manager._entries = [_fake_entry("a", ["text_stats"])]
    import asyncio as _asyncio

    _asyncio.run(manager.shutdown_all())
    assert not manager._entries[0].thread.is_alive()
    _asyncio.run(manager.shutdown_all())  # 第二次不抛
