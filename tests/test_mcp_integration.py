"""MCP 层集成测试：真 stdio 子进程（.venv python 拉起 mcp_servers/research_server.py）。

覆盖：工具发现（>=3）、真实调用、错误路径（未知单位 / 缺必填参数）、
子进程被杀后调用不挂死、shutdown 后无孤儿线程。所有 fixture 用 try/finally 保证清理。

Windows 找子进程 PID 用 PowerShell Get-CimInstance 按命令行匹配（wmic 在新系统已移除）；
stdio_client 不暴露子进程对象，按命令行匹配是唯一不侵入 provider 的手段。
"""

import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from app.mcp_client.provider import (
    NAMESPACE_SEP,
    MCPProviderManager,
    parse_servers_spec,
)
from app.tools.base import BaseTool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
SERVER_SCRIPT = str(PROJECT_ROOT / "mcp_servers" / "research_server.py")
TOOL_TIMEOUT = 10.0


def _make_manager() -> MCPProviderManager:
    spec = [("research", [VENV_PYTHON, SERVER_SCRIPT])]
    return MCPProviderManager(spec, tool_timeout=TOOL_TIMEOUT)


def _assert_no_provider_threads() -> None:
    leftovers = [t.name for t in threading.enumerate() if t.name.startswith("mcp-server-")]
    assert not leftovers, f"孤儿线程残留: {leftovers}"


def _shutdown_quietly(manager: MCPProviderManager) -> None:
    try:
        asyncio.run(manager.shutdown_all())
    except RuntimeError:
        # server 进程已死等场景下 aclose 可能失败；线程是否清干净由孤儿断言兜底
        pass
    _assert_no_provider_threads()


@pytest.fixture
def provider():
    manager = _make_manager()
    asyncio.run(manager.start_all())
    try:
        yield manager
    finally:
        _shutdown_quietly(manager)


def _find_server_pids() -> list[int]:
    ps_script = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.CommandLine -like '*research_server.py*' } "
        "| Select-Object -ExpandProperty ProcessId"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True, timeout=30, check=False,
        # PowerShell 输出走系统本地代码页（中文系统 GBK），只提取 ASCII 数字行，
        # 解码失败一律 replace，不让 reader 线程崩 UnicodeDecodeError
        encoding="utf-8", errors="replace",
    )
    return [int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()]


# ---------------------------------------------------------------------------
# 工具发现
# ---------------------------------------------------------------------------


def test_provider_discovers_research_tools(provider):
    tools = provider.get_tools()
    assert isinstance(tools, dict)
    assert len(tools) >= 3
    for expected in ("text_stats", "unit_convert", "now_tz"):
        name = f"research{NAMESPACE_SEP}{expected}"
        assert name in tools, f"缺少工具 {name}，实际: {sorted(tools)}"
        tool = tools[name]
        assert isinstance(tool, BaseTool)
        function = tool.describe()
        # OpenAI function schema 形状
        assert function["name"] == name
        assert function["description"]
        params = function["parameters"]
        assert params["type"] == "object"
        assert "properties" in params and "required" in params
        assert tool.to_schema() == {"type": "function", "function": function}


# ---------------------------------------------------------------------------
# 真实调用
# ---------------------------------------------------------------------------


def test_call_text_stats(provider):
    tool = provider.get_tools()[f"research{NAMESPACE_SEP}text_stats"]
    out = tool.call(text="hello world\nsecond line")
    stats = json.loads(out)
    assert stats == {"characters": 23, "words": 4, "lines": 2}


def test_call_unit_convert(provider):
    tools = provider.get_tools()
    convert = tools[f"research{NAMESPACE_SEP}unit_convert"]

    km_to_mile = json.loads(convert.call(value=1.0, from_unit="km", to_unit="mile"))
    assert km_to_mile["value"] == pytest.approx(0.6213711922, rel=1e-9)
    assert km_to_mile["from_unit"] == "km" and km_to_mile["to_unit"] == "mile"

    lb = json.loads(convert.call(value=1.0, from_unit="kg", to_unit="lb"))
    assert lb["value"] == pytest.approx(2.2046226218, rel=1e-9)

    f = json.loads(convert.call(value=100.0, from_unit="c", to_unit="f"))
    assert f["value"] == pytest.approx(212.0)
    f2 = json.loads(convert.call(value=-40.0, from_unit="c", to_unit="f"))
    assert f2["value"] == pytest.approx(-40.0)


def test_call_now_tz(provider):
    tool = provider.get_tools()[f"research{NAMESPACE_SEP}now_tz"]
    payload = json.loads(tool.call(tz_name="UTC"))
    assert payload["timezone"] == "UTC"
    assert "+00:00" in payload["now"]


def test_call_unknown_unit_returns_error(provider):
    convert = provider.get_tools()[f"research{NAMESPACE_SEP}unit_convert"]
    start = time.monotonic()
    out = convert.call(value=1.0, from_unit="km", to_unit="lb")
    elapsed = time.monotonic() - start
    payload = json.loads(out)
    assert "error" in payload
    assert elapsed < TOOL_TIMEOUT  # 错误路径要即时返回，不许挂到硬闸


def test_call_missing_required_arg_returns_error(provider):
    # 绕过 validate 直接 call（模拟 LLM 传参不完整），服务器校验失败 -> error 语义
    tool = provider.get_tools()[f"research{NAMESPACE_SEP}text_stats"]
    payload = json.loads(tool.call())
    assert "error" in payload


def test_call_extra_unknown_param_is_ignored(provider):
    # mcp 2.3.0 服务器实测行为：schema 外参数被忽略而非报错（记录契约边界）
    tool = provider.get_tools()[f"research{NAMESPACE_SEP}text_stats"]
    stats = json.loads(tool.call(text="hi", bogus_param=1))
    assert stats == {"characters": 2, "words": 1, "lines": 1}


def test_validate_blocks_missing_args(provider):
    tool = provider.get_tools()[f"research{NAMESPACE_SEP}text_stats"]
    assert tool.validate(text="x") is None
    message = tool.validate()
    assert message is not None and "text" in message


# ---------------------------------------------------------------------------
# 子进程被杀：调用不得挂死
# ---------------------------------------------------------------------------


def test_server_process_killed_call_does_not_hang():
    manager = _make_manager()
    asyncio.run(manager.start_all())
    try:
        tools = manager.get_tools()
        pids = _find_server_pids()
        assert pids, "按命令行未找到 research_server 子进程"
        kill = subprocess.run(
            ["taskkill", "/PID", str(pids[0]), "/F", "/T"],
            capture_output=True, text=True, timeout=30, check=False,
            # 中文 Windows 的 taskkill 输出是 GBK（"成功: 已终止..."），utf-8 严格解码会崩 reader 线程
            encoding="utf-8", errors="replace",
        )
        assert kill.returncode == 0, kill.stderr
        time.sleep(0.5)  # 给管道 EOF 传播留一点时间，避免竞态假阳性

        tool = tools[f"research{NAMESPACE_SEP}text_stats"]
        start = time.monotonic()
        try:
            out = tool.call(text="x")
            payload = json.loads(out)
            outcome = "error" in payload
        except Exception:  # noqa: BLE001 - 契约断言点：任何可捕获异常（断管/超时）都算"不挂死"
            # BrokenResourceError / TimeoutError 等均可：可捕获即满足契约
            outcome = True
        elapsed = time.monotonic() - start
        assert outcome, f"杀进程后调用既无 error 语义也未抛异常: {out!r}"
        # 硬闸 TOOL_TIMEOUT 之外还挂死 = 实现缺陷（正常断管应立即失败或按时超时）
        assert elapsed < TOOL_TIMEOUT + 2, f"杀进程后调用挂死 {elapsed:.1f}s"
    finally:
        _shutdown_quietly(manager)


# ---------------------------------------------------------------------------
# 生命周期：启动失败与孤儿检测
# ---------------------------------------------------------------------------


def test_start_all_failure_reports_and_cleans_up():
    # 让 python 立刻退出（脚本不存在）-> initialize 失败 -> start_all 汇报
    manager = MCPProviderManager(
        [("bad", [VENV_PYTHON, str(PROJECT_ROOT / "mcp_servers" / "no_such_server.py")])],
        tool_timeout=TOOL_TIMEOUT,
    )
    with pytest.raises(RuntimeError, match="bad"):
        asyncio.run(manager.start_all())
    _shutdown_quietly(manager)


def test_shutdown_all_no_orphan_threads_and_idempotent():
    manager = _make_manager()
    asyncio.run(manager.start_all())
    assert len(manager.get_tools()) >= 3
    asyncio.run(manager.shutdown_all())
    _assert_no_provider_threads()
    asyncio.run(manager.shutdown_all())  # 幂等：第二次不抛


def test_parse_servers_spec_matches_fixture_shape():
    # 消费方链路：config 原始串 -> parse_servers_spec -> MCPProviderManager
    # 每个带空格/反斜杠的参数各自引号包裹（实测：同对引号包两个路径会被并成一个 token）
    spec = parse_servers_spec(f'research="{VENV_PYTHON}" "{SERVER_SCRIPT}"')
    assert spec == [("research", [VENV_PYTHON, SERVER_SCRIPT])]
    manager = MCPProviderManager(spec, tool_timeout=TOOL_TIMEOUT)
    assert manager._entries[0].command == [VENV_PYTHON, SERVER_SCRIPT]
