"""MCP 客户端适配层：把 stdio 子进程 MCP server 的工具适配成本仓 BaseTool 协议。

职责边界：
- MCPProviderManager：每个 server 一条后台线程 + 独立事件循环，线程内一个 serve
  协程持有 stdio_client/ClientSession 的完整生命周期（connect -> 挂起等 stop 事件 ->
  同 task 内关闭），shutdown 置位 stop_event 后 join，并对线程做孤儿检测（残留即 raise）。
- MCPTool：同步 call（ReAct 循环是同步代码），内部
  asyncio.run_coroutine_threadsafe(session.call_tool(...), loop) + future.result(timeout)。

mcp 2.3.0 版本漂移记录（相对 mcp 1.x 的官方文档示例）：
- FastMCP 已改名 MCPServer（mcp.server.mcpserver），服务器侧适配见 mcp_servers/research_server.py；
- ClientSession.read_timeout_seconds 参数类型从 timedelta 变为 float；
- 工具抛异常时 call_tool 不 raise，而是返回 is_error=True 的 CallToolResult
  （text block 只有 "Error executing tool <name>"，具体异常留在服务器 stderr）。
"""

import asyncio
import json
import shlex
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack

from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from app.tools.base import BaseTool

# 命名空间分隔符：{server}__{tool}，与 05-lightagent 的命名空间约定一致
NAMESPACE_SEP = "__"

# 启动期给每个 server 的 connect 预算（进程拉起 + initialize + list_tools）
_STARTUP_TIMEOUT = 30.0
# shutdown 时单 server 的退出预算（stdio_client.__aexit__ 要 terminate 子进程并等退出）
_JOIN_TIMEOUT = 15.0


def parse_servers_spec(raw: str) -> list[tuple[str, list[str]]]:
    """解析 MCP_SERVERS 配置串："name=command arg1 arg2; name2=..."。

    用 shlex.split(posix=True) 处理引号。Windows 反斜杠的实测边界（Python 3.12）：
    - 未加引号的裸路径：`C:\\Users\\x.py` 中的 `\\U`、`\\x` 被当转义符吃掉 -> 禁止裸写反斜杠路径；
    - 双引号内：反斜杠仅在转义 `"` 或 `\\` 时生效，`"C:\\path with space\\x.py"` 原样保留 -> 带空格
      的路径必须加引号，或一律用正斜杠（Windows 的 Python 与 subprocess 都接受正斜杠）。
    """
    spec: list[tuple[str, list[str]]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens = shlex.split(chunk, posix=True)
        if not tokens:
            continue
        head, sep, rest = tokens[0].partition("=")
        if not sep or not head.strip():
            raise ValueError(f"MCP_SERVERS 片段缺少 name= 前缀: {chunk!r}")
        command = [rest] if rest else []
        command.extend(tokens[1:])
        if not command or not command[0]:
            raise ValueError(f"MCP_SERVERS 片段缺少命令: {chunk!r}")
        spec.append((head.strip(), command))
    return spec


class _ServerEntry:
    """单个 server 的连接状态；字段只被所属线程与 manager 主线程按约定读写。"""

    def __init__(self, name: str, command: list[str]):
        self.name = name
        self.command = list(command)
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: ClientSession | None = None
        self.tools: list = []
        self.ready = threading.Event()
        self.start_error: BaseException | None = None
        self.started = False
        # connect 与 context 持有必须在同一个 task（anyio cancel scope 的 task 亲和性），
        # 因此退出不走外部 aclose，而是由 shutdown 线程安全地置位该事件，让 serve
        # 协程自己走出 async with（__aexit__ 在进入它的 task 内执行）
        self.stop_event: asyncio.Event | None = None


class MCPTool(BaseTool):
    """一个 MCP server 工具的 BaseTool 适配。

    call() 是同步的：跨线程把 call_tool 协程投递到 server 专属事件循环，
    future.result(timeout) 是硬闸——即使 server 假死不回包也会按时抛 TimeoutError，
    由 agent 循环统一捕获。协程本身无法从外部强杀，随 shutdown 时 loop 停止而终结。
    """

    def __init__(self, server_name: str, tool, session: ClientSession, loop, timeout: float):
        self._server_name = server_name
        self._tool = tool
        self._session = session
        self._loop = loop
        self._timeout = timeout
        self.name = f"{server_name}{NAMESPACE_SEP}{tool.name}"
        self.description = getattr(tool, "description", "") or ""

    @property
    def remote_tool_name(self) -> str:
        return self._tool.name

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": getattr(self._tool, "input_schema", None) or {
                "type": "object", "properties": {}, "required": [],
            },
        }

    def validate(self, **kwargs) -> str | None:
        required = (getattr(self._tool, "input_schema", None) or {}).get("required", [])
        missing = [k for k in required if kwargs.get(k) is None]
        if missing:
            return f"缺少必填参数: {', '.join(missing)}"
        return None

    def call(self, **kwargs) -> str:
        coro = self._session.call_tool(self._tool.name, arguments=dict(kwargs))
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            result = future.result(timeout=self._timeout)
        except FutureTimeoutError:
            # 协程尚未开始执行时可以取消；已在跑的只能随 loop 生命周期终结
            future.cancel()
            raise TimeoutError(
                f"MCP 工具 {self.name} 超时（>{self._timeout}s）"
            ) from None
        except Exception as e:  # noqa: BLE001 - 契约要求：任何调用期异常都转 error 语义返回
            # 投递/执行期的传输层异常（server 进程死亡、管道断裂等）：
            # 转成带 error 字段的 JSON 串返回，与 agent.py 的 json.dumps 处理兼容
            return json.dumps({"error": f"MCP 调用异常: {e}"}, ensure_ascii=False)

        blocks = getattr(result, "content", None) or []
        text = "\n".join(
            block.text for block in blocks if getattr(block, "text", None) is not None
        )
        if getattr(result, "is_error", False):
            # mcp 2.x：服务器工具异常不 raise，而是 is_error=True 的结果
            return json.dumps(
                {"error": text or f"MCP 工具 {self.name} 执行失败"}, ensure_ascii=False
            )
        return text


class MCPProviderManager:
    """管理多个 stdio MCP server：线程/事件循环生命周期 + 工具聚合发现。"""

    def __init__(self, servers_spec: list[tuple[str, list[str]]], tool_timeout: float | None = None):
        if tool_timeout is None:
            # 延迟导入，让测试可以注入超时而不依赖 config 的加载副作用
            from app.config import MCP_TOOL_TIMEOUT
            tool_timeout = MCP_TOOL_TIMEOUT
        self._tool_timeout = float(tool_timeout)
        self._entries: list[_ServerEntry] = []
        seen: set[str] = set()
        for name, command in servers_spec:
            if name in seen:
                raise ValueError(f"MCP server 名重复: {name}")
            seen.add(name)
            self._entries.append(_ServerEntry(name, command))
        self._shutdown = False

    async def start_all(self) -> None:
        """拉起全部 server 线程并等待 connect 完成；任一失败抛 RuntimeError（附全部失败原因）。

        已成功启动的 server 不回滚，交给 shutdown_all 统一清理（调用方 finally 里调）。
        """
        if self._entries and any(e.thread for e in self._entries):
            raise RuntimeError("MCPProviderManager 已经启动过，禁止重复 start_all")
        for entry in self._entries:
            thread = threading.Thread(
                target=self._thread_main, args=(entry,),
                name=f"mcp-server-{entry.name}", daemon=True,
            )
            entry.thread = thread
            thread.start()

        failures: list[str] = []
        for entry in self._entries:
            if not entry.ready.wait(timeout=_STARTUP_TIMEOUT):
                failures.append(f"{entry.name}: connect 超时（>{_STARTUP_TIMEOUT}s）")
                continue
            if entry.start_error is not None:
                failures.append(f"{entry.name}: {entry.start_error!r}")
        if failures:
            raise RuntimeError("MCP server 启动失败: " + "; ".join(failures))

    def _thread_main(self, entry: _ServerEntry) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        entry.loop = loop
        entry.stop_event = asyncio.Event()
        params = StdioServerParameters(command=entry.command[0], args=list(entry.command[1:]))

        async def serve() -> None:
            # 整个生命周期（connect -> 持有 -> 退出）都在这个 task 里，
            # 满足 anyio cancel scope 的 task 亲和性
            async with AsyncExitStack() as stack:
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                listing = await session.list_tools()
                entry.session = session
                entry.tools = list(listing.tools)
                entry.started = True
                entry.ready.set()
                # call_tool 协程由 MCPTool.call 经 run_coroutine_threadsafe 投到本 loop，
                # 与本 task 并发执行；stop_event 置位后从这里正常退出并关闭全部 context
                await entry.stop_event.wait()

        try:
            loop.run_until_complete(serve())
        except BaseException as e:  # noqa: BLE001 - 线程顶层：任何异常（含 CancelledError）都必须落进 start_error 供主线程上报
            entry.start_error = e
        finally:
            entry.ready.set()
            try:
                loop.close()
            except Exception:  # noqa: BLE001, S110 - close 失败无恢复动作，线程即将退出
                pass

    def get_tools(self) -> dict[str, BaseTool]:
        """聚合所有 server 的工具，键为命名空间化名 {server}__{tool}。"""
        if not any(e.thread for e in self._entries):
            raise RuntimeError("MCPProviderManager 尚未 start_all，无法获取工具")
        tools: dict[str, BaseTool] = {}
        for entry in self._entries:
            if not entry.started:
                continue
            for tool in entry.tools:
                adapter = MCPTool(
                    server_name=entry.name,
                    tool=tool,
                    session=entry.session,
                    loop=entry.loop,
                    timeout=self._tool_timeout,
                )
                tools[adapter.name] = adapter
        return tools

    async def shutdown_all(self) -> None:
        """置位各 server 的 stop 事件（serve 协程在同 task 内关闭全部 context）、
        join 线程；孤儿线程残留即 raise。幂等。"""
        if self._shutdown:
            return
        self._shutdown = True
        errors: list[str] = []
        for entry in self._entries:
            if entry.stop_event is not None and entry.loop is not None:
                try:
                    # asyncio.Event 非线程安全，跨线程置位必须走 call_soon_threadsafe
                    entry.loop.call_soon_threadsafe(entry.stop_event.set)
                except RuntimeError:
                    pass  # loop 已停/已关
            thread = entry.thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=_JOIN_TIMEOUT)
            # 孤儿检测：shutdown 语义要求线程全部死亡，任何残留都是资源泄漏，必须显式失败
            if thread is not None and thread.is_alive():
                errors.append(f"{entry.name}: provider 线程 shutdown 后仍存活（孤儿线程）")
        if errors:
            raise RuntimeError("MCP shutdown 异常: " + "; ".join(errors))
