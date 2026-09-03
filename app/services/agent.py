import json
import time

from app.config import MAX_AGENT_STEPS, MAX_CONTEXT_TOKENS, WINDOW_TURNS
from app.models.schemas import AgentStep, SourceInfo
from app.services.context import estimate_tokens, manage_context
from app.services.llm import _llm_service
from app.services.tracer import Tracer
from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool
from app.tools.search_docs import SearchDocsTool
from app.tools.web_search import WebSearchTool

SYSTEM_PROMPT = """你是 AI 研究助手，能通过工具搜索信息并回答问题。

规则：
1. 回答必须基于工具返回的结果，不要编造
2. 如果本地文档和网络搜索都找不到相关信息，如实告知用户
3. 工具返回的每条结果都要核实相关性后再引用
4. 数学计算使用 calculator 工具，不要心算"""


def _build_tools() -> dict[str, BaseTool]:
    return {
        "calculator": CalculatorTool(),
        "web_search": WebSearchTool(),
        "search_docs": SearchDocsTool(),
    }


def _extract_sources(tool_result) -> list[SourceInfo]:
    sources = []
    if not isinstance(tool_result, list):
        return sources
    for item in tool_result:
        if isinstance(item, dict) and "text" in item:
            sources.append(SourceInfo(
                type="document",
                name=item.get("document_name", ""),
                content=item["text"][:200],
                page=item.get("page"),
            ))
        elif isinstance(item, dict) and "url" in item:
            sources.append(SourceInfo(
                type="web",
                name=item.get("title", ""),
                content=item.get("content", ""),
                url=item.get("url"),
            ))
    return sources


def _summarize(messages: list[dict]) -> str:
    text = "\n".join(
        m.get("content", "")
        for m in messages
        if isinstance(m.get("content"), str) and m["content"]
    )
    if not text:
        return "（无历史内容）"
    try:
        prompt = f"请用不超过两句话总结以下对话的关键信息与结论，保留具体数值：\n{text}"
        return _llm_service.chat([{"role": "user", "content": prompt}], temperature=0.0)
    except Exception:
        return "（历史对话过长，已截断）"


def _log_step(tracer, step: int, action: str, tokens: int, duration_ms: float, success: bool) -> None:
    if tracer is None:
        return
    tracer.log({
        "step": step,
        "action": action,
        "tokens": tokens,
        "duration_ms": round(duration_ms, 2),
        "success": success,
    })


def run_agent(
    question: str,
    max_steps: int | None = None,
    history: list[dict] | None = None,
    tracer: Tracer | None = None,
) -> tuple[str, list[AgentStep], list[SourceInfo]]:
    if max_steps is None:
        max_steps = MAX_AGENT_STEPS

    tools = _build_tools()
    tools_schema = [t.to_schema() for t in tools.values()]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    steps: list[AgentStep] = []
    sources: list[SourceInfo] = []

    start_time = time.time()

    for step_idx in range(max_steps):
        if time.time() - start_time > 120:
            return "思考超时，请简化问题后重试", steps, sources

        messages = manage_context(messages, MAX_CONTEXT_TOKENS, WINDOW_TURNS, _summarize)

        msg = _llm_service.chat_with_tools(messages, tools_schema)

        if not msg.tool_calls:
            answer = msg.content or "无法生成回答"
            _log_step(tracer, step_idx + 1, "answer", estimate_tokens(messages), 0, True)
            return answer, steps, sources

        tool_results: dict[str, str] = {}
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool = tools.get(tool_name)
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            step_start = time.time()
            if tool is None:
                result_obj = {"error": f"未知工具: {tool_name}"}
            else:
                err = tool.validate(**tool_args)
                if err:
                    result_obj = {"error": err}
                else:
                    try:
                        result_obj = tool.call(**tool_args)
                    except Exception as e:
                        result_obj = {"error": str(e)}
            duration_ms = (time.time() - step_start) * 1000

            result_str = json.dumps(result_obj, ensure_ascii=False)[:500] if result_obj is not None else None
            tool_results[tc.id] = result_str or "工具执行失败"

            steps.append(AgentStep(
                step=step_idx + 1,
                thought=f"调用 {tool_name} 工具",
                tool=tool_name,
                tool_input=tool_args,
                tool_result=result_str,
            ))
            sources.extend(_extract_sources(result_obj))
            _log_step(tracer, step_idx + 1, tool_name, estimate_tokens(messages), duration_ms, not isinstance(result_obj, dict) or "error" not in result_obj)

        messages.append(msg)
        for tc in msg.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_results[tc.id],
            })

    return "达到最大步数限制，请简化问题", steps, sources
