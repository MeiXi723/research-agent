"""配置驱动的通用角色执行器（RoleRuntime）。

与 roles/base_role.py 的分工：BaseRole 面向固定业务 schema（Pydantic 输出对象），
RoleRuntime 面向任意流水线 JSON——prompt/工具白名单/输出字段全部来自配置，
输出统一为自由 dict；JSON 解析失败重试 1 次，仍失败兜底 {"_raw": 原文}，
不把非结构化文本静默丢给下游（调用方可见 _raw 标记）。

工具白名单语义：只有配置 tools 列表内的工具会经 asyncio.to_thread 调用
（沿用 researcher 的线程池模式）；注册表里白名单外的工具一律不碰，
白名单内但注册表没有的工具记一条拒绝日志。
"""

import asyncio
import json
import re

from app.multi_agent.roles.base_role import RoleContext

# JSON 解析失败的最大尝试次数：首次 + 重试 1 次（对齐 STRUCTURED_CALL_RETRIES 语义）
JSON_PARSE_RETRIES = 2

_FENCE_START = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_FENCE_END = re.compile(r"\s*```$")


def parse_json_object(text: str) -> dict | None:
    """把 LLM 文本解析成 JSON 对象；容忍 markdown 代码围栏；非 dict 视为失败。"""
    cleaned = _FENCE_END.sub("", _FENCE_START.sub("", text.strip())).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def textualize_output(output: dict) -> str:
    """把节点输出 dict 文本化（report 拼接用）：str 直取，list 逐行，其余 JSON 化。"""
    parts: list[str] = []
    for key, value in output.items():
        if key == "_raw":
            parts.append(str(value))
        elif isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(
                item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                for item in value
            )
        else:
            parts.append(json.dumps(value, ensure_ascii=False))
    return "\n".join(part for part in parts if part).strip()


class RoleRuntime:
    """单个配置角色的执行器：无业务 schema，输入输出都是自由 dict。"""

    def __init__(self, spec: dict):
        self.name = spec["name"]
        self.system_prompt = spec["system_prompt"]
        self.tool_names = list(spec.get("tools") or [])
        self.output_fields = list(spec.get("output_fields") or [])

    async def run(
        self,
        ctx: RoleContext,
        task: str,
        previous_outputs: dict[str, dict],
        revision_notes: list[str] | None = None,
    ) -> tuple[dict, list[str]]:
        """执行角色，返回 (输出 dict, 运行日志行)。

        输入 = task + 前序所有节点输出 + revision_notes（若有）；
        工具结果（若有白名单工具）一并进 payload。
        """
        tool_results, runtime_log = await self._run_allowed_tools(ctx, task)
        payload = self._build_payload(task, previous_outputs, revision_notes, tool_results)
        output = await self._llm_json(ctx, payload)
        return output, runtime_log

    async def _run_allowed_tools(
        self, ctx: RoleContext, query: str
    ) -> tuple[list[dict], list[str]]:
        """只调配置白名单内的工具，返回 (工具结果, 日志行)。

        白名单 = 配置 tools 列表：注册表中白名单外的工具天然不会被遍历到；
        白名单内但注册表缺失的记拒绝日志，不抛错。
        """
        results: list[dict] = []
        log: list[str] = []
        for name in self.tool_names:
            tool = ctx.tools.get(name)
            if tool is None:
                log.append(f"工具 {name} 不在可用注册表，已拒绝调用")
                results.append({"tool": name, "error": "unknown tool"})
                continue
            err = ctx.guard.validate_tool_args(name, {"query": query})
            if err:
                log.append(f"工具 {name} 参数校验未过：{err}")
                results.append({"tool": name, "error": err})
                continue
            try:
                raw = await asyncio.to_thread(tool.call, query=query)
            except Exception as e:  # 单工具失败降级为错误记录，不打死角色主流程
                log.append(f"工具 {name} 调用异常：{type(e).__name__}: {e}")
                results.append({"tool": name, "error": f"{type(e).__name__}: {e}"})
                continue
            log.append(f"工具 {name} 返回 {len(raw) if isinstance(raw, list) else 1} 条结果")
            results.append({"tool": name, "result": raw})
        return results, log

    def _build_payload(
        self,
        task: str,
        previous_outputs: dict[str, dict],
        revision_notes: list[str] | None,
        tool_results: list[dict],
    ) -> str:
        lines = [f"# 任务\n{task}"]
        if previous_outputs:
            lines.append("# 前序节点输出\n" + json.dumps(previous_outputs, ensure_ascii=False, indent=2))
        if revision_notes:
            lines.append(
                "# 上轮校验意见（请据此修改）\n" + "\n".join(f"- {note}" for note in revision_notes)
            )
        if tool_results:
            lines.append("# 工具结果\n" + json.dumps(tool_results, ensure_ascii=False, indent=2))
        lines.append("请只输出一个 JSON 对象，不要输出其它内容。")
        return "\n\n".join(lines)

    async def _llm_json(self, ctx: RoleContext, payload: str) -> dict:
        """调 LLM 要 JSON 对象：解析失败重试 1 次，仍失败兜底 {"_raw": 最后原文}。

        LLM 调用层异常与解析失败同样处理（重试后兜底），与 base_role 的
        fail-soft 语义一致；异常信息进 _raw 保证可观测。
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": payload},
        ]
        last_text = ""
        for _ in range(JSON_PARSE_RETRIES):
            try:
                text = await ctx.llm.chat(messages)
            except Exception as e:
                last_text = f"llm 调用失败: {type(e).__name__}: {e}"
                continue
            parsed = parse_json_object(text)
            if parsed is not None:
                return parsed
            last_text = text
            # 重试时附纠偏提示，提高第二次命中的概率
            messages = [*messages,
                        {"role": "user",
                         "content": "以上输出无法解析为 JSON 对象，请重新回答，只输出一个 JSON 对象。"}]
        return {"_raw": last_text}
