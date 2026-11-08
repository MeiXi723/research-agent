from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from app.multi_agent import STRUCTURED_CALL_RETRIES
from app.services.llm import AsyncLLMService
from app.tools.base import BaseTool


class RoleContext:
    """角色执行上下文：LLM、可用工具白名单、安全守卫。由 graph 构建方注入。

    设计约束：角色只通过 ctx 与 LLM/工具交互，不直接读写自由文本消息。
    所有工具入参先过 guard.validate_tool_args。
    """

    def __init__(
        self,
        llm: AsyncLLMService,
        tools: dict[str, BaseTool],
        guard: Any,
        token_budget: int = 20000,
    ):
        self.llm = llm
        self.tools = tools
        self.guard = guard
        self.token_budget = token_budget


class BaseRole(ABC):
    """角色插件抽象。三角色都继承它，保证接口一致、可迁移到其他业务。

    子类必须实现：name、system_prompt、run。
    run 接收 RoleContext 和它所需的输入，返回对应 Pydantic 输出对象。
    """

    name: str = ""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @property
    @abstractmethod
    def output_schema(self) -> type[BaseModel]:
        ...

    @abstractmethod
    async def run(self, ctx: RoleContext, **inputs) -> BaseModel:
        ...

    async def _structured_call(
        self, ctx: RoleContext, user_payload: str, fallback: BaseModel
    ) -> BaseModel:
        """调 LLM 并要求结构化输出，失败重试，最终不可用则回退到 fallback。

        这是各角色共用的骨架：任何角色输出过不了 Pydantic 校验，就按失败处理，
        不把格式脏数据带给下游。
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_payload},
        ]
        # 只捕获两类异常：LLM 调用层异常与结构化校验失败（ValidationError）。
        # 其余异常（如编程 bug）原样抛出，避免被静默吞掉掩盖真实错误。
        last_error: str | None = None
        for attempt in range(STRUCTURED_CALL_RETRIES):
            try:
                text = await ctx.llm.chat(messages)
                return self.output_schema.model_validate_json(text)
            except ValidationError as e:
                last_error = f"schema: {e}"
            except Exception as e:
                # 视为 LLM 调用层异常（重试区间已由 llm.chat 指数退避处理）
                last_error = f"llm: {type(e).__name__}: {e}"
        # 重试耗尽：返回 fallback，由调用方负责记入 execution_log
        _ = last_error
        return fallback
