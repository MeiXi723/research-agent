"""安全守卫：工具入参沙箱校验。

每个角色调用工具前必须过 validate_tool_args；system prompt 由角色自带注入防护台词。
"""

from typing import Any

from app.tools.base import BaseTool


class Guard:
    def __init__(self, tools: dict[str, BaseTool]):
        self.tools = tools

    def validate_tool_args(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """返回错误消息（有错），否则 None（可安全调用）。"""
        tool = self.tools.get(tool_name)
        if tool is None:
            return f"未知工具: {tool_name}"
        err = tool.validate(**args)
        if err:
            return err
        return None
