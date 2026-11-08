from app.services.memory import get_memory
from app.tools.base import BaseTool


class RecallMemoryTool(BaseTool):
    """让 agent 能按语义找回历史记忆——v1 的 key-value 记忆只能精确 key 查。"""

    name = "recall_memory"
    description = (
        "按语义检索长期记忆（历史研究结论、用户偏好、已确认的事实要点）。"
        "回答前如果怀疑之前研究过相关主题，先调这个工具。"
    )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "语义检索语句"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 3"},
                },
                "required": ["query"],
            },
        }

    def call(self, query: str, top_k: int = 3):
        memory = get_memory()
        if memory is None:
            return {"error": "记忆库未初始化"}
        return memory.recall(query, top_k)
