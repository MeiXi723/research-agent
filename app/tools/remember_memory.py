
from app.services.memory import get_memory
from app.tools.base import BaseTool


class RememberMemoryTool(BaseTool):
    """写入通路（v2.1）：agent 可以沉淀长期记忆。

    写入由 remember_smart 三分支判定——系统自动决定「更新已有记忆 / 新增 /
    丢弃临时提及」，模型不需要自己判断去重。工具描述里明确了「不该写什么」，
    拦住把本轮临时上下文倾倒进长期记忆的滥用（对抗探针 inj10 的防御层之一）。
    """

    name = "remember_memory"
    description = (
        "把值得长期记住的信息写入长期记忆。适用：用户稳定偏好（kind=preference）、"
        "已确认的事实要点（kind=fact）、重要研究结论（kind=conclusion）。"
        "禁止写入：仅本轮任务相关的临时信息、未经验证的猜测、检索到的原始资料——"
        "这些只是当前对话的上下文，写入会被系统丢弃。系统自动判定更新已有记忆还是新增。"
    )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要记住的信息（一句话，自包含，含关键数字与专有名词）",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "preference", "conclusion", "note"],
                        "description": "记忆类型：fact=事实 / preference=用户偏好 / conclusion=结论 / note=其他",
                    },
                },
                "required": ["content", "kind"],
            },
        }

    def validate(self, content: str = "", kind: str = "note") -> str | None:
        if not content or not content.strip():
            return "记忆内容不能为空"
        if kind not in ("fact", "preference", "conclusion", "note"):
            return f"kind 必须是 fact/preference/conclusion/note，收到: {kind}"
        if len(content) > 2000:
            return "记忆内容过长（>2000 字符），请提炼为一句话要点"
        return None

    def call(self, content: str, kind: str = "note"):
        memory = get_memory()
        if memory is None:
            return {"error": "记忆库未初始化"}

        import app.services.llm as llm_module
        llm = llm_module._llm_service  # 运行时读单例（lifespan 回填），judge 用它做中间带判定
        result = memory.remember_smart(content.strip(), kind, llm=llm)
        result["kind"] = kind
        return result
