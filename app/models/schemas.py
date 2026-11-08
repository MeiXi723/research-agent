from datetime import datetime
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(description="用户问题")
    conversation_id: str | None = Field(default=None, description="对话 ID，为空则新建对话")
    force_route: str | None = Field(
        default=None,
        description="强制路由：chat / tool_research / deep_research；缺省走意图识别路由层",
    )


class AgentStep(BaseModel):
    step: int = Field(description="步骤序号")
    thought: str | None = Field(description="LLM 思考过程")
    tool: str | None = Field(description="调用的工具名")
    tool_input: dict | None = Field(description="工具参数")
    tool_result: str | None = Field(description="工具返回结果")


class SourceInfo(BaseModel):
    type: str = Field(description="来源类型: document / web")
    name: str = Field(description="来源名称")
    content: str = Field(description="来源内容摘要")
    url: str | None = Field(default=None, description="URL（web 来源）")
    page: int | None = Field(default=None, description="页码（document 来源）")


class AskResponse(BaseModel):
    answer: str = Field(description="最终答案")
    steps: list[AgentStep] = Field(description="Agent 每一步记录")
    sources: list[SourceInfo] = Field(description="引用来源")
    conversation_id: str = Field(description="对话 ID")
    route: str | None = Field(default=None, description="意图路由结果: chat / tool_research / deep_research")
    metrics: dict | None = Field(
        default=None,
        description="运行级指标：token 真实用量、工具调用成败分布、上下文压缩事件、延迟",
    )


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    created_at: datetime


class ConversationResponse(BaseModel):
    id: str
    title: str
    step_count: int
    created_at: datetime


class ErrorResponse(BaseModel):
    detail: str
