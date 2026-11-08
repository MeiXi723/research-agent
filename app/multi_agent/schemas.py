"""多智能体研究模式的消息 Schema（Pydantic v2，移植自 06-multi-agent-collab）。"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---- 请求 / 响应 ----
class TaskRequest(BaseModel):
    task: str = Field(description="调研主题，如\"2026 年宁德时代动力电池竞争格局\"")
    max_iterations: int = Field(default=2, ge=1, le=5, description="Writer↔Reviewer 最多迭代几轮")
    token_budget: int = Field(default=20000, ge=1000, description="单角色上下文预算")


# ---- 任务拆解 ----
class SubQuestion(BaseModel):
    id: str                            # "sq_1"
    question: str
    rationale: str                     # 为什么拆这个子问题


class TaskPlan(BaseModel):
    goal: str
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=6)


# ---- Researcher 输出（带证据素材）----
class EvidenceItem(BaseModel):
    evidence_id: str                   # 全局唯一证据 ID："{sub_question_id}_{seq}"
    text: str                          # 证据原文（截断到 ~300 字）
    source_type: Literal["document", "web"]
    source_name: str
    url: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


class EvidenceBatch(BaseModel):
    sub_question_id: str
    evidence: list[EvidenceItem] = Field(default_factory=list)


# ---- Writer 输出 ----
class DraftSection(BaseModel):
    sub_question_id: str
    content: str
    used_evidence_ids: list[str] = Field(default_factory=list)


# ---- fact_checker 输出（Reviewer 依赖）----
class FactCheckIssue(BaseModel):
    claim: str                         # 初稿原句
    severity: Literal["unsupported", "conflict"]
    evidence_snippet: Optional[str] = None
    suggested_fix: str                 # 修改建议


class FactCheckResult(BaseModel):
    sub_question_id: str
    score: float = Field(ge=0, le=1)
    issues: list[FactCheckIssue] = Field(default_factory=list)
    passed: bool


class ReviewVerdict(BaseModel):
    overall_passed: bool
    per_section: list[FactCheckResult] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)


# ---- 响应 ----
class SectionResult(BaseModel):
    sub_question_id: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    draft: str = ""
    check: Optional[FactCheckResult] = None


class MultiAgentResponse(BaseModel):
    report: str
    plan: TaskPlan
    sections: list[SectionResult] = Field(default_factory=list)
    iterations: int
    passed: bool
    execution_log: list[str] = Field(default_factory=list)
