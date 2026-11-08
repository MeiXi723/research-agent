"""多智能体研究模式（Researcher -> Writer -> Reviewer 循环）。

常量集中在本包而非 app/config：config.py 不在多智能体功能的可修改清单内，
且这些默认值只被本包消费，收敛到这里避免散落各角色模块。
"""

# 结构化输出（Pydantic 校验失败）的最大重试次数
STRUCTURED_CALL_RETRIES = 2
# 单条证据原文截断长度
EVIDENCE_MAX_CHARS = 300
# 每个子问题最多保留的证据条数
TOP_N = 4
# FactCheckResult.passed 的分数阈值
PASS_THRESHOLD = 0.8
# Writer<->Reviewer 默认迭代上限（可被 TaskRequest.max_iterations 覆盖）
DEFAULT_MAX_ITERATIONS = 2
# 任务拆解子问题数量边界
DEFAULT_SUB_QUESTION_MIN = 1
DEFAULT_SUB_QUESTION_MAX = 6

from app.multi_agent.fact_checker import FactChecker  # noqa: E402
from app.multi_agent.guard import Guard  # noqa: E402
from app.multi_agent.schemas import (  # noqa: E402
    DraftSection,
    EvidenceBatch,
    EvidenceItem,
    FactCheckIssue,
    FactCheckResult,
    MultiAgentResponse,
    ReviewVerdict,
    SectionResult,
    SubQuestion,
    TaskPlan,
    TaskRequest,
)

__all__ = [
    "STRUCTURED_CALL_RETRIES",
    "EVIDENCE_MAX_CHARS",
    "TOP_N",
    "PASS_THRESHOLD",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_SUB_QUESTION_MIN",
    "DEFAULT_SUB_QUESTION_MAX",
    "FactChecker",
    "Guard",
    "TaskRequest",
    "TaskPlan",
    "SubQuestion",
    "EvidenceItem",
    "EvidenceBatch",
    "DraftSection",
    "FactCheckIssue",
    "FactCheckResult",
    "ReviewVerdict",
    "SectionResult",
    "MultiAgentResponse",
]
