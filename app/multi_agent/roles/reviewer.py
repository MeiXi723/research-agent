from pydantic import BaseModel

from app.multi_agent.schemas import (
    DraftSection,
    EvidenceBatch,
    FactCheckResult,
    ReviewVerdict,
)
from app.multi_agent.fact_checker.checker import FactChecker
from app.multi_agent.roles.base_role import BaseRole, RoleContext


class ReviewerRole(BaseRole):
    """Reviewer: 调 fact_checker 对比每段初稿与对应证据，产出核验结论。

    仅负责「初稿 vs 素材」一致性核验，不引入新事实。
    输出 ReviewVerdict（全局 passed + 每段 FactCheckResult + 汇总修改意见）。
    """

    name = "reviewer"

    @property
    def system_prompt(self) -> str:
        # Reviewer 不直接调 LLM 生成内容，prompt 仅作占位；核验逻辑全在 fact_checker
        return "你是事实核验审核员，依据素材判断初稿是否被证据支持。"

    @property
    def output_schema(self) -> type[BaseModel]:
        return ReviewVerdict

    async def run(
        self,
        ctx: RoleContext,
        drafts: dict[str, DraftSection],
        evidence_batches: dict[str, EvidenceBatch],
    ) -> ReviewVerdict:
        checker = FactChecker(ctx.llm)
        per_section: list[FactCheckResult] = []
        revision_notes: list[str] = []
        for sq_id, draft in drafts.items():
            batch = evidence_batches.get(sq_id, EvidenceBatch(sub_question_id=sq_id))
            result = await checker.check(batch, draft.content)
            per_section.append(result)
            for issue in result.issues:
                revision_notes.append(
                    f"[{sq_id}] {issue.claim or '一段表述'} → {issue.suggested_fix}"
                )
        overall_passed = all(r.passed for r in per_section)
        return ReviewVerdict(
            overall_passed=overall_passed,
            per_section=per_section,
            revision_notes=revision_notes,
        )
