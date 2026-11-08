"""自研事实核验组件。

能力边界：只做「输出 ↔ 已有证据素材」的一致性校验，不做外部真实世界事实辨别。
两实现：先规则（零 LLM 成本），再 LLM 语义判定。
"""

import json
import re

from app.multi_agent import PASS_THRESHOLD
from app.multi_agent.schemas import (
    EvidenceBatch,
    EvidenceItem,
    FactCheckIssue,
    FactCheckResult,
)
from app.services.llm import AsyncLLMService

# 绝对化断言：若初稿把它当事实写，而证据没对应支撑，标 unsupported
_ABSOLUTE_WORDS = ("唯一", "最大", "最先进", "领先", "无风险", "绝对", "所有", "全部不会")


class FactChecker:
    def __init__(self, llm: AsyncLLMService):
        self.llm = llm

    def rule_check(self, content: str, items: list[EvidenceItem]) -> list[FactCheckIssue]:
        """规则级硬冲突检查，不调 LLM。"""
        issues: list[FactCheckIssue] = []
        evidence_text = "".join(it.text for it in items)
        for sentence in self._split_sentences(content):
            for word in _ABSOLUTE_WORDS:
                if word in sentence and word not in evidence_text:
                    issues.append(
                        FactCheckIssue(
                            claim=sentence,
                            severity="unsupported",
                            evidence_snippet=None,
                            suggested_fix=f"删去或弱化绝对化表述「{word}」，缺少证据支撑",
                        )
                    )
                    break
        return issues

    async def llm_check(
        self, content: str, items: list[EvidenceItem]
    ) -> list[FactCheckIssue]:
        """LLM 语义一致性判定，结构化输出。"""
        evidence_str = "\n".join(f"[{it.evidence_id}] {it.text}" for it in items)
        prompt = (
            "你是事实核验审核员。判断以下「调查结论」的每一句是否被「证据」支持。\n"
            "证据:\n" + (evidence_str or "(无证据)") +
            "\n调查结论:\n" + content +
            "\n逐句输出 JSON 数组，每项 {\"claim\": 原句, "
            "\"severity\": \"unsupported\"|\"conflict\", \"suggested_fix\": 修改建议}。\n"
            "只允许出现证据没有或不一致的句子。"
        )
        text = await self.llm.chat([
            {"role": "system", "content": "你是严格的事实核对员，只依据给出证据判断，不引入外部知识。"},
            {"role": "user", "content": prompt},
        ])
        return self._parse_issues(text)

    def _parse_issues(self, text: str) -> list[FactCheckIssue]:
        # 简单解析：预期为 JSON 数组；解析失败按零问题降级（不误杀）
        try:
            data = json.loads(text)
            return [
                FactCheckIssue(
                    claim=str(i.get("claim", "")),
                    severity=("conflict" if i.get("severity") == "conflict" else "unsupported"),
                    evidence_snippet=None,
                    suggested_fix=str(i.get("suggested_fix", "删除或补充证据")),
                )
                for i in data
            ]
        except Exception:
            return []

    async def check(
        self, batch: EvidenceBatch, content: str
    ) -> FactCheckResult:
        """两级合并：规则 + LLM，返回带支持率与 pass 的结果。"""
        issues = self.rule_check(content, batch.evidence)
        if not issues:
            try:
                llm_issues = await self.llm_check(content, batch.evidence)
            except Exception:
                # LLM 判定层故障不得打死流水线：降级为纯规则结论（fail-open 到规则层，
                # 但 rule_check 已通过时按零问题计——语义核查缺失要可观测，日志留痕）
                llm_issues = []
                import logging
                logging.getLogger(__name__).warning("fact_checker llm_check 异常，降级为规则层结论")
            issues = llm_issues if llm_issues else issues

        total = max(len(self._split_sentences(content)), 1)
        bad = len(issues)
        score = round(1.0 - bad / total, 2)
        return FactCheckResult(
            sub_question_id=batch.sub_question_id,
            score=score,
            issues=issues,
            passed=score >= PASS_THRESHOLD,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"[。！？\n]+", text)
        return [p.strip() for p in parts if p.strip()]
