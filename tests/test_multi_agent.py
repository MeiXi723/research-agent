"""多智能体研究模式单元测试（全 mock 离线，零网络）。

依赖注入约定：graph/roles 不读 app.services.llm 的模块级全局，
统一由 build_graph(roles, planner, ctx) 工厂注入——测试直接塞 fake llm/tools，
无需 patch app.services.llm._async_llm_service。
"""

import asyncio
import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.multi_agent import guard as guard_module
from app.multi_agent.fact_checker import FactChecker
from app.multi_agent.graph import (
    ResearchPlanner,
    build_graph,
    fallback_plan,
    make_initial_state,
    route_after_review,
)
from app.multi_agent.roles import ResearcherRole, ReviewerRole, RoleContext, WriterRole
from app.multi_agent.schemas import (
    EvidenceBatch,
    EvidenceItem,
    ReviewVerdict,
)
from app.tools.base import BaseTool

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """假 AsyncLLMService：按 system prompt 关键词路由响应，记录全部调用。"""

    def __init__(self, planner_reply="not-a-json", writer_content="依据证据的论述",
                 check_reply="[]"):
        self.calls: list[list[dict]] = []
        self._planner_reply = planner_reply
        self._writer_content = writer_content
        self._check_reply = check_reply

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        system = messages[0]["content"]
        if "任务拆解规划员" in system:
            return self._planner_reply
        if "报告撰写员" in system:
            m = re.search(r"子问题:\s*(sq_\d+)", messages[-1]["content"])
            sq_id = m.group(1) if m else "sq_1"
            return json.dumps({"sub_question_id": sq_id, "content": self._writer_content})
        if "事实核对员" in system:
            return self._check_reply
        return "not-a-json"


class FakeTool(BaseTool):
    def __init__(self, name="web_search", results=None, validate_err=None):
        self.name = name
        self.description = "fake tool"
        self._results = results if results is not None else []
        self._validate_err = validate_err

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def call(self, query: str):
        return self._results

    def validate(self, query: str = ""):
        return self._validate_err


def build_ctx(llm, tools):
    return RoleContext(llm=llm, tools=tools, guard=guard_module.Guard(tools))


def build_roles():
    return {"researcher": ResearcherRole(), "writer": WriterRole(), "reviewer": ReviewerRole()}


# ---------------------------------------------------------------------------
# fallback_plan / planner 兜底
# ---------------------------------------------------------------------------


def test_fallback_plan_produces_three_subquestions():
    plan = fallback_plan("测试主题")
    assert [sq.id for sq in plan.sub_questions] == ["sq_1", "sq_2", "sq_3"]
    assert plan.goal == "测试主题"
    assert all(sq.question and sq.rationale for sq in plan.sub_questions)


def test_planner_falls_back_on_invalid_llm_output():
    llm = FakeLLM(planner_reply="不是 JSON")
    plan = asyncio.run(ResearchPlanner().run(build_ctx(llm, {}), goal="测试主题"))
    assert len(plan.sub_questions) == 3
    # 重试 2 次（STRUCTURED_CALL_RETRIES）都失败后才回退
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# 整图串联（plan -> research -> write -> review -> finalize）
# ---------------------------------------------------------------------------


def test_pipeline_report_nonempty():
    llm = FakeLLM(
        # planner 返回非法 JSON -> 走 fallback_plan（3 子问题）
        planner_reply="not-a-json",
        writer_content="根据证据，情况如下。",
        check_reply="[]",
    )
    tools = {"web_search": FakeTool(results=[{"title": "来源A", "content": "证据内容A"}])}
    graph = build_graph(build_roles(), ResearchPlanner(), build_ctx(llm, tools))

    state = asyncio.run(graph.ainvoke(make_initial_state("测试主题", max_iterations=2)))

    assert state["plan"] is not None
    assert len(state["plan"].sub_questions) == 3
    assert state["report"], "report 不应为空"
    assert len(state["drafts"]) == 3
    assert state["verdict"].overall_passed is True
    assert state["iterations"] == 1
    # planner 收到非法 JSON 重试 2 次后回退 fallback_plan + writer 3 次
    # + review 走 LLM 判定 3 次（规则层零命中）
    assert len(llm.calls) == 2 + 3 + 3
    assert any("finalize" in line for line in state["log"])


def test_pipeline_iterates_when_not_passed_then_maxed():
    # checker 一直判不过 -> 跑满 2 轮后经 maxed 分支退出
    llm = FakeLLM(
        planner_reply="not-a-json",
        writer_content="依据证据的论述。",
        check_reply=json.dumps([
            {"claim": "依据证据的论述。", "severity": "unsupported",
             "suggested_fix": "补充证据"}
        ]),
    )
    graph = build_graph(build_roles(), ResearchPlanner(), build_ctx(llm, {}))

    state = asyncio.run(graph.ainvoke(make_initial_state("测试主题", max_iterations=2)))

    assert state["iterations"] == 2
    assert state["verdict"].overall_passed is False
    assert state["report"], "maxed 退出也应拼出报告"


# ---------------------------------------------------------------------------
# route_after_review 两分支（+ revise 补充）
# ---------------------------------------------------------------------------


def _state_with_verdict(passed: bool, iterations: int, max_iterations: int = 2) -> dict:
    verdict = ReviewVerdict(overall_passed=passed, per_section=[], revision_notes=[])
    return {"verdict": verdict, "iterations": iterations, "max_iterations": max_iterations}


def test_route_after_review_passed_branch():
    assert route_after_review(_state_with_verdict(passed=True, iterations=1)) == "finalize"


def test_route_after_review_maxed_branch():
    assert route_after_review(_state_with_verdict(passed=False, iterations=2)) == "finalize"


def test_route_after_review_revise_branch():
    assert route_after_review(_state_with_verdict(passed=False, iterations=1)) == "revise"


# ---------------------------------------------------------------------------
# fact_checker 规则层（零 LLM）
# ---------------------------------------------------------------------------


def test_fact_checker_rule_hits_absolute_words_without_llm():
    llm = FakeLLM()
    checker = FactChecker(llm)
    items = [EvidenceItem(evidence_id="sq_1_0", text="市场正在增长",
                          source_type="document", source_name="doc")]
    result = asyncio.run(checker.check(
        EvidenceBatch(sub_question_id="sq_1", evidence=items),
        "本产品是市场上最大的玩家。",
    ))

    assert len(result.issues) == 1
    assert result.issues[0].severity == "unsupported"
    assert "最大" in result.issues[0].suggested_fix
    assert result.passed is False
    # 规则层命中后不再触发 LLM：零 LLM 路径
    assert llm.calls == []


def test_fact_checker_rule_passes_when_evidence_supports():
    llm = FakeLLM(check_reply="[]")
    checker = FactChecker(llm)
    items = [EvidenceItem(evidence_id="sq_1_0", text="它是最大的玩家",
                          source_type="document", source_name="doc")]
    result = asyncio.run(checker.check(
        EvidenceBatch(sub_question_id="sq_1", evidence=items),
        "它是最大的玩家。",
    ))
    # 证据中有该词 -> 规则零命中，但走 LLM 判定（返回空数组）-> 通过
    assert result.issues == []
    assert result.passed is True
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# guard 透传
# ---------------------------------------------------------------------------


def test_guard_passes_through_tool_validate_error():
    tool = FakeTool(name="web_search", validate_err="搜索关键词不能为空")
    g = guard_module.Guard({"web_search": tool})
    assert g.validate_tool_args("web_search", {"query": ""}) == "搜索关键词不能为空"


def test_guard_unknown_tool_and_ok_case():
    tool = FakeTool(name="web_search")
    g = guard_module.Guard({"web_search": tool})
    assert "未知工具" in g.validate_tool_args("nope", {"query": "x"})
    assert g.validate_tool_args("web_search", {"query": "x"}) is None


def test_researcher_skips_tool_when_guard_rejects():
    llm = FakeLLM()
    tools = {"web_search": FakeTool(name="web_search", results=[{"content": "x"}],
                                    validate_err="搜索关键词不能为空")}
    from app.multi_agent.schemas import SubQuestion
    batch = asyncio.run(ResearcherRole().run(
        build_ctx(llm, tools), sub_question=SubQuestion(id="sq_1", question="q", rationale="r")
    ))
    assert batch.evidence == []


# ---------------------------------------------------------------------------
# API 端点（自建 app，patch llm 工厂与工具，零网络）
# ---------------------------------------------------------------------------


def test_api_full_endpoint():
    from app.api import research as api_research

    fake_llm = FakeLLM(planner_reply="not-a-json", writer_content="端点冒烟结论。")

    def fake_llm_factory(**kwargs):
        return fake_llm

    def fake_tools():
        return {"web_search": FakeTool(results=[{"title": "来源", "content": "证据"}])}

    monkeypatch_llm = pytest.MonkeyPatch()
    monkeypatch_llm.setattr(api_research, "AsyncLLMService", fake_llm_factory)
    monkeypatch_llm.setattr(api_research, "build_default_tools", fake_tools)
    try:
        app = FastAPI()
        app.include_router(api_research.router)
        client = TestClient(app)
        resp = client.post("/api/research/full",
                           json={"task": "端点冒烟主题", "max_iterations": 1})
    finally:
        monkeypatch_llm.undo()

    assert resp.status_code == 200
    data = resp.json()
    assert data["report"], "端点 report 不应为空"
    assert len(data["plan"]["sub_questions"]) == 3
    assert data["passed"] is True
    assert data["iterations"] == 1
    assert data["sections"], "sections 应按子问题展开"
