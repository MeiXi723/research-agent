"""LangGraph 极简状态图（胶水层，保持极简）。

LangGraph 只做：State 生命周期、节点注册、边与条件路由执行。
业务逻辑（角色、核验、拆解）全部在上层自研代码。

本文件是 LangGraph 责任边界，务必只写调度，不写业务。
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel

from langgraph.graph import END, StateGraph

from app.multi_agent import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SUB_QUESTION_MAX,
    DEFAULT_SUB_QUESTION_MIN,
)
from app.multi_agent.roles.base_role import BaseRole, RoleContext
from app.multi_agent.schemas import (
    DraftSection,
    EvidenceBatch,
    ReviewVerdict,
    SubQuestion,
    TaskPlan,
)


class MultiAgentState(TypedDict):
    """完整状态。log 用 operator.add 作 reducer -> 追加而非覆盖。

    langgraph 1.x 实测偏差：0.x 时代惯用的 add_messages 会把 str 强转成
    HumanMessage 对象，对纯 str 日志造成类型漂移，故改用 operator.add。
    """

    task: str
    plan: TaskPlan | None
    evidence: dict[str, EvidenceBatch]            # sub_question_id -> batch
    drafts: dict[str, DraftSection]               # sub_question_id -> draft
    verdict: ReviewVerdict | None
    revision_notes: list[str]
    iterations: int                               # 初始 0，write 后自增
    report: str | None
    log: Annotated[list[str], operator.add]       # 执行日志（非静默失败的可观测）
    max_iterations: int                           # 入口 state 注入，route_after_review 消费


def make_initial_state(task: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> dict:
    """组装入口 state。图经 ainvoke 驱动，调用方（API/测试）都用它起步。"""
    return {
        "task": task,
        "plan": None,
        "evidence": {},
        "drafts": {},
        "verdict": None,
        "revision_notes": [],
        "iterations": 0,
        "report": None,
        "log": [],
        "max_iterations": max_iterations,
    }


# ---------- 任务拆解模板 ----------
# 06 的 task_template/research.py 未列入本仓可创建文件清单，收敛到 graph.py：
# planner 与 plan_node 强耦合，放胶水层不引入额外目录。

class ResearchPlanner(BaseRole):
    """把 TaskRequest.task 拆成 TaskPlan（goal + sub_questions + rationale）。"""

    name = "planner"

    @property
    def system_prompt(self) -> str:
        return (
            "你是任务拆解规划员。把一个复杂调研主题拆成 1-6 个子问题，"
            "每个子问题给出拆解理由。子问题要具体、可检索、彼此不重叠。"
            "只输出符合要求的 JSON 结构，不输出其它内容。"
        )

    @property
    def output_schema(self) -> type[BaseModel]:
        return TaskPlan

    async def run(self, ctx: RoleContext, goal: str) -> TaskPlan:
        payload = (
            f"调研主题: {goal}\n"
            "请输出 JSON: {\"goal\": 主题, \"sub_questions\": [{\"id\":\"sq_1\","
            "\"question\": 子问题, \"rationale\": 拆解理由}]}\n"
            f"子问题数量在 {DEFAULT_SUB_QUESTION_MIN}~{DEFAULT_SUB_QUESTION_MAX} 之间。"
        )
        result = await self._structured_call(ctx, payload, fallback_plan(goal))
        if not result.sub_questions:
            return fallback_plan(goal)
        return result


def fallback_plan(goal: str) -> TaskPlan:
    """兜底计划：固定 3 个子问题骨架（plan 降级）。"""
    return TaskPlan(
        goal=goal,
        sub_questions=[
            SubQuestion(id="sq_1", question=goal, rationale="问题概述"),
            SubQuestion(id="sq_2", question=goal + "的现状", rationale="现状梳理"),
            SubQuestion(id="sq_3", question=goal + "的挑战与趋势", rationale="前瞻"),
        ],
    )


# ---------- 节点函数（薄包装，业务下沉到角色/模板） ----------

def route_after_review(state: MultiAgentState) -> str:
    """条件路由（循环退出）：核验通过或达迭代上限 -> finalize，否则 revise。"""
    passed = state.get("verdict") is not None and state["verdict"].overall_passed
    maxed = state.get("iterations", 0) >= state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    return "finalize" if (passed or maxed) else "revise"


def build_graph(roles, planner, ctx) -> "StateGraph":
    """构造并编译状态图。roles/planner/ctx 以闭包捕获注入，不落模块级全局——
    每请求新建图时，并发请求各自持有自己的依赖，互不污染。

    本图节点为 async，langgraph 1.x 下同步 invoke 会直接抛 TypeError（实测），
    必须经 ainvoke 驱动。
    """
    planner_ = planner
    researcher = roles["researcher"]
    writer = roles["writer"]
    reviewer = roles["reviewer"]

    async def plan_node(state: MultiAgentState) -> dict:
        """plan 节点：调 ResearchPlanner，结构化失败时内部回退 fallback_plan。"""
        plan = await planner_.run(ctx, goal=state["task"])
        return {"plan": plan, "log": [f"[plan] 拆解出 {len(plan.sub_questions)} 个子问题"]}

    async def research_node(state: MultiAgentState) -> dict:
        """research 节点：对每个子问题跑 ResearcherRole，收集证据。"""
        evidence: dict[str, EvidenceBatch] = {}
        for sq in state["plan"].sub_questions:
            evidence[sq.id] = await researcher.run(ctx, sub_question=sq)
        total = sum(len(b.evidence) for b in evidence.values())
        return {"evidence": evidence, "log": [f"[research] 共收集 {total} 条证据"]}

    async def write_node(state: MultiAgentState) -> dict:
        """write 节点：调 WriterRole 生成初稿；完成后 iterations += 1。

        revision_notes 来自上一轮 review（首轮为空），Writer 据此改写。
        """
        drafts: dict[str, DraftSection] = {}
        prev_notes = state.get("revision_notes") or None
        for sq in state["plan"].sub_questions:
            batch = state["evidence"].get(sq.id, EvidenceBatch(sub_question_id=sq.id))
            drafts[sq.id] = await writer.run(
                ctx,
                sub_question_id=sq.id,
                evidence=batch,
                revision_notes=prev_notes,
            )
        iterations = state.get("iterations", 0) + 1
        return {
            "drafts": drafts,
            "iterations": iterations,
            "log": [f"[write] 生成 {len(drafts)} 段初稿（第 {iterations} 轮）"],
        }

    async def review_node(state: MultiAgentState) -> dict:
        """review 节点：调 ReviewerRole（fact_checker 在其内部），产出 verdict。"""
        verdict = await reviewer.run(
            ctx, drafts=state["drafts"], evidence_batches=state["evidence"]
        )
        return {
            "verdict": verdict,
            "revision_notes": verdict.revision_notes,
            "log": [f"[review] overall_passed={verdict.overall_passed}，"
                    f"issues={sum(len(r.issues) for r in verdict.per_section)}"],
        }

    def finalize_node(state: MultiAgentState) -> dict:
        """finalize 节点：按 plan 顺序拼接各段初稿成报告。"""
        parts: list[str] = []
        plan = state.get("plan")
        drafts = state.get("drafts", {})
        if plan:
            for sq in plan.sub_questions:
                draft = drafts.get(sq.id)
                if draft and draft.content:
                    parts.append(draft.content)
        report = "\n\n".join(parts)
        return {"report": report, "log": [f"[finalize] 报告 {len(report)} 字"]}

    comp = StateGraph(MultiAgentState)

    comp.add_node("plan", plan_node)
    comp.add_node("research", research_node)
    comp.add_node("write", write_node)
    comp.add_node("review", review_node)
    comp.add_node("finalize", finalize_node)

    comp.set_entry_point("plan")
    comp.add_edge("plan", "research")
    comp.add_edge("research", "write")
    comp.add_edge("write", "review")
    comp.add_conditional_edges(
        "review",
        route_after_review,
        {"revise": "write", "finalize": "finalize"},
    )
    comp.add_edge("finalize", END)

    return comp.compile()
