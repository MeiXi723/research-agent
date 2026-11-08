"""LangGraph 极简状态图（胶水层，保持极简）。

LangGraph 只做：State 生命周期、节点注册、边与条件路由执行。
业务逻辑（角色、核验、拆解）全部在上层自研代码。

本文件是 LangGraph 责任边界，务必只写调度，不写业务。
"""

import json
import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel

from langgraph.graph import END, StateGraph

from app.multi_agent import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SUB_QUESTION_MAX,
    DEFAULT_SUB_QUESTION_MIN,
)
from app.multi_agent.pipeline_config import DEFAULT_MAX_LOOPS, validate_pipeline_config
from app.multi_agent.role_runtime import RoleRuntime
from app.multi_agent.roles import ResearcherRole, ReviewerRole, WriterRole
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


# ---------------------------------------------------------------------------
# 配置驱动的通用流水线（任意角色、任意编排）——与上面的三角色专用图并存
# ---------------------------------------------------------------------------

def _merge_dict(left: dict, right: dict) -> dict:
    """dict 合并 reducer：outputs/loops 由多个节点累积写入，整字段覆盖会互相冲掉。"""
    return {**(left or {}), **(right or {})}


class GenericPipelineState(TypedDict):
    """通用链式流水线状态。

    outputs/loops 用合并 reducer（与 log 的 operator.add 同理：多节点累积而非覆盖）；
    loops 只由带 route 的节点写入，值为该节点的执行轮数——回跳数 = 轮数 - 1，
    条件路由据此判上限。revision_notes 为失败节点的校验意见，供回跳目标改写用。
    """

    task: str
    outputs: Annotated[dict[str, dict], _merge_dict]
    revision_notes: list[str]
    loops: Annotated[dict[str, int], _merge_dict]
    execution_log: Annotated[list[str], operator.add]


def make_pipeline_initial_state(task: str) -> dict:
    """通用链的入口 state（图经 ainvoke 驱动，API/测试都用它起步）。"""
    return {
        "task": task,
        "outputs": {},
        "revision_notes": [],
        "loops": {},
        "execution_log": [],
    }


def pipeline_iterations(state: dict) -> int:
    """最大回跳数：loops 值是带 route 节点的执行轮数，回跳数 = 轮数 - 1，取全图最大。"""
    loops = state.get("loops") or {}
    return max(loops.values()) - 1 if loops else 0


def _route_action(output: dict, runs: int, route_cfg: dict) -> str:
    """条件边判定（节点记日志与条件路由共用同一纯函数）。

    runs 为该节点含本次在内的执行轮数；已发生的回跳数 = runs - 1，
    故「还有回跳额度」即 runs <= max_loops。节点与路由读到的 runs 一致
    （节点无条件先自增再返回，路由读的是合并后的同一值），纯函数输入相同
    结果必然相同，两处判定不会漂移。
    """
    passed = output.get(route_cfg["pass_field"]) == route_cfg["pass_value"]
    if passed or runs > route_cfg["max_loops"]:
        return "forward"
    return "revise"


def _make_pipeline_node(runtime: RoleRuntime, route_cfg: dict | None, ctx: RoleContext):
    """为单个 RoleRuntime 生成图节点（route_cfg/ctx 闭包捕获，不落模块级全局）。"""

    async def node(state: GenericPipelineState) -> dict:
        output, runtime_log = await runtime.run(
            ctx,
            task=state["task"],
            previous_outputs=dict(state.get("outputs") or {}),
            revision_notes=state.get("revision_notes") or None,
        )
        log = [f"[{runtime.name}] {line}" for line in runtime_log]
        if "_raw" in output:
            log.append(f"[{runtime.name}] JSON 解析失败，走 _raw 兜底")
        updates: dict = {"outputs": {runtime.name: output}}

        if route_cfg is None:
            log.append(f"[{runtime.name}] 完成，输出字段: {list(output)}")
            return {**updates, "execution_log": log}

        runs = (state.get("loops") or {}).get(runtime.name, 0) + 1
        updates["loops"] = {runtime.name: runs}
        action = _route_action(output, runs, route_cfg)
        if action == "revise":
            updates["revision_notes"] = [
                f"[{runtime.name}] 第 {runs} 次校验未通过（{route_cfg['pass_field']}"
                f" 期望 {route_cfg['pass_value']}），上次输出: "
                f"{json.dumps(output, ensure_ascii=False)[:300]}"
            ]
            log.append(
                f"[{runtime.name}] 校验未通过（第 {runs} 次执行），"
                f"回跳 {route_cfg['revise_to']}（第 {runs}/{route_cfg['max_loops']} 次回跳）"
            )
        else:
            # 通过则清掉过期意见；超限强制前进同样清空并留痕
            updates["revision_notes"] = []
            if "_raw" not in output and output.get(route_cfg["pass_field"]) == route_cfg["pass_value"]:
                log.append(f"[{runtime.name}] 校验通过")
            else:
                log.append(
                    f"[{runtime.name}] 校验未通过（第 {runs} 次执行），"
                    f"达回跳上限 {route_cfg['max_loops']}，强制前进"
                )
        updates["execution_log"] = log
        return updates

    return node


def _make_route_router(runtime: RoleRuntime, route_cfg: dict):
    """条件路由：读节点写好的 runs，用同一纯函数决定 forward/revise。"""

    def router(state: GenericPipelineState) -> str:
        output = (state.get("outputs") or {}).get(runtime.name, {})
        runs = (state.get("loops") or {}).get(runtime.name, 0)
        return _route_action(output, runs, route_cfg)

    return router


def _normalize_route(route: dict) -> dict:
    return {
        "pass_field": route["pass_field"],
        "pass_value": route["pass_value"],
        "revise_to": route["revise_to"],
        "max_loops": route.get("max_loops", DEFAULT_MAX_LOOPS),
    }


def build_pipeline(config: dict, ctx: RoleContext) -> "StateGraph":
    """按流水线配置建图。engine=research 委托三角色专用图；engine=chain 建通用链。

    chain 拓扑：roles 顺序线性连边；带 route 的节点改用条件边
    （forward -> 下一节点/END，revise -> revise_to），其余节点普通连边。
    校验失败抛 ValueError（消息带流水线/角色名）。
    """
    validate_pipeline_config(config)
    engine = config.get("engine")

    if engine == "research":
        roles = {
            "researcher": ResearcherRole(),
            "writer": WriterRole(),
            "reviewer": ReviewerRole(),
        }
        return build_graph(roles, ResearchPlanner(), ctx)

    runtimes = [RoleRuntime(spec) for spec in config["roles"]]
    routes = [
        _normalize_route(spec["route"]) if spec.get("route") is not None else None
        for spec in config["roles"]
    ]

    comp = StateGraph(GenericPipelineState)
    for i, rt in enumerate(runtimes):
        comp.add_node(rt.name, _make_pipeline_node(rt, routes[i], ctx))

    comp.set_entry_point(runtimes[0].name)
    for i, rt in enumerate(runtimes):
        next_key = runtimes[i + 1].name if i + 1 < len(runtimes) else END
        route_cfg = routes[i]
        if route_cfg is None:
            comp.add_edge(rt.name, next_key)
        else:
            comp.add_conditional_edges(
                rt.name,
                _make_route_router(rt, route_cfg),
                {"forward": next_key, "revise": route_cfg["revise_to"]},
            )
    return comp.compile()
