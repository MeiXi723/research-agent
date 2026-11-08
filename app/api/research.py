"""多智能体研究 API：POST /api/research/full。

handler 是同步 def，FastAPI 把它调度到线程池线程——线程内没有运行中的
事件循环，asyncio.run 安全；图每次请求新建（无状态），并发请求互不污染。
"""

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
from app.multi_agent.graph import (
    ResearchPlanner,
    build_graph,
    build_pipeline,
    make_initial_state,
    make_pipeline_initial_state,
    pipeline_iterations,
)
from app.multi_agent.guard import Guard
from app.multi_agent.pipeline_config import load_pipeline, list_pipelines
from app.multi_agent.role_runtime import textualize_output
from app.multi_agent.roles import ResearcherRole, ReviewerRole, RoleContext, WriterRole
from app.multi_agent.schemas import (
    MultiAgentResponse,
    SectionResult,
    TaskRequest,
)
from app.services.llm import AsyncLLMService
from app.tools.search_docs import SearchDocsTool
from app.tools.web_search import WebSearchTool

router = APIRouter(prefix="/api/research")


def build_default_tools() -> dict:
    """每次请求新建工具实例。本仓两个检索工具的 call(query) 无副作用、
    无凭据时返回空列表，实例化与调用都不联网。
    """
    tools = [SearchDocsTool(), WebSearchTool()]
    return {t.name: t for t in tools}


async def _run_multi_agent(req: TaskRequest) -> MultiAgentResponse:
    # 读 lifespan 回填的模块单例（而非每次新建）：e2e/AI_MOCK 才能整体替换 LLM 层
    from app.services.llm import _async_llm_service
    llm = _async_llm_service or AsyncLLMService(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
    )
    tools = build_default_tools()
    ctx = RoleContext(llm=llm, tools=tools, guard=Guard(tools), token_budget=req.token_budget)
    roles = {"researcher": ResearcherRole(), "writer": WriterRole(), "reviewer": ReviewerRole()}
    graph = build_graph(roles, ResearchPlanner(), ctx)

    state = await graph.ainvoke(make_initial_state(req.task, req.max_iterations))

    plan = state["plan"]
    evidence = state.get("evidence", {})
    drafts = state.get("drafts", {})
    verdict = state.get("verdict")
    per_section = {r.sub_question_id: r for r in (verdict.per_section if verdict else [])}
    sections = [
        SectionResult(
            sub_question_id=sq.id,
            evidence=evidence[sq.id].evidence if sq.id in evidence else [],
            draft=drafts[sq.id].content if sq.id in drafts else "",
            check=per_section.get(sq.id),
        )
        for sq in plan.sub_questions
    ]
    return MultiAgentResponse(
        report=state.get("report") or "",
        plan=plan,
        sections=sections,
        iterations=state.get("iterations", 0),
        passed=verdict.overall_passed if verdict else False,
        execution_log=state.get("log", []),
    )


@router.post("/full", response_model=MultiAgentResponse)
def run_full_research(req: TaskRequest) -> MultiAgentResponse:
    return asyncio.run(_run_multi_agent(req))


# ---------------------------------------------------------------------------
# 配置驱动流水线端点（GET /api/pipelines、POST /api/pipelines/{name}/run）
# ---------------------------------------------------------------------------
# 注册方式说明：main.py（不在本功能可改动清单内）只 include 本模块的 router
# （prefix=/api/research），而 FastAPI include_router 会把父前缀拼进所有
# APIRoute——顶栏路径 /api/pipelines 无法经 @router.get 声明产生。实测
# （fastapi 0.115.6）include_router 对 starlette 原生 Route 只拼 include 参数
# prefix（默认空串），故用 add_route 注册完整路径；请求仍走同一 app 的
# 中间件与异常处理链，行为与 APIRoute 端点一致。

async def _run_pipeline(name: str, req: TaskRequest) -> dict:
    """按名跑流水线：research 引擎复用现有三角色链路，chain 引擎跑通用图。"""
    config = load_pipeline(name)
    if config["engine"] == "research":
        return (await _run_multi_agent(req)).model_dump()

    # 读 lifespan 回填的模块单例（而非每次新建）：e2e/测试才能整体替换 LLM 层
    from app.services.llm import _async_llm_service
    llm = _async_llm_service or AsyncLLMService(
        api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
    )
    tools = build_default_tools()
    ctx = RoleContext(llm=llm, tools=tools, guard=Guard(tools), token_budget=req.token_budget)
    graph = build_pipeline(config, ctx)
    state = await graph.ainvoke(make_pipeline_initial_state(req.task))

    outputs = state.get("outputs", {})
    last_role = config["roles"][-1]["name"]
    return {
        "report": textualize_output(outputs.get(last_role, {})),
        "outputs": outputs,
        "execution_log": state.get("execution_log", []),
        "iterations": pipeline_iterations(state),
    }


async def _run_pipeline_http(request: Request) -> JSONResponse:
    name = request.path_params["name"]
    try:
        req = TaskRequest.model_validate(await request.json())
    except (ValidationError, json.JSONDecodeError) as e:
        return JSONResponse({"detail": str(e)}, status_code=422)
    try:
        return JSONResponse(await _run_pipeline(name, req))
    except FileNotFoundError as e:
        return JSONResponse({"detail": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)


def _list_pipelines_http(request: Request) -> JSONResponse:
    return JSONResponse(list_pipelines())


router.add_route("/api/pipelines", _list_pipelines_http, methods=["GET"])
router.add_route("/api/pipelines/{name}/run", _run_pipeline_http, methods=["POST"])
