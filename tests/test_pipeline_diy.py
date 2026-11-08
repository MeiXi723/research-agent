"""配置驱动流水线（任意角色任意编排）单元测试——全 mock 离线，零网络。

覆盖：配置加载/校验错误路径（空 roles、revise_to 不存在、name 穿越）、
RoleRuntime（JSON 重试/兜底、工具白名单）、chain 建图（2/5 角色线性、
route 回跳与强制前进）、research 引擎委托、API 端点。
"""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.multi_agent import guard as guard_module
from app.multi_agent.graph import (
    build_pipeline,
    make_initial_state,
    make_pipeline_initial_state,
    pipeline_iterations,
)
from app.multi_agent.pipeline_config import load_pipeline, list_pipelines, validate_pipeline_config
from app.multi_agent.role_runtime import parse_json_object, textualize_output
from app.multi_agent.roles import RoleContext
from app.tools.base import BaseTool

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class ScriptLLM:
    """按 system_prompt 关键字路由的假 LLM：每个关键字一个回复队列，
    耗尽后重复最后一项；记录全部 messages 便于断言重试次数与 payload。
    """

    def __init__(self, script: dict[str, list[str]]):
        self.script = script
        self.calls: list[list[dict]] = []
        self._cursor = {kw: 0 for kw in script}

    async def chat(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        system = messages[0]["content"]
        for kw, queue in self.script.items():
            if kw in system:
                reply = queue[min(self._cursor[kw], len(queue) - 1)]
                self._cursor[kw] += 1
                return reply
        raise AssertionError(f"ScriptLLM 收到未脚本化的角色 prompt: {system[:50]}")


class RecordingTool(BaseTool):
    def __init__(self, name="web_search", results=None, validate_err=None):
        self.name = name
        self.description = "fake tool"
        self.results = results if results is not None else []
        self.validate_err = validate_err
        self.calls: list[str] = []

    def describe(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate(self, query: str = ""):
        return self.validate_err

    def call(self, query: str):
        self.calls.append(query)
        return self.results


def build_ctx(llm, tools):
    return RoleContext(llm=llm, tools=tools, guard=guard_module.Guard(tools))


def run_chain(config, llm, tools):
    ctx = build_ctx(llm, tools)
    graph = build_pipeline(config, ctx)
    return asyncio.run(graph.ainvoke(make_pipeline_initial_state("配置链冒烟任务")))


def two_role_config():
    return {
        "name": "two_role_test",
        "description": "测试：两角色链",
        "engine": "chain",
        "roles": [
            {"name": "planner", "system_prompt": "你是规划员TWO，把任务拆成要点，输出 JSON {\"points\": [...]}。",
             "tools": [], "output_fields": ["points"]},
            {"name": "writer", "system_prompt": "你是撰写员TWO，输出 JSON {\"draft\": \"...\"}。",
             "tools": [], "output_fields": ["draft"]},
        ],
    }


def tool_role_config(tool_names):
    return {
        "name": "tool_test",
        "description": "测试：工具白名单",
        "engine": "chain",
        "roles": [
            {"name": "searcher", "system_prompt": "你是检索员TL，输出 JSON {\"findings\": [...]}。",
             "tools": tool_names, "output_fields": ["findings"]},
        ],
    }


def three_role_config(max_loops):
    return {
        "name": "route_test",
        "description": "测试：带回跳的链",
        "engine": "chain",
        "roles": [
            {"name": "planner", "system_prompt": "你是规划员RT，输出 JSON {\"points\": [...]}。",
             "tools": [], "output_fields": ["points"]},
            {"name": "analyst", "system_prompt": "你是分析师RT，输出 JSON {\"analysis\": \"...\"}。",
             "tools": [], "output_fields": ["analysis"]},
            {"name": "reviewer",
             "system_prompt": "你是审核员RT，检查可靠性，输出 JSON {\"passed\": true/false, \"notes\": \"...\"}。",
             "tools": [], "output_fields": ["passed", "notes"],
             "route": {"pass_field": "passed", "pass_value": True,
                       "revise_to": "analyst", "max_loops": max_loops}},
        ],
    }


# ---------------------------------------------------------------------------
# 配置校验错误路径
# ---------------------------------------------------------------------------


def test_validate_rejects_empty_roles():
    with pytest.raises(ValueError, match="roles"):
        validate_pipeline_config({"name": "x", "engine": "chain", "roles": []})


def test_validate_rejects_unknown_revise_to():
    config = three_role_config(max_loops=2)
    config["roles"][2]["route"]["revise_to"] = "ghost"
    with pytest.raises(ValueError, match="revise_to"):
        validate_pipeline_config(config)


def test_validate_rejects_pass_field_not_in_output_fields():
    config = three_role_config(max_loops=2)
    config["roles"][2]["route"]["pass_field"] = "ok"
    with pytest.raises(ValueError, match="pass_field"):
        validate_pipeline_config(config)


def test_validate_rejects_duplicate_role_names():
    config = two_role_config()
    config["roles"][1]["name"] = "planner"
    with pytest.raises(ValueError, match="重复"):
        validate_pipeline_config(config)


def test_validate_rejects_bad_engine():
    with pytest.raises(ValueError, match="engine"):
        validate_pipeline_config({"name": "x", "engine": "dag", "roles": []})


def test_build_pipeline_rejects_invalid_config():
    with pytest.raises(ValueError):
        build_pipeline({"name": "x", "engine": "chain", "roles": []}, build_ctx(None, {}))


# ---------------------------------------------------------------------------
# load_pipeline / list_pipelines
# ---------------------------------------------------------------------------


def test_load_pipeline_rejects_traversal_and_illegal_names():
    for bad in ["../../etc", "..\\..\\windows", "Research", "res earch", "", "res/earch", "r.json"]:
        with pytest.raises(ValueError):
            load_pipeline(bad)


def test_load_pipeline_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pipeline("no_such_pipeline")


def test_load_pipeline_bundled_configs():
    research = load_pipeline("research")
    assert research["engine"] == "research"
    assert research["default"] is True

    diy = load_pipeline("diy_example")
    assert diy["engine"] == "chain"
    assert [r["name"] for r in diy["roles"]] == ["planner", "searcher", "analyst", "reviewer"]
    route = diy["roles"][-1]["route"]
    assert route["pass_field"] == "passed"
    assert route["revise_to"] == "analyst"
    assert route["max_loops"] == 2


def test_list_pipelines_contains_bundled():
    items = {item["name"]: item for item in list_pipelines()}
    assert {"research", "diy_example"} <= set(items)
    assert items["research"]["engine"] == "research"
    assert items["diy_example"]["roles"] == ["planner", "searcher", "analyst", "reviewer"]


# ---------------------------------------------------------------------------
# RoleRuntime：JSON 解析重试/兜底
# ---------------------------------------------------------------------------


def test_llm_invalid_json_retried_then_raw_fallback():
    llm = ScriptLLM({"检索员TL": ["这不是JSON", "还不是JSON"]})
    state = run_chain(tool_role_config([]), llm, {})
    assert len(llm.calls) == 2  # 首次 + 重试 1 次
    assert state["outputs"]["searcher"] == {"_raw": "还不是JSON"}
    # 重试时追加了纠偏提示
    assert "JSON" in llm.calls[1][-1]["content"]


def test_llm_json_second_attempt_succeeds():
    llm = ScriptLLM({"检索员TL": ["oops", json.dumps({"findings": ["ok"]})]})
    state = run_chain(tool_role_config([]), llm, {})
    assert len(llm.calls) == 2
    assert state["outputs"]["searcher"] == {"findings": ["ok"]}


def test_parse_json_object_handles_fences_and_non_dict():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object("[1, 2]") is None  # 非 dict 视为失败
    assert parse_json_object(" nonsense ") is None


def test_textualize_output_flattens_common_shapes():
    assert textualize_output({"analysis": "结论A", "confidence": 0.9}) == "结论A\n0.9"
    assert textualize_output({"points": ["p1", "p2"]}) == "p1\np2"
    assert textualize_output({"_raw": "原始文本"}) == "原始文本"
    assert textualize_output({"findings": [{"k": 1}]}) == '{"k": 1}'


# ---------------------------------------------------------------------------
# RoleRuntime：工具白名单
# ---------------------------------------------------------------------------


def test_whitelisted_tool_called_via_thread_and_payload_contains_results():
    tool = RecordingTool(results=[{"title": "来源", "content": "关键证据内容"}])
    llm = ScriptLLM({"检索员TL": [json.dumps({"findings": ["关键证据内容"]})]})
    state = run_chain(tool_role_config(["web_search"]), llm, {"web_search": tool})
    assert tool.calls == ["配置链冒烟任务"]
    assert "关键证据内容" in llm.calls[0][-1]["content"]
    assert state["outputs"]["searcher"] == {"findings": ["关键证据内容"]}
    assert any("web_search" in line for line in state["execution_log"])


def test_tool_outside_whitelist_rejected():
    web = RecordingTool(name="web_search")  # 注册表里有，但不在角色白名单
    llm = ScriptLLM({"检索员TL": [json.dumps({"findings": []})]})
    state = run_chain(tool_role_config(["ghost_tool"]), llm, {"web_search": web})
    assert web.calls == []  # 白名单外零调用
    assert any("ghost_tool" in line and "拒绝" in line for line in state["execution_log"])


def test_role_without_tools_never_calls_registry():
    web = RecordingTool()
    llm = ScriptLLM({"检索员TL": [json.dumps({"findings": []})]})
    state = run_chain(tool_role_config([]), llm, {"web_search": web})
    assert web.calls == []
    assert state["outputs"]["searcher"] == {"findings": []}


def test_tool_guard_rejection_skips_call():
    tool = RecordingTool(validate_err="搜索关键词不能为空")
    llm = ScriptLLM({"检索员TL": [json.dumps({"findings": []})]})
    state = run_chain(tool_role_config(["web_search"]), llm, {"web_search": tool})
    assert tool.calls == []
    assert any("校验未过" in line for line in state["execution_log"])
    assert state["outputs"]["searcher"] == {"findings": []}


# ---------------------------------------------------------------------------
# chain 建图：线性链（2 角色 / 5 角色）
# ---------------------------------------------------------------------------


def test_two_role_chain_runs():
    llm = ScriptLLM({
        "规划员TWO": [json.dumps({"points": ["要点一", "要点二"]})],
        "撰写员TWO": [json.dumps({"draft": "成稿内容"})],
    })
    state = run_chain(two_role_config(), llm, {})
    assert state["outputs"]["planner"] == {"points": ["要点一", "要点二"]}
    assert state["outputs"]["writer"] == {"draft": "成稿内容"}
    assert len(state["execution_log"]) == 2
    assert state["execution_log"][0].startswith("[planner]")
    assert state["execution_log"][1].startswith("[writer]")
    assert state["revision_notes"] == []
    assert pipeline_iterations(state) == 0


def test_five_role_chain_runs_and_passes_all_previous_outputs():
    names = ["r1", "r2", "r3", "r4", "r5"]
    config = {
        "name": "five_role_test", "description": "测试：五角色链", "engine": "chain",
        "roles": [
            {"name": n, "system_prompt": f"你是{n}号角色，输出 JSON {{\"step\": \"{n}\"}}。",
             "tools": [], "output_fields": ["step"]}
            for n in names
        ],
    }
    llm = ScriptLLM({n: [json.dumps({"step": n})] for n in names})
    state = run_chain(config, llm, {})
    assert list(state["outputs"]) == names  # 任意数量角色按序全部执行
    order = [line.split("]")[0] + "]" for line in state["execution_log"]]
    assert order == [f"[{n}]" for n in names]
    # 最后一个角色收到前序所有节点输出
    last_payload = llm.calls[-1][-1]["content"]
    for n in names[:-1]:
        assert f'"step": "{n}"' in last_payload


# ---------------------------------------------------------------------------
# chain 建图：route 回跳环与强制前进
# ---------------------------------------------------------------------------


def test_route_backjump_once_then_pass():
    llm = ScriptLLM({
        "规划员RT": [json.dumps({"points": ["a"]})],
        "分析师RT": [json.dumps({"analysis": "第一版"}), json.dumps({"analysis": "修改版"})],
        "审核员RT": [json.dumps({"passed": False, "notes": "证据不足"}),
                     json.dumps({"passed": True, "notes": "ok"})],
    })
    state = run_chain(three_role_config(max_loops=2), llm, {})
    analyst_runs = sum(1 for c in llm.calls if "分析师RT" in c[0]["content"])
    reviewer_runs = sum(1 for c in llm.calls if "审核员RT" in c[0]["content"])
    assert analyst_runs == 2  # 打回重跑 1 次
    assert reviewer_runs == 2  # 首判 + 复核
    assert state["loops"] == {"reviewer": 2}
    assert pipeline_iterations(state) == 1
    assert state["revision_notes"] == []  # 最终通过后清空
    assert any("回跳" in line for line in state["execution_log"])
    # 打回意见确实流回了 analyst 第二次执行的输入
    second_analyst_payload = [c for c in llm.calls if "分析师RT" in c[0]["content"]][1][-1]["content"]
    assert "证据不足" in second_analyst_payload


def test_route_forced_forward_after_max_loops():
    llm = ScriptLLM({
        "规划员RT": [json.dumps({"points": ["a"]})],
        "分析师RT": [json.dumps({"analysis": "第1版"}), json.dumps({"analysis": "第2版"})],
        "审核员RT": [json.dumps({"passed": False, "notes": "不过"})],  # 队列耗尽后重复最后一项：永远不过
    })
    state = run_chain(three_role_config(max_loops=1), llm, {})
    reviewer_runs = sum(1 for c in llm.calls if "审核员RT" in c[0]["content"])
    assert reviewer_runs == 2  # 1 次回跳 + 1 次强制前进
    assert pipeline_iterations(state) == 1
    assert any("强制前进" in line for line in state["execution_log"])


def test_route_failure_notes_reach_revise_target():
    llm = ScriptLLM({
        "规划员RT": [json.dumps({"points": ["a"]})],
        "分析师RT": [json.dumps({"analysis": "结论"})],
        "审核员RT": [json.dumps({"passed": False, "notes": "不过"})],
    })
    state = run_chain(three_role_config(max_loops=1), llm, {})
    # 失败意见在回跳时流给 analyst；强制前进出口处清空过期意见
    analyst_payloads = [c[-1]["content"] for c in llm.calls if "分析师RT" in c[0]["content"]]
    assert "不过" in analyst_payloads[1]
    assert state["revision_notes"] == []
    assert state["outputs"]["reviewer"] == {"passed": False, "notes": "不过"}


def test_pipeline_iterations_helper():
    assert pipeline_iterations({}) == 0
    assert pipeline_iterations({"loops": {}}) == 0
    assert pipeline_iterations({"loops": {"reviewer": 1}}) == 0
    assert pipeline_iterations({"loops": {"reviewer": 3}}) == 2


# ---------------------------------------------------------------------------
# research 引擎委托内置三角色图
# ---------------------------------------------------------------------------


def test_research_engine_delegates_to_builtin_graph():
    config = load_pipeline("research")
    llm = ScriptLLM({
        "任务拆解规划员": ["不是JSON"],  # 走 fallback_plan（3 子问题）
        "报告撰写员": [json.dumps({"sub_question_id": "sq_1", "content": "依据证据的论述。"})],
        "事实核对员": [json.dumps([])],
    })
    graph = build_pipeline(config, build_ctx(llm, {}))
    state = asyncio.run(graph.ainvoke(make_initial_state("研究引擎委托测试", max_iterations=1)))
    assert len(state["plan"].sub_questions) == 3
    assert state["report"]
    assert state["iterations"] == 1


# ---------------------------------------------------------------------------
# API 端点（自建 app，patch llm 单例与工具，零网络）
# ---------------------------------------------------------------------------


def _build_client():
    from app.api import research as api_research

    app = FastAPI()
    app.include_router(api_research.router)
    return app, api_research


def test_api_list_pipelines():
    app, _ = _build_client()
    resp = TestClient(app).get("/api/pipelines")
    assert resp.status_code == 200
    names = {item["name"] for item in resp.json()}
    assert {"research", "diy_example"} <= names


def test_api_run_diy_chain_endpoint(monkeypatch):
    app, api_research = _build_client()
    llm = ScriptLLM({
        "规划员": [json.dumps({"points": ["p1", "p2"]})],
        "检索员": [json.dumps({"findings": ["f1"]})],
        "分析师": [json.dumps({"analysis": "分析结论", "confidence": 0.8})],
        "审核员": [json.dumps({"passed": True, "notes": "ok"})],
    })
    monkeypatch.setattr("app.services.llm._async_llm_service", llm)
    monkeypatch.setattr(api_research, "build_default_tools",
                        lambda: {"web_search": RecordingTool(), "search_docs": RecordingTool(name="search_docs")})
    resp = TestClient(app).post("/api/pipelines/diy_example/run", json={"task": "DIY 链端点冒烟"})
    assert resp.status_code == 200
    body = resp.json()
    # report = 最后一个节点（reviewer）outputs 的文本化拼接
    assert body["report"] == "true\nok"
    assert body["outputs"]["analyst"] == {"analysis": "分析结论", "confidence": 0.8}
    assert set(body["outputs"]) == {"planner", "searcher", "analyst", "reviewer"}
    assert body["iterations"] == 0
    assert body["execution_log"]


def test_api_run_research_pipeline_endpoint(monkeypatch):
    app, api_research = _build_client()
    llm = ScriptLLM({
        "任务拆解规划员": ["不是JSON"],
        "报告撰写员": [json.dumps({"sub_question_id": "sq_1", "content": "依据证据的论述。"})],
        "事实核对员": [json.dumps([])],
    })
    monkeypatch.setattr("app.services.llm._async_llm_service", llm)
    monkeypatch.setattr(api_research, "build_default_tools",
                        lambda: {"web_search": RecordingTool()})
    resp = TestClient(app).post("/api/pipelines/research/run",
                                json={"task": "研究引擎端点冒烟", "max_iterations": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["report"]
    assert len(body["plan"]["sub_questions"]) == 3
    assert body["iterations"] == 1
    assert body["passed"] is True
    assert body["sections"]


def test_api_run_unknown_pipeline_returns_404():
    app, _ = _build_client()
    resp = TestClient(app).post("/api/pipelines/ghost/run", json={"task": "x"})
    assert resp.status_code == 404


def test_api_run_invalid_body_returns_422():
    app, _ = _build_client()
    resp = TestClient(app).post("/api/pipelines/diy_example/run", json={"nope": 1})
    assert resp.status_code == 422


def test_api_run_illegal_pipeline_name_returns_400():
    # 路径穿越形态经 unit 层 load_pipeline 已拒（含 %2F 形态 URL 匹配不到路由，会 404），
    # 这里用路由可匹配的非法名验证 ValueError -> 400 映射
    app, _ = _build_client()
    resp = TestClient(app).post("/api/pipelines/BadName/run", json={"task": "x"})
    assert resp.status_code == 400


def test_api_existing_full_endpoint_unaffected():
    """既有 /full 行为不变：同 router 内共存，行为与改造前一致。"""
    app, api_research = _build_client()
    llm = ScriptLLM({
        "任务拆解规划员": ["不是JSON"],
        "报告撰写员": [json.dumps({"sub_question_id": "sq_1", "content": "端点回归结论。"})],
        "事实核对员": [json.dumps([])],
    })
    mp = pytest.MonkeyPatch()
    mp.setattr("app.services.llm._async_llm_service", llm)
    mp.setattr(api_research, "build_default_tools", lambda: {"web_search": RecordingTool()})
    try:
        resp = TestClient(app).post("/api/research/full", json={"task": "回归主题", "max_iterations": 1})
    finally:
        mp.undo()
    assert resp.status_code == 200
    assert resp.json()["report"]
