"""工具执行器测试（P1-3 + P2-2/3/4）：超时硬闸、瞬态重试、typed 校验、结构化截断。"""
import json
import time
from unittest.mock import patch


from app.services.agent import _execute_tool_call, _truncate_result
from app.tools.base import BaseTool


class SimpleTool(BaseTool):
    name = "simple"
    description = "测试工具"

    def describe(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"},
                               "top_n": {"type": "integer"}},
                "required": ["query"],
            },
        }

    def validate(self, **kwargs):
        if kwargs.get("query") == "语义拒绝":
            return "语义校验不通过"
        return None

    def call(self, **kwargs):
        return {"ok": True, "query": kwargs.get("query")}


class FlakyTool(BaseTool):
    """前 N 次抛 ConnectionError（瞬态），之后成功。"""

    name = "flaky"
    description = "瞬态失败工具"
    fails_left = 1

    def describe(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def call(self, **kwargs):
        if FlakyTool.fails_left > 0:
            FlakyTool.fails_left -= 1
            raise ConnectionError("网络抖动")
        return {"ok": True}


class HangingTool(BaseTool):
    name = "hanging"
    description = "挂起工具"

    def describe(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def call(self, **kwargs):
        time.sleep(10)  # 超过测试用超时硬闸
        return {"ok": True}


# ── validate_typed（P2-4）──

def test_validate_typed_missing_required():
    tool = SimpleTool()
    err = tool.validate_typed({"top_n": 3})
    assert "缺少必填参数" in err and "query" in err


def test_validate_typed_unknown_param():
    tool = SimpleTool()
    err = tool.validate_typed({"query": "x", "evil": "注入尝试"})
    assert "未知参数" in err and "evil" in err


def test_validate_typed_type_mismatch():
    tool = SimpleTool()
    err = tool.validate_typed({"query": "x", "top_n": "不是数字"})
    assert "integer" in err


def test_validate_typed_bool_not_integer():
    tool = SimpleTool()
    # bool 是 int 子类——显式 integer 参数收到 bool 必须报错
    err = tool.validate_typed({"query": "x", "top_n": True})
    assert "boolean" in err


def test_validate_typed_pass():
    tool = SimpleTool()
    assert tool.validate_typed({"query": "x", "top_n": 3}) is None


def test_validate_semantic_layer_still_runs():
    # 语义校验（子类 validate）与 typed 校验叠加：语义拒绝要生效
    tool = SimpleTool()
    assert tool.validate(query="语义拒绝") == "语义校验不通过"


# ── 超时硬闸与瞬态重试（P1-3）──

def test_hanging_tool_hits_timeout_hard_gate():
    tool = HangingTool()
    with patch("app.services.agent.TOOL_EXEC_TIMEOUT", 0.2):
        result = _execute_tool_call(tool, "hanging", {})
    assert result["error"].startswith("工具执行超时")
    assert result["retryable"] is True


def test_transient_failure_auto_retries_once():
    tool = FlakyTool()
    result = _execute_tool_call(tool, "flaky", {})
    assert result == {"ok": True}  # 瞬态失败自动重试后成功
    assert FlakyTool.fails_left == 0


def test_permanent_failure_no_retry():
    class BrokenTool(BaseTool):
        name = "broken"
        description = "永久失败"
        def describe(self):
            return {"name": "broken", "description": "", "parameters": {"type": "object", "properties": {}}}
        def call(self, **kwargs):
            raise ValueError("永久性逻辑错误")
    result = _execute_tool_call(BrokenTool(), "broken", {})
    assert "ValueError" in result["error"]
    assert result["retryable"] is False  # 非瞬态错误不重试（不打无意义补枪）


# ── 结构化截断（P2-3）──

def test_short_result_untouched():
    obj = {"ok": True}
    assert _truncate_result(obj) == json.dumps(obj, ensure_ascii=False)


def test_long_result_truncated_to_valid_json():
    obj = {"data": "长" * 1000, "key2": 2}
    out = _truncate_result(obj, limit=500)
    assert len(out) <= 550
    parsed = json.loads(out)  # 合法 JSON——v1 切片产出的是残片
    assert parsed["truncated"] is True
    assert "data" in parsed["keys"]


def test_long_list_result_truncated():
    out = _truncate_result(["x" * 300] * 5, limit=500)
    parsed = json.loads(out)
    assert parsed["truncated"] is True


def test_none_result():
    # None 序列化为 "null"（合法 JSON，长度 4 ≤ 500 原样保留）
    assert _truncate_result(None) == "null"
