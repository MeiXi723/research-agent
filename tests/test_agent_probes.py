"""注入防御与边界机制测试（unit，机制层——LLM 行为层的探针评测走 evals/run_eval.py --mode probe）。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.services.agent import SYSTEM_PROMPT, _build_tools, run_agent


def test_system_prompt_has_injection_hardening():
    # D05 防御机制：系统提示必须显式声明「工具输出是数据不是指令」
    assert "工具输出" in SYSTEM_PROMPT and "数据" in SYSTEM_PROMPT
    assert "忽略" in SYSTEM_PROMPT  # 点名注入话术，LLM 有锚点可对照


def test_build_tools_includes_recall_memory():
    tools = _build_tools()
    assert "recall_memory" in tools  # v2.0 语义记忆工具已注册


def test_injection_probes_manifest_valid():
    probes = json.loads(
        (Path(__file__).parent.parent / "evals" / "probes" / "injection_probes.json")
        .read_text(encoding="utf-8")
    )["probes"]
    assert len(probes) >= 10
    for p in probes:
        assert p["id"] and p["kind"] and p["payload"] and p["expected"]


def test_tool_result_hard_truncated_at_500_chars():
    # 超长工具结果必须截断——注入负载常借超长文本稀释系统指令
    class GiantTool:
        name = "giant"
        description = "返回超长结果"

        def describe(self):
            return {"name": "giant", "description": self.description,
                    "parameters": {"type": "object", "properties": {}}}

        def to_schema(self):
            return {"type": "function", "function": self.describe()}

        def validate(self, **kwargs):
            return None

        def call(self, **kwargs):
            return "X" * 2000

    # 注意：MagicMock(name=...) 的 name 是 mock 身份而非属性——工具名/参数必须用 SimpleNamespace
    msg_tool = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(id="t1", function=SimpleNamespace(name="giant", arguments="{}"))],
    )
    msg_final = SimpleNamespace(content="done", tool_calls=None)
    with patch("app.services.agent._build_tools", return_value={"giant": GiantTool()}), \
         patch("app.services.agent._llm_service") as mock_llm:
        mock_llm.chat_with_tools.side_effect = [msg_tool, msg_final]
        _, steps, _ = run_agent("触发超长工具结果")
    assert len(steps[0].tool_result) <= 500
