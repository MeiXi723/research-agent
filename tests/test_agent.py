from unittest.mock import MagicMock, patch

from app.services.agent import run_agent
from app.services.tracer import Tracer


def _make_msg(content=None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def _make_tool_call(tool_id, name, arguments):
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@patch("app.services.agent._llm_service")
def test_run_agent_direct_answer(mock_llm):
    mock_llm.chat_with_tools.return_value = _make_msg(content="答案是 42")

    answer, steps, sources = run_agent("1+1 等于多少")

    assert answer == "答案是 42"
    assert steps == []
    assert sources == []


@patch("app.services.agent._llm_service")
def test_run_agent_with_tool_call(mock_llm):
    mock_llm.chat_with_tools.side_effect = [
        _make_msg(tool_calls=[_make_tool_call("call_1", "calculator", '{"expression": "2+3"}')]),
        _make_msg(content="结果是 5"),
    ]

    answer, steps, sources = run_agent("2+3 等于多少")

    assert answer == "结果是 5"
    assert len(steps) == 1
    assert steps[0].tool == "calculator"
    assert '"result": 5' in steps[0].tool_result


@patch("app.services.agent._llm_service")
def test_run_agent_unknown_tool(mock_llm):
    mock_llm.chat_with_tools.side_effect = [
        _make_msg(tool_calls=[_make_tool_call("call_1", "not_exist", '{}')]),
        _make_msg(content="无法执行"),
    ]

    answer, steps, sources = run_agent("测试")

    assert answer == "无法执行"
    assert steps[0].tool_result == '{"error": "未知工具: not_exist"}'


@patch("app.services.agent._llm_service")
def test_run_agent_max_steps(mock_llm):
    mock_llm.chat_with_tools.side_effect = [
        _make_msg(tool_calls=[_make_tool_call("call_1", "calculator", '{"expression": "1+1"}')]),
    ] * 100

    answer, steps, sources = run_agent("测试", max_steps=3)

    assert answer == "达到最大步数限制，请简化问题"


@patch("app.services.agent._llm_service")
def test_run_agent_with_tracer(mock_llm, tmp_path):
    mock_llm.chat_with_tools.return_value = _make_msg(content="答案")

    tracer = Tracer(str(tmp_path))
    run_agent("问题", tracer=tracer)

    with open(tracer.trace_path, "r", encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    assert len(lines) == 1
    import json
    record = json.loads(lines[0])
    assert record["action"] == "answer"
    assert record["success"] is True
