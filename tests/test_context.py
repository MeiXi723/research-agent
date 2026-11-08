"""上下文引擎 v2 测试：原子组不拆散、单轮 ReAct 压缩触发、tool_calls 计数。"""
import json

from app.services.context import estimate_tokens, manage_context


def _summarizer_ok(messages):
    return "历史摘要"


def _react_sequence(steps: int, filler: str = "观察内容" * 30) -> list[dict]:
    """构造单轮 ReAct 序列：system + user + (assistant(tool_calls)+tool)*steps。"""
    msgs = [{"role": "system", "content": "sys"}]
    msgs.append({"role": "user", "content": "初始问题"})
    for i in range(steps):
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "search_docs",
                                         "arguments": json.dumps({"query": f"查询{i}"})}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": filler + str(i)})
    return msgs


def test_estimate_tokens_counts_tool_calls():
    # P2-1：assistant 的 tool_calls JSON 参数必须计入预算（v1 只数 content）
    base = [{"role": "assistant", "content": None}]
    with_calls = [{"role": "assistant", "content": None,
                  "tool_calls": [{"function": {"arguments": json.dumps({"q": "长" * 200})}}]}]
    diff = estimate_tokens(with_calls) - estimate_tokens(base)
    assert diff > 100, f"tool_calls 参数未被计入（diff={diff}）"


def test_single_react_loop_compression_triggers():
    # P1-1 核心回归：单轮 ReAct（仅 1 个 user）超预算也必须压缩
    msgs = _react_sequence(steps=6)
    events = []
    result = manage_context(msgs, max_tokens=50, window_messages=6,
                            summarizer=_summarizer_ok, events=events)
    assert estimate_tokens(result) < estimate_tokens(msgs)
    assert result[0]["role"] == "system"
    assert any("历史摘要" in m.get("content", "") for m in result if m["role"] == "user")
    assert events and events[0]["compressed_groups"] >= 1


def test_atomic_groups_never_split():
    # 压缩后每条 tool 消息前必须仍有带 tool_calls 的 assistant——原子组不拆对
    msgs = _react_sequence(steps=6)
    result = manage_context(msgs, max_tokens=50, window_messages=4, summarizer=_summarizer_ok)
    for i, m in enumerate(result):
        if m.get("role") == "tool":
            prev = result[i - 1]
            assert prev.get("role") == "assistant" and prev.get("tool_calls"), \
                f"tool 消息与 assistant 被拆散 @ {i}"


def test_keep_recent_window_messages():
    # window_messages 语义：压缩后保留的最近消息数 ≥ 该值（摘要与 system 不计入）
    msgs = _react_sequence(steps=8)
    result = manage_context(msgs, max_tokens=50, window_messages=6, summarizer=_summarizer_ok)
    non_summary = [
        m for m in result
        if m.get("role") != "system"
        and not str(m.get("content", "")).startswith("[历史对话摘要]")
    ]
    assert len(non_summary) >= 6


def test_within_budget_returns_unchanged():
    msgs = _react_sequence(steps=2)
    assert manage_context(msgs, max_tokens=10000, window_messages=14,
                          summarizer=_summarizer_ok) == msgs


def test_single_group_over_budget_returns_unchanged():
    # 只剩一组且超预算：不能压，原样返回（压缩需要丢弃整组）
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "长" * 500}]
    assert manage_context(msgs, max_tokens=10, window_messages=4,
                          summarizer=_summarizer_ok) == msgs


def test_multi_turn_history_compresses():
    # 跨轮 history（多 user）场景依然工作
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"问题{i} " + "内容" * 50})
        msgs.append({"role": "assistant", "content": f"答案{i} " + "内容" * 50})
    result = manage_context(msgs, max_tokens=100, window_messages=6, summarizer=_summarizer_ok)
    assert result[0]["role"] == "system"
    assert any("历史摘要" in m.get("content", "") for m in result)
    assert estimate_tokens(result) < estimate_tokens(msgs)
