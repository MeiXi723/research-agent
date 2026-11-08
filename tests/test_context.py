from app.services import context as ctx
from app.services.context import estimate_tokens, manage_context


def _summarizer_ok(messages):
    return "历史摘要"


def test_estimate_tokens():
    # 双口径：tiktoken 可用（BPE 词表已在缓存/可下载）→ 真编码 10；
    # 词表不可达的离线环境自动降级为字符数//2 → 20。两种实现都合法。
    msgs = [{"content": "你好" * 10}]
    if ctx._get_encoding() is not None:
        assert estimate_tokens(msgs) == 10
    else:
        assert estimate_tokens(msgs) == 20


def test_manage_context_within_budget():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "答案1"},
    ]
    result = manage_context(messages, max_tokens=1000, window_turns=5, summarizer=_summarizer_ok)
    assert result == messages


def test_manage_context_compresses_old():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧答案"},
        {"role": "user", "content": "新问题"},
    ]
    result = manage_context(messages, max_tokens=1, window_turns=1, summarizer=_summarizer_ok)
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert "历史摘要" in result[1]["content"]
    assert result[2]["content"] == "新问题"


def test_manage_context_no_system_returns_unchanged():
    messages = [{"role": "user", "content": "问题"}]
    result = manage_context(messages, max_tokens=1, window_turns=1, summarizer=_summarizer_ok)
    assert result == messages
