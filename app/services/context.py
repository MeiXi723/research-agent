from typing import Callable

_encoding = None
_encoding_failed = False


def _get_encoding():
    """tiktoken cl100k_base 惰性单例；加载失败置标志不再重试。"""
    global _encoding, _encoding_failed
    if _encoding_failed:
        return None
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_failed = True
            return None
    return _encoding


def estimate_tokens(messages: list[dict]) -> int:
    """真实 token 计数（tiktoken cl100k_base）；不可用时降级为字符数//2。

    v1 用 len(content) 字符数冒充 token，中文下 1 字≈1 token、英文 1 token≈4 字符，
    同一预算对两种语言语义完全不同——预算判断必须用真计数。
    """
    enc = _get_encoding()
    total = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if enc is not None:
            total += len(enc.encode(content))
        else:
            total += len(content) // 2
    return total


def manage_context(
    messages: list[dict],
    max_tokens: int,
    window_turns: int,
    summarizer: Callable[[list[dict]], str],
    events: list[dict] | None = None,
) -> list[dict]:
    """滑窗 + 摘要压缩。events 传入时记录压缩事件（可观测性：压了什么、剩多少）。"""
    before = estimate_tokens(messages)
    if before <= max_tokens:
        return messages

    if not messages or messages[0].get("role") != "system":
        return messages

    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= window_turns:
        return messages

    keep_from = user_indices[len(user_indices) - window_turns]
    old = messages[1:keep_from]
    kept = messages[keep_from:]

    summary = summarizer(old)
    compressed = [messages[0], {"role": "user", "content": f"[历史对话摘要] {summary}"}] + kept

    if events is not None:
        events.append({
            "compressed_messages": len(old),
            "kept_turns": window_turns,
            "tokens_before": before,
            "tokens_after": estimate_tokens(compressed),
        })
    return compressed
