# =============================================================================
# context.py — 上下文引擎 v2（原子组压缩）
# =============================================================================
# v1 缺陷（defect-audit P1-1）：压缩以 user 消息为锚，而单轮 ReAct 循环里只有
# 1 个 user 消息 → MAX_CONTEXT_TOKENS 在长任务中永不触发，上下文无限膨胀。
#
# v2 核心变化：
# 1. 按「原子组」切分——OpenAI API 要求 tool 消息紧跟带 tool_calls 的 assistant，
#    压缩若拆开这对消息会直接 400。组规则：user/assistant 开新组，tool 归入
#    当前组。压缩以组为最小单位，永不拆对。
# 2. 触发只看 token 预算——单轮内同样生效。
# 3. 保留语义：从尾部按组累计消息数，保留最近 window_messages 条以上。
# =============================================================================

import json
from typing import Callable


def _get_encoding():
    # 延迟导入包装：测试通过 patch 本函数验证双口径
    from app.services._encoding import get_encoding
    return get_encoding()


def estimate_tokens(messages: list[dict]) -> int:
    """真实 token 计数（tiktoken cl100k_base）；不可用时降级为字符数//2。

    v2：assistant 消息的 tool_calls（JSON 参数可达数百 token）一并计入——
    v1 只数 content，预算系统性低估。
    """
    enc = _get_encoding()
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += _count(content, enc)
        tool_calls = m.get("tool_calls")
        if tool_calls:
            total += _count(json.dumps(tool_calls, ensure_ascii=False), enc)
    return total


def _count(text: str, enc) -> int:
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 2)


def _split_groups(messages: list[dict]) -> list[list[dict]]:
    """原子组切分：user/assistant 开新组，tool 等附属消息归入当前组。

    messages[0] 必须是 system（调用方保证），不参与分组。
    """
    groups: list[list[dict]] = []
    for m in messages[1:]:
        role = m.get("role")
        if role in ("user", "assistant") or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return groups


def manage_context(
    messages: list[dict],
    max_tokens: int,
    window_messages: int,
    summarizer: Callable[[list[dict]], str],
    events: list[dict] | None = None,
) -> list[dict]:
    """原子组滑窗压缩。events 传入时记录压缩事件（可观测性）。"""
    before = estimate_tokens(messages)
    if before <= max_tokens:
        return messages
    if not messages or messages[0].get("role") != "system":
        return messages

    groups = _split_groups(messages)
    if len(groups) <= 1:
        # 只剩一组且超预算：不能压（压缩需要丢弃整组），原样返回
        return messages

    # 从尾部按组保留，直到累计消息数达到 window_messages
    keep_n = 0
    acc = 0
    for g in reversed(groups):
        acc += len(g)
        keep_n += 1
        if acc >= max(1, window_messages):
            break
    keep_n = max(1, keep_n)

    if len(groups) <= keep_n:
        return messages

    dropped = groups[:-keep_n]
    kept = groups[-keep_n:]

    summary = summarizer([m for g in dropped for m in g])
    compressed = (
        [messages[0]]
        + [{"role": "user", "content": f"[历史对话摘要] {summary}"}]
        + [m for g in kept for m in g]
    )

    if events is not None:
        events.append({
            "compressed_groups": len(dropped),
            "compressed_messages": sum(len(g) for g in dropped),
            "kept_messages": sum(len(g) for g in kept),
            "tokens_before": before,
            "tokens_after": estimate_tokens(compressed),
        })
    return compressed
