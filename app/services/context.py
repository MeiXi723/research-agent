from typing import Callable


def estimate_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
    return total


def manage_context(
    messages: list[dict],
    max_tokens: int,
    window_turns: int,
    summarizer: Callable[[list[dict]], str],
) -> list[dict]:
    if estimate_tokens(messages) <= max_tokens:
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
    return [messages[0], {"role": "user", "content": f"[历史对话摘要] {summary}"}] + kept
