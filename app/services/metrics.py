# =============================================================================
# metrics.py — 运行级指标聚合（v2.0 观测层）
# =============================================================================
# 口径：
#   token   —— LLM 层捕获的 OpenAI resp.usage 真实值（不是字符数估算）；
#              AI_MOCK 回放无 usage，token 字段为 0 并在 mock 标记
#   工具    —— 按 tool 名聚合 调用次数 / 成功 / 失败（error 结果计失败）
#   上下文  —— manage_context 的压缩事件（压了几条消息、前后 token）
# =============================================================================


def new_metrics() -> dict:
    # 路由字段（intent/route_via）归 main 层所有——agent 的 sink update 不允许
    # 覆盖它们（并发冒烟实测：sink.update 会把路由字段冲成 None）
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_captured": False,  # False = 本次没有拿到真实 usage（mock/异常路径）
        "tool_calls_total": 0,
        "tool_calls": {},  # name -> {"count": n, "success": n, "fail": n}
        "context_events": [],
        "steps": 0,
    }


def record_usage(metrics: dict, usage: dict | None) -> None:
    if not usage:
        return
    metrics["llm_calls"] += 1
    metrics["prompt_tokens"] += usage.get("prompt_tokens", 0)
    metrics["completion_tokens"] += usage.get("completion_tokens", 0)
    metrics["total_tokens"] += usage.get("total_tokens", 0)
    metrics["usage_captured"] = True


def record_tool(metrics: dict, name: str, ok: bool) -> None:
    stats = metrics["tool_calls"].setdefault(name, {"count": 0, "success": 0, "fail": 0})
    stats["count"] += 1
    stats["success" if ok else "fail"] += 1
    metrics["tool_calls_total"] += 1


def summarize(metrics: dict) -> dict:
    out = dict(metrics)
    total = metrics["tool_calls_total"]
    fail = sum(s["fail"] for s in metrics["tool_calls"].values())
    out["tool_success_rate"] = round((total - fail) / total, 3) if total else None
    return out
