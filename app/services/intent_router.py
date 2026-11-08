# =============================================================================
# intent_router.py — 三层漏斗意图路由（v2.0）
# =============================================================================
# 统一入口 /ask 的分流漏斗（每层命中即出漏斗，via 字段记录命中层）：
#
#   L1 规则/正则   —— 零成本毫秒级：寒暄、计算表达式、显式深研关键词
#   L2 语义识别    —— 热门话题样本库向量匹配（fastembed）：改写/口语化也能命中，
#                     目标与 L1 合计覆盖 90% 热门话题（evals/router_set.json 实测口径）
#   L3 LLM 兜底    —— 剩余长尾全部交 LLM JSON 分类；LLM 也失败 → tool_research
#                     （fail-safe 到最稳路径）
#
# 覆盖率是设计契约不是口号：`python evals/run_eval.py --mode router` 按层出命中率，
# 样本库/规则扩充后必须复跑确认 L1+L2 ≥ 90%。
# =============================================================================

import json
import threading
from pathlib import Path

import numpy as np

import app.services.llm as llm_module

INTENTS = ("chat", "tool_research", "deep_research")

SAMPLES_PATH = Path(__file__).parent.parent.parent / "evals" / "intent_samples.json"

# L2 语义命中双档阈值（依真实 bge 相似度分布定标：改写题 top1 分布 0.48~0.77）
# 高档：直接采信；低档：意图采信但标 low_confidence（可观测、评测分层）；
# margin 双闸隔离并列歧义，其余落 L3。
SEMANTIC_TOP1 = 0.72
SEMANTIC_MARGIN = 0.05
SEMANTIC_LOW_TOP1 = 0.52
SEMANTIC_LOW_MARGIN = 0.04

_DEEP_HINTS = ("研究报告", "深度研究", "全面分析", "调研报告", "综述", "in-depth", "research report")
_CHAT_HINTS = ("你好", "你是谁", "谢谢", "再见", "晚安", "hi", "hello", "thanks")
_CALC_RE = None  # 惰性编译，见 _rule_prefilter


# ---------------------------------------------------------------------------
# L1 规则/正则
# ---------------------------------------------------------------------------

def _rule_prefilter(question: str) -> dict | None:
    import re
    q = question.strip()
    lowered = q.lower()

    if any(h in q or h in lowered for h in _DEEP_HINTS):
        return {"intent": "deep_research", "reason": "命中深度研究关键词", "via": "rule", "layer": 1}
    # 计算表达式：整串只由算术字符组成且至少含一个运算符
    # （"3个方向"含汉字必不命中；含减号的长算式 "3.6 / 1.2 - 0.7" 也能通过）
    if re.fullmatch(r"[\d\s.,+\-*/×÷%^()=？?]+", q) and any(
        op in q for op in ("+", "-", "*", "/", "×", "÷", "%", "^")
    ):
        return {"intent": "tool_research", "reason": "命中计算表达式", "via": "rule", "layer": 1}
    if len(q) <= 12 and any(h in lowered for h in _CHAT_HINTS):
        return {"intent": "chat", "reason": "命中寒暄短语", "via": "rule", "layer": 1}
    return None


# ---------------------------------------------------------------------------
# L2 语义识别：热门话题样本库
# ---------------------------------------------------------------------------

_samples_cache: dict | None = None
_samples_lock = threading.Lock()


def _load_samples() -> list[dict]:
    global _samples_cache
    with _samples_lock:
        if _samples_cache is None:
            data = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))
            _samples_cache = [s for s in data if s.get("intent") in INTENTS and s.get("text")]
        return _samples_cache


def _semantic_match(question: str, embed_fn) -> dict | None:
    if embed_fn is None:
        return None
    samples = _load_samples()
    if not samples:
        return None
    q_vec = np.array([embed_fn(question)], dtype="float32")
    s_vecs = np.array([embed_fn(s["text"]) for s in samples], dtype="float32")
    # 归一化后点积 = 余弦相似度
    q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
    s_norm = s_vecs / (np.linalg.norm(s_vecs, axis=1, keepdims=True) + 1e-9)
    sims = (s_norm @ q_norm.T).flatten()
    order = np.argsort(sims)[::-1]
    top1, top2 = float(sims[order[0]]), float(sims[order[1]]) if len(order) > 1 else 0.0
    best = samples[int(order[0])]

    if top1 >= SEMANTIC_TOP1 and top1 - top2 >= SEMANTIC_MARGIN:
        return {
            "intent": best["intent"],
            "reason": f"语义命中样本「{best['text'][:24]}…」(sim={top1:.2f})",
            "via": "semantic", "layer": 2,
            "confidence": round(top1, 3),
        }
    if top1 >= SEMANTIC_LOW_TOP1 and top1 - top2 >= SEMANTIC_LOW_MARGIN:
        return {
            "intent": best["intent"],
            "reason": f"语义低置信命中「{best['text'][:24]}…」(sim={top1:.2f})",
            "via": "semantic", "layer": 2,
            "confidence": round(top1, 3),
            "low_confidence": True,
        }
    return None


# ---------------------------------------------------------------------------
# L3 LLM 兜底
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _llm_classify(question: str, history_len: int) -> dict:
    llm = llm_module._llm_service
    if llm is None:
        return {"intent": "tool_research", "reason": "LLM 未就绪，回退默认路径", "via": "fallback", "layer": 3}

    prompt = (
        "把用户问题分类到三选一：\n"
        'chat = 寒暄/问候/无需任何资料的直接对话\n'
        'tool_research = 需要查文档/搜索/计算等工具才能回答的具体问题\n'
        'deep_research = 用户明确要一份完整研究报告/多角度深度分析\n'
        f"（对话历史轮数 {history_len}）\n"
        '只输出 JSON：{"intent": "...", "reason": "一句话"}\n\n'
        f"用户问题：{question}"
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        data = _extract_json(raw or "")
        intent = (data or {}).get("intent")
        if intent in INTENTS:
            return {"intent": intent, "reason": (data or {}).get("reason", ""), "via": "llm", "layer": 3}
        return {"intent": "tool_research", "reason": f"LLM 分类越界({intent})，回退默认", "via": "fallback", "layer": 3}
    except Exception as e:
        reason = f"LLM 分类异常({type(e).__name__})，回退默认"
        return {"intent": "tool_research", "reason": reason, "via": "fallback", "layer": 3}


# ---------------------------------------------------------------------------
# 漏斗入口
# ---------------------------------------------------------------------------

def classify_intent(question: str, history_len: int = 0, embed_fn=None) -> dict:
    """三层漏斗：L1 规则 → L2 语义 → L3 LLM。返回 {intent, reason, via, layer[, confidence]}。"""
    rule = _rule_prefilter(question)
    if rule:
        return rule

    if embed_fn is None:
        llm = llm_module._llm_service
        embed_fn = getattr(llm, "embed", None)
    semantic = _semantic_match(question, embed_fn)
    if semantic:
        return semantic

    return _llm_classify(question, history_len)


def reload_samples() -> None:
    """样本库更新后强制重载（测试与运维用）。"""
    global _samples_cache
    with _samples_lock:
        _samples_cache = None
